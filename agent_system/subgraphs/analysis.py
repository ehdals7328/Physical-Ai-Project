# agent_system/subgraphs/analysis.py
import json
from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

from state import ShipAgentState
from tools.weather_mcp_client import get_realtime_weather_from_mcp
from tools.retriever_tools import wps_retriever_tool, safety_retriever_tool

# 1. 대화형 에이전트가 언제든 자유롭게 꺼내 쓸 수 있는 도구 정의
@tool
def get_weather_tool(location: str) -> str:
    """특정 지역(예: 울산, 거제, 부산, 서울 등)의 실시간 기온, 풍속, 습도 날씨 정보를 조회할 때 호출하세요."""
    data = get_realtime_weather_from_mcp(location)
    return json.dumps(data, ensure_ascii=False)

@tool
def search_shipyard_docs_tool(query: str) -> str:
    """조선소 용접 규격(WPS), 전류/전압 조건, 결함 대책, 크레인/안전 규정을 찾아볼 때 호출하세요."""
    wps_res = wps_retriever_tool.invoke({"query": query})
    safe_res = safety_retriever_tool.invoke({"query": query})
    return f"[WPS 규격]: {wps_res}\n[안전 규정]: {safe_res}"

conversational_tools = [get_weather_tool, search_shipyard_docs_tool]
tool_map = {t.name: t for t in conversational_tools}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
llm_with_tools = llm.bind_tools(conversational_tools)


# 2. 대화 후 상태 정리 스키마
class ConversationStateUpdate(BaseModel):
    task_type: Literal["weld_order", "heavy_plan", "general_qa", "chitchat"] = Field(
        description="현재 맥락의 주된 성격 (단순 대화나 날씨 질문 등은 chitchat 또는 general_qa)"
    )
    updated_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="대화 중 파악되거나 수정된 작업 조건 (material, thickness, location, equipment 등)"
    )
    trigger_document_generation: bool = Field(
        default=False,
        description="오직 작업지시서(weld_order)나 인양계획서(heavy_plan)의 필수 조건이 모두 모였고, 사용자에게 조건을 브리핑한 뒤 최종 발행 동의('응 뽑아줘', '진행해')까지 받았을 때만 True로 설정. 그 외 모든 대화/질문/날씨조회는 False."
    )


def interactive_agent_node(state: ShipAgentState) -> dict:
    """도구(날씨, 규격 검색)를 자유롭게 사용하며 사용자와 자연스럽게 대화하는 에이전트 노드"""
    messages = list(state.get("messages", []))
    current_params = state.get("extracted_params") or {}

    sys_msg = SystemMessage(content=(
        "당신은 조선소 현장을 지원하는 유연하고 유능한 AI 파트너입니다.\n"
        f"현재까지 파악된 작업 조건: {current_params}\n\n"
        "[대화 및 행동 원칙]\n"
        "1. 일상적인 인사나 안부에는 친구나 동료처럼 자연스럽게 대화를 이어가세요.\n"
        "2. 사용자가 날씨를 물어보면 거부하지 말고 `get_weather_tool`을 호출해 실시간 날씨를 알려주세요. (만약 어느 지역인지 모르면 어느 지역 날씨를 알려줄지 물어보거나 이전 작업 지역을 활용하세요.)\n"
        "3. 용접 기술이나 안전 기준을 물어보면 `search_shipyard_docs_tool`로 찾아보고 친절히 설명해 주세요.\n"
        "4. 사용자가 '용접 작업지시서'나 '인양 계획서' 작성을 원할 경우:\n"
        "   - 필요한 정보(예: 모재 강종, 두께, 작업 지역 / 또는 장비명, 작업 지역)가 부족하면 대화하면서 자연스럽게 물어보세요.\n"
        "   - 정보가 다 모이면 바로 문서를 발행하지 말고, 수집된 조건을 한 번 정리해서 보여준 뒤 '이대로 작업지시서를 발행할까요, 아니면 수정할 부분이 있나요?'라고 먼저 물어보세요.\n"
        "   - 사용자가 최종 동의하면 '네, 해당 조건으로 표준 문서를 발행하겠습니다.'라고 응답하세요."
    ))

    # 1차 LLM 추론 (필요시 스스로 도구 호출 결정)
    ai_msg = llm_with_tools.invoke([sys_msg] + messages)

    # 만약 LLM이 날씨나 규격 검색 도구를 호출하기로 결정했다면 즉시 실행
    tool_History = [sys_msg] + messages + [ai_msg]
    if ai_msg.tool_calls:
        for tc in ai_msg.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            if tool_fn:
                tool_output = tool_fn.invoke(tc["args"])
                tool_History.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"]))
        # 도구 결과를 보고 최종 자연어 답변 생성
        final_ai_msg = llm.invoke(tool_History)
        response_text = final_ai_msg.content
    else:
        response_text = ai_msg.content

    # 대화 결과를 바탕으로 내부 파라미터 및 '정식 문서 발행 착수 여부'만 판단
    state_extractor = llm.with_structured_output(
        ConversationStateUpdate, method="function_calling"
    )
    status: ConversationStateUpdate = state_extractor.invoke(
        f"기존 조건: {current_params}\n"
        f"사용자 입력: {state.get('user_query')}\n"
        f"에이전트 답변: {response_text}\n"
        "위 대화를 바탕으로 최신 작업 조건(updated_params)과 정식 문서 생성 파이프라인 가동 여부(trigger_document_generation)를 판단하세요."
    )

    # 아직 대화 중이거나(일상 대화, 날씨 질문, 조건 수집 중, 승인 대기 중)
    if not status.trigger_document_generation:
        return {
            "task_type": status.task_type,
            "extracted_params": status.updated_params,
            "is_param_complete": "no",  # Team 2/3(문서 생성)으로 넘어가지 않고 대화 응답 즉시 반환
            "final_answer": response_text
        }

    # 사용자가 최종 문서 발행을 승인한 경우 -> Team 2 & Team 3 문서 발행 파이프라인 가동!
    mat = status.updated_params.get("material", "")
    thk = status.updated_params.get("thickness", "")
    eq = status.updated_params.get("equipment", "")
    search_q = f"{mat} {thk} {eq} 표준 규격 및 안전 지침".strip()

    return {
        "task_type": status.task_type,
        "extracted_params": status.updated_params,
        "search_queries": [search_q],
        "is_param_complete": "yes",  # 이제 정식 문서 발행으로 이동!
        "final_answer": None,
        "retry_count": 0
    }


builder = StateGraph(ShipAgentState)
builder.add_node("interactive_agent", interactive_agent_node)
builder.add_edge(START, "interactive_agent")
builder.add_edge("interactive_agent", END)

team1_subgraph = builder.compile()