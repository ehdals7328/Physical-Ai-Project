"""
Team 1: 요구사항 분석 및 도구 실행 계획 서브그래프 (analysis.py)
- 하드코딩 리스트/정규식 없이 Function Calling으로만 의도, 제원 누적, 검색어, 기상 조회를 계획
"""

import json
from typing import Any, Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from modules.agent.state import ShipAgentState


@tool
def request_weather_tool(city_en: Optional[str] = None) -> str:
    """
    기상 확인이 필요하거나 작업지시서/계획서를 작성할 때 호출합니다.
    - city_en: 질문이나 누적 제원에 포함된 작업 지역의 대표 도시 영문명 (예: 'Ulsan', 'Geoje', 'Pohang' 등).
               어느 맥락에도 위치 정보가 없다면 None으로 비워두십시오.
    """
    return f"Weather check planned for {city_en}"


@tool
def request_rag_search_tool(search_queries: List[str]) -> str:
    """
    사내 기술 규격(WPS), 작업 기준, 안전 지침 검색이 필요할 때 호출합니다.
    - search_queries: 이전 누적 제원과 현재 입력을 종합한 핵심 공학 검색어 리스트 (1~2개)
    """
    return f"RAG search planned: {search_queries}"


@tool
def configure_work_order_tool(
    material: Optional[str] = None,
    thickness: Optional[str] = None,
    process: Optional[str] = None,
    location: Optional[str] = None,
    additional_specs: Optional[Dict[str, Any]] = None
) -> str:
    """
    작업지시서/작업계획서 작성이 요청되었거나, 이전 턴의 문서 작성 요청에 이어 사용자가 작업 제원(재질, 두께, 공정, 전류/전압 등)을 보충 입력할 때 반드시 호출합니다.
    - material: 모재 강종 또는 작업 대상 자재
    - thickness: 판 두께 또는 치수 규격
    - process: 세부 작업 공정 또는 이음 형태
    - location: 작업 지역 영문명 또는 현장명
    - additional_specs: 전류, 전압, 자세, 풍속 조건 등 사용자가 언급한 기타 모든 작업 조건 딕셔너리
    """
    return "Work order parameters configured"


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
planner_tools = [request_weather_tool, request_rag_search_tool, configure_work_order_tool]
llm_with_planner = llm.bind_tools(planner_tools)

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 조선소 엔지니어링 총괄 분석 에이전트입니다.\n"
     "[이전 태스크 상태], [이전 누적 작업 제원], [현재 사용자 입력]을 종합하여 필요한 도구를 모두 호출하세요.\n\n"
     "[핵심 원칙]\n"
     "1. 사용자가 작업지시서/계획서 작성을 요청했거나, [이전 태스크 상태]가 'doc_generation(제원 보충 대기 중)'인 상태에서 추가 제원을 입력한 경우:\n"
     "   - 반드시 `configure_work_order_tool`, `request_weather_tool`, `request_rag_search_tool` 3가지를 함께 호출하세요.\n"
     "   - [이전 누적 작업 제원]에 있던 값과 [현재 사용자 입력]에서 새로 파악된 값을 합쳐서 인자에 모두 채워 넣으세요.\n"
     "2. 문서 작성 없이 단순 기술/규격 질의만 하는 경우: `request_rag_search_tool`만 호출하세요.\n"
     "3. 단순 실시간 기상 질문인 경우: `request_weather_tool`만 호출하세요.\n"
     "4. 일상 인사나 잡담인 경우: 도구를 호출하지 마세요."),
    MessagesPlaceholder(variable_name="messages", optional=True),
    ("human",
     "[이전 태스크 상태]: {prev_status}\n"
     "[이전 누적 작업 제원]: {accumulated_params}\n"
     "[현재 사용자 입력]: {query}")
])


def agentic_intent_planner_node(state: ShipAgentState) -> Dict[str, Any]:
    query = state.get("user_query", "")
    existing_params = dict(state.get("extracted_params", {}) or {})
    messages = state.get("messages", [])
    prev_task_type = state.get("task_type")
    prev_complete = state.get("is_param_complete")

    is_continuing_doc = (prev_task_type == "doc_generation" and prev_complete == "no")
    prev_status_str = "doc_generation(제원 보충 대기 중)" if is_continuing_doc else (prev_task_type or "초기 상태")

    chain = PLANNER_PROMPT | llm_with_planner
    response = chain.invoke({
        "messages": messages[-4:] if messages else [],
        "prev_status": prev_status_str,
        "accumulated_params": json.dumps(existing_params, ensure_ascii=False),
        "query": query
    })

    tool_calls = getattr(response, "tool_calls", [])

    task_type = "doc_generation" if is_continuing_doc else ("chat" if not tool_calls else "knowledge_qa")
    need_weather = (task_type == "doc_generation")
    search_queries: List[str] = []

    for tc in tool_calls:
        name = tc["name"]
        args = tc.get("args", {})

        if name == "request_weather_tool":
            need_weather = True
            city_en = args.get("city_en")
            if city_en and str(city_en).upper() not in ("NONE", "NULL", ""):
                existing_params["location"] = city_en

        elif name == "request_rag_search_tool":
            queries = args.get("search_queries", [])
            if queries:
                search_queries.extend(queries)

        elif name == "configure_work_order_tool":
            task_type = "doc_generation"
            need_weather = True
            for k in ["material", "thickness", "process", "location"]:
                val = args.get(k)
                if val and str(val).upper() not in ("NONE", "NULL", ""):
                    existing_params[k] = val
            extra = args.get("additional_specs")
            if isinstance(extra, dict):
                for ek, ev in extra.items():
                    if ev is not None and str(ev).upper() not in ("NONE", "NULL", ""):
                        existing_params[ek] = ev

    # 작업지시서인데 검색어가 누락된 경우 누적된 제원들을 조합하여 동적 검색어 생성
    if task_type == "doc_generation" and not search_queries:
        combined_keywords = " ".join(
            str(v) for k, v in existing_params.items() if k != "location" and v
        ).strip()
        search_queries = [f"{combined_keywords} 작업 규격".strip() if combined_keywords else query]

    return {
        "task_type": task_type,
        "target_categories": [],
        "need_weather": need_weather,
        "extracted_params": existing_params,
        "search_queries": search_queries,
        "missing_params": [],
        "is_param_complete": "idle",
        "search_status": "idle"
    }


builder = StateGraph(ShipAgentState)
builder.add_node("agentic_intent_planner", agentic_intent_planner_node)
builder.add_edge(START, "agentic_intent_planner")
builder.add_edge("agentic_intent_planner", END)

team1_subgraph = builder.compile()