"""
Team 1: 순수 요구사항 의도 분석 및 도구 실행 계획 서브그래프 (analysis.py)
- OpenAI Tool Calling 시 Dict[str, Any] 스키마 회피 버그를 방지하기 위해 List[str] 구조로 작업 조건을 수집합니다.
"""

import json
from typing import Any, Dict, List, Literal, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from modules.agent.state import ShipAgentState


@tool
def search_technical_knowledge_tool(search_queries: List[str]) -> str:
    """
    사내 기술 규격, 선박 건조 공정 절차, 안전 지침, 또는 작업지시서 양식/필수 항목 설명 등
    지식 베이스(ChromaDB) 검색이 필요할 때 호출합니다.
    - search_queries: 검색할 핵심 키워드 또는 구문 리스트 (1~2개)
    """
    return f"Knowledge search planned: {search_queries}"


@tool
def check_realtime_weather_tool(
    city_en: Optional[str] = None,
    location_action: Literal["KEEP_OR_SET", "CLEAR"] = "KEEP_OR_SET"
) -> str:
    """
    실시간 현장 기상(기온, 풍속, 습도, 강수량) 확인이 필요하거나 작업지시서를 발행할 때 호출합니다.
    - city_en: 작업 지역의 대표 도시 영문명 (예: 'Ulsan', 'Geoje', 'Daegu', 'Busan' 등). 언급이 없으면 None.
    - location_action: 기존 위치를 취소할 때만 'CLEAR', 그 외에는 'KEEP_OR_SET'.
    """
    return f"Weather check planned: {city_en} ({location_action})"


@tool
def issue_official_document_tool(
    context_mode: Literal["NEW_REQUEST", "CONTINUE_FILLING"] = "NEW_REQUEST",
    spec_items: Optional[List[str]] = None
) -> str:
    """
    사용자가 실제 '작업지시서' 또는 '작업계획서'의 작성/발행(생성)을 요청했거나,
    직전 턴의 역질문에 대한 답변으로 부족한 작업 조건을 입력했을 때 반드시 이 도구를 호출합니다!
    (단, '작업지시서 필수 항목이 뭐야?'처럼 양식이나 개념만 물어보는 질문에는 호출하지 마십시오.)
    - context_mode:
      * 'NEW_REQUEST': 새로운 작업지시서/계획서 작성을 요청한 경우
      * 'CONTINUE_FILLING': 직전 역질문에 이어 부족한 작업 제원을 보충 답변한 경우
    - spec_items: 사용자가 언급한 작업 조건들을 '항목명:값' 형태의 문자열 리스트로 입력 (예: ['공정:용접', '재질:SS400', '두께:10T'])
    """
    return "Official document issuance planned"


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
planner_tools = [
    search_technical_knowledge_tool,
    check_realtime_weather_tool,
    issue_official_document_tool
]
llm_with_planner = llm.bind_tools(planner_tools)

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 조선소 엔지니어링 총괄 기획 에이전트(Team 1)입니다.\n"
     "[직전 턴 상태], [이전 누적 작업 제원], [현재 사용자 입력]을 분석하여 알맞은 도구를 호출하세요.\n\n"
     "[도구 선택 기준]\n"
     "1. 작업지시서/계획서 작성 요청 (예: '울산에서 용접 작업지시서 작성해줘', '크레인 작업계획서 써줘') 또는 [직전 턴 상태]가 '제원 보충 대기 중'일 때 제원을 답변한 경우:\n"
     "   - 반드시 `issue_official_document_tool`, `check_realtime_weather_tool`, `search_technical_knowledge_tool` 3가지를 모두 호출하세요!\n"
     "2. 문서 발행이 아닌 일반 지식/절차/양식 질문 (예: '선박 건조 과정이 어떻게 돼?', '작업지시서에 들어가는 필수 항목이 뭐야?'):\n"
     "   - `search_technical_knowledge_tool`만 호출하세요.\n"
     "3. 현장 날씨와 안전 지침이 함께 필요한 질문 (예: '오늘 울산 외부에서 용접할 건데 유의사항 있어?'):\n"
     "   - `search_technical_knowledge_tool`과 `check_realtime_weather_tool`을 호출하세요.\n"
     "4. 순수 날씨 질문: `check_realtime_weather_tool`만 호출하세요.\n"
     "5. 단순 인사/잡담: 아무 도구도 호출하지 마세요."),
    MessagesPlaceholder(variable_name="messages", optional=True),
    ("human",
     "[직전 턴 상태]: {prev_complete_status}\n"
     "[이전 누적 작업 제원]: {accumulated_params}\n"
     "[현재 사용자 입력]: {query}")
])


def agentic_intent_planner_node(state: ShipAgentState) -> Dict[str, Any]:
    query = state.get("user_query", "").strip()
    existing_params = dict(state.get("extracted_params", {}) or {})
    messages = state.get("messages", [])
    prev_complete = state.get("is_param_complete")

    status_desc = "제원 보충 대기 중(is_param_complete=no)" if (prev_complete == "no") else "일반 대기"

    response = (PLANNER_PROMPT | llm_with_planner).invoke({
        "messages": messages[-4:] if messages else [],
        "prev_complete_status": status_desc,
        "accumulated_params": json.dumps(existing_params, ensure_ascii=False),
        "query": query
    })

    tool_calls = getattr(response, "tool_calls", [])

    task_type = "chat" if not tool_calls else "knowledge_qa"
    need_weather = False
    search_queries: List[str] = []
    updated_params = dict(existing_params)

    for tc in tool_calls:
        name = tc["name"]
        args = tc.get("args", {})

        if name == "search_technical_knowledge_tool":
            queries = args.get("search_queries") or []
            for q in queries:
                if q and str(q).strip():
                    search_queries.append(str(q).strip())

        elif name == "check_realtime_weather_tool":
            need_weather = True
            loc_action = args.get("location_action", "KEEP_OR_SET")
            city_en = args.get("city_en")
            if loc_action == "CLEAR":
                updated_params.pop("location", None)
            elif city_en and str(city_en).upper() not in ("NONE", "NULL", ""):
                updated_params["location"] = str(city_en).strip()

        elif name == "issue_official_document_tool":
            task_type = "doc_generation"
            need_weather = True
            context_mode = args.get("context_mode", "NEW_REQUEST")
            if context_mode == "NEW_REQUEST" and prev_complete != "no":
                kept_loc = updated_params.get("location")
                updated_params = {"location": kept_loc} if kept_loc else {}

            spec_items = args.get("spec_items") or []
            for item in spec_items:
                if ":" in str(item):
                    k, v = str(item).split(":", 1)
                    if k.strip() and v.strip() and v.strip().upper() not in ("NONE", "NULL"):
                        updated_params[k.strip()] = v.strip()
                elif str(item).strip():
                    updated_params[f"조건_{len(updated_params)+1}"] = str(item).strip()

    if not search_queries and task_type != "chat":
        if task_type == "doc_generation":
            spec_str = " ".join(str(v) for k, v in updated_params.items() if k != "location" and v).strip()
            search_queries = [spec_str if spec_str else query]
        elif not need_weather:
            search_queries = [query]

    print(
        f"🧠 [Team 1 분석 결과]: task_type={task_type} | need_weather={need_weather} "
        f"| 제원={updated_params} | 검색어={search_queries}"
    )

    return {
        "task_type": task_type,
        "target_categories": [],
        "need_weather": need_weather,
        "extracted_params": updated_params,
        "search_queries": search_queries,
        "missing_params": [],
        "is_param_complete": None,
        "search_status": "idle"
    }


builder = StateGraph(ShipAgentState)
builder.add_node("agentic_intent_planner", agentic_intent_planner_node)
builder.add_edge(START, "agentic_intent_planner")
builder.add_edge("agentic_intent_planner", END)

team1_subgraph = builder.compile()