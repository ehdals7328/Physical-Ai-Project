"""
Team 1: 요구사항 분석 및 도구 실행 계획 서브그래프 (analysis.py)
- Pure Function Calling (llm.bind_tools) 아키텍처
- LLM이 사용자 질의를 분석하여 필요한 도구(기상 조회, 규격 검색, 작업지시서 설정)를 자율적으로 호출
"""

from typing import Any, Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from modules.agent.state import ShipAgentState


# ==============================================================================
# 1. Team 1 전용 플래닝 도구(@tool) 정의
# ==============================================================================
@tool
def request_weather_tool(city_en: Optional[str] = None) -> str:
    """
    사용자가 특정 지역의 날씨, 기온, 풍속, 습도 등 기상 정보를 묻거나,
    야외 용접/크레인 작업지시서 작성을 위해 현장 기상 확인이 필요할 때 호출합니다.
    - city_en: 질문에 언급된 지역의 대한민국 대표 도시 영문명 (예: 'Daegu', 'Geoje', 'Ulsan', 'Busan', 'Pohang', 'Seoul' 등).
               만약 사용자가 지역/위치를 전혀 언급하지 않았다면 None으로 비워두세요.
    """
    return f"Weather check planned for {city_en}"


@tool
def request_rag_search_tool(search_queries: List[str]) -> str:
    """
    조선소 용접 규격(WPS), 안전보건 지침, 선급 규정, 결함 방지 기준 등
    사내 기술 문서(ChromaDB) 검색이 필요할 때 호출합니다.
    - search_queries: 벡터 DB 검색 적중률을 높이기 위한 핵심 공학 검색어 리스트 (1~2개)
    """
    return f"RAG search planned: {search_queries}"


@tool
def configure_work_order_tool(
    material: Optional[str] = None,
    thickness: Optional[str] = None,
    process: Optional[str] = None,
    location: Optional[str] = None
) -> str:
    """
    사용자가 '작업지시서' 또는 '작업계획서' 작성을 요청했을 때 호출하여 작업 제원을 등록합니다.
    언급되지 않은 제원은 추측하지 말고 None으로 비워두세요.
    """
    return "Work order parameters configured"


# LLM에 3개 도구 바인딩 (Parallel Tool Calling 활성화)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
planner_tools = [request_weather_tool, request_rag_search_tool, configure_work_order_tool]
llm_with_planner = llm.bind_tools(planner_tools)

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 조선소 엔지니어링 총괄 분석 에이전트입니다.\n"
     "이전 대화 맥락과 현재 사용자의 질문을 분석하여 필요한 도구(Tool)를 모두 호출하세요.\n\n"
     "[도구 호출 원칙]\n"
     "1. 날씨, 기온, 풍속을 물어보거나('~어때?', '~알려줘' 등 어미 무관) 작업지시서 작성 시: 반드시 `request_weather_tool`을 호출하세요.\n"
     "   - 질문이나 이전 대화에 지역명(예: 대구, 거제 조선소, 울산 독 등)이 있으면 `city_en`에 영문 도시명('Daegu', 'Geoje', 'Ulsan' 등)을 넣으세요.\n"
     "   - 지역명이 전혀 없으면 `city_en=None`으로 호출하세요.\n"
     "2. 기술 기준, 용접 규격, 안전 수칙 질문이거나 작업지시서 작성 시: `request_rag_search_tool`을 호출하세요.\n"
     "   - 단, 기술 질문 없이 순수하게 현재 날씨만 묻는 경우에는 `request_rag_search_tool`을 호출하지 마세요.\n"
     "3. 작업지시서 작성 요청 시: `configure_work_order_tool`을 함께 호출하여 언급된 제원을 기록하세요.\n"
     "4. 단순 인사나 일상 잡담인 경우 아무 도구도 호출하지 마세요."),
    MessagesPlaceholder(variable_name="messages", optional=True),
    ("human", "[이전 누적 작업 제원]: {accumulated_params}\n[사용자 질문]: {query}")
])


# ==============================================================================
# 2. 자율 도구 계획 노드 (agentic_intent_planner)
# ==============================================================================
def agentic_intent_planner_node(state: ShipAgentState) -> Dict[str, Any]:
    query = state.get("user_query", "")
    existing_params = dict(state.get("extracted_params", {}) or {})
    messages = state.get("messages", [])

    chain = PLANNER_PROMPT | llm_with_planner
    response = chain.invoke({
        "messages": messages[-4:] if messages else [],
        "accumulated_params": existing_params,
        "query": query
    })

    tool_calls = getattr(response, "tool_calls", [])

    # 기본 상태 초기화
    task_type = "chat" if not tool_calls else "knowledge_qa"
    need_weather = False
    search_queries: List[str] = []
    missing_params: List[str] = []

    # LLM이 선택한 Function Calling(tool_calls) 결과 해독
    for tc in tool_calls:
        name = tc["name"]
        args = tc.get("args", {})

        if name == "request_weather_tool":
            need_weather = True
            city_en = args.get("city_en")
            if city_en and str(city_en).upper() != "NONE":
                existing_params["location"] = city_en
            if not existing_params.get("location"):
                missing_params.append("location(작업 위치/지역명)")

        elif name == "request_rag_search_tool":
            queries = args.get("search_queries", [])
            if queries:
                search_queries.extend(queries)

        elif name == "configure_work_order_tool":
            task_type = "doc_generation"
            need_weather = True  # 작업지시서는 현장 기상 연동 기본 포함
            for k in ["material", "thickness", "process", "location"]:
                if args.get(k):
                    existing_params[k] = args[k]
            if not existing_params.get("location") and "location(작업 위치/지역명)" not in missing_params:
                missing_params.append("location(작업 위치/지역명)")

    # 작업지시서인데 검색어가 비어있다면 기본 검색어 보완
    if task_type == "doc_generation" and not search_queries:
        search_queries = [query]

    return {
        "task_type": task_type,
        "target_categories": [],
        "need_weather": need_weather,
        "extracted_params": existing_params,
        "search_queries": search_queries,
        "missing_params": missing_params,
        "is_param_complete": "no" if missing_params else "yes",
        "search_status": "idle"
    }


# ==============================================================================
# 3. 서브그래프 조립
# ==============================================================================
builder = StateGraph(ShipAgentState)
builder.add_node("agentic_intent_planner", agentic_intent_planner_node)
builder.add_edge(START, "agentic_intent_planner")
builder.add_edge("agentic_intent_planner", END)

team1_subgraph = builder.compile()