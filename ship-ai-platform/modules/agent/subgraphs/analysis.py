"""Team 1: 질의 분석 서브그래프 (analysis.py) - 완전 범용 구조"""

from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from modules.agent.state import DocCategory, ShipAgentState, TaskType

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class UniversalQueryAnalysis(BaseModel):
    task_type: TaskType = Field(
        description="'chat'(일상대화), 'knowledge_qa'(규격/법규/지침/날씨 질의), 'doc_generation'(지시서/계획서 등 문서 작성 요청)"
    )
    need_weather: bool = Field(
        default=False,
        description="실시간 현장 기상청(FastMCP) 조회가 필요한지 여부. 날씨, 기온, 풍속을 직접 묻거나, 현장 작업지시서 작성인 경우 반드시 True"
    )
    extracted_conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="사용자가 질문에서 직접 언급한 모든 기술/환경 조건(예: {'location': '울산', 'equipment': '크레인'} 등. 언급 없는 값은 절대 생성 금지)"
    )

structured_analyzer = llm.with_structured_output(UniversalQueryAnalysis, method="function_calling")

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 조선소 엔지니어링 분석 AI입니다.\n"
     "사용자의 입력을 분석하여 task_type과 작업 조건을 추출하세요.\n\n"
     "[조건 추출 규칙]\n"
     "- location: 사용자가 언급한 작업 지역/조선소/독 명칭(예: 거제, 울산 야외독, 포항 등). 언급이 전혀 없으면 절대 생성하지 마십시오.\n"
     "- 날씨/기상 질문이거나 현장 공정 문서 요청이면 need_weather = True로 설정하십시오.\n"),
    ("human", "[이전 대화 누적 조건]: {accumulated_params}\n[사용자 질의]: {query}")
])


def analyze_query_node(state: ShipAgentState) -> Dict[str, Any]:
    query = state.get("user_query", "")
    existing = state.get("extracted_params", {}) or {}

    chain = ANALYSIS_PROMPT | structured_analyzer
    res: UniversalQueryAnalysis = chain.invoke(
        {"accumulated_params": existing, "query": query}
    )

    merged = dict(existing)
    merged.update(res.extracted_conditions)

    return {
        "task_type": res.task_type,
        "target_categories": [],  # 카테고리 미사용으로 빈 리스트 고정
        "need_weather": res.need_weather,
        "extracted_params": merged,
        "search_status": "idle",
    }


builder = StateGraph(ShipAgentState)
builder.add_node("analyze_query", analyze_query_node)
builder.add_edge(START, "analyze_query")
builder.add_edge("analyze_query", END)
team1_subgraph = builder.compile()