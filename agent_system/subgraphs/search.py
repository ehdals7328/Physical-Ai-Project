# (Team 2: 지식·환경 검색팀)
from typing import Dict, List, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from state import ShipAgentState
# tools/mock_tools.py 또는 실제 도구 모듈 임포트
from tools import mock_chroma_rag, mock_weather_mcp
from tools.mcp_client import get_realtime_weather_from_mcp


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --------------------------------------------------------------------------
# 1. Pydantic 구조화 출력 스키마 정의 (Grader & Rewriter 전용)
# --------------------------------------------------------------------------
class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="검색된 문서들이 사용자 질문을 해결하는 데 충분하고 적합한지 여부 ('yes' 또는 'no')"
    )
    explanation: str = Field(
        ...,
        description="평가 결과에 대한 공학적 근거 요약"
    )

class RewrittenQuery(BaseModel):
    refined_queries: List[str] = Field(
        ...,
        description="검색 정확도를 높이기 위해 재작성된 공학 표준 검색 쿼리 목록 (1~3개)"
    )

# --------------------------------------------------------------------------
# 2. 세부 에이전트 노드 정의
# --------------------------------------------------------------------------
def rag_retriever_node(state: ShipAgentState) -> Dict:
    """[Agent 1: RAG 규격 검색] 1팀의 정제된 쿼리를 기반으로 사내 WPS/지침 문서 인출"""
    search_queries = state.get("search_queries", [])
    user_query = state.get("user_query", "")

    # 검색 쿼리가 없으면 사용자 원본 질문 활용
    target_queries = search_queries if search_queries else [user_query]
    
    retrieved_docs: List[str] = []
    for query in target_queries:
        docs = mock_chroma_rag(query)
        retrieved_docs.extend(docs)

    # 중복 제거
    retrieved_docs = list(dict.fromkeys(retrieved_docs))
    return {"retrieved_docs": retrieved_docs}


def weather_collector_node(state: ShipAgentState) -> Dict:
  """[Agent 2: 현장 기상 수집] FastMCP 서버(streamable-http)와 통신해 실시간 기상 수집"""
  extracted_params = state.get("extracted_params", {})
  location = extracted_params.get("location", "울산 야외 독")

  # 🌟 검증 완료된 streamable-http MCP 클라이언트 호출
  weather_data = get_realtime_weather_from_mcp(location)
  return {"weather_info": weather_data}


def doc_grader_node(state: ShipAgentState) -> Dict:
    """[Agent 3: 문서 적합성 평가] 인출된 사내 규격 문서가 작업자 질의에 적합한지 평가"""
    user_query = state.get("user_query", "")
    retrieved_docs = state.get("retrieved_docs", [])
    retry_count = state.get("retry_count", 0)

    docs_text = "\n---\n".join(retrieved_docs)
    system_prompt = (
        "당신은 조선·해양 생산 규격 검증 감리관(Grader)입니다.\n"
        "제공된 [검색 문서]들이 작업자의 [질문]을 해결하고 작업지시서/계획서를 작성하기에\n"
        "충분하고 적합한 규정인지 엄격히 평가하여 'yes' 또는 'no'로 판정하세요.\n\n"
        f"[질문]: {user_query}\n\n"
        f"[검색 문서]:\n{docs_text}"
    )

    decision = llm.with_structured_output(
        GradeDocuments,
        method="function_calling"
    ).invoke(system_prompt)

    return {
        "is_docs_relevant": decision.binary_score,
        "retry_count": retry_count if decision.binary_score == "yes" else retry_count + 1
    }


def query_rewriter_node(state: ShipAgentState) -> Dict:
    """[Agent 4: 쿼리 재작성] 검색 실패 시 Vector DB 검색 키워드를 공학적으로 재보정"""
    user_query = state.get("user_query", "")
    current_queries = state.get("search_queries", [])

    system_prompt = (
        "당신은 검색 쿼리 최적화 전문가입니다.\n"
        "이전 검색 쿼리로 규격서(Vector DB)에서 적절한 문서를 찾지 못했습니다.\n"
        "질문자의 의도를 다시 분석하여, 용접절차시방서(WPS)나 안전지침에서 정확히 걸릴 수 있는\n"
        "새로운 표준 공학 검색 키워드 1~2개를 생성하세요.\n\n"
        f"[원래 질문]: {user_query}\n"
        f"[실패한 이전 쿼리]: {current_queries}"
    )

    result = llm.with_structured_output(
        RewrittenQuery,
        method="function_calling"
    ).invoke(system_prompt)

    return {"search_queries": result.refined_queries}

# --------------------------------------------------------------------------
# 3. 조건부 분기 라우팅 로직
# --------------------------------------------------------------------------
def route_after_grading(state: ShipAgentState) -> Literal["retry", "pass"]:
    """문서 적합성 판정 결과 및 재시도 한도(최대 2회)를 확인해 분기"""
    is_relevant = state.get("is_docs_relevant")
    retry_count = state.get("retry_count", 0)

    # 적합하거나 최대 재시도(2회) 초과 시 다음 팀(Team 3)으로 이동
    if is_relevant == "yes" or retry_count >= 2:
        return "pass"
    return "retry"

# --------------------------------------------------------------------------
# 4. Team 2 서브그래프 빌드
# --------------------------------------------------------------------------
builder = StateGraph(ShipAgentState)

# 노드 등록
builder.add_node("rag_retriever", rag_retriever_node)
builder.add_node("weather_collector", weather_collector_node)
builder.add_node("doc_grader", doc_grader_node)
builder.add_node("query_rewriter", query_rewriter_node)

# 엣지 연결
builder.add_edge(START, "rag_retriever")
builder.add_edge("rag_retriever", "weather_collector")
builder.add_edge("weather_collector", "doc_grader")

# 적합성 판정에 따른 조건부 분기 (Evaluator-Optimizer 루프)
builder.add_conditional_edges(
    "doc_grader",
    route_after_grading,
    {
        "pass": END,
        "retry": "query_rewriter"
    }
)

# 재작성 노드는 다시 검색 노드로 회귀
builder.add_edge("query_rewriter", "rag_retriever")

team2_subgraph = builder.compile()