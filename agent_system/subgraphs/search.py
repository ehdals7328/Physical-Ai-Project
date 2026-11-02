# agent_system/subgraphs/search.py
from typing import Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from state import ShipAgentState
from tools.mock_tools import mock_chroma_rag #, mock_weather_mcp
from tools.mcp_client import get_realtime_weather_from_mcp
from tools.domain_tools import validate_environmental_rules
from tools.retriever_tools import wps_retriever_tool, safety_retriever_tool

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ----------------------------------------------------
# 1. RAG 규격 검색 노드
# ----------------------------------------------------
def rag_retriever_node(state: ShipAgentState) -> dict:
    """
    RAG 도구(BaseTool)를 명시적으로 호출(invoke)하여 사내 규격 및 안전 지침을 검색하는 노드
    """
    queries = state.get("search_queries") or [state["user_query"]]
    task_type = state.get("task_type", "general_qa")
    all_docs = []

    for q in queries:
        if task_type == "heavy_plan":
            # 중량물 인양 안전계획서는 안전 수칙 도구 호출
            doc_result = safety_retriever_tool.invoke(q)
            all_docs.append(str(doc_result))
        elif task_type == "weld_order":
            # 용접 작업지시서는 사내 WPS 표준 도구 호출
            doc_result = wps_retriever_tool.invoke(q)
            all_docs.append(str(doc_result))
        else:
            # 일반 기술 질의는 두 도구 모두 참조
            doc_wps = wps_retriever_tool.invoke(q)
            doc_safety = safety_retriever_tool.invoke(q)
            all_docs.extend([str(doc_wps), str(doc_safety)])

    # 중복 제거 및 공백 정리
    unique_docs = list(dict.fromkeys([d.strip() for d in all_docs if d.strip()]))
    return {"retrieved_docs": unique_docs}

# ----------------------------------------------------
# 2. 기상 수집 및 도메인 안전 룰 판정 노드 (Tool Invoke 적용)
# ----------------------------------------------------
def weather_collector_node(state: ShipAgentState) -> dict:
    loc = state.get("extracted_params", {}).get("location") or "울산 야외 독"
    task_type = state.get("task_type", "weld_order")
    
    # 기상 수집 (현재 mock, 추후 303호 MCP 연결)
    weather = get_realtime_weather_from_mcp(loc)
    
    # 🛠️ 도구 호출: 기상 데이터와 태스크 기반 안전 규정 자동 판정
    safety_result = validate_environmental_rules.invoke({
        "temperature": float(weather.get("temperature", 20.0)),
        "wind_speed": float(weather.get("wind_speed", 0.0)),
        "task_type": task_type
    })
    
    return {
        "weather_info": weather,
        "safety_evaluation": safety_result  # 👈 도구 실행 결과를 상태에 기록
    }

# ----------------------------------------------------
# 3. 문서 적합성 평가 노드 (Grader)
# ----------------------------------------------------
class DocumentGrade(BaseModel):
    is_relevant: Literal["yes", "no"] = Field(
        description="인출된 사내 규격 문서가 사용자의 질문/조건을 해결하기에 적합한지 여부"
    )

grade_llm = llm.with_structured_output(DocumentGrade, method="function_calling")

def doc_grader_node(state: ShipAgentState) -> dict:
    docs_text = "\n\n".join(state.get("retrieved_docs", []))
    prompt = (
        f"사용자 질문: {state['user_query']}\n\n"
        f"인출된 규격 문서들:\n{docs_text}\n\n"
        "이 문서들이 사용자의 질문에 답하거나 작업 문서를 작성하기에 적합하고 충분한 내용을 담고 있습니까? (yes/no)"
    )
    res: DocumentGrade = grade_llm.invoke(prompt)
    return {"is_docs_relevant": res.is_relevant}

# ----------------------------------------------------
# 4. 쿼리 재작성 노드 (Rewriter)
# ----------------------------------------------------
def query_rewriter_node(state: ShipAgentState) -> dict:
    prompt = (
        f"사내 규격집에서 원하는 내용을 찾지 못했습니다. 검색어를 더 적절한 표준 기술 용어로 보완하세요.\n"
        f"기존 질문: {state['user_query']}\n"
        f"기존 검색어: {state.get('search_queries')}"
    )
    res = llm.invoke(prompt)
    current_retry = state.get("retry_count", 0)
    return {
        "search_queries": [res.content.strip()],
        "retry_count": current_retry + 1
    }

# ----------------------------------------------------
# 분기 및 서브그래프 빌드
# ----------------------------------------------------
def check_grade_result(state: ShipAgentState) -> Literal["pass", "retry", "exit"]:
    if state.get("is_docs_relevant") == "yes":
        return "pass"
    if state.get("retry_count", 0) >= 2:
        return "exit"
    return "retry"

builder = StateGraph(ShipAgentState)
builder.add_node("rag_retriever", rag_retriever_node)
builder.add_node("weather_collector", weather_collector_node)
builder.add_node("doc_grader", doc_grader_node)
builder.add_node("query_rewriter", query_rewriter_node)

builder.add_edge(START, "rag_retriever")
builder.add_edge("rag_retriever", "weather_collector")
builder.add_edge("weather_collector", "doc_grader")
builder.add_conditional_edges(
    "doc_grader",
    check_grade_result,
    {
        "pass": END,
        "exit": END,
        "retry": "query_rewriter"
    }
)
builder.add_edge("query_rewriter", "rag_retriever")

team2_subgraph = builder.compile()