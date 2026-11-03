# agent_system/subgraphs/search.py
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from state import ShipAgentState
from tools.retriever_tools import wps_retriever_tool, safety_retriever_tool
from tools.weather_mcp_client import get_realtime_weather_from_mcp

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class WeatherNeedDecision(BaseModel):
    needs_weather: bool = Field(
        description="현재 작업이나 질문 해결을 위해 실시간 야외 기상 정보(기온, 풍속, 습도) 조회가 필요한지 여부"
    )
    target_location: Optional[str] = Field(
        None,
        description="기상 조회가 필요할 경우 조회할 지역명(예: 거제, 울산, 부산, 목포 등). 지역을 알 수 없으면 None"
    )


class GradeDecision(BaseModel):
    is_relevant: Literal["yes", "no"] = Field(
        description="검색된 사내 규격 문서가 사용자의 질의 및 작업 조건을 처리하기에 적합한지 여부"
    )


class RewriteQuery(BaseModel):
    rewritten_query: str = Field(description="검색 정확도를 높이기 위해 재작성한 검색 쿼리")


def rag_retriever_node(state: ShipAgentState) -> dict:
    """사내 규격 및 안전 지침 RAG 검색 노드"""
    queries = state.get("search_queries") or [state.get("user_query", "")]
    task_type = state.get("task_type", "general_qa")

    docs = []
    for q in queries:
        if task_type == "heavy_plan":
            res = safety_retriever_tool.invoke({"query": q})
        else:
            res = wps_retriever_tool.invoke({"query": q})
        if res:
            docs.append(str(res))

    return {"retrieved_docs": docs}


def weather_collector_node(state: ShipAgentState) -> dict:
    """LLM이 기상 정보 필요 여부를 자율 판단하여 FastMCP를 호출하는 노드"""
    user_query = state.get("user_query", "")
    task_type = state.get("task_type", "")
    params = state.get("extracted_params") or {}

    prompt = (
        f"작업 유형: {task_type}\n"
        f"파악된 작업 조건: {params}\n"
        f"사용자 최신 입력: {user_query}\n\n"
        "이 작업을 수행하기 위해 실시간 기상 데이터를 조회해야 하는지, 조회한다면 어느 지역인지 판단하세요."
    )
    decision: WeatherNeedDecision = llm.with_structured_output(WeatherNeedDecision).invoke(prompt)

    if not decision.needs_weather or not decision.target_location:
        return {"weather_info": None}

    weather_data = get_realtime_weather_from_mcp(decision.target_location)
    return {"weather_info": weather_data}


def doc_grader_node(state: ShipAgentState) -> dict:
    """검색된 문서의 적합성을 평가하는 노드"""
    docs = state.get("retrieved_docs") or []
    if not docs:
        return {"is_docs_relevant": "no"}

    prompt = (
        f"작업 목표: {state.get('task_type')}\n"
        f"작업 조건: {state.get('extracted_params')}\n"
        f"검색된 규격 문서:\n{docs}\n\n"
        "검색된 문서가 해당 작업 조건이나 질문에 답변하기에 유의미한 정보를 포함하고 있는지 평가하세요."
    )
    res: GradeDecision = llm.with_structured_output(GradeDecision).invoke(prompt)
    return {"is_docs_relevant": res.is_relevant}


def query_rewriter_node(state: ShipAgentState) -> dict:
    """부적합 판정 시 검색어를 스스로 개선하는 노드"""
    retry = state.get("retry_count", 0) + 1
    prompt = (
        f"기존 검색어 {state.get('search_queries')}로 적합한 사내 규격을 찾지 못했습니다.\n"
        f"작업 조건({state.get('extracted_params')})과 질문({state.get('user_query')})을 바탕으로 "
        "더 핵심적인 기술 키워드 중심의 검색어를 새로 작성하세요."
    )
    res: RewriteQuery = llm.with_structured_output(RewriteQuery).invoke(prompt)
    return {
        "search_queries": [res.rewritten_query],
        "retry_count": retry
    }


def check_relevance(state: ShipAgentState) -> str:
    if state.get("is_docs_relevant") == "yes" or state.get("retry_count", 0) >= 2:
        return "pass"
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
    check_relevance,
    {
        "pass": END,
        "retry": "query_rewriter"
    }
)
builder.add_edge("query_rewriter", "rag_retriever")

team2_subgraph = builder.compile()