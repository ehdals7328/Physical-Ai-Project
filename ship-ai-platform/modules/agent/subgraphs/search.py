"""
Team 2: 지식·환경 수집 및 Function Calling 감리 서브그래프 (search.py)
- Node 1 (rag_retriever): Team 1이 계획한 검색어로 ChromaDB 인출
- Node 2 (weather_collector): Team 1이 계획한 영문 도시명으로 FastMCP 기상 수집
- Node 3 (semantic_doc_grader): 무관한 RAG 문서 필터링
- Node 4 (env_safety_critic): evaluate_environmental_safety 도구를 Function Calling으로 직접 호출하여 안전 판정
- Node 5 (spec_gap_evaluator): RAG 규격서 기반 동적 필수 제원 누락 추론
"""

import json
from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from modules.agent.state import ShipAgentState
from modules.agent.tools.retriever_tools import search_knowledge_by_categories
from modules.agent.tools.weather_mcp_client import get_realtime_weather_from_mcp
from modules.agent.tools.domain_tools import evaluate_environmental_safety

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ==============================================================================
# 1. RAG 규격 문서 인출 노드 (rag_retriever)
# ==============================================================================
def dynamic_rag_retriever_node(state: ShipAgentState) -> Dict[str, Any]:
    search_queries = state.get("search_queries", [])
    task_type = state.get("task_type", "knowledge_qa")

    # Team 1에서 RAG 검색 도구를 호출하지 않은 경우(순수 날씨 질의, 일상 대화) 즉시 스킵
    if not search_queries:
        return {"retrieved_docs": [], "search_status": "empty"}

    target_query = search_queries[0]
    collected_docs: List[Dict[str, Any]] = []

    try:
        raw_items = search_knowledge_by_categories(target_query, top_k=4)
        for item in raw_items:
            collected_docs.append({
                "category": item.get("category", "규격"),
                "content": item.get("chunk_text", "") or str(item),
                "metadata": {
                    "document_name": item.get("document_name"),
                    "revision": item.get("revision"),
                    "page": item.get("page_start"),
                    "similarity": item.get("similarity_score"),
                    "source": item.get("source"),
                    "doc_id": item.get("chunk_id")
                }
            })
        status = "completed" if collected_docs else "empty"
        return {"retrieved_docs": collected_docs, "search_status": status}
    except Exception as e:
        print(f"[Team 2 Search Error]: {e}")
        return {"retrieved_docs": [], "search_status": "error"}


# ==============================================================================
# 2. FastMCP 기상 도구 실행 노드 (weather_collector)
# ==============================================================================
def conditional_weather_node(state: ShipAgentState) -> Dict[str, Any]:
    if not state.get("need_weather", False):
        return {"weather_data": None}

    location = (state.get("extracted_params") or {}).get("location")
    if not location:
        return {
            "weather_data": {
                "status": "NEED_LOCATION",
                "message": "기상 정보 확인을 위해 작업 위치(조선소 및 지역명)를 입력해 주세요."
            },
            "need_weather": True
        }

    try:
        weather_info = get_realtime_weather_from_mcp(location)
        return {"weather_data": weather_info, "need_weather": True}
    except Exception as e:
        return {
            "weather_data": {"location": location, "status": "ERROR", "error": str(e)},
            "need_weather": True
        }


# ==============================================================================
# 3. 시맨틱 문서 적합성 평가 노드 (semantic_doc_grader)
# ==============================================================================
class RelevantDocIndices(BaseModel):
    relevant_indices: List[int] = Field(
        default_factory=list,
        description="사용자 질의 및 작업 목적과 실제로 연관이 있는 문서의 인덱스 번호(0부터 시작) 목록. 없으면 빈 리스트([])"
    )


structured_grader = llm.with_structured_output(RelevantDocIndices, method="function_calling")

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 기술 문서 적합성 감사 AI입니다.\n"
     "검색된 문서 청크들이 [사용자 질의]에 답변하거나 [작업지시서]를 작성하는 데 실질적인 기술 근거가 되는지 평가하세요.\n"
     "- 질문 의도와 무관한 문서는 제외하고, 관련성이 인정되는 문서의 인덱스 번호(0, 1, 2...)만 반환하세요."),
    ("human", "[사용자 질의]: {query}\n\n[검색된 문서 목록]:\n{docs_preview}")
])


def semantic_doc_grader_node(state: ShipAgentState) -> Dict[str, Any]:
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"retrieved_docs": [], "search_status": "empty"}

    query = state.get("user_query", "")
    preview_lines = [
        f"[{idx}] {d.get('metadata', {}).get('document_name', '문서')}: {str(d.get('content', ''))[:300].replace(chr(10), ' ')}"
        for idx, d in enumerate(docs)
    ]

    try:
        res: RelevantDocIndices = (GRADER_PROMPT | structured_grader).invoke({
            "query": query,
            "docs_preview": "\n".join(preview_lines)
        })
        filtered = [docs[i] for i in res.relevant_indices if 0 <= i < len(docs)]
        return {
            "retrieved_docs": filtered,
            "search_status": "completed" if filtered else "empty"
        }
    except Exception as e:
        print(f"[Team 2 Grader Warning]: {e}")
        return {"retrieved_docs": docs}


# ==============================================================================
# 4. Function Calling 기반 환경·안전 룰 판정 노드 (env_safety_critic) ⭐
# ==============================================================================
llm_with_safety_tool = llm.bind_tools([evaluate_environmental_safety], tool_choice="evaluate_environmental_safety")

SAFETY_TOOL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 조선소 환경 안전 감리 AI입니다.\n"
     "제공된 [실측 기상 데이터]와 [참조 규격 문서]를 확인하고, 반드시 `evaluate_environmental_safety` 도구를 호출하세요.\n"
     "규격 문서에 외기온도 예열 기준이나 풍속 제한 기준이 명시되어 있다면 `thresholds` 인자에 반영하고, 없으면 기본값을 사용하세요."),
    ("human",
     "[실측 기온]: {temp}℃\n[실측 풍속]: {wind}m/s\n[작업 맥락]: {task_type}\n[참조 규격 문서]:\n{docs_text}")
])


def env_safety_critic_node(state: ShipAgentState) -> Dict[str, Any]:
    weather = state.get("weather_data")
    if not weather or weather.get("status") != "SUCCESS":
        return {}

    temp = weather.get("temperature")
    wind = weather.get("wind_speed")
    if temp is None or wind is None:
        return {}

    docs = state.get("retrieved_docs", [])
    docs_text = "\n".join(str(d.get("content", ""))[:400] for d in docs) if docs else "참조 규격 없음"

    try:
        # LLM이 Function Calling으로 evaluate_environmental_safety 도구 호출 인자 구성
        ai_msg = (SAFETY_TOOL_PROMPT | llm_with_safety_tool).invoke({
            "temp": float(temp),
            "wind": float(wind),
            "task_type": state.get("task_type", "용접 작업"),
            "docs_text": docs_text
        })

        tool_calls = getattr(ai_msg, "tool_calls", [])
        if tool_calls:
            safety_eval = evaluate_environmental_safety.invoke(tool_calls[0]["args"])
        else:
            safety_eval = evaluate_environmental_safety.invoke({
                "current_temp": float(temp),
                "current_wind": float(wind)
            })
    except Exception:
        safety_eval = evaluate_environmental_safety.invoke({
            "current_temp": float(temp),
            "current_wind": float(wind)
        })

    enriched_weather = dict(weather)
    enriched_weather["safety_evaluation"] = safety_eval
    return {"weather_data": enriched_weather}


# ==============================================================================
# 5. RAG 문서 기반 동적 필수 제원 검증 노드 (spec_gap_evaluator)
# ==============================================================================
class SpecGapAnalysis(BaseModel):
    missing_fields: List[str] = Field(
        default_factory=list,
        description="인출된 RAG 규격/양식을 적용하기 위해 필수적이나 현재 사용자 입력에 누락된 항목 목록. 모두 충족되었으면 빈 리스트([])"
    )
    is_ready_to_generate: bool = Field(
        description="필수 파라미터가 충분하여 지시서를 작성할 수 있는지 여부"
    )


structured_spec_evaluator = llm.with_structured_output(SpecGapAnalysis, method="function_calling")

SPEC_EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 사내 표준 규격 감사 엔지니어입니다.\n"
     "인출된 [사내 규격/양식 문서(RAG)]를 확인하고, [현재 확보된 작업 제원]과 대조하여 "
     "사용자가 반드시 추가로 제공해야 하는 핵심 누락 항목(missing_fields)을 판정하세요."),
    ("human",
     "[인출된 사내 규격 문서(RAG)]:\n{docs}\n\n"
     "[현재 확보된 작업 제원]: {params}\n"
     "[사용자 원문 질의]: {query}")
])


def spec_gap_evaluator_node(state: ShipAgentState) -> Dict[str, Any]:
    if state.get("task_type") != "doc_generation" or not state.get("retrieved_docs"):
        return {}

    docs = state.get("retrieved_docs", [])
    params = state.get("extracted_params", {}) or {}
    preview_docs = "\n".join(f"[{d.get('category')}]: {str(d.get('content'))[:500]}" for d in docs)

    try:
        res: SpecGapAnalysis = (SPEC_EVAL_PROMPT | structured_spec_evaluator).invoke({
            "docs": preview_docs,
            "params": json.dumps(params, ensure_ascii=False),
            "query": state.get("user_query", "")
        })
        current_missing = list(state.get("missing_params", []))
        for m in res.missing_fields:
            if m not in current_missing:
                current_missing.append(m)

        return {
            "missing_params": current_missing,
            "is_param_complete": "yes" if res.is_ready_to_generate else "no"
        }
    except Exception as e:
        print(f"[Team 2 Spec Gap Warning]: {e}")
        return {}


# ==============================================================================
# 6. 서브그래프 조립
# ==============================================================================
builder = StateGraph(ShipAgentState)

builder.add_node("rag_retriever", dynamic_rag_retriever_node)
builder.add_node("weather_collector", conditional_weather_node)
builder.add_node("semantic_doc_grader", semantic_doc_grader_node)
builder.add_node("env_safety_critic", env_safety_critic_node)
builder.add_node("spec_gap_evaluator", spec_gap_evaluator_node)

builder.add_edge(START, "rag_retriever")
builder.add_edge(START, "weather_collector")
builder.add_edge("rag_retriever", "semantic_doc_grader")
builder.add_edge("semantic_doc_grader", "env_safety_critic")
builder.add_edge("weather_collector", "env_safety_critic")
builder.add_edge("env_safety_critic", "spec_gap_evaluator")
builder.add_edge("spec_gap_evaluator", END)

team2_subgraph = builder.compile()