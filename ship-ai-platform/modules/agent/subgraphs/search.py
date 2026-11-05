"""
Team 2: 지식·환경 수집 및 Function Calling 감리 서브그래프 (search.py)
- Node 1 (rag_retriever): Team 1이 계획한 검색어로 ChromaDB 인출
- Node 2 (weather_collector): FastMCP 실시간 기상 수집
- Node 3 (semantic_doc_grader): 무관한 RAG 문서 필터링
- Node 4 (env_safety_critic): evaluate_environmental_safety 도구 자율 실행
- Node 5 (spec_gap_evaluator): 인출된 RAG 규격서와 현재 확보된 제원을 대조하여 누락 항목 동적 판정
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
    search_queries = state.get("search_queries") or []
    user_query = state.get("user_query", "").strip()

    target_query = search_queries[0] if search_queries else user_query
    if not target_query:
        return {"retrieved_docs": [], "search_status": "empty"}

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
        description="사용자 질의 및 누적 작업 제원과 연관이 있는 문서의 인덱스 번호(0부터 시작) 목록"
    )


structured_grader = llm.with_structured_output(RelevantDocIndices, method="function_calling")

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 기술 문서 적합성 감사 AI입니다.\n"
     "검색된 문서 청크들이 [사용자 질의] 및 [누적 작업 제원]에 대한 기술 근거가 되는지 평가하여 관련 문서의 인덱스 번호(0, 1, 2...)를 반환하세요."),
    ("human",
     "[사용자 질의]: {query}\n"
     "[누적 작업 제원]: {params}\n\n"
     "[검색된 문서 목록]:\n{docs_preview}")
])


def semantic_doc_grader_node(state: ShipAgentState) -> Dict[str, Any]:
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"retrieved_docs": [], "search_status": "empty"}

    query = state.get("user_query", "")
    params = state.get("extracted_params", {}) or {}
    preview_lines = [
        f"[{idx}] {d.get('metadata', {}).get('document_name', '문서')}: {str(d.get('content', ''))[:300].replace(chr(10), ' ')}"
        for idx, d in enumerate(docs)
    ]

    try:
        res: RelevantDocIndices = (GRADER_PROMPT | structured_grader).invoke({
            "query": query,
            "params": json.dumps(params, ensure_ascii=False),
            "docs_preview": "\n".join(preview_lines)
        })
        filtered = [docs[i] for i in res.relevant_indices if 0 <= i < len(docs)]
        # 만약 필터링 결과가 전부 비어버렸다면 원본 문서를 유지하여 규격 참조 단절 방지
        final_docs = filtered if filtered else docs
        return {
            "retrieved_docs": final_docs,
            "search_status": "completed"
        }
    except Exception as e:
        print(f"[Team 2 Grader Warning]: {e}")
        return {"retrieved_docs": docs}


# ==============================================================================
# 4. Function Calling 기반 환경·안전 룰 판정 노드 (env_safety_critic)
# ==============================================================================
llm_with_safety_tool = llm.bind_tools([evaluate_environmental_safety], tool_choice="evaluate_environmental_safety")

SAFETY_TOOL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 조선소 환경 안전 감리 AI입니다.\n"
     "주어진 [실측 기상], [작업 맥락], [규격 문서]를 검토하여 `evaluate_environmental_safety` 도구를 실행하세요.\n"
     "- 만약 [작업 맥락]에서 사용자가 특정 가상 기상 조건(예: '풍속 12m/s 강풍 상황')을 가정하여 요청했다면, 실측 기상 대신 사용자가 가정한 수치를 current_wind / current_temp에 우선 적용하여 안전성을 평가하십시오.\n"
     "- thresholds 인자에는 단일 숫자(float) 값들로 구성된 딕셔너리(예: {{'work_stop_wind_limit': 10.0, 'weld_wind_screen_limit': 2.0, 'preheat_temp_limit': 5.0}})를 전달하십시오."),
    ("human",
     "[실측 기상]: 기온 {temp}℃, 풍속 {wind}m/s\n"
     "[작업 맥락]: {query}\n"
     "[확보된 제원]: {params}\n"
     "[규격 문서]:\n{docs_text}")
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
    query = state.get("user_query", "")
    params = state.get("extracted_params", {}) or {}

    try:
        ai_msg = (SAFETY_TOOL_PROMPT | llm_with_safety_tool).invoke({
            "temp": float(temp),
            "wind": float(wind),
            "query": query,
            "params": json.dumps(params, ensure_ascii=False),
            "docs_text": docs_text
        })

        tool_calls = getattr(ai_msg, "tool_calls", [])
        if tool_calls:
            safety_eval = evaluate_environmental_safety.invoke(tool_calls[0]["args"])
        else:
            actual_process = params.get("process") or query
            safety_eval = evaluate_environmental_safety.invoke({
                "current_temp": float(temp),
                "current_wind": float(wind),
                "task_context": actual_process,
                "task_type": actual_process
            })
    except Exception as e:
        print(f"[Team 2 Safety Critic Warning]: {e}")
        actual_process = params.get("process") or query
        safety_eval = evaluate_environmental_safety.invoke({
            "current_temp": float(temp),
            "current_wind": float(wind),
            "task_context": actual_process,
            "task_type": actual_process
        })

    enriched_weather = dict(weather)
    enriched_weather["safety_evaluation"] = safety_eval
    return {"weather_data": enriched_weather}


# ==============================================================================
# 5. RAG 문서 기반 동적 필수 제원 검증 노드 (spec_gap_evaluator)
# ==============================================================================
class SpecGapAnalysis(BaseModel):
    required_by_rag: List[str] = Field(
        default_factory=list,
        description="인출된 RAG 규격서에서 작업 조건을 특정하기 위해 기준 키(Key)로 삼고 있는 현장 입력 항목들 (예: 모재 재질, 판 두께, 작업 자세 등)"
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="required_by_rag 중 현재 [확보된 제원]과 [사용자 입력]에 아직 제공되지 않은 필수 항목 목록. 모두 충족되었으면 빈 리스트([])"
    )
    is_ready_to_generate: bool = Field(
        description="누락된 필수 항목이 없어 즉시 작업지시서를 발행할 수 있는지 여부 (missing_fields가 비어있으면 True)"
    )


structured_spec_evaluator = llm.with_structured_output(SpecGapAnalysis, method="function_calling")

SPEC_EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 사내 표준 규격(RAG) 대조 심사 엔지니어 AI입니다.\n"
     "[인출된 사내 규격 문서(RAG)]를 읽고, 해당 규격서의 표나 기준을 적용하기 위해 작업자가 알려주어야 하는 현장 입력 조건과 현재 [확보된 제원]을 대조하세요.\n\n"
     "[대조 및 판정 원칙]\n"
     "1. 규격서에서 전류/전압/안전수칙 등의 결과값을 찾기 위한 '입력 조건(예: 대상 모재/강종, 판 두께, 이음 형태 등)'이 [확보된 제원]이나 [사용자 입력]에 빠져 있다면, 그 부족한 조건들을 구체적으로 `missing_fields`에 나열하고 `is_ready_to_generate=False`로 판정하십시오.\n"
     "2. 반대로 [확보된 제원]에 대상 모재와 두께/공정 정보가 이미 들어있다면, 즉시 `missing_fields=[]`, `is_ready_to_generate=True`로 승인하십시오.\n"
     "3. 규격서 본문에서 찾아서 지시서에 적어주어야 하는 결과값(권장 전류, 전압, 와이어경, 예열온도 등)은 작업자에게 묻는 항목이 아니므로 절대 `missing_fields`에 넣지 마십시오."),
    ("human",
     "[현재 확보된 제원 (이전 대화 누적 포함)]: {params}\n"
     "[현재 사용자 입력]: {query}\n"
     "[인출된 사내 규격 문서(RAG)]:\n{docs}")
])


def spec_gap_evaluator_node(state: ShipAgentState) -> Dict[str, Any]:
    if state.get("task_type") != "doc_generation":
        return {}

    docs = state.get("retrieved_docs", [])
    params = state.get("extracted_params", {}) or {}
    preview_docs = "\n".join(
        f"[{d.get('metadata', {}).get('document_name', '규격')}]: {str(d.get('content'))[:500]}"
        for d in docs
    ) if docs else "인출된 규격 문서 없음"

    try:
        res: SpecGapAnalysis = (SPEC_EVAL_PROMPT | structured_spec_evaluator).invoke({
            "docs": preview_docs,
            "params": json.dumps(params, ensure_ascii=False),
            "query": state.get("user_query", "")
        })

        missing = list(res.missing_fields) if not res.is_ready_to_generate else []
        if not params.get("location") and "작업 위치(조선소/현장명)" not in missing:
            missing.append("작업 위치(조선소/현장명)")

        is_complete = (len(missing) == 0)
        print(f"📋 [Spec Gap 분석]: 확보된 제원={params} | RAG 대조 누락 항목={missing}")

        return {
            "missing_params": missing,
            "is_param_complete": "yes" if is_complete else "no"
        }
    except Exception as e:
        print(f"[Team 2 Spec Gap Warning]: {e}")
        return {"missing_params": [], "is_param_complete": "yes"}


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