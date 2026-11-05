"""
Team 2: 지식·환경 수집 및 Function Calling 감리 서브그래프 (search.py)
- Node 1 (rag_retriever): 복수 검색어 순회 및 카테고리 연동 ChromaDB 인출
- Node 2 (weather_collector): FastMCP 실시간 기상 수집
- Node 3 (semantic_doc_grader): 무관한 RAG 문서 필터링 (오류/부적합 시 무관 문서 복구 금지)
- Node 4 (env_safety_critic): evaluate_environmental_safety 도구 자율 실행 (기온/풍속/강수량 연동)
- Node 5 (spec_gap_evaluator): 인출된 RAG 규격서와 확보된 제원을 대조하여 누락 항목 동적 판정 (Fail-Closed)
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
    print("dynamic_rag_retriever_node")
    search_queries = [q.strip() for q in (state.get("search_queries") or []) if q and q.strip()]
    user_query = state.get("user_query", "").strip()
    categories = state.get("target_categories") or None

    # 일상 대화(chat)이거나 순수 날씨 질문(검색어 없음 & 문서생성 아님)인 경우 불필요한 RAG 검색 스킵
    if not search_queries and state.get("task_type") != "doc_generation":
        if state.get("need_weather") or state.get("task_type") == "chat":
            return {"retrieved_docs": [], "search_status": "empty"}
        if user_query:
            search_queries = [user_query]

    if not search_queries and user_query:
        search_queries = [user_query]

    if not search_queries:
        return {"retrieved_docs": [], "search_status": "empty"}

    collected_docs: List[Dict[str, Any]] = []
    seen_doc_ids = set()

    try:
        # 첫 번째 검색어만 쓰지 않고 계획된 검색어 전체(최대 3개)를 순회하여 중복 없이 수집
        for target_query in search_queries[:3]:
            raw_items = search_knowledge_by_categories(
                query=target_query,
                categories=categories,
                top_k=4
            )
            for item in raw_items:
                doc_id = item.get("chunk_id") or item.get("id") or str(item.get("chunk_text", ""))[:80]
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
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

        collected_docs.sort(
            key=lambda d: float((d.get("metadata") or {}).get("similarity") or 0.0),
            reverse=True
        )
        final_docs = collected_docs[:4]
        status = "completed" if final_docs else "empty"
        return {"retrieved_docs": final_docs, "search_status": status}
    except Exception as e:
        print(f"[Team 2 Search Error]: {e}")
        return {"retrieved_docs": [], "search_status": "error"}


# ==============================================================================
# 2. FastMCP 기상 도구 실행 노드 (weather_collector)
# ==============================================================================
def conditional_weather_node(state: ShipAgentState) -> Dict[str, Any]:
    print("conditional_weather_node")
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
        return {
            "retrieved_docs": [],
            "search_status": "error" if state.get("search_status") == "error" else "empty"
        }

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
        return {
            "retrieved_docs": filtered,
            "search_status": "completed" if filtered else "empty"
        }
    except Exception as e:
        print(f"[Team 2 Grader Warning]: {e}")
        return {"retrieved_docs": [], "search_status": "error"}


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
     "[실측 기상]: 기온 {temp}℃, 풍속 {wind}m/s, 강수량 {rain}mm/h\n"
     "[작업 맥락]: {query}\n"
     "[확보된 제원]: {params}\n"
     "[규격 문서]:\n{docs_text}")
])


def env_safety_critic_node(state: ShipAgentState) -> Dict[str, Any]:
    print("env_safety_critic_node")
    weather = state.get("weather_data")
    if not weather or weather.get("status") != "SUCCESS":
        return {}

    temp = weather.get("temperature")
    wind = weather.get("wind_speed")
    rain = weather.get("rainfall", 0.0)

    docs = state.get("retrieved_docs", [])
    docs_text = "\n".join(str(d.get("content", ""))[:400] for d in docs) if docs else "참조 규격 없음"
    query = state.get("user_query", "")
    params = state.get("extracted_params", {}) or {}
    actual_process = params.get("process") or query

    # 기온이나 풍속이 결측된 경우 즉시 Fail-Closed 판정 실행
    if temp is None or wind is None:
        safety_eval = evaluate_environmental_safety.invoke({
            "current_temp": temp,
            "current_wind": wind,
            "rainfall": rain,
            "task_context": actual_process,
            "task_type": actual_process
        })
        enriched_weather = dict(weather)
        enriched_weather["safety_evaluation"] = safety_eval
        return {"weather_data": enriched_weather}

    try:
        ai_msg = (SAFETY_TOOL_PROMPT | llm_with_safety_tool).invoke({
            "temp": float(temp),
            "wind": float(wind),
            "rain": float(rain or 0.0),
            "query": query,
            "params": json.dumps(params, ensure_ascii=False),
            "docs_text": docs_text
        })

        tool_calls = getattr(ai_msg, "tool_calls", [])
        if tool_calls:
            call_args = dict(tool_calls[0]["args"])
            call_args.setdefault("rainfall", rain)
            safety_eval = evaluate_environmental_safety.invoke(call_args)
        else:
            safety_eval = evaluate_environmental_safety.invoke({
                "current_temp": float(temp),
                "current_wind": float(wind),
                "rainfall": rain,
                "task_context": actual_process,
                "task_type": actual_process
            })
    except Exception as e:
        print(f"[Team 2 Safety Critic Warning]: {e}")
        safety_eval = evaluate_environmental_safety.invoke({
            "current_temp": float(temp),
            "current_wind": float(wind),
            "rainfall": rain,
            "task_context": actual_process,
            "task_type": actual_process
        })

    enriched_weather = dict(weather)
    enriched_weather["safety_evaluation"] = safety_eval
    return {"weather_data": enriched_weather}


# ==============================================================================
# 5. RAG 문서 기반 동적 필수 제원 검증 노드 (spec_gap_evaluator) - Fail-Closed 적용
# ==============================================================================
class SpecGapAnalysis(BaseModel):
    newly_extracted_params: Dict[str, str] = Field(
        default_factory=dict,
        description="현재 사용자 입력에서 추가로 확인된 작업 제원 (예: {'material': 'SS400', 'thickness': '10T', 'process': '맞대기'})"
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="인출된 RAG 규격서를 적용하기 위해 필수적이나 [현재 확보된 제원]과 [사용자 입력] 어디에도 없는 항목 목록. 모두 충족되었으면 빈 리스트([])"
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
     "1. [현재 사용자 입력]에 새로운 제원(강종, 두께, 자세, 공정 등)이 있다면 `newly_extracted_params`에 추출하세요.\n"
     "2. 규격서에서 전류/전압 등의 결과값을 찾기 위한 '현장 입력 조건(예: 대상 모재 강종, 판 두께 등)'이 [확보된 제원]과 [현재 사용자 입력]을 합쳐도 여전히 빠져 있다면, 그 부족한 항목만 `missing_fields`에 적고 `is_ready_to_generate=False`로 판정하십시오.\n"
     "3. [확보된 제원]과 [현재 사용자 입력]에 대상 모재와 두께/공정 정보가 이미 들어있다면, 즉시 `missing_fields=[]`, `is_ready_to_generate=True`로 승인하십시오.\n"
     "4. 규격서 본문에 적힌 권장 전류, 전압, 와이어경, 예열온도 등은 AI가 규격서를 보고 작성할 결과값이므로 절대 사용자에게 되묻지 마십시오."),
    ("human",
     "[현재 확보된 제원 (이전 대화 누적 포함)]: {params}\n"
     "[현재 사용자 입력]: {query}\n"
     "[인출된 사내 규격 문서(RAG)]:\n{docs}")
])


def spec_gap_evaluator_node(state: ShipAgentState) -> Dict[str, Any]:
    if state.get("task_type") != "doc_generation":
        return {"is_param_complete": "yes"}

    docs = state.get("retrieved_docs", [])
    params = dict(state.get("extracted_params", {}) or {})
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

        for k, v in (res.newly_extracted_params or {}).items():
            if v and str(v).upper() not in ("NONE", "NULL", ""):
                params[k] = v

        missing = list(res.missing_fields) if not res.is_ready_to_generate else []
        if not params.get("location") and "작업 위치(조선소/현장명)" not in missing:
            missing.append("작업 위치(조선소/현장명)")

        is_complete = (len(missing) == 0)
        print(f"📋 [Spec Gap 분석]: 통합 제원={params} | RAG 대조 누락 항목={missing}")

        return {
            "extracted_params": params,
            "missing_params": missing,
            "is_param_complete": "yes" if is_complete else "no"
        }
    except Exception as e:
        print(f"[Team 2 Spec Gap Error]: {e}")
        return {
            "missing_params": ["규격 제원 검증 실패 (재입력 필요)"],
            "is_param_complete": "no"
        }


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