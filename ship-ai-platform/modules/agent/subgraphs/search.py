"""
Team 2: 지식·환경 수집 및 정합성 감리 서브그래프 (search.py)
- Node 1 (rag_retriever): 검색어 순회 및 ChromaDB 문서 인출
- Node 2 (weather_collector): need_weather=True일 때만 FastMCP 실시간 기상 수집
- Node 3 (semantic_doc_grader): 질문 의도에 맞는 문서 선별 (일반 질문 시 이전 제원 간섭 배제 및 0건 탈락 방지)
- Node 4 (env_safety_critic): 기상 데이터 확보 시 환경 안전 기준 판정
- Node 5 (spec_gap_evaluator): 오직 doc_generation일 때만 인출된 RAG 규격서와 사용자 입력을 대조해 부족한 현장 조건 도출
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
    task_type = state.get("task_type", "knowledge_qa")
    if task_type == "chat":
        return {"retrieved_docs": [], "search_status": "empty"}

    search_queries = [q.strip() for q in (state.get("search_queries") or []) if q and q.strip()]
    user_query = state.get("user_query", "").strip()
    categories = state.get("target_categories") or None

    # 순수 날씨 질문(검색어 없음 & 날씨 필요 & 문서생성 아님)인 경우 불필요한 RAG 검색 스킵
    if not search_queries:
        if state.get("need_weather") and task_type != "doc_generation":
            return {"retrieved_docs": [], "search_status": "empty"}
        if user_query:
            search_queries = [user_query]
        else:
            return {"retrieved_docs": [], "search_status": "empty"}
    
    # LLM 생성 검색어뿐만 아니라 사용자 원문 질의(user_query)도 검색 후보에 포함하여 검색 누락 방지
    if user_query and user_query not in search_queries:
        search_queries.append(user_query)

    collected_docs: List[Dict[str, Any]] = []
    seen_doc_ids = set()

    try:
        for target_query in search_queries[:3]:
            raw_items = search_knowledge_by_categories(
                query=target_query,
                categories=categories,
                top_k=4
            )
            print(f"   🔎 [ChromaDB 조회] 검색어='{target_query}' -> {len(raw_items)}건 인출됨")
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
        description="사용자 질의 및 작업 목적과 직간접적으로 연관된 문서의 인덱스 번호(0부터 시작) 목록"
    )


structured_grader = llm.with_structured_output(RelevantDocIndices, method="function_calling")

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 기술 문서 적합성 감사 AI입니다.\n"
     "검색된 문서 청크들이 [사용자 질의]에 답변하거나 작업을 수행하는 데 참고 자료가 되는지 평가하여 인덱스 번호(0, 1, 2...)를 반환하세요.\n"
     "- 조금이라도 관련된 기술 개념, 공정 절차, 안전 수칙, 규격 표가 포함되어 있다면 탈락시키지 말고 포함하십시오."),
    ("human",
     "[작업 목적]: {task_type}\n"
     "[사용자 질의]: {query}\n"
     "[작업 제원(문서 발행 시에만 참고)]: {params}\n\n"
     "[검색된 문서 목록]:\n{docs_preview}")
])


def semantic_doc_grader_node(state: ShipAgentState) -> Dict[str, Any]:
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {
            "retrieved_docs": [],
            "search_status": "error" if state.get("search_status") == "error" else "empty"
        }

    task_type = state.get("task_type", "knowledge_qa")
    query = state.get("user_query", "")
    params = (state.get("extracted_params", {}) or {}) if task_type == "doc_generation" else {}

    preview_lines = [
        f"[{idx}] {d.get('metadata', {}).get('document_name', '문서')}: {str(d.get('content', ''))[:300].replace(chr(10), ' ')}"
        for idx, d in enumerate(docs)
    ]

    try:
        res: RelevantDocIndices = (GRADER_PROMPT | structured_grader).invoke({
            "task_type": task_type,
            "query": query,
            "params": json.dumps(params, ensure_ascii=False),
            "docs_preview": "\n".join(preview_lines)
        })
        # LLM이 적합하다고 판정한 인덱스의 문서만 필터링 (엉뚱한 문서는 정직하게 버림)
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
     "- 만약 [작업 맥락]에서 사용자가 특정 가상 기상 조건을 가정했다면 실측 기상 대신 가정한 수치를 우선 적용하십시오.\n"
     "- thresholds 인자에는 단일 숫자(float) 값들로 구성된 딕셔너리를 전달하십시오."),
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
# 5. RAG 문서 기반 동적 필수 제원 검증 노드 (spec_gap_evaluator)
# ==============================================================================
class SpecGapAnalysis(BaseModel):
    newly_extracted_params: Dict[str, str] = Field(
        default_factory=dict,
        description="현재 사용자 입력에서 추가로 확인된 현장 작업 조건"
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="문서 발행을 위해 작업자가 지정해야 하나 아직 확인되지 않은 기본 작업 대상 항목 목록. 모두 충족되었으면 빈 리스트([])"
    )
    is_ready_to_generate: bool = Field(
        description="누락된 기본 작업 대상이 없어 즉시 작업지시서/계획서를 발행할 수 있는지 여부 (missing_fields가 비어있으면 True)"
    )


structured_spec_evaluator = llm.with_structured_output(SpecGapAnalysis, method="function_calling")

SPEC_EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 사내 표준 규격(RAG) 대조 심사 엔지니어 AI(Team 2)입니다.\n"
     "[인출된 사내 규격 문서(RAG)]를 바탕으로 공식 작업지시서/계획서를 발행하기 위해 현장 작업자가 알려주어야 하는 기본 대상 정보가 [현재 확보된 제원]과 [현재 사용자 입력]에 갖추어져 있는지 판정하세요.\n\n"
     "[대조 및 판정 원칙]\n"
     "1. [현재 사용자 입력]에 새로운 작업 조건이 있다면 `newly_extracted_params`에 추출하여 기록하세요.\n"
     "2. 규격서를 적용하기 위해 작업자가 반드시 알려주어야 하는 '기본 작업 대상(예: 어떤 자재/두께를 쓰는지 또는 어떤 대상을 작업하는지)'이 전혀 확인되지 않는 경우에만 그 항목을 `missing_fields`에 적고 `is_ready_to_generate=False`로 판정하십시오.\n"
     "3. ⚠️ 절대 주의: 규격서(RAG) 본문이나 표에 적혀 있는 세부 기술 결과값(예: 권장 전류/전압 범위, 와이어 직경/규격명, 용착금속 화학성분, 기계적 성질, 예열 온도, 가스 유량, 승인 기준 등)은 AI가 문서를 보고 지시서에 채워 넣을 값입니다. **절대 사용자에게 요구(`missing_fields`에 추가)하지 마십시오!**\n"
     "4. 기본 작업 대상 정보가 이미 확인되었다면 사족을 달지 말고 즉시 `missing_fields=[]`, `is_ready_to_generate=True`로 승인하십시오."),
    ("human",
     "[현재 확보된 제원 (이전 대화 누적 포함)]: {params}\n"
     "[현재 사용자 입력]: {query}\n"
     "[인출된 사내 규격 문서(RAG)]:\n{docs}")
])


def spec_gap_evaluator_node(state: ShipAgentState) -> Dict[str, Any]:
    # 문서 생성(doc_generation)이 아닌 모든 질의(knowledge_qa, chat 등)는 제원 누락 검사를 즉시 통과(Bypass)
    if state.get("task_type") != "doc_generation":
        return {"missing_params": [], "is_param_complete": "yes"}

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
                params[k] = str(v).strip()

        missing = list(res.missing_fields) if not res.is_ready_to_generate else []
        if not params.get("location") and not any("위치" in m for m in missing):
            missing.append("작업 위치(조선소/현장 지역명)")

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
# 6. 서브그래프 조립 (직렬 파이프라인으로 병렬 덮어쓰기 충돌 방지)
# ==============================================================================
builder = StateGraph(ShipAgentState)

builder.add_node("rag_retriever", dynamic_rag_retriever_node)
builder.add_node("semantic_doc_grader", semantic_doc_grader_node)
builder.add_node("weather_collector", conditional_weather_node)
builder.add_node("env_safety_critic", env_safety_critic_node)
builder.add_node("spec_gap_evaluator", spec_gap_evaluator_node)

builder.add_edge(START, "rag_retriever")
builder.add_edge("rag_retriever", "semantic_doc_grader")
builder.add_edge("semantic_doc_grader", "weather_collector")
builder.add_edge("weather_collector", "env_safety_critic")
builder.add_edge("env_safety_critic", "spec_gap_evaluator")
builder.add_edge("spec_gap_evaluator", END)

team2_subgraph = builder.compile()