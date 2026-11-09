"""
modules/agent/subgraphs/search.py
Team 2: 지식(RAG)·환경(Weather MCP)·공급망(303호 B2B MCP) 자율 감리 서브그래프
"""

import json
from typing import Any, Dict, List, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from modules.agent.state import ShipAgentState
from modules.agent.tools.domain_tools import evaluate_environmental_safety
from modules.agent.tools.erp_tools import query_partner_erp_b2b
from modules.agent.tools.retriever_tools import search_knowledge_by_categories
from modules.agent.tools.weather_mcp_client import get_realtime_weather_from_mcp

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ==============================================================================
# 1. 동적 RAG 지식 인출 노드 (rag_retriever)
# ==============================================================================
def dynamic_rag_retriever_node(state: ShipAgentState) -> Dict[str, Any]:
    task_type = state.get("task_type", "knowledge_qa")
    # 순수 ERP/인사/외주 단답 조회일 때는 무거운 RAG 검색 생략
    if task_type in ("erp_resource_qa", "hr_safety_qa", "scm_partner_qa", "chat"):
        return {"retrieved_docs": [], "search_status": "completed", "doc_grade": "yes"}

    search_queries = list(state.get("search_queries") or [])
    user_query = state.get("user_query", "").strip()
    categories = state.get("target_categories") or None

    if user_query and user_query not in search_queries:
        search_queries.append(user_query)

    collected_docs: List[Dict[str, Any]] = []
    seen_ids = set()

    for q in search_queries[:3]:
        try:
            raw_items = search_knowledge_by_categories(query=q, categories=categories, top_k=4)
            # 카테고리 필터 결과 0건이면 전체 카테고리 폴백 검색
            if not raw_items and categories:
                raw_items = search_knowledge_by_categories(query=q, categories=None, top_k=4)
            print(f"   🔎 [ChromaDB 조회] 쿼리={q} ➔ {len(raw_items)}건 인출됨")
            for item in raw_items:
                cid = item.get("chunk_id") or str(hash(item.get("chunk_text", "")))
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    collected_docs.append({
                        "category": item.get("category", "규격"),
                        "content": item.get("chunk_text", "") or str(item),
                        "metadata": {
                            "document_name": item.get("document_name"),
                            "revision": item.get("revision"),
                            "page": item.get("page_start"),
                            "similarity": item.get("similarity_score"),
                            "source": item.get("source"),
                            "doc_id": cid,
                        },
                    })
        except Exception as e:
            print(f"[Team 2 RAG Search Warning]: {e}")
    status = "completed" if collected_docs else "empty"
    return {"retrieved_docs": collected_docs, "search_status": status}


# ==============================================================================
# 2. 시맨틱 문서 적합성 평가 노드 (semantic_doc_grader) - Evaluator
# ==============================================================================
class RelevantDocIndices(BaseModel):
    relevant_indices: List[int] = Field(
        default_factory=list,
        description="사용자 질의 및 작업 맥락과 연관이 있는 문서의 인덱스 번호(0부터 시작) 목록",
    )


structured_grader = llm.with_structured_output(RelevantDocIndices, method="function_calling")

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 기술 문서 적합성 감사 AI(Evaluator)입니다.\n"
        "검색된 문서 청크들이 [사용자 질의]에 답변하거나 작업을 수행하는 데 참고 자료가 되는지 평가하여 인덱스 번호를 반환하세요.\n"
        "- 일반 기술/과정 질문인 경우 [누적 작업 제원]과 무관하더라도 질문과 관련된 문서라면 반드시 포함하십시오.",
    ),
    (
        "human",
        "[작업 유형]: {task_type}\n"
        "[사용자 질의]: {query}\n"
        "[누적 작업 제원]: {params}\n\n"
        "[검색된 문서 목록]:\n{docs_preview}",
    ),
])


def semantic_doc_grader_node(state: ShipAgentState) -> Dict[str, Any]:
    task_type = state.get("task_type", "knowledge_qa")
    if task_type in ("erp_resource_qa", "hr_safety_qa", "scm_partner_qa", "chat"):
        return {"doc_grade": "yes", "search_status": "completed"}

    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"retrieved_docs": [], "search_status": "empty", "doc_grade": "no"}

    query = state.get("user_query", "")
    params = (state.get("extracted_params") or {}) if task_type == "doc_generation" else {}

    preview_lines = [
        f"[{idx}] {d.get('metadata', {}).get('document_name', '문서')}: {str(d.get('content', ''))[:300].replace(chr(10), ' ')}"
        for idx, d in enumerate(docs)
    ]

    try:
        res: RelevantDocIndices = (GRADER_PROMPT | structured_grader).invoke({
            "task_type": task_type,
            "query": query,
            "params": json.dumps(params, ensure_ascii=False),
            "docs_preview": "\n".join(preview_lines),
        })
        filtered = [docs[i] for i in res.relevant_indices if 0 <= i < len(docs)]

        # 재시도 후에도 필터가 비어있다면 벡터 유사도 상위 2건 보존
        if not filtered and state.get("retry_count", 0) >= 1:
            filtered = docs[:2]
        is_good = "yes" if filtered else "no"
        print(f"   ⚖️ [Doc Grader] 전체 {len(docs)}건 중 {len(filtered)}건 채택 (판정: {is_good})")
        return {
            "retrieved_docs": filtered,
            "search_status": "completed" if filtered else "empty",
            "doc_grade": is_good,
        }
    except Exception as e:
        print(f"[Team 2 Grader Warning]: {e}")
        return {"retrieved_docs": docs[:2], "search_status": "completed", "doc_grade": "yes"}


# ==============================================================================
# 3. 자율 쿼리 재작성 노드 (query_rewriter) - Optimizer Loop
# ==============================================================================
class RewrittenQueries(BaseModel):
    queries: List[str] = Field(description="재검색을 위한 조선/용접 공학 표준 키워드 쿼리 1~2개")


structured_rewriter = llm.with_structured_output(RewrittenQueries, method="function_calling")


def rewrite_query_node(state: ShipAgentState) -> Dict[str, Any]:
    query = state.get("user_query", "")
    params = state.get("extracted_params") or {}
    retry = state.get("retry_count", 0) + 1

    prompt = (
        f"이전 검색어({state.get('search_queries')})로 사내 규격 문서를 찾지 못했습니다.\n"
        f"사용자 질문('{query}')과 제원({json.dumps(params, ensure_ascii=False)})을 바탕으로 "
        f"ChromaDB 검색 적중률을 높일 수 있는 핵심 공학 키워드 검색어 2개를 생성하세요."
    )
    try:
        res: RewrittenQueries = structured_rewriter.invoke(prompt)
        new_queries = res.queries if res.queries else [query]
    except Exception:
        new_queries = [query]

    print(f"   🔄 [Agentic Retry #{retry}] 쿼리 재작성 -> {new_queries}")
    return {
        "search_queries": new_queries,
        "target_categories": [],  # 재검색 시 카테고리 제한 해제
        "retry_count": retry,
    }


def route_after_grader(state: ShipAgentState) -> Literal["rewrite_query", "b2b_scm_collector"]:
    if state.get("doc_grade") == "yes":
        return "b2b_scm_collector"
    if state.get("retry_count", 0) < 1:  # 최대 1회 자율 재검색 루프
        return "rewrite_query"
    return "b2b_scm_collector"


# ==============================================================================
# 4. 303호 B2B 협력사 공급망 자율 조회 노드 (b2b_scm_collector)
# ==============================================================================
class OutsourcingRequirement(BaseModel):
    """사용자 질의 및 결품 상태 기반 외주 공급망 필요 도메인 자율 분석"""
    domain: Literal["HEAVY_TRANSPORT", "WELDING_MATERIAL"] = Field(
        description="운송/이송/배차/중량물 이동 지원 필요 시 'HEAVY_TRANSPORT', 용접봉/와이어/강판 등 자재 수급 필요 시 'WELDING_MATERIAL'"
    )

def b2b_scm_collector_node(state: ShipAgentState) -> Dict[str, Any]:
    """하드코딩 키워드 없이 LLM 문맥 추론과 ERP 진단 상태로 외주 협력사를 자율 결정"""
    is_outsourcing = state.get("is_outsourcing_required", False)
    task_type = state.get("task_type")

    if not is_outsourcing and task_type != "scm_partner_qa":
        return {"partner_erp_plan": None}

    erp_status = state.get("erp_resource_status") or {}
    params = state.get("extracted_params") or {}
    query = state.get("user_query", "")

    # 🌟 하드코딩 리스트 완전 삭제: LLM이 질의 맥락과 사내 결품 내역을 보고 외주 성격을 자율 판정
    classifier = llm.with_structured_output(OutsourcingRequirement)
    prompt = [
        ("system", 
         "당신은 스마트 조선소 공급망(SCM) 자원 조정관입니다. "
         "사용자 요청과 본사 자원 진단 결과를 바탕으로 외주 의뢰 대상이 '중량물 운송/배차(HEAVY_TRANSPORT)'인지, "
         "'용접 자재 조달(WELDING_MATERIAL)'인지 자율 판단하십시오."),
        ("human", f"[사용자 질의]: {query}\n[본사 자원 진단 사유]: {erp_status.get('outsourcing_reason')}")
    ]
    
    try:
        decision: OutsourcingRequirement = classifier.invoke(prompt)
        shortage_domain = decision.domain
    except Exception:
        # 안전 폴백: ERP 진단 문자열 원문 그대로 활용
        shortage_domain = "HEAVY_TRANSPORT" if "TRANSPORT" in str(erp_status.get("outsourcing_reason")).upper() else "MATERIAL_SHORTAGE"

    # 외주 협력사 게이트웨이 호출
    partner_plan = query_partner_erp_b2b.invoke({
        "shortage_reason": shortage_domain,
        "material": params.get("material", "SS400"),
        "required_wire_kg": 20.0,
    })
    return {"partner_erp_plan": partner_plan}


# ==============================================================================
# 5. 실시간 기상 수집 및 안전 룰 판정 노드 (weather & env_safety_critic)
# ==============================================================================
def weather_and_safety_node(state: ShipAgentState) -> Dict[str, Any]:
    """
    Team 2: 실시간 환경 기상 수집 및 도메인 안전 룰 엔진 판정 노드
    - 하드코딩 키워드 검사 배제
    - 실제 센서/서버 계측치 기반 정밀 안전 판정 (Fail-Closed 원칙)
    """
    need_weather = state.get("need_weather", False)
    task_type = state.get("task_type", "knowledge_qa")

    # 기상 조회가 불필요한 질의는 스킵
    if not need_weather and task_type != "doc_generation":
        return {"weather_data": None}

    params = state.get("extracted_params") or {}
    location = params.get("location")

    if not location:
        return {
            "weather_data": {
                "status": "NEED_LOCATION",
                "message": "작업 구역(지역명)이 명시되지 않아 기상 조회를 수행할 수 없습니다."
            }
        }

    # FastMCP 클라이언트 호출
    weather_info = dict(get_realtime_weather_from_mcp(location) or {})
    status = weather_info.get("status")

    # 통신 실패 또는 계측치 누락 시 가짜 수치 날조 차단 (Fail-Closed)
    if status != "SUCCESS":
        print(f"⚠️ [FastMCP 기상 조회 비정상]: {weather_info.get('error') or weather_info}")
        return {
            "weather_data": {
                "location": location,
                "status": status or "ERROR",
                "error": weather_info.get("error") or "기상 데이터 결측",
                "raw_text": weather_info.get("raw_text")
            }
        }

    temp = weather_info.get("temperature")
    wind = weather_info.get("wind_speed")
    rain = weather_info.get("precipitation", 0.0)
    hum = weather_info.get("humidity", 50.0)

    try:
        process_ctx = params.get("process") or params.get("equipment") or "야외 용접"
        
        # 환경 안전 룰 엔진 호출
        safety_res = evaluate_environmental_safety.invoke({
            "current_temp": float(temp),
            "current_wind": float(wind),
            "rainfall": float(rain) if rain is not None else 0.0,
            "thresholds": None,
            "task_context": str(process_ctx),
            "is_outdoor": True
        })
        weather_info["safety_evaluation"] = safety_res

        # -------------------------------------------------------------
        # 📌 [디버깅 로그 삽입 위치]
        # -------------------------------------------------------------
        print(f"   ⛅ [Weather MCP] 상태={status}, 기온={temp}℃, 풍속={wind}m/s")
        print(f"   🛡️ [Safety Critic] 최종 판정: {safety_res.get('status')} (사유: {safety_res.get('mandatory_rules') or safety_res.get('prerequisites') or '정상'})")
        # -------------------------------------------------------------

        return {"weather_data": weather_info}

    except Exception as e:
        print(f"⚠️ [도메인 안전 룰 엔진 오류]: {e}")
        return {
            "weather_data": {
                "location": location,
                "status": "EVALUATION_ERROR",
                "error": f"환경 안전 기준 평가 실패: {str(e)}"
            }
        }


# ==============================================================================
# 6. 서브그래프 조립
# ==============================================================================
builder = StateGraph(ShipAgentState)

builder.add_node("rag_retriever", dynamic_rag_retriever_node)
builder.add_node("semantic_doc_grader", semantic_doc_grader_node)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("b2b_scm_collector", b2b_scm_collector_node)
builder.add_node("weather_and_safety", weather_and_safety_node)

builder.add_edge(START, "rag_retriever")
builder.add_edge("rag_retriever", "semantic_doc_grader")
builder.add_conditional_edges(
    "semantic_doc_grader",
    route_after_grader,
    {
        "rewrite_query": "rewrite_query",
        "b2b_scm_collector": "b2b_scm_collector",
    },
)
builder.add_edge("rewrite_query", "rag_retriever")
builder.add_edge("b2b_scm_collector", "weather_and_safety")
builder.add_edge("weather_and_safety", END)

team2_subgraph = builder.compile()