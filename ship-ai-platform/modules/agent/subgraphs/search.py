"""
modules/agent/subgraphs/search.py
Team 2: 지식(RAG + 보안 필터)·환경(Weather MCP)·비전(YOLOv11 궤적/안전거리)·공급망(PostgreSQL B2B) 자율 감리 서브그래프
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
from modules.agent.tools.vision_tools import get_vision_inspection_status
from modules.agent.tools.weather_mcp_client import get_realtime_weather_from_mcp

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ==============================================================================
# 1. 동적 RAG 지식 인출 노드 (rag_retriever + CAN_ACCESS_INTERNAL_RAG 보안 필터)
# ==============================================================================
def dynamic_rag_retriever_node(state: ShipAgentState) -> Dict[str, Any]:
    task_type = state.get("task_type", "knowledge_qa")
    if task_type in ("erp_resource_qa", "hr_safety_qa", "hr_worker_qa", "scm_partner_qa", "chat"):
        return {"retrieved_docs": [], "search_status": "completed", "doc_grade": "yes"}

    search_queries = list(state.get("search_queries") or [])
    user_query = state.get("user_query", "").strip()
    categories = state.get("target_categories") or None
    can_access_internal = state.get("can_access_internal_rag", "Y") != "N"

    if not can_access_internal and categories:
        categories = [c for c in categories if c != "internal_docs"]

    if user_query and user_query not in search_queries:
        search_queries.append(user_query)

    collected_docs = []
    seen_ids = set()

    for q in search_queries[:3]:
        try:
            raw_items = search_knowledge_by_categories(query=q, categories=categories, top_k=4)
            if not raw_items and categories:
                raw_items = search_knowledge_by_categories(query=q, categories=None, top_k=4)
            print(f"   🔎 [ChromaDB 조회] 쿼리={q} ➔ {len(raw_items)}건 인출됨")
            for item in raw_items:
                col_type = str(item.get("collection_type") or item.get("access_level") or "PUBLIC").upper()
                if not can_access_internal and col_type == "INTERNAL":
                    continue

                cid = item.get("chunk_id") or str(hash(item.get("chunk_text", "")))
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    collected_docs.append({
                        "category": item.get("category", "규격"),
                        "content": item.get("chunk_text", "") or str(item),
                        "metadata": {
                            "document_name": item.get("document_name"),
                            "document_code": item.get("document_code"),
                            "collection_type": col_type,
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
# 2. 시맨틱 문서 적합성 평가 노드 (semantic_doc_grader)
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
    if task_type in ("erp_resource_qa", "hr_safety_qa", "hr_worker_qa", "scm_partner_qa", "chat"):
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
# 3. 자율 쿼리 재작성 노드 (query_rewriter)
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
        "target_categories": [],
        "retry_count": retry,
    }


def route_after_grader(state: ShipAgentState) -> Literal["rewrite_query", "b2b_scm_collector"]:
    if state.get("doc_grade") == "yes":
        return "b2b_scm_collector"
    if state.get("retry_count", 0) < 1:
        return "rewrite_query"
    return "b2b_scm_collector"


# ==============================================================================
# 4. PostgreSQL B2B 협력사 공급망 실시간 조회 노드 (b2b_scm_collector)
# ==============================================================================
class OutsourcingRequirement(BaseModel):
    domain: Literal["HEAVY_TRANSPORT", "WELDING_MATERIAL", "OUTSOURCED_WORKER"] = Field(
        description="운송/이송/배차/중량물 이동 지원 필요 시 'HEAVY_TRANSPORT', 용접봉/와이어/강판 등 자재 수급 필요 시 'WELDING_MATERIAL', 외주 용접 인력 요청 시 'OUTSOURCED_WORKER'"
    )


def b2b_scm_collector_node(state: ShipAgentState) -> Dict[str, Any]:
    is_outsourcing = state.get("is_outsourcing_required", False)
    task_type = state.get("task_type")

    if not is_outsourcing and task_type != "scm_partner_qa":
        return {"partner_erp_plan": None}

    erp_status = state.get("erp_resource_status") or {}
    params = state.get("extracted_params") or {}
    query = state.get("user_query", "")

    classifier = llm.with_structured_output(OutsourcingRequirement)
    prompt = [
        (
            "system",
            "당신은 스마트 조선소 공급망(SCM) 자원 조정관입니다. "
            "사용자 요청과 본사 자원 진단 결과를 바탕으로 외주 의뢰 대상이 '중량물 운송/배차(HEAVY_TRANSPORT)'인지, "
            "'용접 자재 조달(WELDING_MATERIAL)'인지, '외주 인력 파견(OUTSOURCED_WORKER)'인지 자율 판단하십시오.",
        ),
        ("human", f"[사용자 질의]: {query}\n[본사 자원 진단 사유]: {erp_status.get('outsourcing_reason')}"),
    ]

    try:
        decision: OutsourcingRequirement = classifier.invoke(prompt)
        shortage_domain = decision.domain
    except Exception:
        shortage_domain = (
            "HEAVY_TRANSPORT"
            if "TRANSPORT" in str(erp_status.get("outsourcing_reason")).upper()
            else "WELDING_MATERIAL"
        )

    # 🌟 PostgreSQL B2B View 실시간 조회 호출
    partner_plan = query_partner_erp_b2b.invoke({
        "shortage_reason": shortage_domain,
        "material": params.get("material", ""),
        "required_wire_kg": 20.0,
        "target_region": params.get("location", "울산"),
    })
    return {"partner_erp_plan": partner_plan}


# ==============================================================================
# 5. 실시간 기상 수집, YOLOv11 비전 검사 연동 및 안전 룰 판정 노드
# ==============================================================================
def weather_and_safety_node(state: ShipAgentState) -> Dict[str, Any]:
    need_weather = state.get("need_weather", False)
    task_type = state.get("task_type", "knowledge_qa")
    params = state.get("extracted_params") or {}

    vision_info = state.get("vision_data")
    if task_type == "doc_generation" and not vision_info:
        vision_info = get_vision_inspection_status.invoke({
            "work_id": state.get("work_id") or params.get("block_id"),
            "camera_code": params.get("camera_code"),
        })

    if not need_weather and task_type != "doc_generation":
        return {"weather_data": None, "vision_data": vision_info}

    location = params.get("location")
    if not location or "ZONE" in str(location).upper():
        branch_code = params.get("branch_code") or state.get("branch_code")
        if branch_code and "GEOJE" in branch_code.upper():
            location = "거제"
        else:
            location = "울산"

    weather_info = dict(get_realtime_weather_from_mcp(location) or {})
    status = weather_info.get("status")

    if status != "SUCCESS":
        print(f"⚠️ [FastMCP 기상 조회 비정상]: {weather_info.get('error') or weather_info}")
        return {
            "weather_data": {
                "location": location,
                "status": status or "ERROR",
                "error": weather_info.get("error") or "기상 데이터 결측",
                "raw_text": weather_info.get("raw_text"),
            },
            "vision_data": vision_info,
        }

    raw_temp = weather_info.get("temperature")
    raw_wind = weather_info.get("wind_speed")
    raw_rain = weather_info.get("precipitation")

    temp = float(raw_temp) if raw_temp is not None else 20.0
    wind = float(raw_wind) if raw_wind is not None else 0.0
    rain = float(raw_rain) if raw_rain is not None else 0.0

    try:
        process_ctx = params.get("process") or params.get("equipment") or "야외 용접"

        safety_res = evaluate_environmental_safety.invoke({
            "current_temp": temp,
            "current_wind": wind,
            "rainfall": rain,
            "thresholds": None,
            "task_context": str(process_ctx),
            "is_outdoor": True,
        })

        if vision_info:
            if vision_info.get("safety_clearance_yn") == "N":
                rules = list(safety_res.get("mandatory_rules") or [])
                interfering = ", ".join(vision_info.get("interfering_objects") or ["간섭 장애물"])
                rules.append(f"비전 안전 인터록(VISION_INSPECTIONS): 용접선 인근 간섭({interfering}) 해소 후 착수 필수")
                safety_res["mandatory_rules"] = rules
                if safety_res.get("status") == "PASS":
                    safety_res["status"] = "CONDITIONAL_PASS"

        weather_info["safety_evaluation"] = safety_res

        print(f"   ⛅ [Weather MCP] 상태={status}, 기온={temp}℃, 풍속={wind}m/s")
        print(
            f"   👁️ [Vision YOLOv11] 상태={vision_info.get('welding_state') if vision_info else 'N/A'}, "
            f"예상용접길이={vision_info.get('estimated_length_mm') if vision_info else 'N/A'}mm, "
            f"안전거리={vision_info.get('safety_clearance_yn') if vision_info else 'Y'}"
        )
        print(f"   🛡️ [Safety Critic] 최종 판정: {safety_res.get('status')}")

        return {"weather_data": weather_info, "vision_data": vision_info}

    except Exception as e:
        print(f"⚠️ [도메인 안전 룰 엔진 오류]: {e}")
        return {
            "weather_data": {
                "location": location,
                "status": "EVALUATION_ERROR",
                "error": f"환경 안전 기준 평가 실패: {str(e)}",
            },
            "vision_data": vision_info,
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