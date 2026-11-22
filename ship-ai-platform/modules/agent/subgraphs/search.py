"""
modules/agent/subgraphs/search.py
Team 2: 내부 Supervisor(Orchestrator) 기반 지식(RAG)·공급망(MCP + A2A)·환경(Weather MCP)·비전(YOLOv11) 자율 감리 서브그래프
"""

import json
from typing import Any, Dict, List, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from modules.agent.state import ShipAgentState
from modules.agent.tools.domain_tools import evaluate_environmental_safety
from modules.agent.tools.erp_tools import get_branch_context_from_db, query_partner_erp_b2b
from modules.agent.tools.partner_mcp_client import negotiate_partner_via_a2a
from modules.agent.tools.retriever_tools import search_knowledge_by_categories
from modules.agent.tools.vision_tools import get_vision_inspection_status, get_vision_overview_for_qa
from modules.agent.tools.weather_mcp_client import get_realtime_weather_from_mcp

# Grader 및 분류 속도 최적화를 위해 즉시 응답 모델 사용
llm = ChatOpenAI(model="gpt-5-mini", reasoning_effort="low")


# ==============================================================================
# 0. Team 2 내부 Supervisor 노드 (동적 워커 스케줄러)
# ==============================================================================
def team2_supervisor_node(state: ShipAgentState) -> Dict[str, Any]:
    """
    현재 State와 아직 실행되지 않은 수집 과업을 점검하여
    다음에 호출할 Team 2 내 전문 워커를 결정합니다.
    """
    plan = state.get("team2_plan")

    if plan is None:
        task_type = state.get("task_type", "knowledge_qa")
        erp_status = state.get("erp_resource_status") or {}
        pref_mode = state.get("preferred_execution_mode", "AUTO")

        # 비전 관제 질의는 RAG·MCP를 전부 생략하고 비전 DB만 즉시 조회
        if task_type == "vision_inspection_qa":
            plan = ["vision_qa_collector"]
        else:
            erp_needs_outsource = bool(erp_status.get("require_outsourcing", False))
            is_outsourcing = (
                pref_mode == "OUTSOURCED_B2B"
                or bool(state.get("is_outsourcing_required", False))
                or erp_needs_outsource
            )
            if pref_mode == "IN_HOUSE" and task_type != "scm_partner_qa":
                is_outsourcing = False

            need_weather = state.get("need_weather", False)

            new_plan: List[str] = []
            if task_type in ("knowledge_qa", "doc_generation"):
                new_plan.append("rag_retriever")
            if task_type == "scm_partner_qa" or (
                task_type == "doc_generation" and is_outsourcing
            ):
                new_plan.append("b2b_scm_collector")
            if task_type == "doc_generation" or (
                need_weather
                and task_type not in ("scm_partner_qa", "erp_resource_qa", "hr_worker_qa", "hr_safety_qa")
            ):
                new_plan.append("weather_and_safety")

            plan = new_plan

    if not plan:
        return {"next": "__end__", "team2_plan": []}

    next_worker = plan[0]
    remaining_plan = plan[1:]
    return {"next": next_worker, "team2_plan": remaining_plan}


def route_from_team2_supervisor(
    state: ShipAgentState,
) -> Literal["rag_retriever", "b2b_scm_collector", "weather_and_safety", "vision_qa_collector", "__end__"]:
    nxt = state.get("next")
    if nxt in ("rag_retriever", "b2b_scm_collector", "weather_and_safety", "vision_qa_collector"):
        return nxt
    return "__end__"


# ==============================================================================
# 0-1. 비전 관제 자연어 질의 전용 경량 워커 (vision_qa_collector)
# ==============================================================================
def vision_qa_collector_node(state: ShipAgentState) -> Dict[str, Any]:
    params = state.get("extracted_params") or {}
    try:
        overview = get_vision_overview_for_qa.invoke({
            "block_id": params.get("block_id"),
            "camera_code": params.get("camera_code"),
        })
    except Exception as e:
        print(f"⚠️ [Vision QA Collector 오류]: {e}")
        overview = {"status": "ERROR", "error_message": str(e), "total_records": 0, "summary_counts": {}}

    print(
        f"   👁️ [Vision QA Collector] 조회 상태={overview.get('status')}, "
        f"전체={overview.get('total_records', 0)}건, "
        f"미용접대기={overview.get('summary_counts', {}).get('pending_weld_count', 0)}건"
    )
    return {"vision_data": overview}


# ==============================================================================
# 1. 동적 RAG 지식 인출 워커 (rag_retriever + DB 확정 제원 기반 멀티쿼리 확장)
# ==============================================================================
def dynamic_rag_retriever_node(state: ShipAgentState) -> Dict[str, Any]:
    task_type = state.get("task_type", "knowledge_qa")
    params = state.get("extracted_params") or {}
    base_queries = list(state.get("search_queries") or [])
    user_query = state.get("user_query", "").strip()
    categories = state.get("target_categories") or None
    can_access_internal = state.get("can_access_internal_rag", "Y") != "N"

    if not can_access_internal and categories:
        categories = [c for c in categories if c != "internal_docs"]

    effective_queries: List[str] = []
    if task_type == "doc_generation":
        if params.get("doc_type") == "HEAVY_LIFTING_PLAN":
            effective_queries.append(
                "중량물 취급 작업계획서 크레인 양중 줄걸이 와이어로프 체결 각도 60도 신호수 안전수칙"
            )
            effective_queries.append(
                "선박 블록 인양 러그 전도 낙하 방지 강풍 작업 중지 풍속 10m/s 안전지침"
            )
        else:
            proc = str(params.get("process") or "").upper().strip()
            mat = str(params.get("material") or "").strip()
            thk = str(params.get("thickness") or "").strip()
            jnt = str(params.get("joint_type") or "").strip()
            spec_tokens = " ".join(str(x) for x in [proc, mat, thk, jnt] if x).strip()

            if proc == "SAW":
                effective_queries.append(
                    f"SAW 서브머지드 아크 용접 {mat} {thk} {jnt} WPS 전류 전압 속도 입상 플럭스 예열 층간온도".strip()
                )
                effective_queries.append(
                    "SAW 서브머지드 아크 용접 플럭스 건조 온도 300℃ 보온 호퍼 흡습 방지 비파괴검사 RT UT"
                )
            elif spec_tokens:
                effective_queries.append(
                    f"{spec_tokens} WPS 용접 시공 조건 전류 전압 속도 보호가스 예열 층간온도"
                )
                effective_queries.append(
                    f"{proc} 용접재료 와이어 플럭스 건조 보관 습도 관리 비파괴검사 RT UT"
                )

    for q in base_queries:
        if q and q not in effective_queries:
            effective_queries.append(q)

    if user_query and user_query not in effective_queries:
        effective_queries.append(user_query)

    max_queries = 3 if task_type == "doc_generation" else 2
    collected_docs = []
    seen_ids = set()

    for q in effective_queries[:max_queries]:
        try:
            search_cats = None if task_type == "doc_generation" else categories
            raw_items = search_knowledge_by_categories(
                query=q, categories=search_cats, top_k=4
            )
            print(f"   🔎 [ChromaDB 조회] 쿼리='{q}' ➔ {len(raw_items)}건 인출됨")
            for item in raw_items:
                col_type = str(
                    item.get("collection_type")
                    or item.get("access_level")
                    or "PUBLIC"
                ).upper()
                if not can_access_internal and col_type in ("INTERNAL", "CONFIDENTIAL"):
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
# 2. 시맨틱 문서 적합성 평가 워커 (semantic_doc_grader)
# ==============================================================================
class RelevantDocIndices(BaseModel):
    relevant_indices: List[int] = Field(
        default_factory=list,
        description="사용자 질의 및 작업 맥락과 연관이 있는 문서의 인덱스 번호(0부터 시작) 목록",
    )


structured_grader = llm.with_structured_output(
    RelevantDocIndices, method="function_calling"
)

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 기술 문서 적합성 감사 AI(Evaluator)입니다.\n"
        "검색된 문서 청크들이 [사용자 질의]에 답변하거나 작업을 수행하는 데 참고 자료가 되는지 평가하여 인덱스 번호를 반환하세요.\n"
        "- [작업 유형]이 'doc_generation'이고 용접 작업지시서인 경우: 대상 용접 공정에 부합하는 용접 시공 조건(전류, 전압, 속도, 가스/플럭스, 예열/층간온도), 용접재료 건조 및 보관 기준, 비파괴검사(NDT), 또는 현장 안전 수칙 문서를 채택하십시오.\n"
        "- [작업 유형]이 'doc_generation'이고 중량물 취급 계획서(HEAVY_LIFTING_PLAN)인 경우: 크레인 양중, 줄걸이, 블록 운반, 낙하/전도 방지, 신호수, 풍속 안전 수칙 문서를 채택하십시오.\n"
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
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"retrieved_docs": [], "search_status": "empty", "doc_grade": "no"}

    query = state.get("user_query", "")
    params = (
        (state.get("extracted_params") or {})
        if task_type == "doc_generation"
        else {}
    )

    preview_lines = [
        f"[{idx}] ({d.get('metadata', {}).get('document_code') or d.get('metadata', {}).get('document_name', '문서')}): "
        f"{str(d.get('content', ''))[:500].replace(chr(10), ' ')}"
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
            filtered = docs[:4]
        is_good = "yes" if filtered else "no"
        print(
            f"   ⚖️ [Doc Grader] 전체 {len(docs)}건 중 {len(filtered)}건 채택 (판정: {is_good})"
        )
        return {
            "retrieved_docs": filtered,
            "search_status": "completed" if filtered else "empty",
            "doc_grade": is_good,
        }
    except Exception as e:
        print(f"[Team 2 Grader Warning]: {e}")
        return {
            "retrieved_docs": docs[:4],
            "search_status": "completed",
            "doc_grade": "yes",
        }


# ==============================================================================
# 3. 자율 쿼리 재작성 워커 (rewrite_query)
# ==============================================================================
class RewrittenQueries(BaseModel):
    queries: List[str] = Field(
        description="재검색을 위한 조선/용접/안전 공학 표준 키워드 쿼리 1~2개"
    )


structured_rewriter = llm.with_structured_output(
    RewrittenQueries, method="function_calling"
)


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


def route_after_grader(
    state: ShipAgentState,
) -> Literal["rewrite_query", "team2_supervisor"]:
    if state.get("doc_grade") == "yes":
        return "team2_supervisor"
    if state.get("retry_count", 0) < 1:
        return "rewrite_query"
    return "team2_supervisor"


# ==============================================================================
# 4. PostgreSQL B2B 협력사 MCP 카탈로그 + A2A 자율 협의 워커 (b2b_scm_collector)
# ==============================================================================
class OutsourcingRequirement(BaseModel):
    domain: Literal["HEAVY_TRANSPORT", "WELDING_MATERIAL", "OUTSOURCED_WORKER"] = Field(
        description="운송/이송/배차/중량물 이동 지원 필요 시 'HEAVY_TRANSPORT', 용접봉/와이어/강판 등 자재 수급 필요 시 'WELDING_MATERIAL', 외주 용접 인력 요청 시 'OUTSOURCED_WORKER'"
    )


def b2b_scm_collector_node(state: ShipAgentState) -> Dict[str, Any]:
    task_type = state.get("task_type", "scm_partner_qa")
    erp_status = state.get("erp_resource_status") or {}
    params = state.get("extracted_params") or {}
    query = state.get("user_query", "")
    worker_qual = erp_status.get("worker_qualification") or {}
    pref_mode = state.get("preferred_execution_mode", "AUTO")

    shortage_domain = state.get("outsourcing_domain")

    # 중량물 취급 계획서에서 외주 연계가 필요하면 무조건 HEAVY_TRANSPORT 우선
    if task_type == "doc_generation" and params.get("doc_type") == "HEAVY_LIFTING_PLAN":
        shortage_domain = "HEAVY_TRANSPORT"
    elif task_type == "doc_generation" or shortage_domain in (None, "NOT_APPLICABLE"):
        if params.get("worker_name") and (
            worker_qual.get("status") == "NOT_FOUND" or not worker_qual.get("is_valid", True)
        ):
            shortage_domain = "OUTSOURCED_WORKER"
        elif (
            pref_mode != "OUTSOURCED_B2B"
            and worker_qual.get("is_valid", False)
            and not erp_status.get("stock_available", True)
        ):
            shortage_domain = "WELDING_MATERIAL"

    if shortage_domain in (None, "NOT_APPLICABLE"):
        classifier = llm.with_structured_output(
            OutsourcingRequirement, method="function_calling"
        )
        prompt = [
            (
                "system",
                "당신은 스마트 조선소 공급망(SCM) 자원 조정관입니다. "
                "사용자 요청과 본사 자원 진단 결과를 바탕으로 외주 의뢰 대상이 '중량물 운송/배차(HEAVY_TRANSPORT)'인지, "
                "'용접 자재 조달(WELDING_MATERIAL)'인지, '외주 인력 파견(OUTSOURCED_WORKER)'인지 자율 판단하십시오.",
            ),
            (
                "human",
                f"[사용자 질의]: {query}\n[본사 자원 진단 사유]: {erp_status.get('outsourcing_reason')}",
            ),
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

    try:
        partner_plan = query_partner_erp_b2b.invoke({
            "outsourcing_domain": shortage_domain,
            "material": params.get("material") or "",
            "process": params.get("process") or "",
            "worker_name": params.get("worker_name") or "",
        })
    except Exception as e:
        print(f"⚠️ [B2B SCM Collector 오류]: {e}")
        partner_plan = {
            "status": "ERROR",
            "outsourcing_domain": shortage_domain,
            "error_message": str(e),
        }

    # [A2A 하이브리드 자율 협의] 전체 협력사 단순 조회(ALL_PARTNERS)를 제외한 외주 업무에서 A2A 협의 실행
    if shortage_domain in ("WELDING_MATERIAL", "HEAVY_TRANSPORT", "OUTSOURCED_WORKER"):
        a2a_res = negotiate_partner_via_a2a(
            outsourcing_domain=shortage_domain,
            target_branch=str(params.get("branch_code") or state.get("branch_code") or "BR-ULSAN-01"),
            requested_qty=params.get("quantity"),
            weight_ton=params.get("weight_ton"),
            material=str(params.get("material") or ""),
            process=str(params.get("process") or ""),
            worker_name=str(params.get("worker_name") or ""),
            requested_item_name=str(params.get("requested_item_name") or ""),
            urgency=str(params.get("urgency") or "NORMAL"),
        )
        partner_plan["a2a_negotiation"] = a2a_res

        # A2A가 중량물 대체 차량, 합의 단가 또는 대체 용접사를 제안한 경우 selected_partner에 동기화
        a2a_prop = a2a_res.get("proposal") or {}
        if a2a_res.get("status") == "SUCCESS" and a2a_prop:
            sel = dict(partner_plan.get("selected_partner") or {})
            if a2a_prop.get("partner_id"):
                sel["partner_id"] = a2a_prop["partner_id"]
                sel["partner_name"] = a2a_prop.get("partner_name") or sel.get("partner_name")
            if a2a_prop.get("assigned_vehicle_code"):
                sel["assigned_vehicle_code"] = a2a_prop["assigned_vehicle_code"]
                sel["max_load_ton"] = a2a_prop.get("max_load_ton")
                sel["business_type"] = "HEAVY_TRANSPORT"
            if a2a_prop.get("agreed_unit_price_krw"):
                sel["unit_price_krw"] = a2a_prop["agreed_unit_price_krw"]
            if a2a_prop.get("recommended_worker") and not sel.get("worker_name"):
                sel.update(a2a_prop["recommended_worker"])
            partner_plan["selected_partner"] = sel

    return {"partner_erp_plan": partner_plan}


# ==============================================================================
# 5. 실시간 기상 수집, YOLOv11 비전 검사 연동 및 안전 룰 판정 워커
# ==============================================================================
def _append_vision_clearance_warning(
    safety_res: Dict[str, Any], vision_info: Dict[str, Any] | None
) -> Dict[str, Any]:
    """
    [패치 2] 비전 관제 상태에 따른 Fail-Closed 안전 판정 병합:
    - 조회 오류(ERROR) 또는 안전 클리어런스 미확보(N) ➔ 즉시 STOP(작업 중지)으로 격상
    - 미계측(NO_DATA) ➔ 차단하지 않고 작업 전 현장 육안 안전거리 확인 안내만 추가
    """
    if not vision_info:
        return safety_res

    v_status = str(vision_info.get("status", "")).upper()
    v_insp = str(vision_info.get("inspection_status", "")).upper()
    v_clearance = str(vision_info.get("safety_clearance_yn") or "").upper()
    rules = list(safety_res.get("mandatory_rules") or [])

    if v_status == "ERROR" or v_insp == "ERROR":
        err_msg = (
            vision_info.get("error")
            or vision_info.get("error_message")
            or "비전 DB/카메라 조회 오류"
        )
        rules.append(
            f"비전 관제 시스템 오류(VISION_INSPECTIONS): {err_msg} — 안전 검증 불가로 Fail-Closed 작업 중지"
        )
        safety_res["mandatory_rules"] = rules
        safety_res["status"] = "STOP"
        safety_res["safety_level"] = "FORBIDDEN"
        safety_res["is_safe"] = False
        safety_res["action_message"] = "비전 관제 조회 오류로 인한 작업 즉시 보류"
    elif v_clearance == "N":
        interfering = ", ".join(
            vision_info.get("interfering_objects") or ["간섭 장애물"]
        )
        rules.append(
            f"비전 안전 인터록(VISION_INSPECTIONS): 작업 구역 인근 간섭({interfering}) 감지 — 장애물 제거 전 착수 금지"
        )
        safety_res["mandatory_rules"] = rules
        safety_res["status"] = "STOP"
        safety_res["safety_level"] = "FORBIDDEN"
        safety_res["is_safe"] = False
        safety_res["action_message"] = "비전 안전거리 미확보(간섭물 감지)로 인한 작업 즉시 보류"
    elif v_status == "NO_DATA" or v_insp == "NO_DATA":
        prereqs = list(safety_res.get("prerequisites") or [])
        prereqs.append(
            "비전 실시간 계측 이력 미존재(계측 대기): 작업 착수 전 현장 작업 구역 안전거리 육안 확인 필수"
        )
        safety_res["prerequisites"] = prereqs

    return safety_res


def weather_and_safety_node(state: ShipAgentState) -> Dict[str, Any]:
    task_type = state.get("task_type", "knowledge_qa")
    params = state.get("extracted_params") or {}

    vision_info = state.get("vision_data")
    if task_type == "doc_generation" and not vision_info:
        try:
            vision_info = get_vision_inspection_status.invoke({
                "work_id": state.get("work_id") or params.get("block_id"),
                "camera_code": params.get("camera_code"),
            })
        except Exception as e:
            print(f"⚠️ [Vision Inspection 조회 경고]: {e}")
            vision_info = {"status": "ERROR", "error_message": str(e)}

    location = params.get("location")
    loc_upper = str(location or "").upper()
    if not location or any(tok in loc_upper for tok in ("ZONE", "DOCK", "BAY")):
        branch_code = params.get("branch_code") or state.get("branch_code")
        b_ctx = get_branch_context_from_db(branch_code)
        location = b_ctx.get("region_name") or "Ulsan"

    try:
        weather_info = dict(get_realtime_weather_from_mcp(location) or {})
    except Exception as e:
        weather_info = {"status": "ERROR", "error": str(e)}

    status = weather_info.get("status")

    if status != "SUCCESS":
        err_msg = weather_info.get("error") or "기상 데이터 결측"
        print(f"⚠️ [FastMCP 기상 조회 비정상]: {err_msg}")
        fail_closed_eval = _append_vision_clearance_warning(
            {
                "status": "STOP",
                "safety_level": "FORBIDDEN",
                "is_safe": False,
                "mandatory_rules": [
                    f"기상 센서/MCP 데이터 수집 실패({status or 'ERROR'}: {err_msg})로 인해 안전성 검증 불가 (Fail-Closed 작업 중지)"
                ],
                "prerequisites": [],
                "action_message": "기상 데이터 확인 불가로 인한 작업 즉시 보류",
            },
            vision_info,
        )
        return {
            "weather_data": {
                "location": location,
                "status": status or "ERROR",
                "error": err_msg,
                "raw_text": weather_info.get("raw_text"),
                "safety_evaluation": fail_closed_eval,
            },
            "vision_data": vision_info,
        }

    raw_temp = weather_info.get("temperature")
    raw_wind = weather_info.get("wind_speed")
    raw_rain = weather_info.get("precipitation")

    temp = float(raw_temp) if raw_temp is not None else None
    wind = float(raw_wind) if raw_wind is not None else None
    rain = float(raw_rain) if raw_rain is not None else None

    try:
        # 중량물 취급 작업계획서는 산업안전보건규칙 크레인/양중 풍속 기준(10m/s 중지, 7m/s 주의) 전용 적용
        if params.get("doc_type") == "HEAVY_LIFTING_PLAN":
            if wind is not None and wind >= 10.0:
                safety_res = {
                    "status": "STOP",
                    "safety_level": "FORBIDDEN",
                    "is_safe": False,
                    "mandatory_rules": [
                        f"순간풍속 {wind}m/s (중량물 양중 작업 중지 기준 10.0m/s 초과) — 크레인·블록 인양 즉시 중지"
                    ],
                    "prerequisites": [],
                    "action_message": "강풍으로 인한 중량물 인양 즉시 중단",
                }
            elif (wind is not None and wind >= 7.0) or (rain is not None and rain > 0):
                rules = []
                if wind is not None and wind >= 7.0:
                    rules.append(
                        f"풍속 {wind}m/s (7~10m/s 주의 구간): 유도 로프(Tag Line) 2개소 체결 및 저속 인양 필수"
                    )
                if rain is not None and rain > 0:
                    rules.append(
                        f"강수({rain}mm/h) 감지: 줄걸이 슬립 방지 및 지반 침하 상태 사전 점검 필수"
                    )
                safety_res = {
                    "status": "CONDITIONAL_PASS",
                    "safety_level": "CAUTION",
                    "is_safe": True,
                    "mandatory_rules": rules,
                    "prerequisites": ["작업지휘자 현장 상주 및 지상 30cm 시험 인양 실시"],
                    "action_message": "중량물 양중 기상 주의 조건부 승인",
                }
            else:
                safety_res = {
                    "status": "PASS",
                    "safety_level": "SAFE",
                    "is_safe": True,
                    "mandatory_rules": [
                        f"실측 풍속 {wind}m/s (중량물 양중 허용 기준 10.0m/s 이내 정상)"
                    ],
                    "prerequisites": ["전담 신호수 배치 및 인양 반경 하부 출입 통제"],
                    "action_message": "중량물 양중 기상 안전 통과",
                }
        else:
            process_ctx = (
                params.get("process") or params.get("equipment") or "야외 용접"
            )
            safety_res = evaluate_environmental_safety.invoke({
                "current_temp": temp,
                "current_wind": wind,
                "rainfall": rain,
                "thresholds": None,
                "task_context": str(process_ctx),
                "is_outdoor": True,
            })

        safety_res = _append_vision_clearance_warning(safety_res, vision_info)
        weather_info["safety_evaluation"] = safety_res

        print(f"   ⛅ [Weather MCP] 상태={status}, 기온={temp}℃, 풍속={wind}m/s")
        print(
            f"   👁️ [Vision YOLOv11] 상태={vision_info.get('welding_state') if vision_info else 'N/A'}, "
            f"안전거리={vision_info.get('safety_clearance_yn') if vision_info else 'Y'}"
        )
        print(f"   🛡️ [Safety Critic] 최종 판정: {safety_res.get('status')}")

        return {"weather_data": weather_info, "vision_data": vision_info}

    except Exception as e:
        print(f"⚠️ [도메인 안전 룰 엔진 오류]: {e}")
        fail_closed_eval = _append_vision_clearance_warning(
            {
                "status": "STOP",
                "safety_level": "FORBIDDEN",
                "is_safe": False,
                "mandatory_rules": [
                    f"환경 안전 기준 평가 중 시스템 오류 발생({str(e)})으로 인한 안전 보류"
                ],
                "prerequisites": [],
                "action_message": "안전 평가 오류로 인한 작업 즉시 보류",
            },
            vision_info,
        )
        return {
            "weather_data": {
                "location": location,
                "status": "EVALUATION_ERROR",
                "error": f"환경 안전 기준 평가 실패: {str(e)}",
                "safety_evaluation": fail_closed_eval,
            },
            "vision_data": vision_info,
        }


# ==============================================================================
# 6. Team 2 SubGraph 조립 (Supervisor 중심 Hub-and-Spoke 아키텍처)
# ==============================================================================
builder = StateGraph(ShipAgentState)

builder.add_node("team2_supervisor", team2_supervisor_node)
builder.add_node("vision_qa_collector", vision_qa_collector_node)
builder.add_node("rag_retriever", dynamic_rag_retriever_node)
builder.add_node("semantic_doc_grader", semantic_doc_grader_node)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("b2b_scm_collector", b2b_scm_collector_node)
builder.add_node("weather_and_safety", weather_and_safety_node)

builder.add_edge(START, "team2_supervisor")
builder.add_conditional_edges(
    "team2_supervisor",
    route_from_team2_supervisor,
    {
        "vision_qa_collector": "vision_qa_collector",
        "rag_retriever": "rag_retriever",
        "b2b_scm_collector": "b2b_scm_collector",
        "weather_and_safety": "weather_and_safety",
        "__end__": END,
    },
)

builder.add_edge("vision_qa_collector", "team2_supervisor")
builder.add_edge("rag_retriever", "semantic_doc_grader")
builder.add_conditional_edges(
    "semantic_doc_grader",
    route_after_grader,
    {
        "rewrite_query": "rewrite_query",
        "team2_supervisor": "team2_supervisor",
    },
)
builder.add_edge("rewrite_query", "rag_retriever")
builder.add_edge("b2b_scm_collector", "team2_supervisor")
builder.add_edge("weather_and_safety", "team2_supervisor")

team2_subgraph = builder.compile()