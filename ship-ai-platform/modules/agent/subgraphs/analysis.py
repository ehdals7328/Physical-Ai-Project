"""
modules/agent/subgraphs/analysis.py
Team 1: 4-Node 전문 분업 서브그래프 (하드코딩 0% - DB 메타데이터 & 스키마 기반 자율 추론)
[1. RBAC/야드 컨텍스트 초기화] -> [2. LLM 의도/슬롯 추출] -> [3. SHIP_ERP 정밀 진단] -> [4. 제원 완결성/외주 판정]
"""

import json
from typing import Any, Dict, List, Literal, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from modules.agent.state import DocCategory, ShipAgentState
from modules.agent.tools.erp_tools import (
    check_hq_erp_resources,
    get_branch_context_from_db,
    list_registered_partners_from_db,
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class UniversalQueryAnalysis(BaseModel):
    task_type: Literal[
        "chat",
        "erp_resource_qa",
        "hr_safety_qa",
        "hr_worker_qa",
        "scm_partner_qa",
        "knowledge_qa",
        "doc_generation",
    ] = Field(
        description=(
            "질의의 핵심 목적 분류:\n"
            "- 'scm_partner_qa': 외부 협력사/하청업체의 가용 자재·단가·외주인력·자격보유자·일당·중장비 배차 조회 및 발주 문의\n"
            "- 'erp_resource_qa': 본사(HQ) 내부 창고의 자재 재고 또는 본사 소유 장비 단순 조회\n"
            "- 'hr_worker_qa': 본사 내부 인사 원장(ERP_WORKER_CERT) 전체 명단 조회\n"
            "- 'hr_safety_qa': 특정 사내 작업자 1인의 선급 자격증 유효성 단건 검증\n"
            "- 'knowledge_qa': WPS 기술 규격, 안전 지침, 기상 등 도메인 지식 문의\n"
            "- 'doc_generation': 공식 작업지시서 또는 작업계획서 작성/발행 요청\n"
            "- 'chat': 일상 인사 및 일반 대화"
        )
    )

    is_action_requested: bool = Field(
        default=False,
        description="단순 조회가 아니라 실제 배차 확정, 발주 실행, 원장 커밋 등 트랜잭션 실행을 요구하면 True",
    )
    preferred_execution_mode: Literal["AUTO", "IN_HOUSE", "OUTSOURCED_B2B"] = Field(
        default="AUTO",
        description="외부 협력사/외주/하청 자원을 조회하거나 활용하려는 의도면 'OUTSOURCED_B2B', 본사 자체 처리를 지정하면 'IN_HOUSE', 별도 구분이 없으면 'AUTO'",
    )
    outsourcing_domain: Literal[
    "OUTSOURCED_WORKER",
    "WELDING_MATERIAL",
    "HEAVY_TRANSPORT",
    "NOT_APPLICABLE",
    ] = Field(
        default="NOT_APPLICABLE",
        description=(
            "외주 연계 또는 협력사 조회(scm_partner_qa) 시 대상 자원 유형 (scm_partner_qa일 때는 반드시 앞의 3개 중 하나를 선택):\n"
            "- 'OUTSOURCED_WORKER': 외주 용접사, 작업자, 인력 명단, 용접 공정 자격 보유 여부, 인건비/일당 문의\n"
            "- 'WELDING_MATERIAL': 와이어, 강판, 가스 등 물리적 자재 재고·단가·납품 문의\n"
            "- 'HEAVY_TRANSPORT': 중량물 운송, 트랜스포터, 크레인 배차 문의\n"
            "- 'NOT_APPLICABLE': 외주나 협력사 자원과 무관한 일반 대화·사내 단순 질의"
        )
    )
    context_action: Literal["KEEP_MERGE", "RESET_NEW"] = Field(
        default="KEEP_MERGE",
        description=(
            "대화 맥락 처리 방식:\n"
            "- 'RESET_NEW': 작업지시서 발행(doc_generation)이 아닌 별도의 단순 자원·협력사·규격 조회(scm_partner_qa, erp_resource_qa, hr_worker_qa, knowledge_qa)이거나 새로운 대상 질문일 때\n"
            "- 'KEEP_MERGE': 직전 작업지시서 작성(doc_generation)의 누락 제원을 보충하거나 동일 작업 조건을 이어갈 때"
        ),
    )
    target_categories: List[DocCategory] = Field(
        default_factory=list,
        description="필요한 RAG 7대 지식 카테고리 목록",
    )
    need_weather: bool = Field(
        default=False,
        description="실시간 기상 조회 필요 여부 (날씨 질의나 'doc_generation'인 경우 True)",
    )
    block_id: Optional[str] = Field(None, description="선박 블록 식별 코드")
    material: Optional[str] = Field(None, description="모재 강종 등급")
    thickness: Optional[str] = Field(None, description="판재 두께 규격")
    location: Optional[str] = Field(None, description="작업 지역/도시명 (구역 코드 제외)")
    process: Optional[str] = Field(None, description="용접 공정 코드")
    weld_position: Optional[str] = Field(None, description="용접 자세 코드")
    worker_name: Optional[str] = Field(
        None,
        description="질의에 명시된 특정 인력의 순수 성명 (직책/호칭 제외, 미명시 시 반드시 None)",
    )
    equipment: Optional[str] = Field(None, description="투입 설비 또는 차량명")
    additional_specs: Dict[str, Any] = Field(
        default_factory=dict,
        description="기타 엔지니어링 조건",
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="벡터 검색용 공학 표준 키워드 쿼리",
    )


structured_analyzer = llm.with_structured_output(
    UniversalQueryAnalysis,
    method="function_calling",
)

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 스마트 조선소 엔지니어링 & ERP 총괄 분석 에이전트(Team 1)입니다.\n"
        "제공된 [DB 등록 외부 협력사 메타데이터]와 대화 맥락을 참고하여 사용자의 질의를 구조화하세요.\n\n"
        "[판단 기준]\n"
        "- 질의 대상이 본사(Internal HQ) 자원인지, 외부 협력사(External Partner/B2B) 자원인지 구분하십시오.\n"
        "- 외부 협력사(또는 등록된 협력사명)의 자재·단가·외주인력·자격보유자·일당·장비 가용성에 대한 질의는 'scm_partner_qa'로 분류하고, "
        "preferred_execution_mode='OUTSOURCED_B2B' 및 질의 대상 자원에 맞는 outsourcing_domain('OUTSOURCED_WORKER', 'WELDING_MATERIAL', 'HEAVY_TRANSPORT' 중 택1)을 반드시 지정하십시오.\n"
        "- 단순 자원/협력사 가용성 질의(scm_partner_qa)일 때는 context_action='RESET_NEW'로 설정하여 직전 작업지시서의 인력/자재 슬롯과 섞이지 않게 하십시오.\n"
        "- 공식 작업지시서/계획서 작성 요청은 외주 연계 여부에 관계없이 'doc_generation'으로 분류하되, "
        "외주 발주 연계가 포함된 경우 preferred_execution_mode='OUTSOURCED_B2B'로 설정하십시오.",
    ),
    MessagesPlaceholder(variable_name="messages", optional=True),
    (
        "human",
        "[직전 턴 역질문 대기 여부]: {is_waiting}\n"
        "[로그인 사용자 기본 작업구역]: {user_workspace} (야드코드: {branch_code})\n"
        "[DB 등록 외부 협력사 메타데이터]: {registered_partners}\n"
        "[이전 누적 작업 제원]: {accumulated_params}\n"
        "[최신 사용자 질의]: {query}",
    ),
])


# ==============================================================================
# Node 1: RBAC 권한 및 ERP_BRANCHES 작업장 컨텍스트 동적 초기화 노드
# ==============================================================================
def rbac_context_initializer_node(state: ShipAgentState) -> Dict[str, Any]:
    existing_params = dict(state.get("extracted_params") or {})
    branch_code = state.get("branch_code") or existing_params.get("branch_code")
    default_workspace = state.get("default_workspace") or existing_params.get("work_zone")

    branch_ctx = get_branch_context_from_db(branch_code)
    resolved_branch = branch_ctx.get("branch_code") or branch_code
    resolved_region = branch_ctx.get("region_name")
    resolved_zone = default_workspace or branch_ctx.get("default_zone")

    if resolved_branch and not existing_params.get("branch_code"):
        existing_params["branch_code"] = resolved_branch
    if resolved_zone and not existing_params.get("work_zone"):
        existing_params["work_zone"] = resolved_zone
    if resolved_region and not existing_params.get("location"):
        existing_params["location"] = resolved_region

    return {
        "branch_code": resolved_branch,
        "default_workspace": resolved_zone,
        "extracted_params": existing_params,
    }


# ==============================================================================
# Node 2: LLM 구조화 의도 분류 및 엔지니어링 슬롯 추출 노드
# ==============================================================================
def intent_slot_extractor_node(state: ShipAgentState) -> Dict[str, Any]:
    query = state.get("user_query", "").strip()
    messages = state.get("messages", [])
    existing_params = dict(state.get("extracted_params") or {})
    is_waiting = state.get("is_param_complete") == "no"

    can_access_erp = state.get("can_access_erp", "Y")
    default_workspace = state.get("default_workspace")
    branch_code = state.get("branch_code")

    # DB에서 현재 등록된 협력사 메타데이터를 동적으로 인출하여 프롬프트에 주입
    partners_meta = list_registered_partners_from_db()

    res: UniversalQueryAnalysis = (ANALYSIS_PROMPT | structured_analyzer).invoke({
        "messages": messages[-6:] if messages else [],
        "is_waiting": "예 (제원 보충 대기 중)" if is_waiting else "아니오",
        "user_workspace": default_workspace or "미지정",
        "branch_code": branch_code or "미지정",
        "registered_partners": json.dumps(partners_meta, ensure_ascii=False),
        "accumulated_params": json.dumps(existing_params, ensure_ascii=False),
        "query": query,
    })

    old_block = existing_params.get("block_id")
    new_block = res.block_id

    if new_block and old_block and new_block != old_block:
        merged_params: Dict[str, Any] = {
            "branch_code": branch_code,
            "work_zone": default_workspace,
            "location": existing_params.get("location"),
        }
    elif res.context_action == "RESET_NEW" and not is_waiting:
        merged_params = {
            "branch_code": branch_code,
            "work_zone": default_workspace,
            "location": existing_params.get("location"),
        }
    else:
        merged_params = dict(existing_params)

    new_fields = {
        "block_id": res.block_id,
        "material": res.material,
        "thickness": res.thickness,
        "location": res.location,
        "process": res.process,
        "weld_position": res.weld_position,
        "worker_name": res.worker_name,
        "equipment": res.equipment,
    }
    for k, v in new_fields.items():
        if v is not None and str(v).strip() != "":
            merged_params[k] = str(v).strip()
    if res.additional_specs:
        merged_params.update(res.additional_specs)

    # 일반 자원/협력사 가용성 조회 시 이전 턴의 사내 작업자 슬롯 간섭 차단 (상태 전이 원칙)
    if res.task_type in ("scm_partner_qa", "erp_resource_qa", "hr_worker_qa") and not res.worker_name:
        merged_params.pop("worker_name", None)

    if can_access_erp == "N" and res.task_type in ("erp_resource_qa", "hr_worker_qa", "scm_partner_qa"):
        blocked_msg = (
            "🔒 **[접근 권한 제한]** 현재 로그인하신 계정은 전사 ERP 자원 및 인사/공급망 원장 조회 권한(`CAN_ACCESS_ERP=N`)이 없습니다. "
            "시스템 관리자(ADMIN) 또는 공정 관리자(MANAGER)에게 권한 승인을 요청해 주세요."
        )
        return {
            "task_type": "chat",
            "extracted_params": merged_params,
            "is_param_complete": "yes",
            "missing_params": [],
            "search_status": "completed",
            "final_answer": blocked_msg,
        }

    if res.task_type == "chat":
        return {
            "task_type": "chat",
            "extracted_params": merged_params,
            "is_param_complete": "yes",
            "missing_params": [],
            "search_status": "completed",
            "final_answer": "안녕하세요! 스마트 조선소 공정 관제 및 작업지시 지원 에이전트입니다. 작업지시서 발행, WPS 규격 조회, ERP 자재/자격증/비전 상태 확인 등 무엇을 도와드릴까요?",
        }

    need_weather_flag = True if res.task_type == "doc_generation" else res.need_weather
    is_outsourcing_init = (
        res.task_type == "scm_partner_qa"
        or res.preferred_execution_mode == "OUTSOURCED_B2B"
        or res.outsourcing_domain is not None
    )

    return {
        "task_type": res.task_type,
        "is_action_requested": res.is_action_requested,
        "preferred_execution_mode": res.preferred_execution_mode,
        "outsourcing_domain": res.outsourcing_domain,
        "is_outsourcing_required": is_outsourcing_init,
        "target_categories": res.target_categories,
        "need_weather": need_weather_flag,
        "extracted_params": merged_params,
        "search_queries": list(res.search_queries),
        "retry_count": 0,
        "search_status": "idle",
    }


def route_after_intent(state: ShipAgentState) -> Literal["hq_erp_diagnostician", "spec_completeness_judge", "__end__"]:
    task_type = state.get("task_type")
    if task_type == "chat":
        return "__end__"
    if task_type in ("doc_generation", "erp_resource_qa", "hr_safety_qa", "scm_partner_qa"):
        return "hq_erp_diagnostician"
    return "spec_completeness_judge"


# ==============================================================================
# Node 3: 본사 SHIP_ERP 마스터 제원 동기화 및 자원 가용성 진단 노드
# ==============================================================================
def hq_erp_diagnostician_node(state: ShipAgentState) -> Dict[str, Any]:
    merged_params = dict(state.get("extracted_params") or {})
    target_block = merged_params.get("block_id")

    erp_status = check_hq_erp_resources.invoke({
        "block_id": target_block,
        "material": merged_params.get("material"),
        "worker_name": merged_params.get("worker_name"),
        "equipment": merged_params.get("equipment"),
        "branch_code": merged_params.get("branch_code"),
    })

    bm = erp_status.get("block_master")
    if bm:
        for bk in ("ship_code", "block_name", "material", "thickness", "joint_type", "process", "branch_code", "weight_ton"):
            if bm.get(bk) is not None:
                merged_params[bk] = bm[bk]

        if bm.get("location"):
            merged_params["location"] = bm["location"]

        erp_status = check_hq_erp_resources.invoke({
            "block_id": target_block,
            "material": merged_params.get("material"),
            "worker_name": merged_params.get("worker_name"),
            "equipment": merged_params.get("equipment"),
            "branch_code": merged_params.get("branch_code"),
        })

    branch_info = erp_status.get("branch_info") or {}
    if not merged_params.get("location") or "ZONE" in str(merged_params.get("location")).upper():
        if branch_info.get("region_name"):
            merged_params["location"] = branch_info["region_name"]

    pref_mode = state.get("preferred_execution_mode", "AUTO")
    is_outsourcing = (
        bool(state.get("is_outsourcing_required", False))
        or bool(erp_status.get("require_outsourcing", False))
        or state.get("task_type") == "scm_partner_qa"
        or pref_mode == "OUTSOURCED_B2B"
    )

    return {
        "extracted_params": merged_params,
        "erp_resource_status": erp_status,
        "is_outsourcing_required": is_outsourcing,
    }


# ==============================================================================
# Node 4: 엔지니어링 제원 완결성 판정 및 검색 쿼리 정제 노드
# ==============================================================================
def spec_completeness_judge_node(state: ShipAgentState) -> Dict[str, Any]:
    task_type = state.get("task_type")
    merged_params = dict(state.get("extracted_params") or {})
    queries = list(state.get("search_queries") or [])
    query = state.get("user_query", "").strip()

    missing: List[str] = []
    is_complete = "yes"
    clarification_answer = None

    if task_type == "doc_generation":
        required_slots = {
            "material": "모재 강종",
            "thickness": "판 두께",
        }
        missing = [k for k in required_slots if not merged_params.get(k)]
        if missing:
            is_complete = "no"
            missing_kor = ", ".join(required_slots[m] for m in missing)
            block_notice = (
                f" (현재 확인된 조건: {json.dumps(merged_params, ensure_ascii=False)})"
                if merged_params else ""
            )
            clarification_answer = (
                f"작업지시서 발행을 위해 필수 제원이 추가로 필요합니다.{block_notice}\n"
                f"누락된 항목(**{missing_kor}**) 또는 대상 **블록 번호(`BLOCK_ID`)**를 입력해 주시면 "
                f"SHIP_ERP 마스터 원장과 WPS 규격을 대조하여 지시서를 작성해 드리겠습니다."
            )

    if not queries and task_type in ("knowledge_qa", "doc_generation"):
        mat = merged_params.get("material", "")
        thk = merged_params.get("thickness", "")
        proc = merged_params.get("process", "용접")
        if task_type == "doc_generation" and (mat or thk):
            queries = [f"{mat} {thk} {proc} WPS 작업 표준 규격".strip()]
        else:
            queries = [query]

    return {
        "missing_params": missing,
        "is_param_complete": is_complete,
        "search_queries": queries,
        "final_answer": clarification_answer,
    }


# ==============================================================================
# Team 1 SubGraph 조립
# ==============================================================================
builder = StateGraph(ShipAgentState)

builder.add_node("rbac_context_initializer", rbac_context_initializer_node)
builder.add_node("intent_slot_extractor", intent_slot_extractor_node)
builder.add_node("hq_erp_diagnostician", hq_erp_diagnostician_node)
builder.add_node("spec_completeness_judge", spec_completeness_judge_node)

builder.add_edge(START, "rbac_context_initializer")
builder.add_edge("rbac_context_initializer", "intent_slot_extractor")
builder.add_conditional_edges(
    "intent_slot_extractor",
    route_after_intent,
    {
        "hq_erp_diagnostician": "hq_erp_diagnostician",
        "spec_completeness_judge": "spec_completeness_judge",
        "__end__": END,
    },
)
builder.add_edge("hq_erp_diagnostician", "spec_completeness_judge")
builder.add_edge("spec_completeness_judge", END)

team1_subgraph = builder.compile()