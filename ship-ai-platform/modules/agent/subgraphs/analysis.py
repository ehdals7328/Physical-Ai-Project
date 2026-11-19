"""
modules/agent/subgraphs/analysis.py
Team 1: 4-Node 전문 분업 서브그래프 (하드코딩 0% - DB 메타데이터 & 스키마 기반 자율 추론)
[1. RBAC/야드 컨텍스트 초기화] -> [2. LLM 의도/슬롯 추출] -> [3. SHIP_ERP 정밀 진단] -> [4. 제원 완결성/외주 판정]
"""

import json
import re
from typing import Any, Dict, List, Literal, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from modules.agent.state import DocCategory, ShipAgentState
from modules.agent.tools.erp_tools import (
    check_hq_erp_resources,
    get_branch_context_from_db,
    list_erp_workers,
    list_registered_partners_from_db,
)

# 의도 분류 및 슬롯 추출 속도 최적화를 위해 즉시 응답 모델 사용
llm = ChatOpenAI(model="gpt-5-mini")

VALID_WELD_PROCESSES = {"FCAW", "SAW", "GMAW", "SMAW", "GTAW", "TIG", "MIG"}
STEEL_GRADES = {"SS400", "AH36", "DH36", "EH36", "AH32", "DH32", "EH32", "AH40", "DH40", "EH40"}

class UniversalQueryAnalysis(BaseModel):
    task_type: Literal[
        "chat",
        "vision_inspection_qa",
        "erp_resource_qa",
        "hr_safety_qa",
        "hr_worker_qa",
        "scm_partner_qa",
        "knowledge_qa",
        "doc_generation",
    ] = Field(
        description=(
            "질의의 핵심 목적 분류:\n"
            "- 'vision_inspection_qa': 현장 비전 탐지 현황, 용접 대기 구역, 용접선 포착 상태, 간섭 위험물, 용접 품질/결함 및 로봇 관제 상태 조회\n"
            "- 'scm_partner_qa': 사외 협력사 소속의 자재·단가·외주인력·일당·장비 가용성 조회 및 외주 발주·배차 실행\n"
            "- 'erp_resource_qa': 본사 내부 창고의 자재 재고 또는 본사 보유 장비 조회\n"
            "- 'hr_worker_qa': 사내외 인사 원장(ERP_WORKER_CERT) 기반 작업자 명단 및 공정별 투입 가능 인력 조회 (사외 협력사 전용 자원을 명시하지 않은 일반 인력 조회 포함)\n"
            "- 'hr_safety_qa': 특정 작업자 1인의 선급 자격증 유효성 단건 검증\n"
            "- 'knowledge_qa': WPS 기술 규격, 안전 지침, 기상 등 도메인 지식 조회\n"
            "- 'doc_generation': 공식 작업지시서 또는 작업계획서 작성 및 발행 (직전 대화에서 작업지시서 작성을 위해 누락 제원·작업자를 안내받은 뒤 대상 블록·제원·작업자를 지정·선택하는 응답 포함)\n"
            "- 'chat': 일상 인사 및 일반 대화"
        )
    )

    is_action_requested: bool = Field(
        default=False,
        description="데이터 조회(Read) 목적이 아닌 원장 기록, 발주 확정, 배차 실행 등 상태 변경 트랜잭션(Write)을 지시하는 경우에만 True",
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
            "- 'NOT_APPLICABLE': 외주나 협력사 자원과 무관한 일반 대화·사내 단순 질의·비전 관제 질의"
        ),
    )
    context_action: Literal["KEEP_MERGE", "RESET_NEW"] = Field(
        default="KEEP_MERGE",
        description=(
            "대화 맥락 처리 방식:\n"
            "- 'KEEP_MERGE': 작업지시서 작성 흐름을 이어가거나 이전 대화의 블록·제원·작업자 조건을 유지·보충할 때\n"
            "- 'RESET_NEW': 작업지시서 작성과 완전히 무관한 새로운 독립 주제 질의일 때"
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
    block_id: Optional[str] = Field(
        None,
        description="질의에 직접 언급된 선박 블록 코드만 추출 (언급되지 않았으면 반드시 None, 강종명 입력 금지)",
    )
    camera_code: Optional[str] = Field(None, description="질의에 직접 언급된 현장 비전 카메라 코드 (없으면 None)")
    material: Optional[str] = Field(None, description="질의에 직접 언급된 모재 강종 등급 (없으면 None)")
    thickness: Optional[str] = Field(None, description="질의에 직접 언급된 판재 두께 규격 (없으면 None)")
    location: Optional[str] = Field(None, description="질의에 직접 언급된 작업 지역/도시명 (없으면 None)")
    process: Optional[str] = Field(
        None,
        description="질의에 직접 언급된 용접 공정 약어(FCAW, SAW, GMAW, SMAW, GTAW 중 하나만 추출, 언급 없으면 None)",
    )
    weld_position: Optional[str] = Field(None, description="질의에 직접 언급된 용접 자세 코드 (없으면 None)")
    worker_name: Optional[str] = Field(
        None,
        description=(
            "질의에 직접 언급된 특정 작업자의 순수 성명만 추출 (언급이 없으면 반드시 None). "
            "한국인 성명(2~4음절)이 일반 명사나 직급처럼 보이더라도(예: '홍길동 반장' -> '홍길동') "
            "사람 이름은 반드시 추출하고 뒤에 붙은 직책·호칭만 제외할 것"
        ),
    )
    equipment: Optional[str] = Field(None, description="질의에 직접 언급된 투입 설비 또는 차량명 (없으면 None)")
    quantity: Optional[float] = Field(None, description="질의에 직접 언급된 자재/품목의 수치 수량 (없으면 None)")
    additional_specs: Dict[str, Any] = Field(
        default_factory=dict,
        description="질의에 명시된 기타 엔지니어링 조건",
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
        "제공된 [DB 등록 외부 협력사 메타데이터]와 이전 대화 맥락을 참고하여 사용자의 질의를 구조화하세요.\n\n"
        "[절대 원칙 - 임의 추론 금지]\n"
        "- 사용자의 현재 발화나 이전 대화에서 **실제로 언급되지 않은 `block_id`, `worker_name`, `material`, `thickness`, `process`는 절대 임의로 채우지 말고 반드시 `null`로 반환**하십시오.\n\n"
        "[판단 기준]\n"
        "1. **작업지시서 작성 멀티턴 맥락 유지**: 최근 대화에서 사용자가 작업지시서 발행을 시작하여 블록·제원·작업자 선택을 진행 중이고, "
        "이번 발화에서 작업자 성명, 블록 번호, 강종, 두께, 공정을 지정·선택했다면 별도의 발행 지시어가 없더라도 `task_type='doc_generation'`, `context_action='KEEP_MERGE'`로 분류하여 즉시 작업지시서 작성으로 진행하십시오.\n"
        "2. **인사 조회 vs 협력사 외주 조회 구분**: 사용자가 사외 협력사, 하청, 외주 또는 등록된 협력사명을 명시적으로 언급하지 않고 투입 가능한 작업자를 묻는 경우에는 `hr_worker_qa`로 분류하십시오.\n"
        "3. **비전 관제 질의**: 현장 비전 탐지 현황, 용접 대기 구역, 용접선 포착 상태, 간섭 장애물, 결함 현황 조회는 `vision_inspection_qa`로 분류하십시오.\n"
        "4. **실행 모드**: 사용자가 명시적으로 외주·협력사·사외 연계를 지정한 경우에만 `preferred_execution_mode='OUTSOURCED_B2B'`로 설정하고, 별도 언급이 없으면 `'AUTO'`로 두십시오.\n"
        "5. **독립 주제 전환(`RESET_NEW`)**: 이전 작업지시서와 무관하게 전체 비전 카메라 현황, 단독 외주 배차/발주, 일반 기술 규격 등을 새로 질문한 경우에는 `context_action='RESET_NEW'`로 분류하십시오.",
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

    partners_meta = list_registered_partners_from_db()

    try:
        res: UniversalQueryAnalysis = (ANALYSIS_PROMPT | structured_analyzer).invoke({
            "messages": messages[-8:] if messages else [],
            "is_waiting": "예 (제원 보충 대기 중)" if is_waiting else "아니오",
            "user_workspace": default_workspace or "미지정",
            "branch_code": branch_code or "미지정",
            "registered_partners": json.dumps(partners_meta, ensure_ascii=False),
            "accumulated_params": json.dumps(existing_params, ensure_ascii=False),
            "query": query,
        })
    except Exception as e:
        print(f"⚠️ [Team 1 의도 분석 LLM 호출 경고]: {e}")
        res = UniversalQueryAnalysis(
            task_type="knowledge_qa" if query else "chat",
            search_queries=[query] if query else [],
        )

    # 최근 사용자 발화 텍스트 모음 (환각 추출 차단용 근거 범위)
    recent_user_texts = [query]
    for m in (messages[-6:] if messages else []):
        if getattr(m, "type", "") == "human":
            recent_user_texts.append(str(getattr(m, "content", "")))
    combined_user_corpus = " ".join(recent_user_texts).upper()

    # 1. block_id 정제 + 실제 사용자 발화 근거 검증 (LLM 임의 생성 차단)
    raw_block_id = str(res.block_id or "").strip().upper()
    cleaned_block_id: Optional[str] = None
    if (
        raw_block_id
        and raw_block_id not in STEEL_GRADES
        and res.task_type != "knowledge_qa"
    ):
        block_num_token = raw_block_id.replace("BLK-", "").replace("B-", "")
        if raw_block_id in combined_user_corpus or (
            len(block_num_token) >= 3 and block_num_token in combined_user_corpus
        ):
            cleaned_block_id = str(res.block_id).strip()

    # 2. process 정제 + 실제 발화 근거 검증
    raw_proc = str(res.process or "").strip().upper()
    cleaned_process = (
        raw_proc
        if (raw_proc in VALID_WELD_PROCESSES and raw_proc in combined_user_corpus)
        else None
    )

    # 3. worker_name 직책 제거 + DB 등록 성명 매칭 + 실제 발화 근거 검증 (LLM 임의 이름 차단)
    cleaned_worker = str(res.worker_name or "").strip()
    if cleaned_worker and cleaned_worker not in " ".join(recent_user_texts):
        cleaned_worker = ""

    if not cleaned_worker and query:
        try:
            workers_res = list_erp_workers.invoke({"branch_code": None})
            for w in workers_res.get("workers", []):
                w_nm = str(w.get("worker_name") or "").strip()
                if w_nm and w_nm in query:
                    cleaned_worker = w_nm
                    break
        except Exception as e:
            print(f"⚠️ [작업자 성명 DB 매칭 참고]: {e}")

    # 4. 이전 누적 슬롯의 비정상 값 정리 및 맥락 병합/초기화 판정
    if str(existing_params.get("block_id") or "").upper() in STEEL_GRADES:
        existing_params.pop("block_id", None)
    if str(existing_params.get("process") or "").upper() not in VALID_WELD_PROCESSES:
        existing_params.pop("process", None)

    old_block = existing_params.get("block_id")
    should_reset_context = (
        (cleaned_block_id and old_block and cleaned_block_id != old_block)
        or (
            not is_waiting
            and res.context_action == "RESET_NEW"
            and res.task_type != "doc_generation"
        )
    )

    # intent_slot_extractor_node 내부 슬롯 병합 부분 교체
    if should_reset_context:
        base_ctx = get_branch_context_from_db(branch_code)
        merged_params: Dict[str, Any] = {
            "branch_code": branch_code,
            "work_zone": default_workspace or base_ctx.get("default_zone"),
            "location": base_ctx.get("region_name") or existing_params.get("location"),
        }
    else:
        merged_params = dict(existing_params)

    # knowledge_qa는 단순 기술 질문이므로 이후 작업지시서 슬롯에 누적하지 않음
    if res.task_type != "knowledge_qa":
        has_explicit_loc = bool(
            res.location and any(tok in query for tok in (str(res.location), "울산", "거제", "부산"))
        )
        new_fields = {
            "block_id": cleaned_block_id,
            "camera_code": res.camera_code,
            "material": res.material,
            "thickness": res.thickness,
            "location": res.location if (not should_reset_context or has_explicit_loc) else None,
            "process": cleaned_process,
            "weld_position": res.weld_position,
            "worker_name": cleaned_worker or None,
            "equipment": res.equipment,
        }
        for k, v in new_fields.items():
            if v is not None and str(v).strip() != "":
                merged_params[k] = str(v).strip()
        if res.quantity is not None:
            merged_params["quantity"] = float(res.quantity)
        if res.additional_specs:
            merged_params.update(res.additional_specs)

    if can_access_erp == "N" and res.task_type in (
        "erp_resource_qa",
        "hr_worker_qa",
        "hr_safety_qa",
        "scm_partner_qa",
    ):
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
    if task_type == "vision_inspection_qa":
        return "spec_completeness_judge"
    return "hq_erp_diagnostician"


# ==============================================================================
# Node 3: 본사 SHIP_ERP 마스터 제원 동기화 및 자원 가용성 진단 노드
# ==============================================================================
def hq_erp_diagnostician_node(state: ShipAgentState) -> Dict[str, Any]:
    task_type = state.get("task_type")
    merged_params = dict(state.get("extracted_params") or {})
    target_block = merged_params.get("block_id")

    erp_status = check_hq_erp_resources.invoke({
        "block_id": target_block,
        "material": merged_params.get("material"),
        "worker_name": merged_params.get("worker_name"),
        "equipment": merged_params.get("equipment"),
        "branch_code": merged_params.get("branch_code"),
        "process": merged_params.get("process"),
    })

    bm = erp_status.get("block_master")
    if bm:
        for bk in (
            "ship_code",
            "block_name",
            "material",
            "thickness",
            "joint_type",
            "process",
            "branch_code",
            "weight_ton",
        ):
            if bm.get(bk) is not None:
                merged_params[bk] = bm[bk]

        # block_master의 work_zone(DOCK-1, ZONE-A-BAY1 등)을 작업구역으로 매핑
        if bm.get("work_zone"):
            merged_params["work_zone"] = bm["work_zone"]

    branch_info = erp_status.get("branch_info") or {}
    loc_upper = str(merged_params.get("location") or "").upper()
    if branch_info.get("region_name"):
        if (
            bm is not None
            or not merged_params.get("location")
            or any(tok in loc_upper for tok in ("ZONE", "DOCK", "BAY"))
        ):
            merged_params["location"] = branch_info["region_name"]

    if not merged_params.get("work_zone") and branch_info.get("default_zone"):
        merged_params["work_zone"] = branch_info["default_zone"]

    pref_mode = state.get("preferred_execution_mode", "AUTO")
    if task_type == "scm_partner_qa" or pref_mode == "OUTSOURCED_B2B":
        is_outsourcing = True
    elif pref_mode == "IN_HOUSE" or task_type in ("erp_resource_qa", "hr_safety_qa", "hr_worker_qa"):
        is_outsourcing = False
    else:
        is_outsourcing = bool(erp_status.get("require_outsourcing", False))

    return {
        "extracted_params": merged_params,
        "erp_resource_status": erp_status,
        "is_outsourcing_required": is_outsourcing,
    }


# ==============================================================================
# Node 4: 엔지니어링 제원 완결성 판정 및 자연스러운 역질문 생성 노드
# ==============================================================================
CLARIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 스마트 조선소 공정 관리 AI 어시스턴트입니다.\n"
        "사용자가 작업지시서 발행을 요청했으나 아직 일부 필수 제원이 입력되지 않았습니다.\n"
        "딱딱한 JSON 덤프나 '강제 차단(HOLD)' 같은 경고성 표현을 쓰지 말고, "
        "현재까지 확인된 조건을 자연스럽게 짚어준 뒤 누락된 항목을 친절하게 안내하세요.\n\n"
        "[안내 지침]\n"
        "1. [ERP 등록 블록 목록]에 있는 블록들은 강종으로 필터링하지 말고 전체 등록 블록(`B-104`, `B-305`, `BLK-E102` 등)을 모두 안내하면서, 블록 번호(`BLOCK_ID`)를 지정하면 강종·두께·공정이 자동 매핑된다고 설명하세요.\n"
        "2. [ERP 가용 작업자 목록]에서 `status='VALID'`이고 용접 공정이 있는 적격 작업자(`최선임`, `김철수`, `오수석`, `장반장` 등)를 함께 추천하여 투입할 작업자 성명을 선택하도록 안내하세요.",
    ),
    (
        "human",
        "[사용자 발화]: {query}\n"
        "[현재까지 확인된 작업 조건]: {params}\n"
        "[누락된 필수 항목]: {missing_kor}\n"
        "[ERP 등록 블록 목록]: {blocks}\n"
        "[ERP 가용 작업자 목록]: {workers}",
    ),
])


def spec_completeness_judge_node(state: ShipAgentState) -> Dict[str, Any]:
    task_type = state.get("task_type")
    merged_params = dict(state.get("extracted_params") or {})
    erp_status = state.get("erp_resource_status") or {}
    queries = list(state.get("search_queries") or [])
    query = state.get("user_query", "").strip()

    missing: List[str] = []
    is_complete = "yes"
    clarification_answer = None

    if task_type == "doc_generation":
        required_slots = {
            "material": "모재 강종",
            "thickness": "판 두께",
            "process": "용접 공정",
            "worker_name": "투입 작업자 성명",
        }
        missing = [k for k in required_slots if not merged_params.get(k)]
        if missing:
            is_complete = "no"
            missing_kor = ", ".join(required_slots[m] for m in missing)
            req_proc = str(merged_params.get("process") or "").upper()
            valid_workers = [
                w for w in (erp_status.get("available_workers") or [])
                if str(w.get("status", "")).upper() == "VALID"
                and str(w.get("weld_process", "NA")).upper() != "NA"
                and (not req_proc or str(w.get("weld_process", "")).upper() == req_proc)
            ]
            try:
                clarification_res = (CLARIFICATION_PROMPT | llm).invoke({
                    "query": query,
                    "params": json.dumps(merged_params, ensure_ascii=False),
                    "missing_kor": missing_kor,
                    "blocks": json.dumps(erp_status.get("available_blocks") or [], ensure_ascii=False),
                    "workers": json.dumps(valid_workers, ensure_ascii=False),
                })
                clarification_answer = clarification_res.content
            except Exception as e:
                print(f"⚠️ [Team 1 역질문 생성 경고]: {e}")
                clarification_answer = (
                    f"작업지시서 발행을 위해 다음 필수 제원을 추가로 알려주세요: **{missing_kor}** "
                    "(대상 블록 코드와 투입할 작업자 성명을 입력해 주시면 즉시 처리해 드립니다.)"
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