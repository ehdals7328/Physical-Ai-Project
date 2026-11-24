"""
modules/agent/subgraphs/analysis.py
Team 1: RBAC 초기화 -> 의도/목표·슬롯 추출 -> 본사 ERP 사전진단 -> 자율 편성/유효성 판정 서브그래프
- 사용자가 블록·작업자를 명시하지 않고 목표만 위임해도 멈추지 않고 auto_assign_mode=True로 Team 2 ReAct 에이전트에 넘김
- CAD 도면 업로드 또는 공정 계획 질의(cad_planning)는 Team 4(Planning)로 즉시 분기
"""

import json
import re
from typing import Any, Dict, List, Literal, Optional
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

from modules.agent.state import DocCategory, ShipAgentState
from modules.agent.tools.erp_tools import (
    check_hq_erp_resources,
    evaluate_heavy_lifting_capacity,
    get_branch_context_from_db,
    list_registered_partners_from_db,
)

llm = ChatOpenAI(model="gpt-5-mini", reasoning_effort="low")

VALID_WELD_PROCESSES = {"FCAW", "SAW", "GMAW", "SMAW", "GTAW", "TIG", "MIG"}
STEEL_GRADES = {"SS400", "AH36", "DH36", "EH36", "AH32", "DH32", "EH32", "AH40", "DH40", "EH40"}


class UniversalQueryAnalysis(BaseModel):
    task_type: Literal[
        "chat",
        "cad_planning",
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
            "- 'cad_planning': CAD 도면 기반 작업계획 수립·분석, 공정 순서 및 대략적 일정 편성, 도면 기반 금일 작업 우선순위 브리핑 요청\n"
            "- 'vision_inspection_qa': 현장 비전 탐지 현황, 용접 대기 구역, 용접선 포착 상태, 간섭 위험물, 용접 품질/결함 및 로봇 관제 상태 조회\n"
            "- 'scm_partner_qa': 사외 협력사 소속의 자재·단가·외주인력·일당·장비 가용성 조회, 납기/조건 협의(A2A) 및 외주 발주·배차 실행\n"
            "- 'erp_resource_qa': 본사 야드(사업장) 현황 및 공정, 등록된 작업 대상 선박 블록 목록, 본사 창고의 자재 재고 또는 본사 보유 장비 조회\n"
            "- 'hr_worker_qa': 사내외 인사 원장(ERP_WORKER_CERT) 기반 전체 작업자 명단 및 공정별 투입 가능 인력 조회\n"
            "- 'hr_safety_qa': 특정 작업자 1인의 선급 자격증 유효성 단건 검증\n"
            "- 'knowledge_qa': 사내 조선·해양 WPS 기술 규격, 안전 지침, 실시간 기상/날씨 조회 등 도메인 정보 질의\n"
            "- 'doc_generation': 공식 용접 작업지시서·중량물 취급 계획서의 작성, 생성, 자동 편성, 발행을 요청한 경우\n"
            "- 'chat': 일상 인사, 플랫폼 사용법, 로그인 사용자의 현재 위치 확인, 조선·해양 도메인 외 잡담"
        )
    )
    auto_assign_requested: bool = Field(
        default=False,
        description=(
            "사용자가 블록이나 작업자/장비를 일일이 지정하지 않고 '알아서 편성해줘', '작업 가능한 블록 찾아서 지시서 써줘', "
            "'적격 작업자 배정해서 발행해줘'처럼 에이전트의 자율 탐색·배정을 위임했거나 일부 자원을 비워둔 채 문서 생성을 요청하면 True"
        ),
    )
    is_action_requested: bool = Field(
        default=False,
        description=(
            "단순 자원·단가·납기 조회(Read) 및 가용 조건 사전 조율·협의 단계는 False. "
            "사용자가 실제 발주서(PO) 생성, 주문 확정, 배차 확정 등 상태 변경 트랜잭션(Write) 실행을 명시적으로 지시한 경우에만 True"
        ),
    )
    preferred_execution_mode: Optional[Literal["AUTO", "IN_HOUSE", "OUTSOURCED_B2B"]] = Field(
        default="AUTO",
        description="외부 협력사/외주/하청 자원을 지정하면 'OUTSOURCED_B2B', 본사 자체 처리를 지정하면 'IN_HOUSE', 별도 구분 없으면 'AUTO'",
    )
    outsourcing_domain: Optional[Literal[
        "ALL_PARTNERS",
        "OUTSOURCED_WORKER",
        "WELDING_MATERIAL",
        "HEAVY_TRANSPORT",
        "NOT_APPLICABLE",
    ]] = Field(
        default="NOT_APPLICABLE",
        description="외주 연계 또는 협력사 조회 대상 자원 유형",
    )
    context_action: Optional[Literal["KEEP_MERGE", "RESET_NEW"]] = Field(
        default="KEEP_MERGE",
        description="이전 작업 흐름 유지/보충이면 'KEEP_MERGE', 완전히 새로운 독립 주제 질의면 'RESET_NEW'",
    )
    target_categories: List[DocCategory] = Field(
        default_factory=list,
        description=(
            "문서 기반 지식 검색(RAG)이 필요한 질의(`knowledge_qa`, `doc_generation`)에 한해 "
            "해당하는 기술·안전·규정 문서 분류 목록을 선택하고, "
            "단순 데이터 원장 조회나 일반 대화일 경우에는 빈 리스트(`[]`)를 반환할 것"
        ),
    )
    need_weather: bool = Field(default=False)

    @field_validator("target_categories", "search_queries", mode="before")
    @classmethod
    def _coerce_none_to_list(cls, v):
        return [] if v is None else v

    block_id: Optional[str] = Field(None, description="직접 언급된 선박 블록 코드만 추출 (없으면 None)")
    camera_code: Optional[str] = Field(None)
    material: Optional[str] = Field(None)
    thickness: Optional[str] = Field(None)
    location: Optional[str] = Field(None, description="질의에 직접 언급된 지역/도시명 (예: 울산, 거제, 부산)")
    process: Optional[str] = Field(None, description="직접 언급된 용접 공정(FCAW, SAW, GMAW, SMAW, GTAW)")
    weld_position: Optional[str] = Field(None)
    worker_name: Optional[str] = Field(None, description="직접 언급된 특정 작업자의 순수 성명 (직책 제외, 없으면 None)")
    equipment: Optional[str] = Field(
        default=None,
        description="사용자가 직접 지정한 양중·운반·시공 장비명 또는 톤수 (예: '50톤 크레인', '15톤 지게차', '200톤 트랜스포터', 언급 없으면 None)",
    )
    quantity: Optional[float] = Field(None)
    additional_specs: Dict[str, Any] = Field(default_factory=dict)
    search_queries: List[str] = Field(default_factory=list)
    direct_chat_reply: Optional[str] = Field(default=None)
    requested_item_name: Optional[str] = Field(default=None)
    worker_scope: Optional[Literal["ALL", "HQ_ONLY", "EXTERNAL_ONLY"]] = Field(default="ALL")
    doc_type: Optional[Literal["WELDING_WORK_ORDER", "HEAVY_LIFTING_PLAN"]] = Field(
        default=None,
        description=(
            "task_type이 'doc_generation'일 때 반드시 둘 중 하나를 선택 (문서 생성이 아닐 때만 null):\n"
            "- 'HEAVY_LIFTING_PLAN': 블록 이동·운반·인양·양중 작업 또는 중량물 취급 계획서·작업계획서 작성 요청 시\n"
            "- 'WELDING_WORK_ORDER': 선박 블록 용접 작업지시서 작성·발행 요청 시"
        ),
    )
    urgency: Optional[Literal["NORMAL", "URGENT"]] = Field(
        default="NORMAL",
        description="긴급, 급행, 당일, 2시간 이내 등 납기 단축이 필요한 요청이면 'URGENT', 일반이면 'NORMAL'",
    )
    target_destination: Optional[str] = Field(default=None)


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
        "- 사용자의 발화나 이전 대화에 언급되지 않은 `block_id`, `worker_name`, `material`, `thickness`, `process`는 절대 임의로 채우지 말고 반드시 `null`로 반환하세요.\n"
        "- 사용자가 특정 블록이나 작업자를 지정하지 않고 '울산 야드 작업 가능한 블록 찾아서 작업지시서 자동 편성해줘', 'B-104 작업지시서 작성해줘'처럼 목표만 제시하면 `task_type='doc_generation'`, `auto_assign_requested=True`로 설정하세요.\n\n"
        "[판단 기준]\n"
        "1. CAD 도면 분석, 공정 계획 수립, 오늘 작업 계획/우선순위 큐 브리핑 요청은 `task_type='cad_planning'`으로 분류하세요.\n"
        "2. 멀티턴 후속 선택(1번/2번, A/B 등) 및 보충 요청은 직전 어시스턴트 응답과 결합하여 `KEEP_MERGE`로 처리하세요.\n"
        "3. 등록된 선박 블록 현황이나 본사 자재·장비 원장 조회는 `task_type='erp_resource_qa'`, `doc_type=null`로 분류하세요.\n"
        "4. 사외 협력사·외주·하청을 명시하지 않고 투입 가능 작업자를 물으면 `hr_worker_qa`, 협력사를 지정하면 `scm_partner_qa`로 분류하세요.\n"
        "5. 비전 탐지 현황/용접 대기 구역/로봇 상태는 `vision_inspection_qa`로 분류하세요.\n"
        "6. 이전 문서 작업과 무관한 새 독립 질문은 `context_action='RESET_NEW'`로 설정하세요.\n"
        "7. 지역명과 함께 현장 기상/풍속/기온을 물으면 `task_type='knowledge_qa'`, `need_weather=True`로 설정하세요.\n"
        "8. 중량물 취급 계획서, 블록 인양/운반 계획서 요청은 `task_type='doc_generation'`, `doc_type='HEAVY_LIFTING_PLAN'`으로 설정하세요.\n"
        "9. 긴급·당일·급행 요청은 `urgency='URGENT'`로 설정하세요.\n"
        "10. `scm_partner_qa`에서 가용 재고·납기·인력·배차 가능 여부를 묻거나 조건을 협의·문의하는 단계는 `is_action_requested=False`로 설정하고, 직전 협의안을 수락하여 실제 발주서(PO) 발행·주문을 실행하라는 지시일 때만 `is_action_requested=True`로 설정하세요.",
    ),
    MessagesPlaceholder(variable_name="messages", optional=True),
    (
        "human",
        "[직전 턴 역질문 대기 여부]: {is_waiting}\n"
        "[현재 로그인 사용자]: {login_user_name}\n"
        "[로그인 사용자 기본 작업구역]: {user_workspace} (야드코드: {branch_code})\n"
        "[DB 등록 외부 협력사 메타데이터]: {registered_partners}\n"
        "[이전 누적 작업 제원]: {accumulated_params}\n"
        "[최신 사용자 질의]: {query}",
    ),
])


def _resolve_branch_from_location_text(loc_text: Optional[str]) -> Optional[str]:
    if not loc_text:
        return None
    up = str(loc_text).strip().upper()
    if "거제" in up or "GEOJE" in up:
        return "BR-GEOJE-01"
    if "울산" in up or "ULSAN" in up:
        return "BR-ULSAN-01"
    return None


# ==============================================================================
# Node 1: RBAC 권한 및 야드 컨텍스트 초기화
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
# Node 2: LLM 구조화 의도/목표 및 슬롯 추출
# ==============================================================================
def intent_slot_extractor_node(state: ShipAgentState) -> Dict[str, Any]:
    query = state.get("user_query", "").strip()
    messages = state.get("messages", [])
    existing_params = dict(state.get("extracted_params") or {})
    is_waiting = state.get("is_param_complete") == "no"

    can_access_erp = state.get("can_access_erp", "Y")
    default_workspace = state.get("default_workspace")
    branch_code = state.get("branch_code")

    # CAD 파일 경로가 명시 전달된 경우 즉시 Team 4(cad_planning)로 라우팅
    if state.get("cad_dxf_path"):
        if can_access_erp == "N":
            blocked_msg = (
                "🔒 **[접근 권한 제한]** 현재 로그인하신 계정은 CAD 공정 계획 편성 및 ERP 원장 접근 권한이 없습니다."
            )
            return {
                "task_type": "chat",
                "extracted_params": existing_params,
                "is_param_complete": "yes",
                "missing_params": [],
                "search_status": "completed",
                "final_answer": blocked_msg,
                "messages": [AIMessage(content=blocked_msg)],
            }
        return {
            "task_type": "cad_planning",
            "extracted_params": existing_params,
            "is_param_complete": "yes",
            "missing_params": [],
            "search_status": "completed",
        }

    partners_meta = list_registered_partners_from_db()
    raw_user_name = str(state.get("user_name") or "").strip()
    login_full_name = raw_user_name.split("(")[0].strip() if raw_user_name else ""

    try:
        res: UniversalQueryAnalysis = (ANALYSIS_PROMPT | structured_analyzer).invoke({
            "messages": messages[-8:] if messages else [],
            "is_waiting": "예 (제원 보충 대기 중)" if is_waiting else "아니오",
            "login_user_name": login_full_name or "미지정",
            "user_workspace": default_workspace or "미지정",
            "branch_code": branch_code or "미지정",
            "registered_partners": json.dumps(partners_meta, ensure_ascii=False, default=str),
            "accumulated_params": json.dumps(existing_params, ensure_ascii=False, default=str),
            "query": query,
        })
    except Exception as e:
        print(f"⚠️ [Team 1 의도 분석 LLM 호출 경고]: {e}")
        res = UniversalQueryAnalysis(
            task_type="knowledge_qa" if query else "chat",
            search_queries=[query] if query else [],
        )

    resolved_doc_type = res.doc_type
    pref_mode = res.preferred_execution_mode or "AUTO"
    out_domain = res.outsourcing_domain or "NOT_APPLICABLE"
    ctx_action = res.context_action or "KEEP_MERGE"
    urgency_val = res.urgency or "NORMAL"

    last_ai_text = ""
    for m in reversed(messages or []):
        if getattr(m, "type", "") == "ai":
            last_ai_text = str(getattr(m, "content", ""))
            break

    recent_user_texts = [query]
    for m in (messages[-6:] if messages else []):
        if getattr(m, "type", "") == "human":
            recent_user_texts.append(str(getattr(m, "content", "")))
    conversation_corpus = (" ".join(recent_user_texts) + " " + last_ai_text).upper()
    conversation_raw = " ".join(recent_user_texts) + " " + last_ai_text

    # 1. block_id 정제 + 근거 검증
    raw_block_id = str(res.block_id or "").strip().upper()
    cleaned_block_id: Optional[str] = None
    if raw_block_id and raw_block_id not in STEEL_GRADES and res.task_type != "knowledge_qa":
        block_num_token = raw_block_id.replace("BLK-", "").replace("B-", "")
        if raw_block_id in conversation_corpus or (
            len(block_num_token) >= 3 and block_num_token in conversation_corpus
        ):
            cleaned_block_id = raw_block_id

    # 2. process 정제
    raw_proc = str(res.process or "").strip().upper()
    cleaned_process = (
        raw_proc
        if (raw_proc in VALID_WELD_PROCESSES and raw_proc in conversation_corpus)
        else None
    )

    # 3. worker_name 정제
    raw_worker = str(res.worker_name or "").strip()
    cleaned_worker = re.sub(r"\s*\([^)]*\)", "", raw_worker).strip()
    if (
        cleaned_worker
        and cleaned_worker not in conversation_raw
        and cleaned_worker != login_full_name
    ):
        cleaned_worker = ""

    # 4. 맥락 병합/초기화 판정
    old_block = str(existing_params.get("block_id") or "").strip().upper() or None
    is_followup_turn = is_waiting or (bool(last_ai_text) and ctx_action == "KEEP_MERGE")
    should_reset_context = (
        not is_followup_turn
        or res.auto_assign_requested
        or (res.task_type == "scm_partner_qa" and not is_waiting)
        or (cleaned_block_id and old_block and cleaned_block_id != old_block)
        or (ctx_action == "RESET_NEW" and res.task_type != "doc_generation")
    )

    if should_reset_context:
        base_ctx = get_branch_context_from_db(branch_code)
        merged_params: Dict[str, Any] = {
            "branch_code": branch_code,
            "work_zone": default_workspace or base_ctx.get("default_zone"),
            "location": base_ctx.get("region_name") or existing_params.get("location"),
        }
    else:
        merged_params = dict(existing_params)

    explicit_branch = _resolve_branch_from_location_text(res.location or query)
    if explicit_branch:
        merged_params["branch_code"] = explicit_branch
        b_ctx = get_branch_context_from_db(explicit_branch)
        if b_ctx.get("region_name"):
            merged_params["location"] = b_ctx["region_name"]
        if b_ctx.get("default_zone"):
            merged_params["work_zone"] = b_ctx["default_zone"]
    elif res.location and str(res.location).strip():
        merged_params["location"] = str(res.location).strip()

    if (
        not should_reset_context
        and existing_params.get("doc_type") == "HEAVY_LIFTING_PLAN"
        and res.task_type == "doc_generation"
        and (resolved_doc_type or "WELDING_WORK_ORDER") == "WELDING_WORK_ORDER"
        and not cleaned_process
    ):
        resolved_doc_type = "HEAVY_LIFTING_PLAN"

    if res.task_type not in ("knowledge_qa", "cad_planning"):
        new_fields = {
            "block_id": cleaned_block_id,
            "camera_code": res.camera_code,
            "material": res.material,
            "thickness": res.thickness,
            "process": cleaned_process,
            "weld_position": res.weld_position,
            "worker_name": cleaned_worker or None,
            "equipment": res.equipment,
            "requested_item_name": res.requested_item_name,
            "doc_type": (
                (resolved_doc_type or "WELDING_WORK_ORDER")
                if res.task_type == "doc_generation"
                else None
            ),
            "urgency": urgency_val,
            "target_destination": res.target_destination,
        }
        for k, v in new_fields.items():
            if v is not None and str(v).strip() != "":
                merged_params[k] = str(v).strip()
        if res.quantity is not None:
            merged_params["quantity"] = float(res.quantity)
        if res.additional_specs:
            merged_params.update(res.additional_specs)
        if res.equipment and str(res.equipment).strip():
            merged_params["user_explicit_equipment"] = True
        elif should_reset_context:
            merged_params["user_explicit_equipment"] = False

    # RBAC 접근 권한 검사
    if can_access_erp == "N" and res.task_type in (
        "cad_planning",
        "erp_resource_qa",
        "hr_worker_qa",
        "hr_safety_qa",
        "scm_partner_qa",
        "doc_generation",
    ):
        blocked_msg = (
            "🔒 **[접근 권한 제한]** 현재 로그인하신 계정은 전사 ERP 자원·인사·공급망 원장 조회 및 "
            "작업지시서·계획서 발행 권한이 부여되어 있지 않습니다."
        )
        return {
            "task_type": "chat",
            "extracted_params": merged_params,
            "is_param_complete": "yes",
            "missing_params": [],
            "search_status": "completed",
            "final_answer": blocked_msg,
            "messages": [AIMessage(content=blocked_msg)],
        }

    if res.task_type == "chat":
        reply_text = (res.direct_chat_reply or "").strip() or (
            "안녕하세요! 스마트 조선소 공정 관제 및 자율 작업지시 지원 AI입니다. 무엇을 도와드릴까요?"
        )
        return {
            "task_type": "chat",
            "extracted_params": merged_params,
            "is_param_complete": "yes",
            "missing_params": [],
            "search_status": "completed",
            "final_answer": reply_text,
            "messages": [AIMessage(content=reply_text)],
        }

    if res.task_type == "cad_planning":
        return {
            "task_type": "cad_planning",
            "extracted_params": merged_params,
            "is_param_complete": "yes",
            "missing_params": [],
            "search_status": "completed",
        }

    # 문서 생성 요청에서 블록이나 작업자가 비어 있으면 자동으로 자율 편성 모드(auto_assign_mode) 활성화
    is_auto_assign = False
    if res.task_type == "doc_generation":
        is_lifting = merged_params.get("doc_type") == "HEAVY_LIFTING_PLAN"
        if res.auto_assign_requested:
            is_auto_assign = True
        elif not merged_params.get("block_id"):
            is_auto_assign = True
        elif not is_lifting and not merged_params.get("worker_name"):
            is_auto_assign = True

    need_weather_flag = True if res.task_type == "doc_generation" else res.need_weather
    is_outsourcing_init = (
        res.task_type == "scm_partner_qa"
        or pref_mode == "OUTSOURCED_B2B"
    )

    return {
        "task_type": res.task_type,
        "auto_assign_mode": is_auto_assign,
        "is_action_requested": res.is_action_requested,
        "preferred_execution_mode": pref_mode,
        "outsourcing_domain": out_domain,
        "worker_scope": res.worker_scope,
        "is_outsourcing_required": is_outsourcing_init,
        "target_categories": res.target_categories,
        "need_weather": need_weather_flag,
        "extracted_params": merged_params,
        "search_queries": list(res.search_queries),
        "retry_count": 0,
        "search_status": "idle",
    }


def route_after_intent(
    state: ShipAgentState,
) -> Literal["hq_erp_diagnostician", "spec_completeness_judge", "__end__"]:
    task_type = state.get("task_type")
    params = state.get("extracted_params") or {}
    if task_type in ("chat", "cad_planning"):
        return "__end__"
    if task_type == "vision_inspection_qa":
        return "spec_completeness_judge"
    if task_type == "knowledge_qa" and not any(
        params.get(k) for k in ("block_id", "material", "worker_name", "equipment")
    ):
        return "spec_completeness_judge"
    return "hq_erp_diagnostician"


# ==============================================================================
# Node 3: 본사 SHIP_ERP 사전 진단 및 블록 마스터 동기화
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
        if bm.get("work_zone"):
            merged_params["work_zone"] = bm["work_zone"]

    branch_info = erp_status.get("branch_info") or {}
    user_query_text = state.get("user_query", "")
    current_loc = str(merged_params.get("location") or "").strip()
    is_user_explicit_loc = bool(current_loc and current_loc in user_query_text)

    if branch_info.get("region_name") and not is_user_explicit_loc:
        if bm is not None or not current_loc or any(
            tok in current_loc.upper() for tok in ("ZONE", "DOCK", "BAY")
        ):
            merged_params["location"] = branch_info["region_name"]

    if not merged_params.get("work_zone") and branch_info.get("default_zone"):
        merged_params["work_zone"] = branch_info["default_zone"]

    outsource_domain = state.get("outsourcing_domain") or "NOT_APPLICABLE"
    if merged_params.get("doc_type") == "HEAVY_LIFTING_PLAN" and merged_params.get("weight_ton"):
        lifting_eval = evaluate_heavy_lifting_capacity(
            block_weight_ton=merged_params.get("weight_ton"),
            vehicles_equipment=erp_status.get("vehicles_equipment") or [],
            target_branch=merged_params.get("branch_code"),
            requested_equipment=merged_params.get("equipment"),
        )
        erp_status["lifting_capacity_eval"] = lifting_eval
        if lifting_eval.get("selected_equipment"):
            erp_status["assigned_vehicle_code"] = lifting_eval["selected_equipment"].get("asset_code")
        if lifting_eval.get("need_outsourcing"):
            erp_status["require_outsourcing"] = True
            erp_status["outsourcing_reason"] = f"HEAVY_TRANSPORT: {lifting_eval.get('reason')}"
            outsource_domain = "HEAVY_TRANSPORT"

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
        "outsourcing_domain": outsource_domain,
        "is_outsourcing_required": is_outsourcing,
    }


# ==============================================================================
# Node 4: 유효성 판정 (미등록 블록 명시 시만 차단, 미지정 시 자율 편성 통과!)
# ==============================================================================
def spec_completeness_judge_node(state: ShipAgentState) -> Dict[str, Any]:
    task_type = state.get("task_type")
    merged_params = dict(state.get("extracted_params") or {})
    erp_status = state.get("erp_resource_status") or {}
    queries = list(state.get("search_queries") or [])
    query = state.get("user_query", "").strip()

    missing: List[str] = []
    is_complete = "yes"
    clarification_answer = None

    if (
        task_type == "doc_generation"
        and merged_params.get("block_id")
        and not erp_status.get("block_master")
    ):
        bad_block = merged_params["block_id"]
        is_complete = "no"
        missing = ["block_id"]
        avail_blocks = erp_status.get("available_blocks") or []
        block_lines = "\n".join(
            f"- `{b.get('block_id')}` ({b.get('block_name', '-')}, {b.get('material', '-')}/{b.get('thickness', '-')}, "
            f"{b.get('process', '-')}, {b.get('branch_code', '-')})"
            for b in avail_blocks
        ) or "- 등록된 블록 없음"
        clarification_answer = (
            f"> 입력하신 블록 번호 **`{bad_block}`**는 본사 ERP 블록 마스터에 등록되어 있지 않아 문서를 생성할 수 없습니다.\n\n"
            f"### 📋 등록된 선박 블록 목록\n{block_lines}\n\n"
            f"위 등록 블록 중 하나를 지정하시거나, **\"현재 야드 작업 가능한 블록 알아서 편성해줘\"**라고 요청해 주세요."
        )

    if not queries and task_type in ("knowledge_qa", "doc_generation"):
        if task_type == "doc_generation" and merged_params.get("doc_type") == "HEAVY_LIFTING_PLAN":
            queries = ["중량물 취급 작업계획서 크레인 양중 줄걸이 와이어로프 신호수 안전 기준"]
        else:
            mat = merged_params.get("material", "")
            thk = merged_params.get("thickness", "")
            proc = merged_params.get("process", "용접")
            queries = [f"{mat} {thk} {proc} WPS 작업 표준 규격".strip()] if (mat or thk) else [query]

    result_dict: Dict[str, Any] = {
        "missing_params": missing,
        "is_param_complete": is_complete,
        "search_queries": queries,
        "final_answer": clarification_answer,
    }
    if clarification_answer:
        result_dict["messages"] = [AIMessage(content=clarification_answer)]
    return result_dict


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