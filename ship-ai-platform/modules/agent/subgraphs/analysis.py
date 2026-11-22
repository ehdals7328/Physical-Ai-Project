"""
modules/agent/subgraphs/analysis.py
Team 1: 4-Node 전문 분업 서브그래프 (하드코딩 0% - DB 메타데이터 & 스키마 기반 자율 추론)
[1. RBAC/야드 컨텍스트 초기화] -> [2. LLM 의도/슬롯 추출] -> [3. SHIP_ERP 정밀 진단 + 중량물 하중 검토] -> [4. 제원 완결성/외주 판정]
"""

import json
import re
from typing import Any, Dict, List, Literal, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage

from modules.agent.state import DocCategory, ShipAgentState
from modules.agent.tools.erp_tools import (
    check_hq_erp_resources,
    evaluate_heavy_lifting_capacity,
    get_branch_context_from_db,
    list_erp_workers,
    list_registered_partners_from_db,
)

# 의도 분류 및 슬롯 추출 속도 최적화를 위해 즉시 응답 모델 사용
llm = ChatOpenAI(model="gpt-5-mini", reasoning_effort="low")

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
            "- 'scm_partner_qa': 사외 협력사 소속의 자재·단가·외주인력·일당·장비 가용성 조회, 납기/조건 협의(A2A) 및 외주 발주·배차 실행\n"
            "- 'erp_resource_qa': 본사 야드(사업장) 현황 및 공정, 본사 창고의 자재 재고 또는 본사 보유 장비 조회\n"
            "- 'hr_worker_qa': 사내외 인사 원장(ERP_WORKER_CERT) 기반 전체 작업자 명단 및 공정별 투입 가능 인력 조회 (야드·구역·설비·본인 위치만 묻는 질문 제외)\n"
            "- 'hr_safety_qa': 특정 작업자 1인의 선급 자격증 유효성 단건 검증\n"
            "- 'knowledge_qa': 사내 조선·해양 WPS 기술 규격, 안전 지침, 실시간 기상/날씨 조회 등 도메인 정보 질의\n"
            "- 'doc_generation': 사용자가 명시적으로 작업지시서·중량물 취급 계획서·작업계획서의 작성, 생성, 발행을 요청한 경우에만 선택 (단순 날씨·규격·자원 문의는 절대 선택 금지)\n"
            "- 'chat': 일상 인사, 플랫폼 사용법, 로그인 사용자의 현재 위치·담당 야드·작업구역 확인, 또는 조선·해양 도메인 외 분야의 지원 가능 여부 확인\n"
        )
    )

    is_action_requested: bool = Field(
        default=False,
        description=(
            "재고·단가·LOT·가용성 '확인'이나 '조회/협의 요청'은 반드시 False! "
            "오직 사용자가 구체적 품목/수량/대수를 지정하여 실제 발주서 발행, 주문 확정, 배차 실행, 원장 커밋을 명령한 경우에만 True"
        ),
    )
    preferred_execution_mode: Literal["AUTO", "IN_HOUSE", "OUTSOURCED_B2B"] = Field(
        default="AUTO",
        description="외부 협력사/외주/하청 자원을 조회하거나 활용하려는 의도면 'OUTSOURCED_B2B', 본사 자체 처리를 지정하면 'IN_HOUSE', 별도 구분이 없으면 'AUTO'",
    )
    outsourcing_domain: Literal[
        "ALL_PARTNERS",
        "OUTSOURCED_WORKER",
        "WELDING_MATERIAL",
        "HEAVY_TRANSPORT",
        "NOT_APPLICABLE",
    ] = Field(
        default="NOT_APPLICABLE",
        description=(
            "외주 연계 또는 협력사 조회(scm_partner_qa) 시 대상 자원 유형:\n"
            "- 'ALL_PARTNERS': 특정 품목·인력·장비 지정 없이 등록된 전체 협력사 목록, 업체 수, 업종 현황을 묻는 일반 조회\n"
            "- 'OUTSOURCED_WORKER': 외주 용접사, 작업자, 인력 명단, 용접 공정 자격 보유 여부, 인건비/일당 문의\n"
            "- 'WELDING_MATERIAL': 와이어, 강판, 가스 등 물리적 자재 재고·단가·납품·긴급 협의 문의\n"
            "- 'HEAVY_TRANSPORT': 중량물 운송, 트랜스포터, 크레인 배차 문의\n"
            "- 'NOT_APPLICABLE': 외주나 협력사 자원과 무관한 일반 대화·사내 단순 질의·비전 관제 질의"
        ),
    )
    context_action: Literal["KEEP_MERGE", "RESET_NEW"] = Field(
        default="KEEP_MERGE",
        description=(
            "대화 맥락 처리 방식:\n"
            "- 'KEEP_MERGE': 작업지시서/중량물 취급 계획서 작성 흐름을 이어가거나 이전 대화의 블록·제원·작업자 조건을 유지·보충할 때\n"
            "- 'RESET_NEW': 이전 문서 작성과 완전히 무관한 새로운 독립 주제 질의일 때"
        ),
    )
    target_categories: List[DocCategory] = Field(
        default_factory=list,
        description=(
            "문서 기반 지식 검색(RAG)이 필요한 질의(`knowledge_qa`, `doc_generation`)에 한해 "
            "해당하는 기술·안전·규정 문서 분류 목록을 선택하고, "
            "단순 데이터 원장 조회나 일반 대화일 경우에는 빈 리스트(`[]`)를 반환할 것"
        ),
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
            "사용자가 1인칭으로 본인을 작업자로 지칭한 경우에는 [현재 로그인 사용자]의 순수 성명을 추출할 것. "
            "한국인 성명(2~4음절)이 일반 명사나 직급처럼 보이더라도(예: '홍길동 반장' -> '홍길동') "
            "사람 이름은 반드시 추출하고 뒤에 붙은 직책·호칭만 제외할 것"
        ),
    )
    equipment: Optional[str] = Field(None, description="질의에 직접 언급된 투입 설비 또는 차량명 (예: '50톤 크레인', '15톤 지게차', '트랜스포터', 없으면 None)")
    quantity: Optional[float] = Field(None, description="질의에 직접 언급된 자재/품목의 수치 수량 (없으면 None)")
    additional_specs: Dict[str, Any] = Field(
        default_factory=dict,
        description="질의에 명시된 기타 엔지니어링 조건",
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="벡터 검색용 공학 표준 키워드 쿼리",
    )
    direct_chat_reply: Optional[str] = Field(
        default=None,
        description=(
            "task_type이 'chat'인 경우에만 작성하는 2~3줄 이내의 간결한 응답. "
            "오직 [최신 사용자 질의]에만 답변하며 이전 대화 내용을 다시 요약하지 말 것. "
            "조선·해양 도메인 외 분야의 지원 가능 여부를 물으면 해당 분야 미지원 안내와 현재 지원 가능한 조선소 업무를 간결히 답변할 것."
            "조선소 업무 외 일상 잡담은 짧게 응대하되 조선소 공정·작업지시 지원 역할을 안내할 것"
        ),
    )
    requested_item_name: Optional[str] = Field(
        default=None,
        description="사용자가 조회하거나 발주·주문·협의하려는 구체적인 자재/품목명 원문 (예: '용접봉', '와이어', 'AH36 강판', '탄산가스'). 언급이 없으면 None",
    )
    worker_scope: Optional[Literal["ALL", "HQ_ONLY", "EXTERNAL_ONLY"]] = Field(
        default="ALL",
        description=(
            "인사 원장(hr_worker_qa) 조회 범위 (인사 질의가 아니면 'ALL' 또는 None):\n"
            "- 'HQ_ONLY': '우리 회사', '본사', '자사', '직영', '정규직' 등 본사 소속 인력만 물었을 때\n"
            "- 'EXTERNAL_ONLY': '외주', '협력사', '하청', '프리랜서' 인력만 물었을 때\n"
            "- 'ALL': 소속 구분 없이 전체 인력을 물었을 때"
        ),
    )
    doc_type: Literal["WELDING_WORK_ORDER", "HEAVY_LIFTING_PLAN"] = Field(
        default="WELDING_WORK_ORDER",
        description=(
            "task_type이 'doc_generation'일 때 발급할 문서 종류:\n"
            "- 'HEAVY_LIFTING_PLAN': 중량물 취급 계획서, 블록 인양/양중 계획서, 크레인/트랜스포터 이동 안전 계획서 작성·발급 요청 시\n"
            "- 'WELDING_WORK_ORDER': 일반 용접 작업지시서 작성·발급 요청 시"
        ),
    )
    urgency: Literal["NORMAL", "URGENT"] = Field(
        default="NORMAL",
        description="긴급, 급행, 당일, 2시간 이내 등 납기 단축이 필요한 요청이면 'URGENT', 일반이면 'NORMAL'",
    )
    target_destination: Optional[str] = Field(
        default=None,
        description="중량물 이동/인양 시 최종 목적지 구역 또는 도크 (예: 'DOCK-1', '탑재장', 언급 없으면 None)",
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
        "1. **멀티턴 후속 선택(1번/2번, A/B 등) 및 보충 통합 처리**: "
        "직전 어시스턴트 응답이 선택지(1/2, A/B 등)를 제시했거나 추가 조건을 역질문한 경우, "
        "사용자의 짧은 후속 응답('2번해줘', 'A로 진행할게' 등)만 단독으로 보지 말고 **직전 어시스턴트가 제시한 해당 번호 항목의 실제 내용과 이전 대화의 주제(`task_type`, `doc_type`)를 그대로 이어받아 분류**하십시오.\n"
        "   - 직전 안내가 협력사 조회(`scm_partner_qa`) 후속 선택지였고 사용자가 자재 납품사·재고·LOT 확인 항목을 선택했다면 `task_type='scm_partner_qa'`, `outsourcing_domain='WELDING_MATERIAL'`, `preferred_execution_mode='OUTSOURCED_B2B'`로 설정하십시오.\n"
        "   - 작업자 교체, 장비 교체, 블록 변경, 제원 보충, 외주 전환 등 문서 발행 진행을 선택한 경우: "
        "`task_type='doc_generation'`, `context_action='KEEP_MERGE'`로 설정하고 직전 어시스턴트 응답과 사용자 발화에서 확정된 변경 값(`worker_name`, `equipment`, `block_id`, `material`, `thickness`, `process`, `preferred_execution_mode`, `doc_type` 등)을 모두 반영하십시오.\n"
        "   - 안전 체크리스트, 예열 기준, 기술 규정 확인을 선택한 경우: `task_type='knowledge_qa'`로 분류하여 해당 안내를 제공하십시오.\n"
        "2. **인사 조회 vs 협력사 외주 조회 구분**: 사용자가 사외 협력사, 하청, 외주 또는 등록된 협력사명을 명시적으로 언급하지 않고 투입 가능한 작업자를 묻는 경우에는 `hr_worker_qa`로 분류하십시오.\n"
        "3. **비전 관제 질의**: 현장 비전 탐지 현황, 용접 대기 구역, 용접선 포착 상태, 간섭 장애물, 결함 현황 조회는 `vision_inspection_qa`로 분류하십시오.\n"
        "4. **실행 모드**: 사용자가 명시적으로 외주·협력사·사외 연계를 지정하거나 직전 안내의 외주 전환 대안을 선택한 경우 `preferred_execution_mode='OUTSOURCED_B2B'`로 설정하고, 별도 언급이 없으면 `'AUTO'`로 두십시오.\n"
        "5. **독립 주제 전환(`RESET_NEW`)**: 이전 작업지시서와 무관하게 전체 비전 카메라 현황, 단독 외주 배차/발주, 일반 기술 규격 등을 새로 질문한 경우에는 `context_action='RESET_NEW'`로 분류하십시오.\n"
        "6. **기상·환경 조건 질의(`knowledge_qa`)**: 지역명과 함께 현장 기상·기온·풍속·강수 상태를 묻는 질문은 일상 잡담(`chat`)이 아니라 반드시 `task_type='knowledge_qa'` 및 `need_weather=True`로 분류하십시오.\n"
        "7. **사내 ERP 자원·블록 마스터 조회(`erp_resource_qa`)**: 등록된 선박 블록 전체 목록이나 특정 공정(SAW/FCAW/GMAW 등)에 해당하는 블록을 묻는 질문은 비전 관제(`vision_inspection_qa`)가 아니라 반드시 `task_type='erp_resource_qa'`로 분류하십시오.\n"
        "8. **인사 조회 범위(`worker_scope`)**: `task_type='hr_worker_qa'`일 때는 반드시 `worker_scope`를 판단하십시오. "
        "'우리 회사', '우리 지금 회사', '본사', '자사', '직영', '정규직'처럼 자사 소속 인력만 물으면 `'HQ_ONLY'`, "
        "'다른 회사', '타사', '외주', '협력사', '프리랜서' 소속을 확인하거나 물으면 `'EXTERNAL_ONLY'`, "
        "소속 구분 없이 전체를 물으면 `'ALL'`로 설정하십시오.\n"
        "9. **중량물 취급 계획서(`HEAVY_LIFTING_PLAN`)**: 중량물 취급 계획서, 블록 인양·양중·운반 계획서, 크레인/트랜스포터 이동 작업계획서 작성·발급 요청은 반드시 `task_type='doc_generation'`, `doc_type='HEAVY_LIFTING_PLAN'`으로 분류하십시오.\n"
        "10. **긴급도(`urgency`)**: 사용자가 긴급, 급행, 당일, 빠르게, 즉시 납품·배차 협의를 요구하면 `urgency='URGENT'`로 설정하십시오.",
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

    # 직전 어시스턴트 응답 텍스트 확보 (멀티턴 후속 선택 근거 검증용)
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

    # 1. block_id 정제 + 대화/선택지 근거 검증
    raw_block_id = str(res.block_id or "").strip().upper()
    cleaned_block_id: Optional[str] = None
    if (
        raw_block_id
        and raw_block_id not in STEEL_GRADES
        and res.task_type != "knowledge_qa"
    ):
        block_num_token = raw_block_id.replace("BLK-", "").replace("B-", "")
        if raw_block_id in conversation_corpus or (
            len(block_num_token) >= 3 and block_num_token in conversation_corpus
        ):
            cleaned_block_id = raw_block_id

    # 2. process 정제 + 대화/선택지 근거 검증
    raw_proc = str(res.process or "").strip().upper()
    cleaned_process = (
        raw_proc
        if (raw_proc in VALID_WELD_PROCESSES and raw_proc in conversation_corpus)
        else None
    )

    # 3. worker_name 정제 (대화 내역, 직전 AI 선택지, 또는 로그인 본인 성명 허용)
    raw_worker = str(res.worker_name or "").strip()
    cleaned_worker = re.sub(r"\s*\([^)]*\)", "", raw_worker).strip()
    if (
        cleaned_worker
        and cleaned_worker not in conversation_raw
        and cleaned_worker != login_full_name
    ):
        cleaned_worker = ""

    # 4. 이전 누적 슬롯의 비정상 값 정리 및 맥락 병합/초기화 판정
    if str(existing_params.get("block_id") or "").upper() in STEEL_GRADES:
        existing_params.pop("block_id", None)
    if str(existing_params.get("process") or "").upper() not in VALID_WELD_PROCESSES:
        existing_params.pop("process", None)

    old_block = str(existing_params.get("block_id") or "").strip().upper() or None
    if old_block:
        existing_params["block_id"] = old_block

    is_followup_turn = is_waiting or (bool(last_ai_text) and res.context_action == "KEEP_MERGE")
    should_reset_context = (
        not is_followup_turn
        or (cleaned_block_id and old_block and cleaned_block_id != old_block)
        or (res.context_action == "RESET_NEW" and res.task_type != "doc_generation")
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

    # 날씨 질의(knowledge_qa)에서도 사용자가 명시한 도시명(예: '대구')은 기상 MCP 조회를 위해 반영
    if res.location and str(res.location).strip():
        merged_params["location"] = str(res.location).strip()

    # 멀티턴 보충 턴에서 직전 중량물 취급 계획서(HEAVY_LIFTING_PLAN) 흐름 보호
    resolved_doc_type = res.doc_type
    if (
        not should_reset_context
        and existing_params.get("doc_type") == "HEAVY_LIFTING_PLAN"
        and res.task_type == "doc_generation"
        and res.doc_type == "WELDING_WORK_ORDER"
        and not cleaned_process
    ):
        resolved_doc_type = "HEAVY_LIFTING_PLAN"

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
            "requested_item_name": res.requested_item_name,
            "doc_type": resolved_doc_type if res.task_type == "doc_generation" else merged_params.get("doc_type"),
            "urgency": res.urgency,
            "target_destination": res.target_destination,
        }
        for k, v in new_fields.items():
            if v is not None and str(v).strip() != "":
                merged_params[k] = str(v).strip()
        if res.quantity is not None:
            merged_params["quantity"] = float(res.quantity)
        if res.additional_specs:
            merged_params.update(res.additional_specs)

    # [패치 4] ERP 원장 조회·예약을 수반하는 doc_generation에도 RBAC 접근 권한 검사 적용
    if can_access_erp == "N" and res.task_type in (
        "erp_resource_qa",
        "hr_worker_qa",
        "hr_safety_qa",
        "scm_partner_qa",
        "doc_generation",
    ):
        blocked_msg = (
            "🔒 **[접근 권한 제한]** 현재 로그인하신 계정은 전사 ERP 자원·인사·공급망 원장 조회 및 "
            "작업지시서·계획서 발행 권한이 부여되어 있지 않습니다. "
            "시스템 관리자 또는 공정 관리자에게 권한 승인을 요청해 주세요."
        )
        return {
            "task_type": "chat",
            "need_erp_check": False,
            "extracted_params": merged_params,
            "is_param_complete": "yes",
            "missing_params": [],
            "search_status": "completed",
            "final_answer": blocked_msg,
        }

    if res.task_type == "chat":
        reply_text = (res.direct_chat_reply or "").strip() or (
            "안녕하세요! 스마트 조선소 공정 관제 및 작업지시 지원 AI입니다. 무엇을 도와드릴까요?"
        )
        return {
            "task_type": "chat",
            "need_erp_check": False,
            "extracted_params": merged_params,
            "is_param_complete": "yes",
            "missing_params": [],
            "search_status": "completed",
            "final_answer": reply_text,
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
    if task_type == "chat":
        return "__end__"
    if task_type == "vision_inspection_qa":
        return "spec_completeness_judge"
    if task_type == "knowledge_qa" and not any(
        params.get(k) for k in ("block_id", "material", "worker_name", "equipment")
    ):
        return "spec_completeness_judge"
    return "hq_erp_diagnostician"


# ==============================================================================
# Node 3: 본사 SHIP_ERP 마스터 제원 동기화 및 자원·중량물 하중 진단 노드
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
    loc_upper = current_loc.upper()

    is_user_explicit_loc = bool(current_loc and current_loc in user_query_text)

    if branch_info.get("region_name") and not is_user_explicit_loc:
        if (
            bm is not None
            or not current_loc
            or any(tok in loc_upper for tok in ("ZONE", "DOCK", "BAY"))
        ):
            merged_params["location"] = branch_info["region_name"]

    if not merged_params.get("work_zone") and branch_info.get("default_zone"):
        merged_params["work_zone"] = branch_info["default_zone"]

    # 중량물 취급 작업계획서(HEAVY_LIFTING_PLAN) 요청 시 정격하중 안전율 사전 진단
    outsource_domain = state.get("outsourcing_domain") or "NOT_APPLICABLE"
    if merged_params.get("doc_type") == "HEAVY_LIFTING_PLAN":
        lifting_eval = evaluate_heavy_lifting_capacity(
            block_weight_ton=merged_params.get("weight_ton"),
            vehicles_equipment=erp_status.get("vehicles_equipment") or [],
            target_branch=merged_params.get("branch_code"),
            requested_equipment=merged_params.get("equipment"),
        )
        erp_status["lifting_capacity_eval"] = lifting_eval
        if lifting_eval.get("selected_equipment"):
            erp_status["assigned_vehicle_code"] = lifting_eval["selected_equipment"].get("asset_code")
        # 사내 장비 정격하중 초과(OVERLOAD) 또는 미보유 시 외주 중량물 A2A 배차 자동 연계
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
# Node 4: 엔지니어링 제원 완결성 판정 및 자연스러운 역질문 생성 노드
# ==============================================================================
CLARIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 스마트 조선소 공정 관리 AI 어시스턴트입니다.\n"
        "사용자가 문서(용접 작업지시서 또는 중량물 취급 작업계획서) 발행을 요청했으나 아직 일부 필수 제원이 입력되지 않았습니다.\n"
        "딱딱한 경고성 표현이나 영문 DB 컬럼명·상태 코드(`BLOCK_ID`, `VALID`, `is_busy`, `weld_process` 등)를 절대 쓰지 말고, "
        "현장 실무자가 한눈에 보고 바로 답할 수 있도록 아래 구조로 간결하게 안내하세요.\n\n"
        "[출력 구조]\n"
        "1. 상단 `>` 인용구(1줄): 문서 작성을 위해 **추가 확인이 필요한 필수 항목**을 굵게 요약\n"
        "2. `### 📋 선택 가능한 등록 블록 및 가용 자원`: [ERP 등록 블록 목록] 전체(블록 지정 시 강종·두께·중량·공정 자동 매핑 안내)와, 용접 작업지시서인 경우 [ERP 가용 작업자 목록] 중 자격이 유효하고 대기 중인 적격 작업자를 간결한 불릿으로 추천\n"
        "3. `💡 입력 예시`: 사용자가 바로 따라 입력할 수 있는 한 줄 문장 예시 제시",
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
        is_lifting = merged_params.get("doc_type") == "HEAVY_LIFTING_PLAN"
        if is_lifting:
            required_slots = {
                "block_id": "작업 대상 블록 번호",
                "weight_ton": "대상 블록 중량(톤)",
            }
        else:
            required_slots = {
                "block_id": "작업 대상 블록 번호",
                "material": "모재 강종",
                "thickness": "판 두께",
                "process": "용접 공정",
                "worker_name": "투입 작업자 성명",
            }
        missing = [k for k in required_slots if not merged_params.get(k)]

        # [패치 1] 입력된 block_id가 실제 ERP_BLOCK_MASTER에 등록된 유효 블록인지 엄격 검증
        if merged_params.get("block_id") and not erp_status.get("block_master"):
            if "block_id" not in missing:
                missing.insert(0, "block_id")
            required_slots["block_id"] = (
                f"유효한 등록 블록 번호 (입력값 '{merged_params.get('block_id')}' 미등록)"
            )

        if missing:
            is_complete = "no"
            missing_kor = ", ".join(required_slots[m] for m in missing)
            req_proc = str(merged_params.get("process") or "").upper()
            valid_workers = [] if is_lifting else [
                w for w in (erp_status.get("available_workers") or [])
                if str(w.get("status", "")).upper() == "VALID"
                and str(w.get("weld_process", "NA")).upper() != "NA"
                and (not req_proc or str(w.get("weld_process", "")).upper() == req_proc)
            ]
            try:
                clarification_res = (CLARIFICATION_PROMPT | llm).invoke({
                    "query": query,
                    "params": json.dumps(merged_params, ensure_ascii=False, default=str),
                    "missing_kor": missing_kor,
                    "blocks": json.dumps(erp_status.get("available_blocks") or [], ensure_ascii=False, default=str),
                    "workers": json.dumps(valid_workers, ensure_ascii=False, default=str),
                })
                clarification_answer = clarification_res.content
            except Exception as e:
                print(f"⚠️ [Team 1 역질문 생성 경고]: {e}")
                clarification_answer = (
                    f"문서 발행을 위해 다음 필수 제원을 추가로 알려주세요: **{missing_kor}** "
                    "(ERP에 등록된 대상 블록 코드를 입력해 주시면 관련 제원이 자동으로 연동됩니다.)"
                )

    if not queries and task_type in ("knowledge_qa", "doc_generation"):
        if task_type == "doc_generation" and merged_params.get("doc_type") == "HEAVY_LIFTING_PLAN":
            queries = ["중량물 취급 작업계획서 크레인 양중 줄걸이 와이어로프 신호수 안전 기준"]
        else:
            mat = merged_params.get("material", "")
            thk = merged_params.get("thickness", "")
            proc = merged_params.get("process", "용접")
            if task_type == "doc_generation" and (mat or thk):
                queries = [f"{mat} {thk} {proc} WPS 작업 표준 규격".strip()]
            else:
                queries = [query]

    result_dict: Dict[str, Any] = {
        "missing_params": missing,
        "is_param_complete": is_complete,
        "search_queries": queries,
        "final_answer": clarification_answer,
    }
    if clarification_answer:
        result_dict["messages"] = [AIMessage(content=clarification_answer)]
    return result_dict


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