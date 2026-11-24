"""
modules/agent/subgraphs/docgen.py
Team 3: 거버넌스 게이트키퍼(Supervisor) 기반 용접 작업지시서(WORK_INSTRUCTIONS) &
중량물 취급 작업계획서(HEAVY_LIFTING_PLAN) 생성, 자율 트러블슈팅, 인사/ERP/비전/A2A 브리핑
& [Logical Bridge: Work Authorization] 트랜잭션 마감 서브그래프
"""

import json
import re
import uuid
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage

from modules.agent.state import ShipAgentState
from modules.agent.tools.document_tools import (
    export_heavy_lifting_plan_pdf,
    export_work_order_file,
    log_inspection_history,
)
from modules.agent.tools.erp_tools import (
    calculate_required_wire_kg,
    commit_erp_work_order,
    get_active_partner_from_db,
    list_erp_workers,
)

# 일반 QA 응답은 빠른 모델로, 공식 작업지시서·중량물 계획서 구조화 설계는 정밀 모델로 분리
qa_llm = ChatOpenAI(model="gpt-5-mini", reasoning_effort="low")
llm = ChatOpenAI(model="gpt-5.1")


class DecimalEncoder(json.JSONEncoder):
    """Oracle DB의 Decimal 및 날짜/시간 객체를 JSON 안전 타입으로 자동 변환"""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return super().default(obj)


class WorkOrderGenerationOutput(BaseModel):
    brief_summary: str = Field(
        description="채팅창에 표시할 3줄 이내의 간결한 작업지시서 발행 핵심 요약 (세부 제원 나열 생략, 판정 결과 및 주요 주의사항만 요약)"
    )
    wps_code: Optional[str] = Field(
        default=None,
        description="참조 문서나 작업 제원에 명시된 절차서/표준 코드 ('.pdf' 파일명 입력 금지)",
    )
    current_amp: str = Field(
        description="대상 공정(FCAW/SAW 등) 및 판두께에 부합하는 적정 전류 숫자 또는 범위만 기입 (단위 'A' 절대 포함 금지, 예: FCAW 200~280 / SAW 550~700)"
    )
    voltage_v: str = Field(
        description="대상 공정(FCAW/SAW 등) 및 판두께에 부합하는 적정 전압 숫자 또는 범위만 기입 (단위 'V' 절대 포함 금지, 예: FCAW 24~30 / SAW 28~34)"
    )
    travel_speed_cm_min: str = Field(
        description="대상 공정 및 판두께에 부합하는 적정 용접 이송 속도 숫자 또는 범위만 기입 (단위 'cm/min' 절대 포함 금지, 예: 25~45)"
    )
    shielding_gas: str = Field(
        description="SAW 공정인 경우 반드시 '입상 플럭스 차폐 (보호가스 미사용)'으로 기입하고, FCAW/GMAW인 경우에만 보호가스 종류 및 유량 기입"
    )
    preheat_temp_c: str = Field(
        description="인출된 RAG 문서 및 모재 강종·두께 기준 예열 및 층간온도 관리 기준"
    )
    consumable_spec: str = Field(
        description=(
            "절대 용접와이어 품목명(예: 플럭스코어드 용접와이어 등)을 적지 마십시오! "
            "인출된 RAG 문서(와이어/플럭스 카탈로그 및 선박건조 안전지침)에 명시된 "
            "용접재료의 현장 보관 온도·습도, 지면/벽 이격 거리, 흡습 및 오염 방지 관리 수칙 문장을 추출해 기입하십시오."
        )
    )
    ndt_requirement: str = Field(
        description="인출된 RAG 문서(Hull Weld Inspection 등)에 명시된 비파괴검사(VT/RT/UT/MT) 요건"
    )


class HeavyLiftingPlanOutput(BaseModel):
    brief_summary: str = Field(
        description="채팅창에 표시할 2~3줄 이내 중량물 취급 작업계획서 핵심 요약 (정격하중률, 투입 장비, 핵심 안전 주의사항)"
    )
    travel_route: str = Field(
        description="A4 1페이지 양식용 1~2줄 핵심 요약(80자 이내): 출발 구역 ➔ 목적지 동선 및 노면·상부 간섭 점검"
    )
    supervisor_name: str = Field(
        description="본사 소속 현장 작업지휘자 성명/직책만 1줄 기입 (예: 최선임(EMP-1002) / 중조립 A동 반장, 외주 용접사 절대 기입 금지)"
    )
    signalman_assignment: str = Field(
        description="A4 1페이지 양식용 1줄 요약(60자 이내): 전담 신호수 1인 배치, 표준 수신호·전용 무전 채널 교신"
    )
    rigging_spec: str = Field(
        description="A4 1페이지 양식용 2줄 이내 핵심 규격(90자 이내): 와이어로프(안전율 5 이상) 4줄 걸이, 정격 샤클, 러그 균열 점검"
    )
    sling_angle_and_cog: str = Field(
        description="A4 1페이지 양식용 2줄 이내 핵심 기준(90자 이내): 줄걸이 각도 60° 이내 준수, 지상 30cm 시험 인양으로 무게중심 확인"
    )
    fall_and_tip_prevention: str = Field(
        description="A4 1페이지 양식용 2줄 이내(100자 이내): 하부 출입금지 구역 설정, 유도 로프(Tag Line) 2본 체결, 서행 인양·지지목 고정"
    )
    emergency_procedure: str = Field(
        description="A4 1페이지 양식용 2줄 이내(100자 이내): 풍속 10m/s 초과·이상 진동 시 즉시 비상정지 후 저속 하강 및 대피"
    )
    safety_checklist: List[str] = Field(
        default_factory=list,
        description="각 항목 40자 이내의 간결한 1줄 안전 점검 항목 4개만 기입",
    )


class DomainQAResponse(BaseModel):
    answer: str = Field(
        description="사용자 질의에 대한 핵심 답변 (불필요한 장문 서론·반복 요약·전체 표 재작성 제외)"
    )
    is_supported_by_internal_docs: bool = Field(
        default=True,
        description=(
            "인출된 사내 규격 문서(RAG)에 사용자 질문의 대상 분야·공법에 해당하는 근거가 실제로 존재하면 True, "
            "사내 규격에 해당 분야 규정이 없어 지원 한계만 고지하는 경우 False"
        ),
    )


# ==============================================================================
# 0. Team 3 거버넌스 게이트키퍼 노드 (Supervisor / Safety & Process Interlock)
# ==============================================================================
def governance_gatekeeper_node(state: ShipAgentState) -> Dict[str, Any]:
    task_type = state.get("task_type", "knowledge_qa")

    if task_type != "doc_generation":
        print(
            f"\n👉 [Team 3 Gatekeeper]: domain_qa_writer "
            f"(사유: 일반 기술/ERP/인사/비전 질의, task_type='{task_type}')\n"
        )
        return {"next": "domain_qa_writer"}

    params = state.get("extracted_params") or {}
    missing = state.get("missing_params", [])
    weather = state.get("weather_data") or {}
    weather_status = weather.get("status")
    safety_eval = weather.get("safety_evaluation") or {}
    # Fail-Closed 원칙: 기상/안전 평가 결과가 없으면 기본값을 'STOP'으로 간주
    safety_status = safety_eval.get("status") or "STOP"

    erp_status = dict(state.get("erp_resource_status") or {})
    vision = state.get("vision_data") or {}
    partner_plan = state.get("partner_erp_plan") or {}

    # [패치 2] 비전 조회 오류(ERROR) 및 안전거리 미확보(N) 공통 Fail-Closed 판별 (NO_DATA는 허용)
    vision_status_upper = str(vision.get("status", "")).upper()
    vision_insp_upper = str(vision.get("inspection_status", "")).upper()
    vision_clearance_upper = str(vision.get("safety_clearance_yn") or "").upper()
    is_vision_error = (
        vision_status_upper == "ERROR"
        or vision_insp_upper == "ERROR"
    )
    is_clearance_blocked = vision_clearance_upper == "N"

    # --------------------------------------------------------------------------
    # [분기 1] 중량물 취급 작업계획서(HEAVY_LIFTING_PLAN) 전용 안전 인터록
    # --------------------------------------------------------------------------
    if params.get("doc_type") == "HEAVY_LIFTING_PLAN":
        cap_eval = dict(erp_status.get("lifting_capacity_eval") or {})
        a2a_prop = (partner_plan.get("a2a_negotiation") or {}).get("proposal") or {}

        # 사용자가 직접 과하중 장비(예: 15톤 지게차)를 강제 지정한 경우에만 즉시 차단
        explicit_overload = bool(params.get("user_explicit_equipment")) and cap_eval.get("status") == "OVERLOAD"
        a2a_safe_vehicle = (
            not explicit_overload
            and bool(a2a_prop.get("assigned_vehicle_code"))
            and float(a2a_prop.get("load_ratio_pct") or 999) <= 90.0
        )

        is_capacity_ok = bool(cap_eval.get("is_safe")) or a2a_safe_vehicle
        is_weather_ok = (
            weather_status == "SUCCESS"
            and safety_status in ("PASS", "CONDITIONAL_PASS")
        )
        has_vision_hazard = (
            is_vision_error
            or is_clearance_blocked
            or any(
                det.get("is_hazard") is True
                or any(
                    k in str(det.get("class_name", "")).upper()
                    for k in ["OBSTACLE", "CLAMP_CONFLICT"]
                )
                for det in (vision.get("detections") or [])
            )
        )

        fail_reasons = []
        if not params.get("block_id"):
            fail_reasons.append("대상블록미지정(블록 번호 미확인)")
        if not params.get("weight_ton"):
            fail_reasons.append("블록중량미확인(정격하중 검토 불가)")
        if missing:
            fail_reasons.append(f"필수제원누락({missing})")
        if not is_capacity_ok:
            fail_reasons.append(
                f"양중장비정격하중부적합({cap_eval.get('reason', '정격하중 초과')})"
            )
        if weather_status != "SUCCESS":
            fail_reasons.append(f"기상조회실패({weather_status or 'MISSING'})")
        elif not is_weather_ok:
            fail_reasons.append(f"중량물기상안전차단({safety_status})")
        if is_vision_error:
            fail_reasons.append(
                f"비전관제조회오류({vision.get('error') or vision.get('error_message') or 'ERROR'})"
            )
        elif has_vision_hazard:
            fail_reasons.append("비전인양구역장애물간섭감지(안전거리미확보)")

        print("\n" + "=" * 65)
        print("🏗️ [Team 3 중량물 취급 계획서 거버넌스 인터록 검사 현황]")
        print(f"   1. 대상 블록 / 중량        : {params.get('block_id')} ({params.get('weight_ton')}톤)")
        print(f"   2. 정격하중 안전율 판정    : {is_capacity_ok} (사내판정={cap_eval.get('status')}, A2A외주안전확보={a2a_safe_vehicle})")
        print(f"   3. 중량물 풍속/기상 판정   : 상태='{weather_status}' | 판정='{safety_status}'")
        print(f"   4. 비전 오류/장애물 여부   : {has_vision_hazard} (상태={vision_status_upper or 'N/A'}, 안전거리={vision_clearance_upper or 'NO_DATA'})")

        # [중량물 분기 내 if fail_reasons: 교체]
        if fail_reasons:
            erp_status["interlock_fail_reasons"] = fail_reasons
            replan_count = int(state.get("replan_count", 0))
            auto_assign = bool(state.get("auto_assign_mode", False))
            # 사용자가 과하중 장비를 강제 지정한 게 아니거나, 자율 편성 모드에서 다른 블록/장비 우회가 가능한 경우 Self-Correction!
            # 비전 DB 장애나 기상 MCP 통신 실패 같은 시스템 오류는 재편성해도 해결 불가 -> 즉시 차단
            is_system_error = is_vision_error or (weather_status != "SUCCESS")
            can_self_correct = (
                (not is_system_error)
                and replan_count < 2
                and (
                    (not is_capacity_ok and not explicit_overload)
                    or (auto_assign and (not is_weather_ok or has_vision_hazard))
                )
            )
            if can_self_correct:
                excl_blocks = list(state.get("excluded_blocks") or [])
                if auto_assign and (not is_weather_ok or has_vision_hazard) and params.get("block_id"):
                    excl_blocks.append(str(params["block_id"]).upper())
                feedback = (
                    f"중량물 계획서 인터록 차단({', '.join(fail_reasons)}). "
                    f"외주 중량물 트랜스포터(A2A) 배차 또는 대체 안전 블록/장비로 자율 재편성하세요."
                )
                print(f"↩️ [Self-Correction Triggered #{replan_count + 1}]: {feedback}")
                return {
                    "next": "__end__",
                    "replan_needed": True,
                    "replan_count": replan_count + 1,
                    "replan_feedback": feedback,
                    "excluded_blocks": excl_blocks,
                    "erp_resource_status": erp_status,
                }

            print(f"👉 [최종 결정]: troubleshooter_agent (차단 사유: {', '.join(fail_reasons)})")
            print("=" * 65 + "\n")
            return {"next": "troubleshooter_agent", "replan_needed": False, "erp_resource_status": erp_status}

        print("👉 [최종 결정]: heavy_lifting_plan_writer (모든 중량물 안전 인터록 통과)")
        print("=" * 65 + "\n")
        return {"next": "heavy_lifting_plan_writer", "replan_needed": False, "erp_resource_status": erp_status}
    # --------------------------------------------------------------------------
    # [분기 2] 선박 블록 용접 작업지시서(WELDING_WORK_ORDER) 거버넌스 인터록
    # --------------------------------------------------------------------------
    worker_qual = dict(erp_status.get("worker_qualification") or {})
    selected_partner = partner_plan.get("selected_partner") or {}
    ext_worker_id = (
        selected_partner.get("assigned_worker_id")
        or selected_partner.get("ext_worker_id")
        or selected_partner.get("worker_id")
    )

    if worker_qual.get("status") == "NOT_FOUND" and ext_worker_id:
        biz_type = str(selected_partner.get("business_type") or "OUTSOURCED_WORKER").upper()
        ext_valid = (
            biz_type == "OUTSOURCED_WORKER"
            and bool(selected_partner.get("can_accept_order", True))
        )
        worker_qual = {
            "worker_id": ext_worker_id,
            "worker_name": selected_partner.get("worker_name") or params.get("worker_name"),
            "role": selected_partner.get("role", "WELDER"),
            "company_code": selected_partner.get("partner_id", "VEND-DONGHAE-WELD"),
            "branch_code": params.get("branch_code"),
            "cert_code": selected_partner.get("cert_code"),
            "weld_process": selected_partner.get("weld_process"),
            "cert_expiry_date": str(selected_partner.get("cert_expiry_date", "")),
            "status": "VALID" if ext_valid else "UNAVAILABLE",
            "is_valid": ext_valid,
            "is_process_matched": True,
            "mismatch_reason": None,
        }

    req_process = params.get("process")
    worker_process = worker_qual.get("weld_process")

    is_proc_matched = worker_qual.get("is_process_matched", True)
    if (
        req_process
        and worker_process
        and str(req_process).upper() != "NA"
        and str(worker_process).upper() != "NA"
    ):
        if str(req_process).upper() != str(worker_process).upper():
            is_proc_matched = False
            worker_qual["is_process_matched"] = False
            worker_qual["is_valid"] = False
            worker_qual["status"] = "PROCESS_MISMATCH"
            worker_qual["mismatch_reason"] = (
                f"공정 불일치: 블록 요구 공정({req_process}) != 작업자 인증 공정({worker_process})"
            )

    is_yard_matched = worker_qual.get("is_yard_matched", True)

    if worker_qual:
        erp_status["worker_qualification"] = worker_qual
        erp_status["worker_qualified"] = bool(
            worker_qual.get("is_valid", False) and is_proc_matched and is_yard_matched
        )

    is_worker_valid = (
        erp_status.get("worker_qualified", True)
        and worker_qual.get("is_valid", False)
        and is_proc_matched
        and is_yard_matched
    )
    if not worker_qual or not worker_qual.get("worker_id"):
        is_worker_valid = False

    robot_exec = vision.get("robot_execution") or {}
    has_vision_hazard = (
        is_vision_error
        or is_clearance_blocked
        or vision_insp_upper == "DEFECT"
        or str(vision.get("welding_state", "")).upper() == "IN_WELDING"
        or str(robot_exec.get("emergency_stop_yn", "N")).upper() == "Y"
        or any(
            det.get("is_hazard") is True
            or any(
                k in str(det.get("class_name", "")).upper()
                for k in ["OBSTACLE", "DEFECT", "CRACK", "CLAMP_CONFLICT"]
            )
            for det in (vision.get("detections") or [])
        )
    )
    # [신규] 자재(WELDING_MATERIAL) 부족 시 선행 PO 승인 여부 인터록 검증 (이중 발주 방지)
    outsourcing_domain = str(state.get("outsourcing_domain") or "").upper()
    stock_available = erp_status.get("stock_available", True)
    existing_approved_po = None
    existing_draft_po = None

    if not stock_available and outsourcing_domain in ("", "WELDING_MATERIAL"):
        try:
            import os
            from sqlalchemy import create_engine, text
            from config import Config

            binds = getattr(Config, "SQLALCHEMY_BINDS", {}) or {}
            user_db_uri = (
                binds.get("ship_user")
                or binds.get("user")
                or os.getenv("SHIP_USER_DATABASE_URL")
            )
            engine_opts = getattr(Config, "SQLALCHEMY_ENGINE_OPTIONS", {"pool_pre_ping": True})
            user_eng = create_engine(user_db_uri, **engine_opts) if user_db_uri else None

            target_br = str(
                worker_qual.get("target_branch_code")
                or params.get("branch_code")
                or state.get("branch_code")
                or "BR-ULSAN-01"
            ).upper()
            wire_code = str(erp_status.get("selected_wire_code") or "WIRE-").upper()

            if user_eng:
                with user_eng.connect() as conn:
                    po_rows = conn.execute(
                        text("""
                            SELECT WI.APPROVAL_STATUS, WO.WORK_NO
                              FROM WORK_INSTRUCTIONS WI
                              JOIN WORK_ORDERS WO ON WI.WORK_ID = WO.WORK_ID
                             WHERE UPPER(WO.WORK_REGION) = :branch
                               AND (UPPER(WO.WORK_NO) LIKE 'PO-MRP-%' OR UPPER(WO.WORK_NO) LIKE 'PO-%')
                               AND DBMS_LOB.INSTR(UPPER(WI.INSTRUCTION_TEXT), :wire_code) > 0
                             ORDER BY WI.CREATED_AT DESC
                        """),
                        {"branch": target_br, "wire_code": wire_code},
                    ).mappings().all()

                for row in po_rows:
                    st = str(row.get("approval_status") or "").upper()
                    wno = row.get("work_no")
                    if st == "APPROVED" and not existing_approved_po:
                        existing_approved_po = wno
                    elif st == "DRAFT" and not existing_draft_po:
                        existing_draft_po = wno
        except Exception as po_chk_err:
            print(f"⚠️ [Gatekeeper 선행 PO 조회 경고]: {po_chk_err}")
        
    is_weather_failed = (
        weather_status != "SUCCESS"
        or safety_status not in ("PASS", "CONDITIONAL_PASS")
    )

    is_blocked = (
        bool(missing)
        or is_weather_failed
        or not is_worker_valid
        or not is_yard_matched
        or has_vision_hazard
    )

    print("\n" + "=" * 65)
    print("🔍 [Team 3 거버넌스 게이트키퍼 인터록 검사 현황]")
    print(f"   1. 작업 유형 (task_type)    : {task_type}")
    print(f"   2. 필수 제원 누락 여부      : {bool(missing)} (누락 목록: {missing})")
    print(f"   3. 기상 상태 및 안전 판정   : 상태='{weather_status}' | 판정='{safety_status}'")
    print(
        f"   4. 작업자 선급 자격 유효성  : {is_worker_valid} "
        f"(작업자: {worker_qual.get('worker_name')}, 상태: {worker_qual.get('status')})"
    )
    print(
        f"   5. 공정(WPS) 상호 일치 여부 : {is_proc_matched} "
        f"(블록요구:{req_process} vs 작업자인증:{worker_process})"
    )
    print(
        f"   6. 야드(권역) 일치 여부     : {is_yard_matched} "
        f"(블록권역:{worker_qual.get('target_region') or '-'}({worker_qual.get('target_branch_code') or '-'}) "
        f"vs 작업자권역:{worker_qual.get('worker_region') or '-'}({worker_qual.get('branch_code') or '-'}))"
    )
    print(
        f"   7. 비전/로봇 위험 감지 여부 : {has_vision_hazard} "
        f"(상태={vision_status_upper or 'N/A'}, 안전거리={vision_clearance_upper or 'NO_DATA'})"
    )

    if is_blocked:
        fail_reasons = []
        if missing:
            fail_reasons.append(f"제원누락({missing})")
        if weather_status != "SUCCESS":
            fail_reasons.append(f"기상조회실패({weather_status or 'MISSING'})")
        elif safety_status not in ("PASS", "CONDITIONAL_PASS"):
            fail_reasons.append(f"기상안전차단({safety_status})")
        if not is_proc_matched:
            fail_reasons.append(
                f"용접공정불일치(블록요구:{req_process} != 작업자인증:{worker_process})"
            )
        elif not is_yard_matched:
            fail_reasons.append(
                f"야드지역불일치(블록:{worker_qual.get('target_region')}({worker_qual.get('target_branch_code')}) "
                f"!= 작업자:{worker_qual.get('worker_region')}({worker_qual.get('branch_code')}))"
            )
        elif not is_worker_valid:
            if not worker_qual or not worker_qual.get("worker_id"):
                fail_reasons.append("투입작업자미배정(작업자 성명 미확인)")
            else:
                fail_reasons.append(
                    f"작업자선급자격결격({worker_qual.get('worker_name')}: {worker_qual.get('status')})"
                )
        if is_vision_error:
            fail_reasons.append(
                f"비전관제조회오류({vision.get('error') or vision.get('error_message') or 'ERROR'})"
            )
        elif is_clearance_blocked:
            fail_reasons.append("비전안전거리미확보(간섭장애물감지)")
        elif has_vision_hazard:
            fail_reasons.append("비전결함/용접중Lock/로봇비상정지")

        erp_status["interlock_fail_reasons"] = fail_reasons
        if not is_worker_valid:
            erp_status["worker_qualified"] = False

        replan_count = int(state.get("replan_count", 0))
        auto_assign = bool(state.get("auto_assign_mode", False))
        is_system_error = is_vision_error or (weather_status != "SUCCESS")

        # 복구 가능 조건: 시스템 장애가 아니면서, 작업자 결격 또는 자율 편성 모드에서 타 안전 블록으로 우회 가능할 때
        worker_only_issue = (
            (not is_worker_valid or not is_proc_matched or not is_yard_matched)
            and not is_weather_failed
            and not has_vision_hazard
        )
        auto_block_issue = auto_assign and (is_weather_failed or has_vision_hazard or not params.get("block_id"))

        if (not is_system_error) and replan_count < 2 and (worker_only_issue or auto_block_issue):
            excl_workers = list(state.get("excluded_workers") or [])
            excl_blocks = list(state.get("excluded_blocks") or [])
            if worker_only_issue and worker_qual.get("worker_name"):
                excl_workers.append(str(worker_qual["worker_name"]))
            if auto_block_issue and params.get("block_id"):
                excl_blocks.append(str(params["block_id"]).upper())

            feedback = (
                f"용접 작업지시서 인터록 차단({', '.join(fail_reasons)}). "
                + (
                    f"제외된 작업자({excl_workers}) 대신 블록({params.get('block_id')})의 공정({req_process})·야드에 부합하는 "
                    f"대기 작업자를 찾거나(사내 없으면 외주 A2A 협의) `tool_sync_erp_plan`으로 교체하세요."
                    if worker_only_issue
                    else f"제외된 블록({excl_blocks}) 대신 현재 기상·비전 안전을 충족하는 다른 대기 블록과 적격 작업자를 탐색하세요."
                )
            )
            print(f"↩️ [Self-Correction Triggered #{replan_count + 1}]: {feedback}")
            print("=" * 65 + "\n")
            return {
                "next": "__end__",
                "replan_needed": True,
                "replan_count": replan_count + 1,
                "replan_feedback": feedback,
                "excluded_workers": excl_workers,
                "excluded_blocks": excl_blocks,
                "erp_resource_status": erp_status,
            }

        chosen_node = "troubleshooter_agent"
        print(f"👉 [최종 결정]: {chosen_node} (차단 사유: {', '.join(fail_reasons)})")
        print("=" * 65 + "\n")
        return {
            "next": chosen_node,
            "replan_needed": False,
            "erp_resource_status": erp_status,
        }

    chosen_node = "work_order_writer"
    print(f"👉 [최종 결정]: {chosen_node} (모든 안전 및 ERP 인터록 통과)")
    print("=" * 65 + "\n")
    return {
        "next": chosen_node,
        "erp_resource_status": erp_status,
    }


def route_from_gatekeeper(
    state: ShipAgentState,
) -> Literal["troubleshooter_agent", "work_order_writer", "heavy_lifting_plan_writer", "domain_qa_writer", "__end__"]:
    nxt = state.get("next")
    if nxt == "__end__" and state.get("replan_needed"):
        return "__end__"
    if nxt in (
        "troubleshooter_agent",
        "work_order_writer",
        "heavy_lifting_plan_writer",
        "domain_qa_writer",
    ):
        return nxt
    return "domain_qa_writer"


# ==============================================================================
# 1. 도메인 QA, ERP 재고, 인사, 비전 관제 및 A2A 협력사 브리핑 워커 (domain_qa_writer)
# ==============================================================================
QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 스마트 조선소 엔지니어링 & ERP·비전·SCM 관제 기술 자문 AI입니다.\n"
        "제공된 [현재 누적 작업 제원], [본사 SHIP_ERP 자원 및 인사 현황], [협력사 B2B 및 A2A 협의 결과], [인출된 사내 규격 문서(RAG)] 및 이전 대화 맥락만을 근거로 "
        "현장 실무자가 화면을 보자마자 한눈에 파악할 수 있도록 시각적으로 정돈된 마크다운으로 답변하세요.\n\n"
        "[공통 가독성 및 표현 원칙]\n"
        "1. **DB 컬럼명·영문 상태 코드 노출 절대 금지**:\n"
        "   - JSON 데이터에 포함된 영문 컬럼명·테이블명·상태 코드(`STANDBY`, `IN_USE`, `VALID`, `EXPIRED`, `BEFORE_WELD`, `IN_WELDING`, `PASS`, `CONDITIONAL_PASS`, `STOP`, `IN_HOUSE`, `OUTSOURCED_B2B`, `branch_code`, `operation_status` 등)는 **절대 원문 그대로 쓰지 마세요.**\n"
        "   - 반드시 자연스러운 현장 한국어(`대기 중`, `운행 중`, `유효`, `만료`, `용접 전 대기`, `용접 진행 중`, `정상`, `조건부 승인`, `작업 중지`, `사내 직영`, `외주 연계`)로 변환하여 표기하세요.\n"
        "2. **한눈에 들어오는 스캔형 레이아웃 (긴 줄글 금지)**:\n"
        "   - 인사 전체 명단 조회(`hr_worker_qa`)를 제외한 모든 답변은 맨 위에 **`>` 인용구(1~2줄)**로 질문에 대한 핵심 결론을 먼저 제시하세요.\n"
        "   - 세부 내용은 긴 문단 줄글 대신 **`###` 소제목 + 마크다운 표 또는 굵은 키워드 불릿**으로 정리하고, 같은 내용을 중복 설명하지 마세요.\n\n"
        "3. **질문 범위 엄수 및 표준 마크다운 표 문법 강제**:\n"
        "   - 사용자가 작업 대상 블록이나 블록 목록만 물었으면 [본사 SHIP_ERP 자원 현황]의 `available_blocks`를 기준으로 **블록 표(블록ID | 블록명 | 강종/두께 | 공정 | 중량 | 소속 야드)와 착수 가능 안내만** 출력하고, 묻지 않은 자재 재고·장비·인력 명단·현장 체크리스트는 절대 추가하지 마세요.\n"
        "   - 마크다운 표의 구분선은 콜론(`:`)을 절대 넣지 말고 반드시 `|---|---|---|---|---|---|` 형태로만 작성하며, 표 위·아래에는 빈 줄 1줄을 반드시 넣으세요.\n"
        "[질의 유형별 작성 기준]\n"
        "1. **인사 원장 및 작업자 추천 (`hr_worker_qa`, `hr_safety_qa`)**:\n"
        "   - 전체 인사 원장 마크다운 표는 시스템이 응답 상단에 자동으로 부착하므로, 답변 본문에 전체 표를 절대 다시 작성하지 마세요.\n"
        "   - 대신 제공된 [사내외 인사 원장 마크다운 표]와 [본사 SHIP_ERP 자원 현황]을 근거로 질문 조건(야드, 공정, 대기 여부, 자격 유효성)에 맞는 핵심 추천 인원만 `### 👷 조건 부합 추천 인원` 아래 3~5줄 불릿으로 간결하게 정리하세요.\n"
        "2. **기술 규격·기상·자격 요건 질의 (`knowledge_qa`)**:\n"
        "   - 상단 `>` 인용구에 핵심 기준(예열 온도, 전류·전압 범위, 기상 안전 판정 등)을 먼저 요약하고, 세부 조건을 마크다운 표 또는 불릿으로 정리하세요.\n"
        "   - 강종·두께 문의 시 [본사 SHIP_ERP 자원 현황]의 등록 블록 및 자재 재고 규격을 최우선으로 안내하고 RAG 규격을 보충하세요.\n"
        "   - 작업자 자격 요건 문의 시 대상 용접 공정과 일치하는 인증 공정 및 유효기간 내의 선급 자격 규격을 설명하고 적격 작업자를 안내하세요.\n"
        "3. **비전 관제 질의 (`vision_inspection_qa`)**:\n"
        "   - 상단 `>` 인용구에 용접 가능/대기 구역 수를 요약한 뒤, [비전(YOLOv11) 실시간 관제 및 검사 현황]을 근거로 카메라별 탐지 상태·용접선 포착·안전 클리어런스·로봇 상태를 마크다운 표로 브리핑하세요.\n"
        "4. **사내 ERP 자원 및 협력사 B2B·A2A 협의 조회 (`erp_resource_qa`, `scm_partner_qa`)**:\n"
        "   - 상단 `>` 인용구에 조회 또는 A2A 협의 결과 핵심 요약을 1~2줄로 제시하세요.\n"
        "   - 전사 야드 목록 문의 시 전체 야드 마스터를 기준으로, 특정 야드 자원 문의 시 해당 야드 자원만 필터링하여 `###` 소제목과 **마크다운 표**로 정리하세요(장비의 대기 중 / 운행 중 상태 명확 구분).\n"
        "   - 협력사 목록, 위치, 취급 품목, 외주 인력·장비 문의 시 [협력사 B2B 조회 결과]의 등록 협력사 및 상세 가용 목록을 반드시 표 위·아래 빈 줄을 띄워 깨지지 않는 **마크다운 표**로 정리하세요.\n"
        "   - A2A 자율 협의 결과(`a2a_negotiation`)에 분할 납품이나 긴급 ETA 역제안이 포함된 경우, 협의 옵션별 조건(납기, 수량, 단가)을 현장 실무자가 선택하기 쉽게 정리하세요.\n"
        "   - **[시스템 지원 한계 엄수 — 없는 기능 제안 금지]**:\n"
        "     * 우리 시스템은 협력사로 단순 재고·견적 문의 메일/문자를 발송하거나 회신을 수신하는 기능이 **없습니다.**\n"
        "     * 따라서 외부 발송용 이메일·문자 템플릿(복사용 요청문)을 작성하거나, '연락처/메일 방법을 알려주시면 대신 발송해 드리겠다'는 식의 실행 불가능한 선택지를 절대 제시하지 마세요.\n"
        "     * 대신 DB·A2A에서 조회된 **실시간 재고·LOT·단가·협의 납기 정보만 정확히 브리핑**하고, 시스템에서 바로 처리 가능한 후속 단계인 **'발주할 품목과 수량(또는 배차 대수)을 지정해 주시면 공식 외주 발주 주문서(PO) 초안을 생성해 드립니다'**로만 안내하세요.\n"
        "5. **규정 범위 한계 명시 및 불필요한 자원 나열 금지**:\n"
        "   - 사용자가 사내 자재, 블록, 장비, 작업자 현황을 명시적으로 묻지 않은 일반 기술·규정 질문에서는 사내 자원 상세 표나 전체 목록을 나열하지 마세요.\n"
        "   - 질문 대상 분야나 공법이 [인출된 사내 규격 문서(RAG)]에 존재하지 않는 경우, 장황한 일반론을 생성하지 말고 아래 3가지만 4~5줄 이내로 간결하게 답변하세요:\n"
        "       (1) 사내 표준 규격에는 해당 분야 규정이 없음을 명확히 고지\n"
        "       (2) 사내 ERP에 실제 등록된 지원 가능 공정 및 강종 범위를 동적으로 요약 안내\n"
        "       (3) 현재 사내 규격 내에서 확인이 필요한 항목 역질문\n"
    ),
    MessagesPlaceholder(variable_name="messages", optional=True),
    (
        "human",
        "[질의 유형]: {task_type}\n"
        "[현재 누적 작업 제원]: {params}\n"
        "[사용자 질문]: {query}\n\n"
        "[사내외 인사 원장 마크다운 표]:\n{worker_table}\n\n"
        "[비전(YOLOv11) 실시간 관제 및 검사 현황]:\n{vision_status}\n\n"
        "[본사 SHIP_ERP 자원 현황]:\n{erp_status}\n\n"
        "[협력사 B2B 조회 및 A2A 협의 결과]:\n{partner_plan}\n\n"
        "[인출된 사내 규격 문서(RAG)]:\n{docs}\n\n"
        "[실시간 기상 정보]:\n{weather}",
    ),
])


def domain_qa_writer_node(state: ShipAgentState) -> Dict[str, Any]:
    """
    일반 기술 규격(knowledge_qa), ERP 자원(erp_resource_qa), 인사 원장(hr_worker_qa/hr_safety_qa),
    비전 관제(vision_inspection_qa), 외주 협력사 조회·A2A 협의 및 단독 PO 초안(scm_partner_qa) 응답 생성 노드
    """
    task_type = state.get("task_type", "knowledge_qa")
    params = dict(state.get("extracted_params") or {})
    erp_status = dict(state.get("erp_resource_status") or {})
    partner_plan = dict(state.get("partner_erp_plan") or {})
    docs = state.get("retrieved_docs") or []
    weather = state.get("weather_data") or {}
    vision_data = state.get("vision_data") or {}
    messages = state.get("messages") or []
    user_query = state.get("user_query", "")

    # 1. 사내외 인사 원장 마크다운 표 생성 (실시간 투입 상태 '🔄 작업 중' 반영)
    worker_table_md = "해당 없음"
    table_title = "👷 사내외 인사 원장 실시간 현황"
    if task_type in ("hr_worker_qa", "hr_safety_qa"):
        try:
            workers = erp_status.get("available_workers") or []
            if not workers:
                workers_res = list_erp_workers.invoke({"branch_code": None})
                workers = (
                    workers_res.get("workers", [])
                    if isinstance(workers_res, dict)
                    else (workers_res or [])
                )

            worker_scope = state.get("worker_scope") or "ALL"
            if worker_scope == "HQ_ONLY":
                workers = [
                    w for w in workers
                    if str(w.get("company_code", "")).upper() == "COMP-HQ"
                    or str(w.get("employment_type", "")).upper() == "REGULAR"
                ]
                table_title = "👷 본사(COMP-HQ) 정규직 인사 원장 실시간 현황"
            elif worker_scope == "EXTERNAL_ONLY":
                workers = [
                    w for w in workers
                    if str(w.get("company_code", "")).upper() != "COMP-HQ"
                ]
                table_title = "👷 협력사·프리랜서 인사 원장 실시간 현황"

            if workers:
                erp_status["available_workers"] = workers

            if workers:
                lines = [
                    "| 작업자명 | 사번 | 고용형태 | 소속업체 | 상주야드 | 직책 | 선급 자격 규격 | 인증공정 | 자격 만료일 | 상태 |",
                    "|---|---|---|---|---|---|---|---|---|---|",
                ]
                for w in workers:
                    status_val = str(
                        w.get("status") or w.get("cert_status") or ""
                    ).upper()
                    is_ok = (status_val == "VALID") or (w.get("is_valid") is True)
                    if w.get("is_busy"):
                        busy_blk = w.get("assigned_block_id") or "타 블록"
                        status_badge = f"🔄 작업 중 ({busy_blk})"
                    elif is_ok:
                        status_badge = "✅ 투입 가능"
                    else:
                        status_badge = "❌ 투입 불가"

                    expiry_val = w.get("expiry_date") or w.get("cert_expiry_date") or "-"
                    lines.append(
                        f"| {w.get('worker_name', '-')} | {w.get('worker_id', '-')} | {w.get('employment_type', '-')} | "
                        f"{w.get('company_code', '-')} | {w.get('branch_code', '-')} | {w.get('role', '-')} | "
                        f"{w.get('cert_code', '-')} | {w.get('weld_process', '-')} | {expiry_val} | {status_badge} |"
                    )
                worker_table_md = "\n".join(lines)
            else:
                worker_table_md = "조회된 작업자가 없습니다."
        except Exception as e:
            worker_table_md = f"인사 원장 조회 오류: {e}"

    # 2. 비전 관제 데이터 및 RAG 문서 포맷팅
    vision_text = (
        json.dumps(vision_data, ensure_ascii=False, cls=DecimalEncoder)
        if vision_data
        else "조회된 비전 관제 데이터 없음"
    )

    if docs:
        doc_chunks = []
        for idx, d in enumerate(docs, 1):
            meta = d.get("metadata") or {}
            src = meta.get("document_name") or meta.get("source") or "사내문서"
            page = meta.get("page", "-")
            doc_chunks.append(f"[{idx}] ({src}, p.{page})\n{d.get('content', '')}")
        docs_text = "\n\n".join(doc_chunks)
    else:
        docs_text = "인출된 사내 규격 문서 없음"

    # 질문 유형에 따라 LLM에 넘길 ERP 컨텍스트 경량화 (토큰 80% 절감 -> 응답 속도 4~6초대 단축)
    compact_erp = dict(erp_status)
    if task_type == "erp_resource_qa":
        compact_erp.pop("available_workers", None)
        if any(k in user_query for k in ("블록", "BLOCK", "공정")):
            compact_erp.pop("inventory", None)
            compact_erp.pop("vehicles_equipment", None)
    elif task_type == "hr_worker_qa":
        compact_erp.pop("inventory", None)
        compact_erp.pop("vehicles_equipment", None)
        compact_erp.pop("available_blocks", None)
    elif task_type == "scm_partner_qa":
        compact_erp.pop("available_workers", None)
        compact_erp.pop("vehicles_equipment", None)
        compact_erp.pop("available_blocks", None)

    # 3. LLM 응답 생성
    qa_inputs = {
        "messages": messages[-6:] if messages else [],
        "task_type": task_type,
        "params": json.dumps(params, ensure_ascii=False, cls=DecimalEncoder),
        "query": user_query,
        "worker_table": worker_table_md,
        "vision_status": vision_text,
        "erp_status": json.dumps(compact_erp, ensure_ascii=False, cls=DecimalEncoder),
        "partner_plan": json.dumps(partner_plan, ensure_ascii=False, cls=DecimalEncoder),
        "docs": docs_text,
        "weather": json.dumps(weather, ensure_ascii=False, cls=DecimalEncoder),
    }

    if task_type == "knowledge_qa":
        structured_qa = qa_llm.with_structured_output(
            DomainQAResponse,
            method="function_calling",
        )
        qa_out: DomainQAResponse = (QA_PROMPT | structured_qa).invoke(qa_inputs)
        answer = (qa_out.answer or "").strip()
        is_supported_docs = qa_out.is_supported_by_internal_docs
    else:
        raw_res = (QA_PROMPT | qa_llm).invoke(qa_inputs)
        answer = (raw_res.content or "").strip()
        is_supported_docs = False

    # 4. hr_worker_qa 시 표 상단 결합 — 직전 AI 메시지에 이미 동일한 표가 있으면 중복 생략
    last_ai_content = ""
    for m in reversed(messages or []):
        if getattr(m, "type", "") == "ai":
            last_ai_content = str(getattr(m, "content", ""))
            break

    already_shown_same_table = f"### {table_title}" in last_ai_content

    if (
        task_type == "hr_worker_qa"
        and worker_table_md
        and worker_table_md != "해당 없음"
        and "조회된 작업자가 없습니다" not in worker_table_md
        and not already_shown_same_table
    ):
        answer = (
            f"### {table_title}\n\n"
            f"{worker_table_md}\n\n---\n\n{answer}"
        )

    # 5. scm_partner_qa 단독 외주 배차/발주 초안(DRAFT) 생성 — LLM 판정(is_action_requested) 및 등록 품목 토큰 대조
    generated_doc: Optional[Dict[str, Any]] = None
    selected_partner = partner_plan.get("selected_partner") or {}
    is_action_req = bool(state.get("is_action_requested"))
    domain = state.get("outsourcing_domain") or "NOT_APPLICABLE"

    has_qty = params.get("quantity") is not None and float(params.get("quantity") or 0) > 0
    if task_type == "scm_partner_qa" and is_action_req:
        if domain == "ALL_PARTNERS":
            is_action_req = False
        elif domain == "WELDING_MATERIAL" and not has_qty:
            is_action_req = False
        elif domain == "HEAVY_TRANSPORT" and not (has_qty or params.get("equipment")):
            is_action_req = False
        elif domain == "OUTSOURCED_WORKER" and not (
            has_qty or params.get("worker_name") or params.get("block_id")
        ):
            is_action_req = False

    # 요청 품목명과 실제 DB(사내 ERP + 협력사) 등록 품목 토큰 단위 대조 (수량/단위 제거)
    req_item_name = str(params.get("requested_item_name") or "").strip()
    is_unregistered_item = False
    if (
        task_type == "scm_partner_qa"
        and is_action_req
        and domain == "WELDING_MATERIAL"
        and req_item_name
    ):
        registered_items = []
        for code, info in (erp_status.get("inventory") or {}).items():
            registered_items.append({
                "code": code,
                "name": str(info.get("item_name") or ""),
                "unit": str(info.get("unit") or ""),
            })
        for p_item in (partner_plan.get("available_items") or partner_plan.get("items") or []):
            registered_items.append({
                "code": str(p_item.get("hq_item_code") or p_item.get("item_code") or ""),
                "name": str(p_item.get("item_name") or ""),
                "unit": str(p_item.get("unit") or ""),
            })

        if registered_items:
            cleaned_req = re.sub(
                r"\d+(?:\.\d+)?\s*(?:kg|g|t|ton|톤|ea|개|장|통|병|sheet)?",
                "",
                req_item_name,
                flags=re.IGNORECASE,
            ).strip()
            req_tokens = [
                tok.lower()
                for tok in re.split(r"[\s/\-(),]+", cleaned_req)
                if len(tok) >= 2
            ]
            matched = []
            for it in registered_items:
                haystack = f"{it['name']} {it['code']}".lower()
                if (cleaned_req and cleaned_req.lower() in haystack) or (
                    req_tokens and all(t in haystack for t in req_tokens)
                ):
                    matched.append(it)
            if not matched:
                is_unregistered_item = True
                seen = set()
                item_lines = []
                for it in registered_items:
                    if it["code"] and it["code"] not in seen:
                        seen.add(it["code"])
                        unit_txt = f", 단위: {it['unit']}" if it["unit"] else ""
                        item_lines.append(f"- `{it['code']}` — {it['name']}{unit_txt}")
                answer = (
                    f"> 요청하신 **'{req_item_name}'** 품목은 현재 사내 ERP 및 협력사 자재 원장에 등록되어 있지 않아 발주서를 생성할 수 없습니다.\n\n"
                    f"### 📦 현재 발주 가능한 등록 자재 목록\n"
                    + "\n".join(item_lines)
                    + "\n\n위 등록 품목 중 발주할 품목과 수량을 지정해 주시면 즉시 발주 초안을 작성해 드립니다."
                )

    if (
        not is_unregistered_item
        and task_type == "scm_partner_qa"
        and is_action_req
        and selected_partner.get("partner_id")
    ):
        from modules.agent.tools.document_tools import export_outsourcing_po_pdf

        branch_info = erp_status.get("branch_info") or {}
        block_master = erp_status.get("block_master") or {}

        block_id = params.get("block_id") or block_master.get("block_id") or "BLK-DIRECT"
        po_draft_no = f"PO-DRAFT-{block_id}-{uuid.uuid4().hex[:4].upper()}"
        partner_id = selected_partner.get("partner_id")
        partner_name = selected_partner.get("partner_name") or partner_id
        biz_type = selected_partner.get("business_type") or "OUTSOURCED_B2B"

        delivery_zone = (
            params.get("work_zone")
            or branch_info.get("default_zone")
            or state.get("default_workspace")
            or "ZONE-A-BAY1"
        )
        params["work_zone"] = delivery_zone
        if not params.get("branch_code"):
            params["branch_code"] = (
                branch_info.get("branch_code")
                or state.get("branch_code")
                or "BR-ULSAN-01"
            )

        wire_code = (
            selected_partner.get("hq_item_code")
            or erp_status.get("selected_wire_code")
            or "WIRE-CSF71T"
        )
        req_wire_kg = float(params.get("quantity") or params.get("wire_kg") or 100.0)

        generated_doc = {
            "is_standalone_po": True,
            "work_id": state.get("work_id"),
            "work_order_number": po_draft_no,
            "title": f"[DRAFT] {params['branch_code']} 외주 발주/배차 요청 ({partner_name})",
            "status": "DRAFT",
            "approval_status": "DRAFT",
            "execution_type": "OUTSOURCED_B2B",
            "business_type": biz_type,
            "wire_item_code": wire_code,
            "reserve_wire_kg": req_wire_kg,
            "assigned_vehicle_code": selected_partner.get("assigned_vehicle_code"),
            "reviewed_by": state.get("user_name") or "현장검토대기",
            "parameters": params,
            "erp_resource_status": erp_status,
            "partner_erp_plan": partner_plan,
            "instruction_text": answer,
            "content": answer,
        }

        try:
            po_export_raw = export_outsourcing_po_pdf.invoke({
                "document_data": generated_doc,
                "file_prefix": f"outsourcing_po_{block_id}",
            })
            po_meta = (
                json.loads(po_export_raw)
                if isinstance(po_export_raw, str)
                else (po_export_raw or {})
            )
            generated_doc["files"] = po_meta
        except Exception as po_err:
            print(f"⚠️ [단독 외주 PO PDF 생성 경고]: {po_err}")

    # 6. A2A 자율 협의 결과 요약 배너 자동 부착 (협력사 협의 질의 시)
    a2a_neg = partner_plan.get("a2a_negotiation") or {}
    a2a_prop = a2a_neg.get("proposal") or {}
    if (
        not is_unregistered_item
        and task_type == "scm_partner_qa"
        and a2a_neg.get("status") == "SUCCESS"
        and a2a_prop
    ):
        decision_kor_map = {
            "ACCEPTED": "협의 수락 (즉시 이행 가능)",
            "COUNTER_OFFER": "조건부 역제안 (검토 필요)",
            "REJECTED": "공급 불가",
        }
        dec_label = decision_kor_map.get(
            str(a2a_prop.get("negotiation_decision", "")).upper(),
            str(a2a_prop.get("negotiation_decision", "-")),
        )
        a2a_banner = (
            f"\n\n---\n### 🤝 본사-협력사 A2A 자율 협의 결과 (`{a2a_neg.get('a2a_task_id')}`)\n"
            f"- **협력사 에이전트**: {a2a_neg.get('agent_card_name')} (**{dec_label}**)\n"
            f"- **협의 요약**: {a2a_prop.get('summary')}\n"
            f"- **기밀 보호 경계**: {a2a_prop.get('privacy_boundary_notice')}"
        )
        if "A2A 자율 협의 결과" not in answer:
            answer = answer + a2a_banner

    # 7. 사내 규격 미보유/부적합 시 하단 PDF 출처 배지 제거
    is_doc_irrelevant = (
        str(state.get("doc_grade", "")).lower() == "no"
        or not is_supported_docs
    )

    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "generated_document": generated_doc,
        "erp_commit_result": generated_doc,
        "retrieved_docs": [] if is_doc_irrelevant else docs,
    }


# ==============================================================================
# 2. Self-Healing 트러블슈터 워커 (troubleshooter_agent)
# ==============================================================================
TROUBLESHOOT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 스마트 조선소 현장 공정 및 안전 안내 어시스턴트입니다.\n"
        "현재 요청하신 작업지시서 또는 중량물 취급 작업계획서는 안전·자격·정격하중·야드 운영 기준에 따라 **발행이 보류**되었습니다.\n"
        "현장 작업자와 관리자가 화면을 보자마자 **무엇이 문제이고, 어떻게 해결하면 되는지 한눈에 파악**할 수 있도록 시각적으로 정돈된 마크다운으로 안내하세요.\n\n"
        "[절대 준수 — 표현 및 가독성 규칙]\n"
        "1. **코드·DB 필드명·영문 상태값 노출 절대 금지**:\n"
        "   - `region_name`, `status`, `VALID`, `is_busy`, `false`, `weld_process`, `available_workers`, `YARD_MISMATCH`, `PROCESS_MISMATCH`, `ALREADY_ASSIGNED`, `OVERLOAD`, `CONDITIONAL_PASS`, `BEFORE_WELD`, `IDLE`, `Gatekeeper` 등 영문 변수명·DB 컬럼명·내부 시스템 용어를 **절대 그대로 쓰지 마세요.**\n"
        "   - 반드시 자연스러운 현장 한국어로 풀어서 표기하세요 (예: `상주 야드`, `자격 유효`, `투입 가능`, `인증 공정`, `야드 지역 불일치`, `정격하중 초과`, `조건부 승인`, `용접 전 대기 상태`).\n"
        "2. **한눈에 들어오는 4단 스캔 구조 (같은 말 반복 금지)**:\n"
        "   - **상단 핵심 요약 (`>` 인용구 2줄)**: 보류된 핵심 이유와 즉시 해결 방법을 굵은 글씨로 요약 (절대 발행 가능하다고 오인하게 쓰지 말 것)\n"
        "   - **### 🛑 발행 보류 사유 상세**: 대상 블록 기준과 요청 조건(작업자 또는 양중장비 정격하중)의 차이점만 불릿 2~3줄로 명확히 대비\n"
        "   - **### 👷 대체 투입 가능 자원 추천**:\n"
        "     * 용접 작업지시서 보류 시: [본사 ERP 자격/자원 검증]의 작업자 목록 중 **대상 블록과 동일 야드 권역 상주 + 자격 유효 + 현재 대기 중(미투입) + 인증 공정 일치** 조건을 모두 충족하는 작업자를 **마크다운 표**(`추천 순위 | 성명 (사번) | 소속 / 직책 | 보유 자격 (인증공정)`)로 정리\n"
        "     * 중량물 취급 계획서 과하중/장비 보류 시: [본사 ERP 자격/자원 검증]의 대체 양중장비 목록(`recommended_alternatives`) 중 총 인양하중(안전율 90% 이내)을 감당할 수 있는 크레인·트랜스포터를 **마크다운 표**(`추천 순위 | 장비명 (자산코드) | 정격하중 | 소속 야드 / 상태`)로 정리\n"
        "   - **### ⚠️ 병행해야 할 현장 안전 조치**: 기상/비전 조건에 따른 필수 현장 조치(예: 저온 시 모재 사전 예열, 강풍 시 방풍막 설치 또는 양중 작업 중단) 1~2줄 요약\n"
        "3. **없는 절차 지어내기 금지 & 명확한 후속 선택지 제공**:\n"
        "   - 시스템에 없는 가상 서류 절차(스크린샷 제출, 사진 업로드 등)는 절대 지어내지 마세요.\n"
        "   - 답변 마지막에는 사용자가 후속 선택만으로 바로 이어갈 수 있도록 선택지를 제시하되, 추천 표 1순위 작업자 성명/사번 또는 1순위 대체 장비명이 명확히 드러나도록 작성하세요.",
    ),
    (
        "human",
        "[사용자 요청]: {query}\n"
        "[차단 사유]: {fail_reasons}\n"
        "[확인된 작업 제원]: {params}\n"
        "[누락된 필수 제원]: {missing}\n"
        "[본사 ERP 자격/자원 검증]: {erp_status}\n"
        "[기상 및 안전 판정]: {weather}\n"
        "[비전/로봇 검사 결과]: {vision}\n"
        "[관련 안전 지침(RAG)]: {docs}",
    ),
])


def troubleshooter_node(state: ShipAgentState) -> Dict[str, Any]:
    missing = state.get("missing_params", [])
    erp_status = state.get("erp_resource_status") or {}
    fail_reasons = erp_status.get("interlock_fail_reasons") or ["거버넌스 안전/자격 인터록 조건 미충족"]
    docs = state.get("retrieved_docs", [])
    docs_text = "\n".join(str(d.get("content", ""))[:400] for d in docs[:2])

    try:
        res = (TROUBLESHOOT_PROMPT | qa_llm).invoke({
            "query": state.get("user_query", ""),
            "fail_reasons": ", ".join(fail_reasons),
            "params": json.dumps(state.get("extracted_params") or {}, ensure_ascii=False, cls=DecimalEncoder),
            "missing": missing,
            "erp_status": json.dumps(erp_status, ensure_ascii=False, cls=DecimalEncoder),
            "weather": json.dumps(state.get("weather_data") or {}, ensure_ascii=False, cls=DecimalEncoder),
            "vision": json.dumps(state.get("vision_data") or {}, ensure_ascii=False, cls=DecimalEncoder),
            "docs": docs_text,
        })
        content_text = res.content
    except Exception as e:
        print(f"⚠️ [Team 3 Troubleshooter LLM 호출 오류]: {e}")
        content_text = (
            f"### 🛑 문서 발행 보류 (Safety & Governance Interlock HOLD)\n"
            f"- **차단 사유**: {', '.join(fail_reasons)}\n"
            f"- 위 안전·자격·정격하중·환경 인터록 차단 사유를 해소한 후 다시 요청해 주십시오."
        )

    # 현장 물리적 위험(비전/로봇 안전 위험, 강풍 STOP, 중량물 과하중) 차단일 때만 긴급 경보 메일 발송
    weather_data = state.get("weather_data") or {}
    safety_status = (weather_data.get("safety_evaluation") or {}).get("status", "PASS")
    vision_data = state.get("vision_data") or {}
    robot_exec = vision_data.get("robot_execution") or {}

    is_physical_safety_hazard = (
        safety_status == "STOP"
        or str(vision_data.get("safety_clearance_yn", "Y")).upper() == "N"
        or str(vision_data.get("inspection_status", "")).upper() == "DEFECT"
        or str(robot_exec.get("emergency_stop_yn", "N")).upper() == "Y"
        or any("비전" in r or "기상" in r or "로봇" in r or "정격하중" in r for r in fail_reasons)
    )

    if is_physical_safety_hazard:
        try:
            from modules.agent.tools.gmail_mcp_client import send_safety_interlock_alert_email

            params_hold = state.get("extracted_params") or {}
            send_safety_interlock_alert_email(
                block_id=params_hold.get("block_id") or "미지정",
                branch_name=state.get("branch_code") or params_hold.get("branch_code") or "BR-ULSAN-01",
                work_zone=params_hold.get("work_zone") or state.get("default_workspace") or "ZONE-A-BAY1",
                worker_name=params_hold.get("worker_name") or "미지정",
                block_reasons=fail_reasons,
                sop_guidance_text=content_text,
            )
        except Exception as alert_err:
            print(f"⚠️ [안전 인터록 경보 메일 알림(비차단)]: {alert_err}")
    else:
        print(f"ℹ️ [안전 경보 메일 생략]: 물리적 현장 위험이 아닌 행정/자격 사유({fail_reasons})이므로 채팅창 안내만 수행")

    return {
        "messages": [AIMessage(content=content_text)],
        "final_answer": content_text,
        "generated_document": {
            "title": "작업 보류 및 현장 안전/자원 조치 지시서 (HOLD)",
            "status": "REJECTED",
            "approval_status": "REJECTED",
            "parameters": state.get("extracted_params") or {},
            "weather": state.get("weather_data"),
            "vision": state.get("vision_data"),
            "content": content_text,
            "instruction_text": content_text,
        },
    }


# ==============================================================================
# 3. 공식 용접 작업지시서 작성 및 멀티-ERP 트랜잭션 마감 워커 (work_order_writer)
# ==============================================================================
WORK_ORDER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 스마트 조선소 수석 공정 엔지니어입니다.\n"
        "제공된 [확정 작업 제원], [ERP 사전 검토/커밋 결과], [협력사 B2B 연계 내역], [YOLOv11 비전 계측 결과], [기상 및 안전 판정], [참조 WPS 및 기술 규격서(RAG)]를 "
        "바탕으로 공식 **표준 작업지시서(Work Order & Work Instruction)**와 구조화된 시공 파라미터 필드를 모두 빈칸 없이 작성하십시오.\n\n"
        "[작성 및 데이터 추출 원칙]\n"
        "1. **RAG 문서 기반 적극 추출**: [참조 WPS 및 기술 규격서(RAG)]에 포함된 용접 와이어/플럭스 카탈로그, 선체 검사 지침(Hull Weld Inspection), 안전보건 지침의 수치와 규정을 면밀히 읽어 각 필드를 채우십시오. 특히 `consumable_spec`에는 와이어 이름이 아니라 보관소 온·습도 유지, 이격 보관, 개봉 후 흡습 방지 등 실제 관리 수칙을 적으십시오.\n"
        "2. **공정별 물리/시공 정합성 엄수**:\n"
        "   - 대상 공정이 `SAW`(서브머지드 아크 용접)인 경우 보호가스(`CO2`, `Ar`)를 절대 기재하지 말고 `'입상 플럭스 차폐 (보호가스 미사용)'`으로 기입하며, `FCAW` 카탈로그의 저전류(180~250A) 수치를 SAW에 혼용하지 말고 해당 강종·두께에 맞는 SAW 시공 조건(예: 550~700A, 28~34V)을 기재하십시오.\n"
        "3. **WPS 번호(`wps_code`) 규칙**: 절대 `.pdf` 파일 이름을 `wps_code`에 넣지 마십시오. 문서 내 절차서 코드가 없으면 `WPS-{{공정}}-{{강종}}-STD` 형식으로 표준 식별 코드를 부여하십시오.\n"
        "4. **비전 계측 표기 원칙**: 로봇 전용 픽셀 좌표(`start_point_json`, `end_point_json`)는 제외하고, 비전 탐지 상태와 안전 클리어런스 확보 여부만 반영하십시오.",
    ),
    (
        "human",
        "[사용자 요청]: {query}\n"
        "[확정 작업 제원]: {params}\n"
        "[ERP 사전 검토/커밋 결과]: {erp_commit}\n"
        "[본사 SHIP_ERP 진단 상세]: {erp_status}\n"
        "[협력사 B2B 연계 내역]: {partner_plan}\n"
        "[YOLOv11 비전 계측 결과(VISION_INSPECTIONS)]: {vision}\n"
        "[기상 및 안전 판정]: {weather}\n"
        "[참조 WPS 및 기술 규격서(RAG)]:\n{docs}",
    ),
])


def work_order_writer_node(state: ShipAgentState) -> Dict[str, Any]:
    params = dict(state.get("extracted_params") or {})
    erp_status = dict(state.get("erp_resource_status") or {})
    partner_plan = state.get("partner_erp_plan") or {}
    weather = state.get("weather_data") or {}
    vision = state.get("vision_data") or {}

    selected_partner = (
        (partner_plan.get("selected_partner") or {}) if partner_plan else {}
    )

    hq_code = erp_status.get("hq_company_code")
    if not hq_code:
        hq_meta = get_active_partner_from_db("HQ")
        hq_code = hq_meta.get("company_code") or "COMP-HQ"

    pref_mode = state.get("preferred_execution_mode", "AUTO")
    worker_qual = erp_status.get("worker_qualification") or {}
    hq_worker_missing = worker_qual.get("status") == "NOT_FOUND"
    hq_worker_found_valid = (
        bool(worker_qual.get("worker_id"))
        and worker_qual.get("status") == "VALID"
        and str(worker_qual.get("company_code", "COMP-HQ")).upper() == "COMP-HQ"
    )

    outsourcing_domain = str(state.get("outsourcing_domain") or "").upper()
    stock_shortage_only = (
        not erp_status.get("stock_available", True)
        and hq_worker_found_valid
        and pref_mode != "OUTSOURCED_B2B"
        and outsourcing_domain in ("", "WELDING_MATERIAL")
    )

    # 외주 역무(인력 파견/중량물 배차)만 작업지시서와 외주 PO를 동시 발행하고,
    # 단순 자재 부족은 선행 승인된 PO(PO-MRP-...) 참조만 연결하여 중복 발주 차단
    is_service_outsourced = (
        pref_mode == "OUTSOURCED_B2B"
        or (erp_status.get("require_outsourcing") is True and not stock_shortage_only)
        or hq_worker_missing
    )
    is_outsourced = is_service_outsourced or not erp_status.get("stock_available", True)

    if pref_mode == "IN_HOUSE" or (
        pref_mode == "AUTO"
        and hq_worker_found_valid
        and erp_status.get("stock_available", True)
        and not erp_status.get("require_outsourcing", False)
    ):
        is_outsourced = False
        is_service_outsourced = False

    fallback_wire_code = erp_status.get("selected_wire_code")
    if not fallback_wire_code:
        for inv_code, inv_item in (erp_status.get("inventory") or {}).items():
            if str(inv_item.get("material_category", "")).upper() == "FILLER_WIRE":
                fallback_wire_code = inv_code
                break

    if is_outsourced:
        if (
            hq_worker_found_valid
            and pref_mode != "OUTSOURCED_B2B"
            and selected_partner.get("business_type") == "OUTSOURCED_WORKER"
        ):
            partner_id = (
                erp_status.get("selected_wire_supplier")
                or get_active_partner_from_db("STEEL_SUPPLY").get("company_code")
            )
        else:
            partner_id = selected_partner.get("partner_id") or erp_status.get("selected_wire_supplier")

        if not partner_id:
            domain_map = {
                "HEAVY_TRANSPORT": "LOGISTICS",
                "OUTSOURCED_WORKER": "BLOCK_WELDING",
                "WELDING_MATERIAL": "STEEL_SUPPLY",
            }
            spec_field = domain_map.get(
                state.get("outsourcing_domain") or "WELDING_MATERIAL",
                "STEEL_SUPPLY",
            )
            fallback_comp = get_active_partner_from_db(spec_field)
            partner_id = fallback_comp.get("company_code")

        wire_item_code = (
            fallback_wire_code
            or selected_partner.get("hq_item_code")
            or selected_partner.get("supplier_sku")
        )
    else:
        partner_id = hq_code
        wire_item_code = fallback_wire_code

    lot_no = (
        selected_partner.get("partner_lot_no")
        if is_outsourced and selected_partner.get("partner_lot_no")
        else erp_status.get("assigned_lot_no")
    )

    worker_id = worker_qual.get("worker_id") or selected_partner.get("assigned_worker_id")
    worker_name = (
        params.get("worker_name")
        or worker_qual.get("worker_name")
        or selected_partner.get("worker_name")
    )

    branch_info = erp_status.get("branch_info") or {}
    target_branch = str(
        params.get("branch_code") or branch_info.get("branch_code") or state.get("branch_code") or ""
    ).upper()

    vehicle_code = (
        selected_partner.get("assigned_vehicle_code")
        if is_outsourced and selected_partner.get("assigned_vehicle_code")
        else erp_status.get("assigned_vehicle_code")
    )
    if not vehicle_code:
        for eq in (erp_status.get("vehicles_equipment") or []):
            if (
                str(eq.get("branch_code", "")).upper() == target_branch
                and str(eq.get("operation_status", "")).upper() == "STANDBY"
            ):
                vehicle_code = eq.get("asset_code")
                erp_status["assigned_vehicle_code"] = vehicle_code
                break

    block_code = params.get("block_id")
    exec_type_str = "OUTSOURCED_B2B" if is_service_outsourced else "IN_HOUSE"
    linked_po_no = (
        erp_status.get("linked_po_number")
        or (f"PO-PLAN-{partner_id}" if is_service_outsourced else None)
    )
    bm_meta = erp_status.get("block_master") or {}

    req_wire_kg = calculate_required_wire_kg(
        estimated_length_mm=vision.get("estimated_length_mm"),
        weight_ton=params.get("weight_ton"),
        plan_wire_kg=bm_meta.get("est_wire_kg"),
    )

    delivery_zone = (
        params.get("work_zone")
        or branch_info.get("default_zone")
        or state.get("default_workspace")
    )
    if delivery_zone:
        params["work_zone"] = delivery_zone

    if state.get("hitl_approved") is True:
        erp_commit = commit_erp_work_order.invoke({
            "block_id": block_code,
            "material": params.get("material", ""),
            "allocated_lot_no": lot_no,
            "worker_name": worker_name,
            "worker_id": worker_id,
            "assigned_vehicle_code": vehicle_code,
            "wire_item_code": wire_item_code,
            "is_outsourced": is_outsourced,
            "partner_id": partner_id if is_service_outsourced else hq_code,
            "reserve_wire_kg": req_wire_kg,
            "delivery_yard_zone": delivery_zone,
        })
        erp_commit["execution_type"] = exec_type_str
        if is_outsourced and not erp_commit.get("b2b_po_ack"):
            erp_commit["b2b_po_ack"] = linked_po_no
        approval_status = (
            "APPROVED"
            if erp_commit.get("erp_transaction_status") == "COMMITTED"
            else "DRAFT"
        )
    else:
        wo_num = state.get("work_no") or f"WO-DRAFT-{block_code}-{uuid.uuid4().hex[:4].upper()}"
        erp_commit = {
            "work_order_number": wo_num,
            "execution_type": exec_type_str,
            "partner_id": partner_id if is_service_outsourced else hq_code,
            "allocated_lot_no": lot_no,
            "worker_id": worker_id,
            "worker_name": worker_name,
            "assigned_vehicle_code": vehicle_code,
            "erp_transaction_status": "DRAFT",
            "b2b_po_ack": linked_po_no,
            "reserved_wire_kg": req_wire_kg,
        }
        approval_status = "DRAFT"

    docs = state.get("retrieved_docs", [])
    docs_text = (
        "\n\n".join(
            f"[출처문서: {d.get('metadata', {}).get('document_name', '-')} "
            f"(p.{d.get('metadata', {}).get('page', 1)})]:\n{d.get('content', '')}"
            for d in docs
        )
        or "인출된 RAG 규격 문서 없음"
    )

    def _clean_ref_title(raw_title: Any, doc_code: Any = None) -> str:
        title = str(raw_title or "").strip()
        # 영문, 숫자, 한글, 기본 문장부호 외 손상된 인코딩 문자(아랍어·키릴·고립자모 등) 일괄 제거
        cleaned = re.sub(r"[^A-Za-z0-9가-힣\s\-\._()/&]", " ", title)
        cleaned = re.sub(r"\s*\|\s*|\s*-\s*PDFCOFFEE\.COM", " ", cleaned, flags=re.IGNORECASE)
        if re.search(r"[A-Za-z]{4,}", cleaned):
            cleaned = re.sub(r"(?:^|\s)[가-힣]{1,2}(?=\s|$|[A-Za-z(])", " ", cleaned)
        cleaned = re.sub(r"^[\s\-_|:;.,()0-9]+(?=[A-Za-z가-힣])", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_|")
        valid_chars = re.sub(r"[^A-Za-z0-9가-힣]", "", cleaned)
        if len(valid_chars) >= 4:
            return cleaned
        code_clean = re.sub(r"[^A-Za-z0-9가-힣\-_.]", "", str(doc_code or "")).strip()
        return code_clean if len(code_clean) >= 4 else "선체 용접 및 비파괴검사(NDT) 표준 지침"

    source_refs_list = [
        {
            "document_code": d.get("metadata", {}).get("document_code"),
            "document_title": _clean_ref_title(
                d.get("metadata", {}).get("document_name"),
                d.get("metadata", {}).get("document_code"),
            ),
            "page": d.get("metadata", {}).get("page", 1),
            "similarity": d.get("metadata", {}).get("similarity"),
        }
        for d in docs
    ]

    vision_for_prompt = {
        k: v
        for k, v in vision.items()
        if k
        not in (
            "start_point_json",
            "end_point_json",
            "centerline_waypoints",
            "centerline_waypoints_json",
        )
    }

    proc_upper = str(params.get("process") or "FCAW").upper()
    try:
        structured_writer = llm.with_structured_output(WorkOrderGenerationOutput)
        wo_out: WorkOrderGenerationOutput = (
            WORK_ORDER_PROMPT | structured_writer
        ).invoke({
            "query": state.get("user_query", ""),
            "params": json.dumps(params, ensure_ascii=False, cls=DecimalEncoder),
            "erp_commit": json.dumps(
                erp_commit, ensure_ascii=False, cls=DecimalEncoder
            ),
            "erp_status": json.dumps(
                erp_status, ensure_ascii=False, cls=DecimalEncoder
            ),
            "partner_plan": json.dumps(
                partner_plan if is_outsourced else {}, ensure_ascii=False, cls=DecimalEncoder
            ),
            "vision": json.dumps(
                vision_for_prompt, ensure_ascii=False, cls=DecimalEncoder
            ),
            "weather": json.dumps(weather, ensure_ascii=False, cls=DecimalEncoder),
            "docs": docs_text,
        })
    except Exception as e:
        print(f"⚠️ [Team 3 WorkOrderWriter LLM 호출 경고]: {e}")
        is_saw = proc_upper == "SAW"
        wo_out = WorkOrderGenerationOutput(
            brief_summary=f"{block_code} 블록 {proc_upper} 표준 용접 시공 조건 및 현장 안전 수칙 준수 요망.",
            wps_code=f"WPS-{proc_upper}-{params.get('material') or 'STD'}-001",
            current_amp="550~700" if is_saw else "200~260",
            voltage_v="28~34" if is_saw else "24~30",
            travel_speed_cm_min="30~45" if is_saw else "25~40",
            shielding_gas="입상 플럭스 차폐 (보호가스 미사용)" if is_saw else "CO2 100% (20~25 L/min)",
            preheat_temp_c="50℃ 이상 예열 및 층간온도 250℃ 이하 유지",
            consumable_spec="용접재료 보관소 실내 온도 20℃ 이상, 상대습도 50% 이하 유지 및 지면/벽면 30cm 이상 이격 보관",
            ndt_requirement="전 구간 육안검사(VT) 100% 및 주요 맞대기/필렛 용접부 비파괴검사(UT/RT/MT) 실시",
        )

    safety_eval = weather.get("safety_evaluation") or {}
    safety_status_str = safety_eval.get("status", "PASS")
    safety_kor_map = {
        "PASS": "정상 (작업 가능)",
        "CONDITIONAL_PASS": "조건부 승인 (안전 조치 필수)",
        "STOP": "작업 중지",
    }
    safety_display = safety_kor_map.get(str(safety_status_str).upper(), str(safety_status_str))

    a2a_prop = (partner_plan.get("a2a_negotiation") or {}).get("proposal") or {}
    a2a_summary_clean = str(a2a_prop.get("summary") or "").replace("➔", "→").replace("➜", "→")
    a2a_line = (
        f"\n- **🤝 협력사 A2A 협의 결과**: {a2a_summary_clean}"
        if (is_service_outsourced and a2a_summary_clean)
        else ""
    )
    replan_cnt = int(state.get("replan_count", 0))
    reasoning_note = (state.get("agent_reasoning_summary") or "").strip()
    agent_note_line = (
        f"\n- **🤖 AI 자율 편성/복구 근거"
        f"{f' (자율 재편성 {replan_cnt}회)' if replan_cnt > 0 else ''}**: {reasoning_note}"
        if reasoning_note
        else ""
    )

    answer = (
        f"### 📋 공식 작업지시서 발행 브리핑 (`{erp_commit.get('work_order_number')}`)\n"
        f"- **대상 블록 / 제원**: `{block_code}` ({params.get('material', '-')} {params.get('thickness', '-')}, {params.get('process', '-')})\n"
        f"- **투입 인력 / 자재**: {worker_name} (`{worker_id}`) / `{wire_item_code or '-'}` ({req_wire_kg} kg 선점)\n"
        f"- **현장 안전 및 시공 요약**: **{safety_display}** — {wo_out.brief_summary}"
        f"{a2a_line}"
        f"{agent_note_line}"
    )

    if approval_status == "DRAFT":
        admin_notice = (
            "\n\n> ⚠️ **[관리자 결재 대기]** 본 작업지침서는 초안 상태입니다. "
            "관리자 승인 시 ERP 자원 배정 및 로봇 파라미터 인가가 최종 확정됩니다."
        )
    else:
        admin_notice = (
            "\n\n> ✅ **[원장 커밋 완료]** 공식 작업지시 번호 "
            f"`{erp_commit.get('work_order_number')}`로 ERP 자원 배정 및 자재 선점이 확정되었습니다."
        )

    full_answer = answer + admin_notice

    candidate_wps = (
        params.get("wps_code")
        or (erp_status.get("block_master") or {}).get("wps_code")
        or wo_out.wps_code
    )
    if not candidate_wps or str(candidate_wps).lower().endswith(".pdf"):
        proc_str = params.get("process") or "WELD"
        mat_str = params.get("material") or "STD"
        resolved_wps_code = f"WPS-{proc_str}-{mat_str}-001"
    else:
        resolved_wps_code = candidate_wps

    resolved_shielding_gas = wo_out.shielding_gas
    if str(params.get("process") or "").upper() == "SAW":
        resolved_shielding_gas = "입상 플럭스 차폐 (보호가스 미사용)"

    welding_params_payload = {
        "wps_code": resolved_wps_code,
        "process_code": params.get("process"),
        "material_grade": params.get("material"),
        "thickness_mm": params.get("thickness"),
        "weld_position": params.get("weld_position") or worker_qual.get("cert_code"),
        "current_amp": wo_out.current_amp,
        "voltage_v": wo_out.voltage_v,
        "travel_speed_cm_min": wo_out.travel_speed_cm_min,
        "shielding_gas": resolved_shielding_gas,
        "preheat_temp_c": wo_out.preheat_temp_c,
        "consumable_spec": wo_out.consumable_spec,
        "ndt_requirement": wo_out.ndt_requirement,
        "safety_clearance_yn": vision.get("safety_clearance_yn"),
    }

    generated_doc = {
        "work_id": state.get("work_id"),
        "plan_id": bm_meta.get("plan_id"),
        "work_order_number": erp_commit.get("work_order_number"),
        "doc_type": "WELDING_WORK_ORDER",
        "wps_code": resolved_wps_code,
        "revision_no": "Rev.1",
        "title": f"[{approval_status}] {params.get('block_id', '')} {params.get('material', '')} {params.get('thickness', '')} 작업지시서".strip(),
        "status": approval_status,
        "approval_status": approval_status,
        "reviewed_by": state.get("user_name") or "현장검토대기",
        "execution_type": exec_type_str,
        "allocated_lot_no": lot_no,
        "assigned_vehicle_code": vehicle_code,
        "wire_item_code": wire_item_code,
        "reserve_wire_kg": req_wire_kg,
        "erp_commit": erp_commit,
        "partner_erp_plan": partner_plan if is_outsourced else {},
        "parameters": params,
        "welding_params_json": welding_params_payload,
        "source_refs_json": source_refs_list,
        "external_env_json": weather,
        "vision_inspection": vision,
        "weather": weather,
        "content": full_answer,
        "instruction_text": full_answer,
        "erp_resource_status": erp_status,
    }

    try:
        export_raw = export_work_order_file.invoke({
            "document_data": generated_doc,
            "file_prefix": f"weld_order_{block_code}",
        })
        export_meta = (
            json.loads(export_raw) if isinstance(export_raw, str) else {}
        )

        if is_service_outsourced:
            try:
                from modules.agent.tools.document_tools import export_outsourcing_po_pdf

                po_export_raw = export_outsourcing_po_pdf.invoke({
                    "document_data": generated_doc,
                    "file_prefix": f"po_order_{block_code}",
                })
                po_meta = (
                    json.loads(po_export_raw)
                    if isinstance(po_export_raw, str)
                    else (po_export_raw or {})
                )
                if po_meta.get("pdf_path"):
                    export_meta["po_pdf_path"] = po_meta["pdf_path"]
                if po_meta.get("download_pdf_url"):
                    export_meta["download_po_pdf_url"] = po_meta["download_pdf_url"]
            except Exception as po_err:
                print(f"⚠️ [외주 발주 주문서(PO) PDF 생성 경고]: {po_err}")

        generated_doc["files"] = export_meta

        audit_status = safety_eval.get("status", "PASS")
        full_query_log = (
            f"[제원: {json.dumps(params, ensure_ascii=False, cls=DecimalEncoder)}] "
            f"| 발화: {state.get('user_query', '')}"
        )

        log_inspection_history.invoke({
            "session_id": erp_commit.get("work_order_number", "DRAFT_SESSION"),
            "query": full_query_log,
            "status": audit_status,
        })
    except Exception as e:
        print(f"⚠️ [Document Tools 오류]: {e}")

    return {
        "messages": [AIMessage(content=full_answer)],
        "final_answer": full_answer,
        "generated_document": generated_doc,
        "erp_commit_result": erp_commit,
        "audit_saved": True,
    }


# ==============================================================================
# 4. 중량물 취급 작업계획서 작성 워커 (heavy_lifting_plan_writer)
# ==============================================================================
HEAVY_LIFTING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 조선소 중량물 취급 및 안전공학 수석 엔지니어입니다.\n"
        "산업안전보건기준에 관한 규칙 제38조에 의거해 제공된 [블록 및 작업 제원], [정격하중 안전율 검토], "
        "[협력사 A2A 배차 협의], [실시간 기상 및 안전 판정], [참조 사내 안전 지침(RAG)]을 바탕으로 "
        "현장 반장과 크레인·트랜스포터 운전원이 즉시 적용할 수 있는 **중량물 취급 작업계획서** 항목을 구체적으로 작성하십시오.\n"
        "- 영문 DB 컬럼명이나 상태 코드를 쓰지 말고 현장 한국어 표준 용어로 작성하십시오.\n"
        "- 줄걸이 각도 60도 이내, 지상 30cm 시험 인양(무게중심 확인), 하부 출입통제 및 전담 신호수 배치를 반드시 반영하십시오.",
    ),
    (
        "human",
        "[사용자 요청]: {query}\n"
        "[블록 및 작업 제원]: {params}\n"
        "[정격하중 안전율 검토]: {cap_eval}\n"
        "[협력사 A2A 배차 협의]: {a2a_plan}\n"
        "[실시간 기상 및 안전 판정]: {weather}\n"
        "[참조 사내 안전 지침(RAG)]:\n{docs}",
    ),
])


def heavy_lifting_plan_writer_node(state: ShipAgentState) -> Dict[str, Any]:
    params = dict(state.get("extracted_params") or {})
    erp_status = dict(state.get("erp_resource_status") or {})
    cap_eval = dict(erp_status.get("lifting_capacity_eval") or {})
    partner_plan = state.get("partner_erp_plan") or {}
    a2a_neg = partner_plan.get("a2a_negotiation") or {}
    a2a_prop = a2a_neg.get("proposal") or {}
    weather = state.get("weather_data") or {}
    vision = state.get("vision_data") or {}
    docs = state.get("retrieved_docs") or []

    block_code = params.get("block_id") or "BLK-HEAVY"
    is_outsourced = bool(a2a_prop.get("assigned_vehicle_code")) or bool(cap_eval.get("need_outsourcing"))
    exec_type_str = "OUTSOURCED_B2B" if is_outsourced else "IN_HOUSE"

    selected_eq = cap_eval.get("selected_equipment") or {}
    vehicle_code = (
        a2a_prop.get("assigned_vehicle_code")
        or selected_eq.get("asset_code")
        or erp_status.get("assigned_vehicle_code")
        or "-"
    )
    partner_id = (
        a2a_prop.get("partner_id")
        or (partner_plan.get("selected_partner") or {}).get("partner_id")
        or erp_status.get("hq_company_code")
        or "COMP-HQ"
    )
    delivery_zone = (
        params.get("work_zone")
        or (erp_status.get("branch_info") or {}).get("default_zone")
        or state.get("default_workspace")
        or "ZONE-A-BAY1"
    )

    if state.get("hitl_approved") is True:
        erp_commit = commit_erp_work_order.invoke({
            "block_id": block_code,
            "material": params.get("material", "AH36"),
            "allocated_lot_no": "HEAVY-LIFTING",
            "worker_name": state.get("user_name") or erp_status.get("foreman_name") or "작업지휘자",
            "worker_id": selected_eq.get("driver_worker_id"),
            "assigned_vehicle_code": None if vehicle_code == "-" else vehicle_code,
            "wire_item_code": None,
            "is_outsourced": is_outsourced,
            "partner_id": partner_id,
            "reserve_wire_kg": 0.0,
            "delivery_yard_zone": delivery_zone,
        })
        erp_commit["execution_type"] = exec_type_str
        approval_status = (
            "APPROVED"
            if erp_commit.get("erp_transaction_status") == "COMMITTED"
            else "DRAFT"
        )
        plan_no = erp_commit.get("work_order_number") or f"HLP-{block_code}-{uuid.uuid4().hex[:4].upper()}"
    else:
        plan_no = f"HLP-DRAFT-{block_code}-{uuid.uuid4().hex[:4].upper()}"
        erp_commit = {
            "work_order_number": plan_no,
            "erp_transaction_status": "DRAFT",
            "execution_type": exec_type_str,
            "partner_id": partner_id,
            "assigned_vehicle_code": vehicle_code,
        }
        approval_status = "DRAFT"
    max_load_ton = a2a_prop.get("max_load_ton") or cap_eval.get("max_load_ton") or "-"
    load_ratio_pct = a2a_prop.get("load_ratio_pct") or cap_eval.get("load_ratio_pct") or "-"

    docs_text = (
        "\n\n".join(str(d.get("content", ""))[:500] for d in docs[:3])
        or "사내 중량물 취급 안전 지침 기본 적용"
    )
    source_refs_list = [
        {
            "document_code": d.get("metadata", {}).get("document_code"),
            "document_title": d.get("metadata", {}).get("document_name"),
            "page": d.get("metadata", {}).get("page", 1),
            "similarity": d.get("metadata", {}).get("similarity"),
        }
        for d in docs
    ]

    try:
        structured_hlp = llm.with_structured_output(HeavyLiftingPlanOutput)
        hlp_out: HeavyLiftingPlanOutput = (HEAVY_LIFTING_PROMPT | structured_hlp).invoke({
            "query": state.get("user_query", ""),
            "params": json.dumps(params, ensure_ascii=False, cls=DecimalEncoder),
            "cap_eval": json.dumps(cap_eval, ensure_ascii=False, cls=DecimalEncoder),
            "a2a_plan": json.dumps(a2a_prop, ensure_ascii=False, cls=DecimalEncoder),
            "weather": json.dumps(weather, ensure_ascii=False, cls=DecimalEncoder),
            "docs": docs_text,
        })
    except Exception as e:
        print(f"⚠️ [HeavyLiftingPlan LLM 경고]: {e}")
        hlp_out = HeavyLiftingPlanOutput(
            brief_summary=(
                f"{block_code}({params.get('weight_ton', '-')}톤) 블록을 {vehicle_code}"
                f"(정격 {max_load_ton}톤, 하중률 {load_ratio_pct}%)으로 안전 인양·이송하는 계획서입니다."
            ),
            travel_route=f"{params.get('work_zone', '작업장')} ➔ {params.get('target_destination') or '지정 탑재 구역'} (경로 장애물 사전 제거)",
            supervisor_name=erp_status.get("foreman_name") or "현장반장(작업지휘자)",
            signalman_assignment="전담 신호수 1인 배치 (표준 수신호 및 전용 무전기 교신)",
            rigging_spec="선급 인증 와이어로프(안전율 5 이상) 4줄 걸이 및 정격 샤클 체결, 리프팅 러그 육안/MT 균열 확인",
            sling_angle_and_cog="줄걸이 체결 각도 60° 이내 준수, 지상 30cm 시험 인양 후 편심·브레이크 정상 여부 확인",
            fall_and_tip_prevention="인양 반경 하부 안전펜스 설치 및 출입 전면 통제, 유도 로프(Tag Line) 2개소 체결",
            emergency_procedure="풍속 10m/s 이상 돌풍 또는 이상 진동 감지 시 즉시 저속 하강·비상정지 및 작업자 대피",
            safety_checklist=[
                "줄걸이 와이어로프 소선 단선·킹크 및 샤클 핀 체결 상태 점검",
                "블록 무게중심 확인을 위한 지상 30cm 시험 인양 및 정지 테스트",
                "인양 경로 하부 작업자 대피 및 출입금지 구역 설정 완료",
                "실시간 풍속 10m/s 미만 확인 및 전담 신호수·운전원 무전 감도 점검",
            ],
        )

    safety_eval = weather.get("safety_evaluation") or {}
    safety_status_str = safety_eval.get("status", "PASS")
    safety_kor_map = {
        "PASS": "정상 (인양 가능)",
        "CONDITIONAL_PASS": "조건부 승인 (안전 조치 필수)",
        "STOP": "작업 중지",
    }
    safety_display = safety_kor_map.get(str(safety_status_str).upper(), str(safety_status_str))

    a2a_line = (
        f"\n- **🤝 외주 배차 A2A 협의**: {a2a_prop.get('summary')}"
        if a2a_prop.get("summary")
        else ""
    )

    notice_box = (
        "\n\n> ⚠️ **[관리자 결재 대기]** 본 중량물 취급 작업계획서는 초안 상태입니다. "
        "관리자 최종 승인 시 양중장비 배정(및 외주 배차 PO)이 확정됩니다."
        if approval_status == "DRAFT"
        else f"\n\n> ✅ **[승인 완료]** 중량물 취급 작업계획서(`{plan_no}`)가 최종 승인되었습니다."
    )

    replan_cnt = int(state.get("replan_count", 0))
    reasoning_note = (state.get("agent_reasoning_summary") or "").strip()
    agent_note_line = (
        f"\n- **🤖 AI 자율 편성/복구 근거"
        f"{f' (자율 재편성 {replan_cnt}회)' if replan_cnt > 0 else ''}**: {reasoning_note}"
        if reasoning_note
        else ""
    )

    answer = (
        f"### 🏗️ 중량물 취급 작업계획서 발행 브리핑 (`{plan_no}`)\n"
        f"- **대상 블록 / 총 인양하중**: `{block_code}` ({params.get('weight_ton', '-')} 톤 / 줄걸이 포함 총 {cap_eval.get('total_lifting_ton', '-')} 톤)\n"
        f"- **투입 양중장비 / 하중률**: `{vehicle_code}` (정격 {max_load_ton} 톤 / **하중률 {load_ratio_pct}%**)\n"
        f"- **기상 및 안전 요약**: **{safety_display}** — {hlp_out.brief_summary}"
        f"{a2a_line}"
        f"{agent_note_line}"
        f"{notice_box}"
    )

    hq_foreman = erp_status.get("foreman_name") or "현장반장(작업지휘자)"
    lifting_dump = hlp_out.model_dump()
    if not lifting_dump.get("supervisor_name") or "외주" in str(lifting_dump.get("supervisor_name")):
        lifting_dump["supervisor_name"] = hq_foreman

    generated_doc = {
        "work_id": state.get("work_id"),
        "plan_id": (erp_status.get("block_master") or {}).get("plan_id"),
        "work_order_number": plan_no,
        "doc_type": "HEAVY_LIFTING_PLAN",
        "title": f"[{approval_status}] {block_code} ({params.get('weight_ton', '-')}T) 중량물 취급 작업계획서",
        "status": approval_status,
        "approval_status": approval_status,
        "reviewed_by": state.get("user_name") or erp_status.get("foreman_name") or "현장검토대기",
        "execution_type": exec_type_str,
        "assigned_vehicle_code": vehicle_code,
        "parameters": params,
        "lifting_capacity_eval": cap_eval,
        "lifting_plan_json": lifting_dump,
        "source_refs_json": source_refs_list,
        "partner_erp_plan": partner_plan if is_outsourced else {},
        "erp_resource_status": erp_status,
        "weather": weather,
        "external_env_json": weather,
        "vision_inspection": vision,
        "content": answer,
        "instruction_text": answer,
    }

    try:
        export_raw = export_heavy_lifting_plan_pdf.invoke({
            "document_data": generated_doc,
            "file_prefix": f"heavy_lifting_plan_{block_code}",
        })
        generated_doc["files"] = json.loads(export_raw) if isinstance(export_raw, str) else {}
        log_inspection_history.invoke({
            "session_id": plan_no,
            "query": f"[중량물계획서: {block_code} / {vehicle_code}] {state.get('user_query', '')}",
            "status": safety_status_str,
        })
    except Exception as e:
        print(f"⚠️ [Heavy Lifting PDF 생성 오류]: {e}")

    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "generated_document": generated_doc,
        "erp_commit_result": erp_commit,
        "audit_saved": True,
    }


# ==============================================================================
# 5. Team 3 SubGraph 조립
# ==============================================================================
builder = StateGraph(ShipAgentState)

builder.add_node("governance_gatekeeper", governance_gatekeeper_node)
builder.add_node("domain_qa_writer", domain_qa_writer_node)
builder.add_node("troubleshooter_agent", troubleshooter_node)
builder.add_node("work_order_writer", work_order_writer_node)
builder.add_node("heavy_lifting_plan_writer", heavy_lifting_plan_writer_node)

builder.add_edge(START, "governance_gatekeeper")
builder.add_conditional_edges(
    "governance_gatekeeper",
    route_from_gatekeeper,
    {
        "domain_qa_writer": "domain_qa_writer",
        "troubleshooter_agent": "troubleshooter_agent",
        "work_order_writer": "work_order_writer",
        "heavy_lifting_plan_writer": "heavy_lifting_plan_writer",
        "__end__": END,
    },
)
builder.add_edge("domain_qa_writer", END)
builder.add_edge("troubleshooter_agent", END)
builder.add_edge("work_order_writer", END)
builder.add_edge("heavy_lifting_plan_writer", END)

team3_subgraph = builder.compile()