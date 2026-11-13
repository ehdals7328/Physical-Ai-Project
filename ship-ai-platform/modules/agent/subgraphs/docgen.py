"""
modules/agent/subgraphs/docgen.py
Team 3: 거버넌스 게이트키퍼(Supervisor) 기반 작업지시서(WORK_INSTRUCTIONS) 생성,
자율 트러블슈팅, 인사/ERP 실시간 브리핑 & [Logical Bridge: Work Authorization] 트랜잭션 마감 서브그래프
"""

import json
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from modules.agent.state import ShipAgentState
from modules.agent.tools.document_tools import (
    export_work_order_file,
    log_inspection_history,
)
from modules.agent.tools.erp_tools import (
    calculate_required_wire_kg,
    commit_erp_work_order,
    get_active_partner_from_db,
    list_erp_workers,
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)


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
        description="참조 문서나 작업 제원에 명시된 절차서/표준 코드 ('.pdf' 파일명 입력 금지)"
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


# ==============================================================================
# 0. Team 3 거버넌스 게이트키퍼 노드 (Supervisor / Safety & Process Interlock)
# ==============================================================================
def governance_gatekeeper_node(state: ShipAgentState) -> Dict[str, Any]:
    task_type = state.get("task_type", "knowledge_qa")

    if task_type != "doc_generation":
        print(
            f"\n👉 [Team 3 Gatekeeper]: domain_qa_writer "
            f"(사유: 일반 기술/ERP/인사 질의, task_type='{task_type}')\n"
        )
        return {"next": "domain_qa_writer"}

    params = state.get("extracted_params") or {}
    missing = state.get("missing_params", [])
    weather = state.get("weather_data") or {}
    safety_eval = weather.get("safety_evaluation") or {}
    safety_status = safety_eval.get("status", "PASS")
    erp_status = dict(state.get("erp_resource_status") or {})
    vision = state.get("vision_data") or {}

    # 1. 작업자 선급 자격 유효성 및 블록 요구 공정(WPS) 상호 일치 검증
    worker_qual = dict(erp_status.get("worker_qualification") or {})
    partner_plan = state.get("partner_erp_plan") or {}
    selected_partner = partner_plan.get("selected_partner") or {}
    ext_worker_id = (
        selected_partner.get("assigned_worker_id")
        or selected_partner.get("ext_worker_id")
        or selected_partner.get("worker_id")
    )

    # 본사 원장에 없으나(NOT_FOUND) 외주 협력사 B2B 원장에서 적격 작업자가 확인된 경우 자격 승계
    if worker_qual.get("status") == "NOT_FOUND" and ext_worker_id:
        ext_valid = bool(selected_partner.get("can_accept_order", True))
        worker_qual = {
            "worker_id": ext_worker_id,
            "worker_name": selected_partner.get("worker_name"),
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
        erp_status["worker_qualification"] = worker_qual
        erp_status["worker_qualified"] = ext_valid

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

    is_worker_valid = (
        erp_status.get("worker_qualified", True)
        and worker_qual.get("is_valid", False)
        and is_proc_matched
    )
    if not worker_qual or not worker_qual.get("worker_id"):
        is_worker_valid = False

    # 2. 비전 결함, 아크 발생(용접 중 Lock), 로봇 비상정지(E-STOP) 인터록 검사
    robot_exec = vision.get("robot_execution") or {}
    has_vision_hazard = (
        str(vision.get("inspection_status", "")).upper() == "DEFECT"
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

    weather_status = weather.get("status")

    is_blocked = (
        bool(missing)
        or safety_status == "STOP"
        or not is_worker_valid
        or has_vision_hazard
        or weather_status in ("NEED_LOCATION", "NOT_FOUND", "PARSE_ERROR", "ERROR")
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
        f"   6. 비전/로봇 위험 감지 여부 : {has_vision_hazard} "
        f"(안전거리={vision.get('safety_clearance_yn', 'Y')})"
    )

    if is_blocked:
        fail_reasons = []
        if missing:
            fail_reasons.append(f"제원누락({missing})")
        if safety_status == "STOP":
            fail_reasons.append("기상STOP(작업중지)")
        if not is_proc_matched:
            fail_reasons.append(
                f"용접공정불일치(블록요구:{req_process} != 작업자인증:{worker_process})"
            )
        elif not is_worker_valid:
            if not worker_qual or not worker_qual.get("worker_id"):
                fail_reasons.append("투입작업자미배정(작업자 성명 미확인)")
            else:
                fail_reasons.append(
                    f"작업자선급자격결격({worker_qual.get('worker_name')}: {worker_qual.get('status')})"
                )
        if has_vision_hazard:
            fail_reasons.append("비전결함/용접중Lock/로봇비상정지")
        if weather_status in ("NEED_LOCATION", "NOT_FOUND", "PARSE_ERROR", "ERROR"):
            fail_reasons.append(f"기상조회실패({weather_status})")

        erp_status["interlock_fail_reasons"] = fail_reasons
        if not is_worker_valid:
            erp_status["worker_qualified"] = False

        chosen_node = "troubleshooter_agent"
        print(f"👉 [최종 결정]: {chosen_node} (차단 사유: {', '.join(fail_reasons)})")
        print("=" * 65 + "\n")
        return {
            "next": chosen_node,
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
) -> Literal["troubleshooter_agent", "work_order_writer", "domain_qa_writer"]:
    nxt = state.get("next")
    if nxt in ("troubleshooter_agent", "work_order_writer", "domain_qa_writer"):
        return nxt
    return "domain_qa_writer"


# ==============================================================================
# 1. 도메인 QA, ERP 재고 및 인사 DB 실시간 브리핑 워커 (domain_qa_writer)
# ==============================================================================
QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 스마트 조선소 엔지니어링 & ERP 기술 자문 AI입니다.\n"
        "제공된 [사내 규격 문서(RAG)], [본사 SHIP_ERP 자원 현황], [협력사 B2B 조회 결과], [실시간 기상 정보]를 바탕으로 "
        "사용자의 질문에 명확하고 체계적인 마크다운으로 답변하세요.\n\n"
        "[안내 원칙]\n"
        "1. 사내 ERP 재고(`ERP_BOM_INVENTORY`), 대기 차량(`ERP_VEHICLES_EQUIPMENT`), 작업자 자격증(`ERP_WORKER_CERT`), 협력사(`ERP_COMPANIES`) 문의인 경우 제공된 DB 조회 데이터를 왜곡 없이 그대로 인용하세요.\n"
        "2. 사용자가 외주 발주나 배차 확정을 요청한 경우:\n"
        "   - 만약 대상 장비/차량의 `operation_status`가 이미 `'IN_USE'`(사용/배차 중)라면, '이미 배차가 완료되어 현재 운행(IN_USE) 중이므로 중복 배차할 수 없다'고 안내하세요.\n"
        "   - 대상 장비/차량이 `'STANDBY'`(대기 중)라면, 절대 '배차가 최종 확정 완료되었다'고 단정 짓지 말고 **'외주 배차/발주 요청안(DRAFT)이 생성되었으며, 하단 관리자 최종 승인(HITL) 클릭 시 원장에 확정 커밋됩니다'**라고 안내하세요.",
    ),
    (
        "human",
        "[질의 유형]: {task_type}\n"
        "[사용자 질문]: {query}\n\n"
        "[본사 SHIP_ERP 자원 현황]:\n{erp_status}\n\n"
        "[협력사 B2B 조회 결과]:\n{partner_plan}\n\n"
        "[인출된 사내 규격 문서(RAG)]:\n{docs}\n\n"
        "[실시간 기상 정보]:\n{weather}",
    ),
])


def domain_qa_writer_node(state: ShipAgentState) -> Dict[str, Any]:
    task_type = state.get("task_type")

    if task_type == "hr_worker_qa":
        db_res = list_erp_workers.invoke({"branch_code": None})
        workers = db_res.get("workers", [])

        table_md = "## 👷 사내외 인사 원장 (`ERP_WORKER_CERT`) 실시간 작업자 현황\n\n"
        table_md += "| 작업자명 | 사번 (`WORKER_ID`) | 고용형태 | 소속업체 | 상주야드 | 직책 (`ROLE`) | 선급 자격 규격 | 인증공정 | 자격 만료일 | 상태 |\n"
        table_md += "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"

        for w in workers:
            status_badge = "✅ 투입 가능" if w["is_valid"] else "❌ 투입 불가"
            table_md += (
                f"| {w.get('worker_name', '-')} | {w.get('worker_id', '-')} | {w.get('employment_type', '-')} "
                f"| {w.get('company_code', '-')} | {w.get('branch_code', '-')} | {w.get('role', '-')} "
                f"| {w.get('cert_code', '-')} | {w.get('weld_process', '-')} | {w.get('cert_expiry_date', '-')} | {status_badge} |\n"
            )

        table_md += f"\n> 💡 *현재 Oracle `ERP_WORKER_CERT`에 등록된 작업자는 총 {len(workers)}명입니다.*"

        return {
            "final_answer": table_md,
            "generated_document": None,
        }

    docs = state.get("retrieved_docs", [])
    docs_text = (
        "\n\n".join(
            f"- [{d.get('metadata', {}).get('document_name', '-')} p.{d.get('metadata', {}).get('page', 1)}]: {d.get('content', '')}"
            for d in docs
        )
        or "해당 없음"
    )
    include_weather = (
        state.get("need_weather", False)
        and task_type not in ("scm_partner_qa", "erp_resource_qa", "hr_worker_qa")
    )
    weather_context_str = (
        json.dumps(
            state.get("weather_data") or {},
            ensure_ascii=False,
            indent=2,
            cls=DecimalEncoder,
        )
        if include_weather and state.get("weather_data")
        else "해당 없음 (기상 무관 질의)"
    )

    res = (QA_PROMPT | llm).invoke({
        "task_type": task_type,
        "query": state.get("user_query", ""),
        "erp_status": json.dumps(
            state.get("erp_resource_status") or {},
            ensure_ascii=False,
            indent=2,
            cls=DecimalEncoder,
        ),
        "partner_plan": json.dumps(
            state.get("partner_erp_plan") or {},
            ensure_ascii=False,
            indent=2,
            cls=DecimalEncoder,
        ),
        "docs": docs_text,
        "weather": weather_context_str,
    })

    return {
        "final_answer": res.content,
        "generated_document": None,
    }


# ==============================================================================
# 2. Self-Healing 트러블슈터 워커 (troubleshooter_agent)
# ==============================================================================
TROUBLESHOOT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 조선소 안전 거버넌스 및 공정 트러블슈팅 에이전트입니다.\n"
        "거버넌스 게이트키퍼에 의해 작업지시서 발행이 **강제 차단(HOLD)**되었습니다.\n"
        "제공된 [게이트키퍼 인터록 차단 사유]를 최우선으로 명확히 설명하고 현장 복구 대안(Self-Healing SOP)을 제시하세요.\n"
        "⚠️ 절대 '보류해야 하는 사유가 없다'거나 '작업지시서를 발행할 수 있다'고 답변하지 마십시오.\n\n"
        "[처방 원칙]\n"
        "1. [누락된 필수 제원]이 있는 경우: 해당 항목만 친절하게 요청하세요.\n"
        "2. 투입 작업자가 미배정되었거나 인사 원장에서 확인되지 않은 경우: 투입할 적격 작업자 성명(예: 최선임, 김철수 등)을 명시하여 재요청하도록 안내하세요.\n"
        "3. 작업자 자격증이 만료(`EXPIRED`), 휴직(`ON_LEAVE`), 또는 블록 요구 공정과 작업자 인증 공정이 불일치(`PROCESS_MISMATCH`)하는 경우: 선급 규정에 따라 작업지시서 발행 차단(인터록)을 선언하고 해당 공정 자격을 보유한 적격 작업자로 교체 배정하거나 전문 외주 인력을 투입하도록 권고하세요.\n"
        "4. [기상 안전 판정]이 'STOP'인 경우: 작업 중지 명령과 함께 사내 SOP에 따른 크레인 붐대 고정 및 방풍/보양 조치를 안내하세요.\n"
        "5. [비전 검사 결과]에서 불량(`DEFECT`), 용접 중(`IN_WELDING`), 또는 로봇 비상정지(`EMERGENCY_STOP_YN='Y'`)가 감지된 경우: 즉시 원인 해소 및 재검사를 안내하세요.",
    ),
    (
        "human",
        "[사용자 요청]: {query}\n"
        "[게이트키퍼 인터록 차단 사유]: {fail_reasons}\n"
        "[확정 작업 제원(블록 스펙)]: {params}\n"
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

    res = (TROUBLESHOOT_PROMPT | llm).invoke({
        "query": state.get("user_query", ""),
        "fail_reasons": ", ".join(fail_reasons),
        "params": json.dumps(state.get("extracted_params") or {}, ensure_ascii=False, cls=DecimalEncoder),
        "missing": missing,
        "erp_status": json.dumps(erp_status, ensure_ascii=False, cls=DecimalEncoder),
        "weather": json.dumps(state.get("weather_data") or {}, ensure_ascii=False, cls=DecimalEncoder),
        "vision": json.dumps(state.get("vision_data") or {}, ensure_ascii=False, cls=DecimalEncoder),
        "docs": docs_text,
    })

    return {
        "final_answer": res.content,
        "generated_document": {
            "title": "작업 보류 및 현장 안전/자원 조치 지시서 (HOLD)",
            "status": "REJECTED",
            "approval_status": "REJECTED",
            "parameters": state.get("extracted_params") or {},
            "weather": state.get("weather_data"),
            "vision": state.get("vision_data"),
            "content": res.content,
            "instruction_text": res.content,
        },
    }


# ==============================================================================
# 3. 공식 작업지시서 작성 및 멀티-ERP 트랜잭션 마감 워커 (work_order_writer)
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

    # 1. 본사(HQ) 코드 DB 동적 조회
    hq_code = erp_status.get("hq_company_code")
    if not hq_code:
        hq_meta = get_active_partner_from_db("HQ")
        hq_code = hq_meta.get("company_code")

    # 2. 외주 수행 여부 종합 판정
    pref_mode = state.get("preferred_execution_mode", "AUTO")
    is_outsourced = (
        pref_mode == "OUTSOURCED_B2B"
        or bool(selected_partner.get("partner_id"))
        or (
            state.get("is_outsourcing_required") is True
            and not erp_status.get("stock_available", True)
        )
    )

    # 3. 협력사 코드 및 품목 코드 동적 매핑
    if is_outsourced:
        partner_id = selected_partner.get("partner_id") or erp_status.get(
            "selected_wire_supplier"
        )
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
            selected_partner.get("hq_item_code")
            or erp_status.get("selected_wire_code")
            or selected_partner.get("supplier_sku")
        )
    else:
        partner_id = hq_code
        wire_item_code = erp_status.get("selected_wire_code")

    # 4. 자재 Lot, 작업자, 야드별 대기 장비, 와이어 소요량 동적 산출
    lot_no = (
        selected_partner.get("partner_lot_no")
        if is_outsourced and selected_partner.get("partner_lot_no")
        else erp_status.get("assigned_lot_no")
    )

    worker_qual = erp_status.get("worker_qualification") or {}
    worker_id = worker_qual.get("worker_id") or selected_partner.get(
        "assigned_worker_id"
    )
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
    exec_type_str = "OUTSOURCED_B2B" if is_outsourced else "IN_HOUSE"

    req_wire_kg = calculate_required_wire_kg(
        estimated_length_mm=vision.get("estimated_length_mm"),
        weight_ton=params.get("weight_ton"),
    )

    # 블록/지부 기본 작업 구역 우선 적용 (관리자 기본 구역과 야드가 다른 경우 오매핑 방지)
    delivery_zone = (
        params.get("work_zone")
        or branch_info.get("default_zone")
        or state.get("default_workspace")
    )
    if delivery_zone:
        params["work_zone"] = delivery_zone

    # 5. ERP 원장 커밋 (반드시 관리자 HITL 최종 승인 시에만 실행)
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
            "partner_id": partner_id,
            "reserve_wire_kg": req_wire_kg,
            "delivery_yard_zone": delivery_zone,
        })
        erp_commit["execution_type"] = exec_type_str
        approval_status = "APPROVED"
    else:
        wo_num = state.get("work_no") or f"WO-DRAFT-{block_code}"
        erp_commit = {
            "work_order_number": wo_num,
            "execution_type": exec_type_str,
            "partner_id": partner_id,
            "allocated_lot_no": lot_no,
            "worker_id": worker_id,
            "worker_name": worker_name,
            "assigned_vehicle_code": vehicle_code,
            "erp_transaction_status": "DRAFT",
            "b2b_po_ack": f"PO-PLAN-{partner_id}" if is_outsourced else None,
            "reserved_wire_kg": req_wire_kg,
        }
        approval_status = "DRAFT"

    # 6. RAG 규격 및 프롬프트 인출
    docs = state.get("retrieved_docs", [])
    docs_text = (
        "\n\n".join(
            f"[출처문서: {d.get('metadata', {}).get('document_name', '-')} "
            f"(p.{d.get('metadata', {}).get('page', 1)})]:\n{d.get('content', '')}"
            for d in docs
        )
        or "인출된 RAG 규격 문서 없음"
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

    vision_for_prompt = {
        k: v
        for k, v in vision.items()
        if k
        not in (
            "start_point_json",
            "end_point_json",
            "centerline_waypoints_json",
        )
    }

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
            partner_plan, ensure_ascii=False, cls=DecimalEncoder
        ),
        "vision": json.dumps(
            vision_for_prompt, ensure_ascii=False, cls=DecimalEncoder
        ),
        "weather": json.dumps(weather, ensure_ascii=False, cls=DecimalEncoder),
        "docs": docs_text,
    })

    safety_eval = weather.get("safety_evaluation") or {}
    safety_status_str = safety_eval.get("status", "PASS")

    answer = (
        f"### 📋 공식 작업지시서 발행 브리핑 (`{erp_commit.get('work_order_number')}`)\n"
        f"- **대상 블록 / 제원**: `{block_code}` ({params.get('material', '-')} {params.get('thickness', '-')}, {params.get('process', '-')})\n"
        f"- **투입 인력 / 자재**: {worker_name} (`{worker_id}`) / `{wire_item_code}` ({req_wire_kg} kg 선점)\n"
        f"- **현장 안전 및 시공 요약**: `{safety_status_str}` — {wo_out.brief_summary}"
    )

    if approval_status == "DRAFT":
        admin_notice = (
            "\n\n> ⚠️ **[관리자 결재 대기 (`APPROVAL_STATUS: DRAFT`)]** 본"
            " 작업지침서는 초안 상태입니다. 관리자(MANAGER/ADMIN) 승인 시"
            " `MES_WORK_ORDER_DISPATCH` 원장 커밋 및 로봇 파라미터 인가가"
            " 확정됩니다."
        )
    else:
        admin_notice = (
            "\n\n> ✅ **[원장 커밋 완료 (`STATUS: COMMITTED`)]** 공식 작업지시"
            f" 번호 `{erp_commit.get('work_order_number')}`로"
            " `MES_WORK_ORDER_DISPATCH` 배정 및 자재 선점이 완료되었습니다."
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

    # SAW 공정 보호가스 물리 정합성 보정
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
        "work_order_number": erp_commit.get("work_order_number"),
        "wps_code": resolved_wps_code,
        "revision_no": "Rev.1",
        "title": f"[{approval_status}] {params.get('block_id', '')} {params.get('material', '')} {params.get('thickness', '')} 작업지시서".strip(),
        "status": approval_status,
        "approval_status": approval_status,
        "execution_type": exec_type_str,
        "allocated_lot_no": lot_no,
        "assigned_vehicle_code": vehicle_code,
        "wire_item_code": wire_item_code,
        "reserve_wire_kg": req_wire_kg,
        "erp_commit": erp_commit,
        "partner_erp_plan": partner_plan,
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
        generated_doc["files"] = export_meta

        safety_eval = weather.get("safety_evaluation") or {}
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
        "final_answer": full_answer,
        "generated_document": generated_doc,
        "erp_commit_result": erp_commit,
        "audit_saved": True,
    }


# ==============================================================================
# 4. Team 3 SubGraph 조립
# ==============================================================================
builder = StateGraph(ShipAgentState)

builder.add_node("governance_gatekeeper", governance_gatekeeper_node)
builder.add_node("domain_qa_writer", domain_qa_writer_node)
builder.add_node("troubleshooter_agent", troubleshooter_node)
builder.add_node("work_order_writer", work_order_writer_node)

builder.add_edge(START, "governance_gatekeeper")
builder.add_conditional_edges(
    "governance_gatekeeper",
    route_from_gatekeeper,
    {
        "domain_qa_writer": "domain_qa_writer",
        "troubleshooter_agent": "troubleshooter_agent",
        "work_order_writer": "work_order_writer",
    },
)
builder.add_edge("domain_qa_writer", END)
builder.add_edge("troubleshooter_agent", END)
builder.add_edge("work_order_writer", END)

team3_subgraph = builder.compile()