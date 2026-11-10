"""
modules/agent/subgraphs/docgen.py
Team 3: 작업지시서(WORK_INSTRUCTIONS) 생성, 자율 트러블슈팅,
인사/ERP DB 실시간 브리핑 & [Logical Bridge: Work Authorization] 트랜잭션 마감 서브그래프
"""

import json
from decimal import Decimal
from typing import Any, Dict, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from modules.agent.state import ShipAgentState
from modules.agent.tools.document_tools import export_work_order_file, log_inspection_history
from modules.agent.tools.erp_tools import commit_erp_work_order, list_erp_workers

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)


class DecimalEncoder(json.JSONEncoder):
    """Oracle DB의 Decimal 및 날짜/시간 객체를 JSON 안전 타입으로 자동 변환"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return super().default(obj)


# ==============================================================================
# 1. 라우팅 분기 함수 (route_generation_task)
# ==============================================================================
def route_generation_task(
    state: ShipAgentState,
) -> Literal["troubleshooter_agent", "work_order_writer", "domain_qa_writer"]:
    task_type = state.get("task_type", "knowledge_qa")

    if task_type != "doc_generation":
        print(f"\n👉 [Team 3 라우팅]: domain_qa_writer (사유: 일반 기술/ERP/인사 질의, task_type='{task_type}')\n")
        return "domain_qa_writer"

    missing = state.get("missing_params", [])
    weather = state.get("weather_data") or {}
    safety_eval = weather.get("safety_evaluation") or {}
    safety_status = safety_eval.get("status", "PASS")
    erp_status = state.get("erp_resource_status") or {}
    vision = state.get("vision_data") or {}

    # 🌟 작업자 자격 인터록 검사 강화 (worker_qualified와 worker_qualification.is_valid 둘 다 검사)
    worker_qual = erp_status.get("worker_qualification") or {}
    is_worker_valid = erp_status.get("worker_qualified", True) and worker_qual.get("is_valid", False)
    
    # 만약 worker_qualification 정보 자체가 비어있거나 worker_id가 없으면 부적격
    if not worker_qual or not worker_qual.get("worker_id"):
        is_worker_valid = False

    robot_exec = vision.get("robot_execution") or {}
    has_vision_hazard = (
        str(vision.get("inspection_status", "")).upper() == "DEFECT"
        or str(vision.get("welding_state", "")).upper() == "IN_WELDING"
        or str(robot_exec.get("emergency_stop_yn", "N")).upper() == "Y"
        or any(
            det.get("is_hazard") is True
            or any(k in str(det.get("class_name", "")).upper() for k in ["OBSTACLE", "DEFECT", "CRACK", "CLAMP_CONFLICT"])
            for det in (vision.get("detections") or [])
        )
    )

    weather_status = weather.get("status")

    is_blocked = (
        bool(missing)
        or safety_status == "STOP"
        or not is_worker_valid              # 👈 자격 만료/미배정 시 차단!
        or has_vision_hazard
        or weather_status in ("NEED_LOCATION", "NOT_FOUND", "PARSE_ERROR", "ERROR")
    )

    print("\n" + "=" * 65)
    print("🔍 [Team 3 라우터 거버넌스 인터록 검사 현황]")
    print(f"   1. 작업 유형 (task_type)    : {task_type}")
    print(f"   2. 필수 제원 누락 여부      : {bool(missing)} (누락 목록: {missing})")
    print(f"   3. 기상 상태 및 안전 판정   : 상태='{weather_status}' | 판정='{safety_status}'")
    print(f"   4. 작업자 선급 자격 유효성  : {is_worker_valid} (작업자: {worker_qual.get('worker_name')}, 상태: {worker_qual.get('status')})")
    print(f"   5. 비전/로봇 위험 감지 여부 : {has_vision_hazard} (안전거리={vision.get('safety_clearance_yn', 'Y')})")

    if is_blocked:
        fail_reasons = []
        if missing:
            fail_reasons.append(f"제원누락({missing})")
        if safety_status == "STOP":
            fail_reasons.append("기상STOP(작업중지)")
        if not is_worker_valid:
            fail_reasons.append(f"작업자선급자격결격({worker_qual.get('worker_name', '미배정')})")
        if has_vision_hazard:
            fail_reasons.append("비전결함/용접중Lock/로봇비상정지")
        if weather_status in ("NEED_LOCATION", "NOT_FOUND", "PARSE_ERROR", "ERROR"):
            fail_reasons.append(f"기상조회실패({weather_status})")

        chosen_node = "troubleshooter_agent"
        print(f"👉 [최종 결정]: {chosen_node} (차단 사유: {', '.join(fail_reasons)})")
        print("=" * 65 + "\n")
        return chosen_node

    chosen_node = "work_order_writer"
    print(f"👉 [최종 결정]: {chosen_node} (모든 안전 및 ERP 인터록 통과)")
    print("=" * 65 + "\n")
    return chosen_node


# ==============================================================================
# 2. 도메인 QA, ERP 재고 및 인사 DB 실시간 브리핑 노드 (domain_qa_writer)
# ==============================================================================
QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 스마트 조선소 엔지니어링 & ERP 기술 자문 AI입니다.\n"
        "제공된 [사내 규격 문서(RAG)], [본사 SHIP_ERP 자원 현황], [협력사 B2B 조회 결과], [실시간 기상 정보]를 바탕으로 "
        "사용자의 질문에 명확하고 체계적인 마크다운으로 답변하세요.\n\n"
        "[안내 원칙]\n"
        "- 사내 ERP 재고(`ERP_BOM_INVENTORY`), 대기 차량(`ERP_VEHICLES_EQUIPMENT`), 작업자 자격증(`ERP_WORKER_CERT`), 협력사(`ERP_COMPANIES`) 문의인 경우 제공된 DB 조회 데이터를 왜곡 없이 그대로 인용하세요.",
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
        # branch_code 필터 없이 전체 작업자 조회 (모든 야드 포함)
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
    docs_text = "\n\n".join(
        f"- [{d.get('metadata', {}).get('document_name', '규격서')} p.{d.get('metadata', {}).get('page', 1)}]: {d.get('content', '')}"
        for d in docs
    ) or "해당 없음"

    res = (QA_PROMPT | llm).invoke({
        "task_type": task_type,
        "query": state.get("user_query", ""),
        "erp_status": json.dumps(state.get("erp_resource_status") or {}, ensure_ascii=False, indent=2, cls=DecimalEncoder),
        "partner_plan": json.dumps(state.get("partner_erp_plan") or {}, ensure_ascii=False, indent=2, cls=DecimalEncoder),
        "docs": docs_text,
        "weather": json.dumps(state.get("weather_data") or {}, ensure_ascii=False, indent=2, cls=DecimalEncoder),
    })

    return {
        "final_answer": res.content,
        "generated_document": None,
    }


# ==============================================================================
# 3. Self-Healing 트러블슈터 & 거버넌스 인터록 노드 (troubleshooter_agent)
# ==============================================================================
TROUBLESHOOT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 조선소 안전 거버넌스 및 공정 트러블슈팅 에이전트입니다.\n"
        "현재 작업지시서 발행을 보류해야 하는 사유를 진단하고 현장 복구 대안(Self-Healing SOP)을 제시하세요.\n\n"
        "[처방 원칙]\n"
        "1. [누락된 필수 제원]이 있는 경우: 해당 항목만 친절하게 요청하세요.\n"
        "2. [본사 ERP 자격 검증]에서 작업자 자격증이 만료 또는 부적격(`is_valid=False`)인 경우: 선급 규정에 따라 무자격자 투입 인터록(발행 차단)을 선언하고 `ERP_WORKER_CERT` 내 유효 자격 보유자로 교체 배정을 권고하세요.\n"
        "3. [기상 안전 판정]이 'STOP'(풍속 10m/s 초과 또는 우천)인 경우: 작업 중지 명령과 함께 사내 SOP에 따른 크레인 붐대 고정 및 방풍/보양 조치를 안내하세요.\n"
        "4. [비전 검사 결과]에서 불량(`DEFECT`), 용접 중(`IN_WELDING`), 또는 로봇 비상정지(`EMERGENCY_STOP_YN='Y'`)가 감지된 경우: 즉시 원인 해소 및 재검사를 안내하세요.",
    ),
    (
        "human",
        "[사용자 요청]: {query}\n"
        "[누락된 필수 제원]: {missing}\n"
        "[본사 ERP 자격/자원 검증]: {erp_status}\n"
        "[기상 및 안전 판정]: {weather}\n"
        "[비전/로봇 검사 결과]: {vision}\n"
        "[관련 안전 지침(RAG)]: {docs}",
    ),
])


def troubleshooter_node(state: ShipAgentState) -> Dict[str, Any]:
    missing = state.get("missing_params", [])
    docs = state.get("retrieved_docs", [])
    docs_text = "\n".join(str(d.get("content", ""))[:400] for d in docs[:2])

    res = (TROUBLESHOOT_PROMPT | llm).invoke({
        "query": state.get("user_query", ""),
        "missing": missing,
        "erp_status": json.dumps(state.get("erp_resource_status") or {}, ensure_ascii=False, cls=DecimalEncoder),
        "weather": json.dumps(state.get("weather_data") or {}, ensure_ascii=False, cls=DecimalEncoder),
        "vision": json.dumps(state.get("vision_data") or {}, ensure_ascii=False, cls=DecimalEncoder),
        "docs": docs_text,
    })

    return {
        "final_answer": res.content,
        "generated_document": {
            "title": "작업 보류 및 현장 안전/자원 조치 지시서 (HOLD)",
            "status": "HOLD",
            "approval_status": "REJECTED",
            "parameters": state.get("extracted_params") or {},
            "weather": state.get("weather_data"),
            "vision": state.get("vision_data"),
            "content": res.content,
            "instruction_text": res.content,
        },
    }


# ==============================================================================
# 4. 공식 작업지시서 작성 노드 (work_order_writer)
# ==============================================================================
WORK_ORDER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 스마트 조선소 수석 공정 엔지니어입니다.\n"
        "제공된 [확정 작업 제원], [ERP 사전 검토/커밋 결과], [협력사 B2B 연계 내역], [YOLOv11 비전 계측 결과], [기상 및 안전 판정], [참조 WPS 규격(RAG)]만을 "
        "엄격한 팩트(Fact)로 참조하여 공식 **표준 작업지시서(Work Order & Work Instruction)**를 마크다운 표와 체크리스트 형식으로 작성하십시오.\n\n"
        "[작성 및 데이터 바인딩 원칙]\n"
        "1. **절대 금지**: 프롬프트 외부의 가상 수치나 기본값을 날조하지 마십시오.\n"
        "2. **문서 헤더 및 ERP 오더 정보 (`WORK_ORDERS` & `MES_WORK_ORDER_DISPATCH`)**:\n"
        "   - 작업지시 번호(`WO_NUMBER`), 호선 번호(`SHIP_CODE`), 블록 번호(`BLOCK_ID`), 작업 야드/구역(`BRANCH_CODE` / `WORK_ZONE`), 실행 구분(`IN_HOUSE` vs `OUTSOURCED_B2B`), 승인 상태를 명시하십시오.\n"
        "3. **자재 Lot 추적성, 인력 자격 & 대기차량 배차 표**:\n"
        "   - 배정 강재/와이어 Lot 번호(`ALLOCATED_LOT_NO`), 와이어 선점 수량, 투입 작업자명/사번(`WORKER_ID`) 및 선급 자격증 번호(`CERT_CODE`), 배차된 운반/지원 차량 코드(`ASSIGNED_VEHICLE_CODE`)를 바인딩하십시오.\n"
        "   - 외주 연계 시 [협력사 B2B 연계 내역]의 협력사 코드(`PARTNER_ID`), 협력사명, 외주 발주(PO) 번호를 정확히 기재하십시오.\n"
        "4. **WPS 공정 조건 및 YOLOv11 비전 용접 궤적 표**:\n"
        "   - [확정 작업 제원]의 모재(`STEEL_TYPE`), 두께(`THICKNESS_MM`), 공정(`PROCESS`)과 [참조 WPS 규격(RAG)]의 공학 수치(전류, 전압, 속도, 보호가스 등)를 매핑하십시오.\n"
        "   - [YOLOv11 비전 계측 결과]의 용접 시작점(`START_POINT_JSON`), 종료점(`END_POINT_JSON`), 예상 용접 길이(`ESTIMATED_LENGTH_MM`), 안전거리 확보 여부(`SAFETY_CLEARANCE_YN`)를 함께 표기하십시오.\n"
        "5. **실시간 현장 기상 및 안전 조치 체크리스트**:\n"
        "   - [기상 및 안전 판정]의 실측 기온, 풍속, 습도, 강수량과 필수 안전 준수사항(`mandatory_rules`), 선행 조치(`prerequisites`)를 체크리스트(`- [ ]`)로 작성하십시오."
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
        "[참조 WPS 및 안전 규격서(RAG)]:\n{docs}"
    ),
])


def work_order_writer_node(state: ShipAgentState) -> Dict[str, Any]:
    params = state.get("extracted_params") or {}
    erp_status = state.get("erp_resource_status") or {}
    partner_plan = state.get("partner_erp_plan") or {}
    weather = state.get("weather_data") or {}
    vision = state.get("vision_data") or {}
    is_outsourced = state.get("is_outsourcing_required", False)

    selected_partner = (partner_plan.get("selected_partner") or {}) if partner_plan else {}
    lot_no = (
        selected_partner.get("partner_lot_no")
        if is_outsourced and selected_partner.get("partner_lot_no")
        else erp_status.get("assigned_lot_no")
    ) or "LOT-UNASSIGNED"

    worker_qual = erp_status.get("worker_qualification") or {}
    worker_id = worker_qual.get("worker_id") or "UNASSIGNED"
    worker_name = params.get("worker_name") or worker_qual.get("worker_name") or "미배정"

    vehicle_code = (
        selected_partner.get("assigned_vehicle_code")
        if is_outsourced and selected_partner.get("assigned_vehicle_code")
        else erp_status.get("assigned_vehicle_code")
    )
    partner_id = selected_partner.get("partner_id", "COMP-HQ") if is_outsourced else "COMP-HQ"
    block_code = params.get("block_id") or "UNSPECIFIED"

    if state.get("hitl_approved") or state.get("is_action_requested"):
        erp_commit = commit_erp_work_order.invoke({
            "block_id": block_code,
            "material": params.get("material", ""),
            "allocated_lot_no": lot_no,
            "worker_name": worker_name,
            "worker_id": worker_id,
            "work_no": state.get("work_no"),
            "assigned_vehicle_code": vehicle_code,
            "wire_item_code": erp_status.get("selected_wire_code"),
            "is_outsourced": is_outsourced,
            "partner_id": partner_id,
            "reserve_wire_kg": 15.0,
        })
        approval_status = "APPROVED"
    else:
        wo_num = state.get("work_no") or f"WO-DRAFT-{block_code}"
        erp_commit = {
            "work_order_number": wo_num,
            "execution_type": "OUTSOURCED_B2B" if is_outsourced else "IN_HOUSE",
            "partner_id": partner_id,
            "allocated_lot_no": lot_no,
            "worker_id": worker_id,
            "worker_name": worker_name,
            "assigned_vehicle_code": vehicle_code,
            "erp_transaction_status": "DRAFT",   # 👈 'WAITING_APPROVAL' 대신 DDL 규격과 일치하는 명칭 사용
            "b2b_po_ack": f"PO-PLAN-{partner_id}" if is_outsourced else None,
            "reserved_wire_kg": 15.0,
        }
        approval_status = "DRAFT"

    docs = state.get("retrieved_docs", [])
    docs_text = "\n\n".join(
        f"[{d.get('metadata', {}).get('document_name', 'WPS 규격서')}]: {d.get('content', '')}"
        for d in docs
    ) or "사내 표준 WPS 규격 적용"

    source_refs_list = [
        {
            "document_code": d.get("metadata", {}).get("document_code"),
            "document_title": d.get("metadata", {}).get("document_name"),
            "page": d.get("metadata", {}).get("page", 1),
            "similarity": d.get("metadata", {}).get("similarity"),
        }
        for d in docs
    ]

    res = (WORK_ORDER_PROMPT | llm).invoke({
        "query": state.get("user_query", ""),
        "params": json.dumps(params, ensure_ascii=False, cls=DecimalEncoder),
        "erp_commit": json.dumps(erp_commit, ensure_ascii=False, cls=DecimalEncoder),
        "erp_status": json.dumps(erp_status, ensure_ascii=False, cls=DecimalEncoder),
        "partner_plan": json.dumps(partner_plan, ensure_ascii=False, cls=DecimalEncoder),
        "vision": json.dumps(vision, ensure_ascii=False, cls=DecimalEncoder),
        "weather": json.dumps(weather, ensure_ascii=False, cls=DecimalEncoder),
        "docs": docs_text,
    })
    answer = res.content

    if approval_status == "DRAFT":
        admin_notice = "\n\n> ⚠️ **[관리자 결재 대기 (`APPROVAL_STATUS: DRAFT`)]** 본 작업지침서는 초안 상태입니다. 관리자(MANAGER/ADMIN) 승인 시 `MES_WORK_ORDER_DISPATCH` 원장 커밋 및 로봇 파라미터 인가가 확정됩니다."
    else:
        admin_notice = f"\n\n> ✅ **[원장 커밋 완료 (`STATUS: COMMITTED`)]** 공식 작업지시 번호 `{erp_commit.get('work_order_number')}`로 `MES_WORK_ORDER_DISPATCH` 배정 및 자재 선점이 완료되었습니다."

    full_answer = answer + admin_notice

    welding_params_payload = {
        "process_code": params.get("process"),
        "material_grade": params.get("material"),
        "thickness_mm": params.get("thickness"),
        "weld_position": params.get("weld_position"),
        "start_point": vision.get("start_point_json"),
        "end_point": vision.get("end_point_json"),
        "estimated_length_mm": vision.get("estimated_length_mm"),
        "safety_clearance_yn": vision.get("safety_clearance_yn", "Y"),
    }

    generated_doc = {
        "work_id": state.get("work_id"),
        "work_order_number": erp_commit.get("work_order_number"),
        "revision_no": "Rev.1",
        "title": f"[{approval_status}] {params.get('block_id', '')} {params.get('material', '')} {params.get('thickness', '')} 작업지시서".strip(),
        "status": "APPROVED" if approval_status == "APPROVED" else "WAITING_APPROVAL",
        "approval_status": approval_status,
        "execution_type": erp_commit.get("execution_type"),
        "allocated_lot_no": lot_no,
        "assigned_vehicle_code": vehicle_code,
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
    }

    try:
        export_work_order_file.invoke({
            "document_data": generated_doc,
            "file_prefix": f"weld_order_{block_code}",
        })

        safety_eval = weather.get("safety_evaluation") or {}
        audit_status = safety_eval.get("status", "PASS")
        full_query_log = f"[제원: {json.dumps(params, ensure_ascii=False, cls=DecimalEncoder)}] | 발화: {state.get('user_query', '')}"

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
# 5. 서브그래프 조립
# ==============================================================================
builder = StateGraph(ShipAgentState)

builder.add_node("domain_qa_writer", domain_qa_writer_node)
builder.add_node("troubleshooter_agent", troubleshooter_node)
builder.add_node("work_order_writer", work_order_writer_node)

builder.add_conditional_edges(
    START,
    route_generation_task,
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