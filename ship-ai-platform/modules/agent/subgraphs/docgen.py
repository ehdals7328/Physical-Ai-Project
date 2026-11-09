"""
modules/agent/subgraphs/docgen.py
Team 3: 작업지시서 생성, 자율 트러블슈팅, 인사/ERP DB 실시간 브리핑 & HITL 트랜잭션 마감 서브그래프
"""

import json
from typing import Any, Dict, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from modules.agent.state import ShipAgentState
from modules.agent.tools.erp_tools import list_erp_workers
from modules.agent.tools.document_tools import export_work_order_file, log_inspection_history

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)


# ==============================================================================
# 1. 라우팅 분기 함수 (route_generation_task)
# ==============================================================================
def route_generation_task(
    state: ShipAgentState,
) -> Literal["troubleshooter_agent", "work_order_writer", "domain_qa_writer"]:
    task_type = state.get("task_type", "knowledge_qa")

    # 1) 지식 문의, 사내 ERP 재고/인사 문의, 협력사 외주 문의는 전문 브리핑 노드로 직행
    if task_type != "doc_generation":
        print(f"\n👉 [Team 3 라우팅]: domain_qa_writer (사유: 일반 기술/ERP/인사 질의, task_type='{task_type}')\n")
        return "domain_qa_writer"

    # 2) 작업지시서 발행(doc_generation) 시 안전 & 거버넌스 인터록 점검
    missing = state.get("missing_params", [])
    weather = state.get("weather_data") or {}
    safety_eval = weather.get("safety_evaluation") or {}
    safety_status = safety_eval.get("status", "PASS")
    erp_status = state.get("erp_resource_status") or {}
    vision = state.get("vision_data") or {}

    # 선급 자격증 만료 여부 인터록 (무자격자 배정 시 즉시 차단!)
    worker_qualified = erp_status.get("worker_qualified", True)

    # 비전 결함/장애물 여부
    has_vision_hazard = any(
        det.get("is_hazard") is True
        or any(k in str(det.get("class_name", "")).upper() for k in ["OBSTACLE", "DEFECT", "CRACK", "CLAMP_CONFLICT"])
        for det in (vision.get("detections") or [])
    )

    # 기상 상태
    weather_status = weather.get("status")

    # 🚨 인터록 조건 판정
    is_blocked = (
        bool(missing)
        or safety_status == "STOP"
        or not worker_qualified
        or has_vision_hazard
        or weather_status in ("NEED_LOCATION", "NOT_FOUND")
    )

    print("\n" + "=" * 65)
    print("🔍 [Team 3 라우터 거버넌스 인터록 검사 현황]")
    print(f"   1. 작업 유형 (task_type)    : {task_type}")
    print(f"   2. 필수 제원 누락 여부      : {bool(missing)} (누락 목록: {missing})")
    print(f"   3. 기상 상태 및 안전 판정   : 상태='{weather_status}' | 판정='{safety_status}'")
    print(f"   4. 작업자 선급 자격 유효성  : {worker_qualified} (worker_qualified)")
    print(f"   5. 비전 장애물/결함 감지    : {has_vision_hazard}")

    if is_blocked:
        fail_reasons = []
        if missing: fail_reasons.append(f"제원누락({missing})")
        if safety_status == "STOP": fail_reasons.append("기상STOP(작업중지)")
        if not worker_qualified: fail_reasons.append("작업자선급자격미달/만료")
        if has_vision_hazard: fail_reasons.append("비전위험감지")
        if weather_status in ("NEED_LOCATION", "NOT_FOUND"): fail_reasons.append(f"기상위치결측({weather_status})")

        chosen_node = "troubleshooter_agent"
        print(f"👉 [최종 결정]: {chosen_node} (차단 사유: {', '.join(fail_reasons)})")
        print("=" * 65 + "\n")
        return chosen_node

    # ✅ 정상(PASS) 또는 조건부 승인(CONDITIONAL_PASS) 시 작업지시서 작성 노드로 진입
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
        "제공된 [사내 규격 문서(RAG)], [본사 ERP 자원 현황], [303호 협력사 B2B 조회 결과], [실시간 기상 정보]를 바탕으로 "
        "사용자의 질문에 명확하고 체계적인 마크다운으로 답변하세요.\n\n"
        "[외주 협력사 안내 원칙]\n"
        "우리 스마트 조선소의 사외 1차 협력사는 도메인별로 2곳이 독립 등록되어 있습니다:\n"
        "1. 자재공급 협력사: (주)대한용재 (용접 와이어 CSF-71T, 강판 등 소모 자재 긴급 수급, 리드타임 2시간)\n"
        "2. 중량물 운송 협력사: (주)한국운송 (사내 트랜스포터 T/P 부족 시 중량물 블록 외주 배차, 특수면허 운전원/신호수 배정, 리드타임 1.5시간)\n"
        "- 사용자가 외주 협력사 전체 현황이나 특정 분야(자재 vs 운송)를 물을 경우 위 2대 협력사 체계를 명확히 구분하여 사실대로 설명하십시오.\n"
        "- (주)대한용재를 운송업체로 설명하거나, (주)한국운송을 자재공급사로 왜곡하는 행위를 절대 금지합니다.\n"
        "- 사내 ERP 재고나 작업자 자격증 문의인 경우 [본사 ERP 자원 현황] 데이터를 정확히 인용하세요.",
    ),
    (
        "human",
        "[질의 유형]: {task_type}\n"
        "[사용자 질문]: {query}\n\n"
        "[본사 ERP 자원 현황]:\n{erp_status}\n\n"
        "[303호 협력사 B2B 조회 결과]:\n{partner_plan}\n\n"
        "[인출된 사내 규격 문서(RAG)]:\n{docs}\n\n"
        "[실시간 기상 정보]:\n{weather}",
    ),
])


def domain_qa_writer_node(state: ShipAgentState) -> Dict[str, Any]:
    task_type = state.get("task_type")

    # 🌟 [인사 DB 실시간 조회] ERP_WORKER_CERT 테이블에서 인출하여 즉시 마크다운 표 생성
    if task_type == "hr_worker_qa":
        db_res = list_erp_workers.invoke({})
        workers = db_res.get("workers", [])

        table_md = "## 👷 사내 ERP (`ERP_WORKER_CERT`) 실시간 작업자 현황\n\n"
        table_md += "| 작업자명 | 사번 | 직책 | 선급 자격 규격 | 자격 만료일 | 상태 |\n"
        table_md += "|:---:|:---:|:---:|:---:|:---:|:---:|\n"

        for w in workers:
            status_badge = "✅ 투입 가능" if w["is_valid"] else "❌ 투입 불가 (만료)"
            table_md += f"| {w['worker_name']} | {w['worker_id']} | {w['role']} | {w['cert_code']} | {w['cert_expiry_date']} | {status_badge} |\n"

        table_md += f"\n> 💡 *현재 등록된 작업자는 총 {len(workers)}명이며, 사내 ERP 인사 원장에서 실시간 인출된 팩트 데이터입니다.*"

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
        "erp_status": json.dumps(state.get("erp_resource_status") or {}, ensure_ascii=False, indent=2),
        "partner_plan": json.dumps(state.get("partner_erp_plan") or {}, ensure_ascii=False, indent=2),
        "docs": docs_text,
        "weather": json.dumps(state.get("weather_data") or {}, ensure_ascii=False, indent=2),
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
        "1. [누락된 필수 제원]이 있는 경우: 해당 항목만 친절하게 요청하세요. (규격서 내부 수치인 전류, 전압, 화학성분 등은 절대 요구 금지)\n"
        "2. [본사 ERP 자격 검증]에서 작업자 자격증이 만료(is_valid=False)된 경우: 선급 규정에 따라 무자격자 투입 인터록(발행 차단)을 선언하고 유효 자격 보유자(예: 김철수 반장, 이영희 용접사)로 교체 배정을 권고하세요.\n"
        "3. [기상 안전 판정]이 'STOP'(풍속 10m/s 초과 또는 우천)인 경우: 작업 중지 명령과 함께 사내 SOP에 따른 크레인 붐대 고정 및 천막 보양 조치를 안내하세요.",
    ),
    (
        "human",
        "[사용자 요청]: {query}\n"
        "[누락된 필수 제원]: {missing}\n"
        "[본사 ERP 자격/자원 검증]: {erp_status}\n"
        "[기상 및 안전 판정]: {weather}\n"
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
        "erp_status": json.dumps(state.get("erp_resource_status") or {}, ensure_ascii=False),
        "weather": json.dumps(state.get("weather_data") or {}, ensure_ascii=False),
        "docs": docs_text,
    })

    return {
        "final_answer": res.content,
        "generated_document": {
            "title": "작업 보류 및 현장 안전/자원 조치 지시서 (HOLD)",
            "status": "HOLD",
            "parameters": state.get("extracted_params") or {},
            "weather": state.get("weather_data"),
            "content": res.content,
        },
    }


# ==============================================================================
# 4. 공식 작업지시서 작성 노드 (work_order_writer - HITL 승인 대기 초안 생성)
# ==============================================================================
WORK_ORDER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 스마트 조선소 수석 공정 엔지니어입니다.\n"
        "제공된 [확정 작업 제원], [ERP 사전 검토 결과], [303호 협력사 B2B 연계 내역], [기상 및 안전 판정], [참조 WPS 규격(RAG)]만을 "
        "엄격한 팩트(Fact)로 참조하여 공식 **표준 작업지시서(Work Order)**를 마크다운 표와 체크리스트 형식으로 작성하십시오.\n\n"
        "[작성 및 데이터 바인딩 원칙]\n"
        "1. **절대 금지**: 프롬프트 외부의 가상 수치나 기본값을 날조하지 마십시오. 모든 값은 아래 제공된 실제 컨텍스트 데이터에서만 추출해야 합니다.\n"
        "2. **문서 헤더 및 ERP 오더 정보**:\n"
        "   - 작업지시 번호, 실행 구분(사내 직영 vs B2B 외주), 블록 번호, 최종 승인 상태를 명시하십시오.\n"
        "3. **자재 Lot 추적성 & 인력/외주 배정 표**:\n"
        "   - 배정 강재/와이어 Lot 번호, 와이어 선점 수량, 투입 작업자명 및 자격증 번호를 바인딩하십시오.\n"
        "   - 외주 연계 시 [303호 협력사 B2B 내역]의 협력사명과 외주 발주(PO) 번호를 정확히 기재하십시오.\n"
        "4. **WPS 공정 조건 표**:\n"
        "   - [확정 작업 제원]의 모재, 두께, 공정과 [참조 WPS 규격(RAG)]에 인출된 실제 공학 수치(전류, 전압, 속도, 가스 등)를 매핑하십시오.\n"
        "   - 만약 RAG 문서에 특정 공정(예: SAW)의 정확한 수치가 누락된 경우 'TBD'로 비워두지 말고, 해당 모재/두께/공정에 공학적으로 부합하는 표준 권장 조건(예: SAW 후판 기준 전류 300~350A, 전압 28~32V, 속도 0.4~0.6m/min, Flx/Ar-CO2 등)을 산출하여 기재하십시오.\n"
        "   - 와이어 규격란에는 [ERP 사전 검토 결과] 또는 [303호 B2B 내역]에서 실제 배정된 자재명/Lot를 일치시켜 기재하십시오.\n"
        "5. **실시간 현장 기상 및 안전 조치 체크리스트**:\n"
        "   - [기상 및 안전 판정]에 명시된 실측 환경 수치(기온, 풍속, 습도, 강수량)를 표기하십시오.\n"
        "   - 체크리스트(`- [ ]`) 항목에는 룰 엔진이 통보한 필수 안전 준수사항(`mandatory_rules`) 및 선행 조치(`prerequisites`)를 그대로 반영하십시오."
    ),
    (
        "human",
        "[사용자 요청]: {query}\n"
        "[확정 작업 제원]: {params}\n"
        "[ERP 사전 검토 결과]: {erp_commit}\n"
        "[본사 ERP 진단 상세]: {erp_status}\n"
        "[303호 협력사 B2B 연계 내역]: {partner_plan}\n"
        "[기상 및 안전 판정]: {weather}\n"
        "[참조 WPS 및 안전 규격서(RAG)]:\n{docs}"
    ),
])


def work_order_writer_node(state: ShipAgentState) -> Dict[str, Any]:
    params = state.get("extracted_params") or {}
    erp_status = state.get("erp_resource_status") or {}
    partner_plan = state.get("partner_erp_plan") or {}
    weather = state.get("weather_data") or {}
    is_outsourced = state.get("is_outsourcing_required", False)

    # 1) 가상 배정 Lot 추출
    selected_partner = (partner_plan.get("selected_partner") or {}) if partner_plan else {}
    lot_no = (
        selected_partner.get("partner_lot_no")
        if is_outsourced and selected_partner.get("partner_lot_no")
        else erp_status.get("assigned_lot_no", "LOT-2026-POSCO-04")
    )

    # 2) 임시 승인 대기 프리뷰 플랜 수립
    preview_commit = {
        "work_order_number": f"WO-DRAFT-{params.get('block_id', 'BLK')}",
        "execution_type": "OUTSOURCED_B2B" if is_outsourced else "INTERNAL_HQ",
        "allocated_lot_no": lot_no,
        "erp_transaction_status": "WAITING_APPROVAL",
        "b2b_po_ack": f"PO-PLAN-{selected_partner.get('partner_id', 'HQ')}" if is_outsourced else None,
        "remaining_hq_wire_kg": 0 if is_outsourced else 105.0,
        "reserved_wire_kg": 15.0,
    }

    # 3) RAG 문서 텍스트 직렬화
    docs = state.get("retrieved_docs", [])
    docs_text = "\n\n".join(
        f"[{d.get('metadata', {}).get('document_name', 'WPS 규격서')}]: {d.get('content', '')}"
        for d in docs
    ) or "사내 표준 WPS 규격 적용"

    # 4) 작업지시서 초안 마크다운 생성
    res = (WORK_ORDER_PROMPT | llm).invoke({
        "query": state.get("user_query", ""),
        "params": json.dumps(params, ensure_ascii=False),
        "erp_commit": json.dumps(preview_commit, ensure_ascii=False),
        "erp_status": json.dumps(erp_status, ensure_ascii=False),
        "partner_plan": json.dumps(partner_plan, ensure_ascii=False),
        "weather": json.dumps(weather, ensure_ascii=False),
        "docs": docs_text,
    })
    answer = res.content

    admin_notice = "\n\n> ⚠️ **[관리자 결재 대기]** 본 작업지시서와 외주 자재/장비 계획은 아직 ERP 원장에 확정되지 않았습니다. 관리자 승인 시 공식 발주 및 원장 커밋이 완료됩니다."
    full_answer = answer + admin_notice

    generated_doc = {
        "work_order_number": preview_commit.get("work_order_number"),
        "title": f"[초안] {params.get('block_id', '')} {params.get('material', '')} {params.get('thickness', '')} 작업지시서".strip(),
        "status": "WAITING_APPROVAL",
        "execution_type": preview_commit.get("execution_type"),
        "allocated_lot_no": lot_no,
        "erp_commit": preview_commit,
        "partner_erp_plan": partner_plan,
        "parameters": params,
        "weather": weather,
        "content": full_answer,
    }

    # =========================================================================
    # 🌟 [도구 실제 호출 연결부]: 마크다운 파일 저장 및 감사 로그 기록
    # =========================================================================
    try:
        # 1. 작업지시서 .md 파일 저장 도구 호출
        export_work_order_file.invoke({
            "document_data": generated_doc,
            "file_prefix": f"weld_order_{params.get('block_id', 'BLK')}"
        })

        # 2. 감사(Audit) 로그 파일 기록 도구 호출
        safety_eval = weather.get("safety_evaluation") or {}
        audit_status = safety_eval.get("status", "PASS")
        full_query_log = f"[제원: {json.dumps(params, ensure_ascii=False)}] | 발화: {state.get('user_query', '')}"

        log_inspection_history.invoke({
            "session_id": preview_commit.get("work_order_number", "DRAFT_SESSION"),
            "query": full_query_log,
            "status": audit_status
        })
        print(f"📁 [Audit Log]: audit_history.log 및 .md 파일 저장 완료 ({preview_commit.get('work_order_number')})")
    except Exception as e:
        print(f"⚠️ [Document Tools 오류]: {e}")

    return {
        "final_answer": full_answer,
        "generated_document": generated_doc,
        "erp_commit_result": preview_commit,
        "admin_approval_required": True,
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