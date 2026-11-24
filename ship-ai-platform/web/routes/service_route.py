# web/routes/service_route.py
import sys
import json
import time
import uuid
import ast
import threading
import traceback
from pathlib import Path
from functools import wraps
import math
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from flask import (
    Blueprint,
    render_template,
    request,
    session,
    redirect,
    url_for,
    flash,
    jsonify,
    abort,
    send_from_directory,
    send_file,
    current_app,
)
from sqlalchemy import func, or_, text
from langchain_core.messages import HumanMessage

# 프로젝트 루트 경로 및 .env 로드
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from web.models import (
    db,
    User,
    WorkOrder,
    WorkInstruction,
    AgentChatHistory,
    AgentExecution,
    CommunityBoard,
    CadDrawing,
    WorkPlan,
    RobotExecution,
    VisionAnalysis,
    VisionDetection,
    VisionInspection,
    ErpBranch,
    ErpBlockMaster,
    ErpBomInventory,
    ErpWorkerCert,
    ErpVehiclesEquipment,
    MesWorkOrderDispatch,
    DocumentMetadata,
    ErpCompany,
    B2BCompanyLocation,
    B2BDaehanMaterial,
    B2BDonghaeWorker,
    B2BHankookEquipment,
    B2BSalesOrder,
)

service_bp = Blueprint("service", __name__, url_prefix="/service")


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("해당 서비스는 임직원 및 협력사 로그인 후 이용하실 수 있습니다.", "warning")
            return redirect(url_for("auth.login"))
        curr_user = User.query.get(session["user_id"])
        if not curr_user or curr_user.is_active != "Y":
            session.clear()
            flash("비활성화(사용 중지)된 계정입니다. 관리자에게 문의하세요.", "danger")
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated_function



def _build_page_items(page: int, total_pages: int) -> list:
    """10페이지 초과 시 중간을 '...'으로 축약하는 스마트 페이지네이션"""
    if total_pages <= 10:
        return list(range(1, total_pages + 1))
    items = [1]
    if page > 4:
        items.append("...")
    for p in range(max(2, page - 2), min(total_pages, page + 3)):
        items.append(p)
    if page < total_pages - 3:
        items.append("...")
    items.append(total_pages)
    return items



import re as _re

_SVC_MOJIBAKE_RE = _re.compile(r"[\u2500-\u259f\u0400-\u04ff\u0080-\u00ff\u0100-\u024f\ufffd\u2116]")


def _clean_doc_title(raw_title: str | None, original_file_name: str | None) -> str:
    title = (raw_title or "").strip()
    fname = (original_file_name or "").strip()
    fallback = Path(fname).stem.replace("_", " ").strip() if fname else title
    if not title or _SVC_MOJIBAKE_RE.search(title):
        return fallback or title or "제목 없음"
    return title


def _infer_yard_stage(branch, br_orders: list | None = None) -> str:
    """DB의 DEFAULT_VIDEO 및 WORK_ORDERS 상태 기반으로 야드 공정 단계를 동적 판별 (하드코딩 제거)"""
    raw_vid = (getattr(branch, "default_video", None) or "").strip().lower()
    if raw_vid in ("ocr_test.mp4", "video2.mp4") or "ocr" in raw_vid:
        return "TRANSPORT"
    if br_orders:
        has_in_prog = any(o.status_code == "IN_PROGRESS" for o in br_orders)
        has_ready = any(o.status_code == "READY" for o in br_orders)
        if not has_in_prog and has_ready:
            return "TRANSPORT"
    return "WORKING"


def _match_cad_for_block(blk, cad_docs: list[dict], idx: int) -> dict | None:
    """블록명·호선에 포함된 선박 모델 키워드로 실제 CAD 도면을 동적 매칭"""
    if not cad_docs:
        return None
    hint = f"{blk.ship_code or ''} {blk.block_name or ''} {blk.block_id or ''}".lower()
    for vk in ("gunnerus", "kcs", "kvlcc2", "kvlcc1", "5415"):
        if vk in hint:
            matched = [
                c for c in cad_docs
                if vk in f"{c.get('document_title','')} {c.get('original_file_name','')} {c.get('vessel_name','')}".lower()
            ]
            if matched:
                return matched[idx % len(matched)]
    return cad_docs[idx % len(cad_docs)]


def _get_default_branch_code(user=None) -> str:
    if user and getattr(user, "erp_branch_code", None):
        return user.erp_branch_code
    first_br = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).first()
    return first_br.branch_code if first_br else ""

INQUIRY_CATEGORIES = ("공식문의", "계정/권한", "자재/장비", "현장기술", "오류신고")

def _read_clob(val) -> str:
    """Oracle CLOB(oracledb.LOB) 객체 또는 문자열을 안전하게 str로 변환"""
    if val is None:
        return ""
    if hasattr(val, "read"):
        return val.read() or ""
    return str(val)


def _parse_json_or_literal(val) -> list:
    """Oracle CLOB에서 읽은 JSON(쌍따옴표/홑따옴표 모두 허용)을 안전하게 list로 파싱"""
    raw_str = _read_clob(val).strip()
    if not raw_str:
        return []
    try:
        parsed = json.loads(raw_str)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        try:
            parsed = ast.literal_eval(raw_str)
            return parsed if isinstance(parsed, list) else []
        except Exception as e:
            print(f"⚠️ [retrieval_chunks_json 파싱 경고]: {e}")
            return []


def _check_session_running(user_id: int, thread_id: str) -> bool:
    """해당 대화방에 현재 백그라운드 실행 중(RUNNING)인 작업이 있는지 확인합니다."""
    latest_exec = (
        AgentExecution.query
        .filter_by(user_id=user_id, session_id=thread_id)
        .order_by(AgentExecution.execution_id.desc())
        .first()
    )
    if not latest_exec or latest_exec.status_code != "RUNNING":
        return False

    # 서버 강제 종료 등으로 5분(300초) 이상 멈춘 고아 RUNNING 상태는 FAILED로 자동 정리
    if latest_exec.created_at:
        try:
            elapsed_sec = (datetime.utcnow() - latest_exec.created_at).total_seconds()
            if elapsed_sec > 300:
                latest_exec.status_code = "ERROR"
                latest_exec.current_state = "TIMEOUT"
                latest_exec.error_message = "백그라운드 작업 시간 초과(300초)"
                db.session.commit()
                return False
        except Exception:
            db.session.rollback()
    return True


# =====================================================================
# 1. AI 통합 어시스턴트 (/service/assistant) - 대화방 목록 & 스마트 빌더 연동
# =====================================================================
def _get_user_session_list(user_id: int, hidden_list: list = None):
    """
    DB에 저장된 대화 세션 목록을 추출하되,
    사용자가 목록에서 지운 세션(ROLE_CODE='SYSTEM', CONTENT='HIDDEN_BY_USER')은 영구 제외!
    """
    rows = (
        AgentChatHistory.query
        .filter(
            AgentChatHistory.user_id == user_id,
            AgentChatHistory.session_id != "BOARD_FAQ",
        )
        .order_by(AgentChatHistory.message_id.asc())
        .all()
    )

    db_hidden_sids = {
        r.session_id for r in rows
        if r.role_code == "SYSTEM" and str(r.content or "").strip() == "HIDDEN_BY_USER"
    }
    if hidden_list:
        db_hidden_sids.update(hidden_list)

    sessions_map = {}
    for r in rows:
        sid = r.session_id
        if sid in db_hidden_sids or r.role_code == "SYSTEM":
            continue
        if sid not in sessions_map:
            title = r.content.strip().split("\n")[0][:26] if r.role_code == "USER" else "새 대화"
            sessions_map[sid] = {
                "session_id": sid,
                "title": title,
                "updated_at": r.created_at.strftime("%m/%d %H:%M") if r.created_at else "",
                "last_id": r.message_id,
            }
        else:
            if sessions_map[sid]["title"] == "새 대화" and r.role_code == "USER":
                sessions_map[sid]["title"] = r.content.strip().split("\n")[0][:26]
            sessions_map[sid]["updated_at"] = r.created_at.strftime("%m/%d %H:%M") if r.created_at else ""
            sessions_map[sid]["last_id"] = r.message_id

    return sorted(sessions_map.values(), key=lambda x: x["last_id"], reverse=True)


@service_bp.route("/assistant", methods=["GET"])
@service_bp.route("/qna", methods=["GET"])
@login_required
def assistant_page():
    prompt = request.args.get("prompt", "").strip()
    if prompt:
        return redirect(url_for("service.workspace_page", prompt=prompt))
    return redirect(url_for("service.workspace_page"))


@service_bp.route("/api/assistant/history", methods=["GET"])
@login_required
def get_global_assistant_history():
    """모든 HTML 페이지에서 동일한 대화 세션을 유지하기 위한 서버 이력 조회 API"""
    user = User.query.get_or_404(session["user_id"])
    thread_id = session.get("assistant_thread_id")
    if not thread_id:
        last_msg = (
            AgentChatHistory.query
            .filter_by(user_id=user.user_id)
            .order_by(AgentChatHistory.message_id.desc())
            .first()
        )
        if last_msg:
            thread_id = last_msg.session_id
            session["assistant_thread_id"] = thread_id
        else:
            thread_id = f"ast_{user.username}_{uuid.uuid4().hex[:6]}"
            session["assistant_thread_id"] = thread_id

    rows = (
        AgentChatHistory.query
        .filter_by(user_id=user.user_id, session_id=thread_id)
        .order_by(AgentChatHistory.message_id.asc())
        .all()
    )
    messages = []
    for r in rows[-40:]:
        sources, files_meta, approval_st, doc_meta = [], None, "DRAFT", None
        if r.retrieval_chunks_json:
            for item in _parse_json_or_literal(r.retrieval_chunks_json):
                if isinstance(item, dict) and item.get("_type") == "WORK_ORDER_FILES":
                    files_meta = item.get("files")
                    approval_st = item.get("approval_status", "DRAFT")
                    doc_meta = item.get("doc")
                else:
                    sources.append(item)
        messages.append({
            "role": "user" if r.role_code == "USER" else "assistant",
            "content": _read_clob(r.content),
            "sources": sources,
            "files": files_meta,
            "approval_status": approval_st,
            "doc": doc_meta,
        })
    return jsonify({
        "status": "SUCCESS",
        "session_id": thread_id,
        "is_running": _check_session_running(user.user_id, thread_id),
        "messages": messages,
    })


def _unused_legacy_assistant_page():
    user = User.query.get_or_404(session["user_id"])
    hidden_sessions = list(session.get("hidden_assistant_sessions", []))

    req_sid = request.args.get("sid", "").strip()
    if req_sid:
        session["assistant_thread_id"] = req_sid
    elif "assistant_thread_id" not in session or session["assistant_thread_id"] in hidden_sessions:
        session["assistant_thread_id"] = f"ast_{user.username}_{uuid.uuid4().hex[:6]}"

    thread_id = session["assistant_thread_id"]
    is_running = _check_session_running(user.user_id, thread_id)

    chat_logs = (
        AgentChatHistory.query
        .filter(
            AgentChatHistory.user_id == user.user_id,
            AgentChatHistory.session_id == thread_id,
            AgentChatHistory.role_code != "SYSTEM",
        )
        .order_by(AgentChatHistory.message_id.asc())
        .all()
    )

    formatted_logs = []
    for log in chat_logs:
        chunks = []
        saved_files = None
        saved_status = "DRAFT"
        saved_doc = None
        raw_list = _parse_json_or_literal(log.retrieval_chunks_json)
        for item in raw_list:
            if isinstance(item, dict) and item.get("_type") == "WORK_ORDER_FILES":
                saved_files = item.get("files")
                saved_status = item.get("approval_status", "DRAFT")
                saved_doc = item.get("doc")
            else:
                chunks.append(item)

        formatted_logs.append({
            "role_code": log.role_code,
            "content": _read_clob(log.content),
            "sources": chunks,
            "files": saved_files,
            "approval_status": saved_status,
            "pending_doc": saved_doc,
            "created_at": log.created_at.strftime("%H:%M") if log.created_at else "",
        })

    session_list = _get_user_session_list(user.user_id, hidden_sessions)

    active_branches = (
        ErpBranch.query
        .filter_by(status_code="ACTIVE")
        .order_by(ErpBranch.branch_code.asc())
        .all()
    )

    return render_template(
        "service/assistant.html",
        me=user,
        thread_id=thread_id,
        chat_logs=formatted_logs,
        session_list=session_list,
        branches=active_branches,
        is_running=is_running,
    )


@service_bp.route("/api/assistant/reset", methods=["POST"])
@login_required
def reset_assistant_session():
    """새 대화방 생성"""
    user = User.query.get_or_404(session["user_id"])
    new_sid = f"ast_{user.username}_{uuid.uuid4().hex[:6]}"
    session["assistant_thread_id"] = new_sid
    return jsonify({"status": "SUCCESS", "thread_id": new_sid})


@service_bp.route("/api/assistant/hide-session", methods=["POST"])
@login_required
def hide_assistant_session():
    """X 버튼 클릭 시 대화방 영구 숨김 처리"""
    data = request.get_json(silent=True) or {}
    target_sid = str(data.get("session_id") or "").strip()
    user = User.query.get_or_404(session["user_id"])

    hidden = list(session.get("hidden_assistant_sessions", []))
    if target_sid and target_sid not in hidden:
        hidden.append(target_sid)
        session["hidden_assistant_sessions"] = hidden
        session.modified = True

    if target_sid:
        try:
            exists = AgentChatHistory.query.filter_by(
                session_id=target_sid,
                user_id=user.user_id,
                role_code="SYSTEM",
            ).first()
            if not exists:
                hide_marker = AgentChatHistory(
                    session_id=target_sid,
                    user_id=user.user_id,
                    role_code="SYSTEM",
                    content="HIDDEN_BY_USER",
                    retrieval_chunks_json="[]",
                )
                db.session.add(hide_marker)
                db.session.commit()
        except Exception as e:
            db.session.rollback()
            print("세션 숨김 DB 마커 저장 참고:", e)

    is_current = (session.get("assistant_thread_id") == target_sid)
    new_sid = None
    if is_current:
        new_sid = f"ast_{user.username}_{uuid.uuid4().hex[:6]}"
        session["assistant_thread_id"] = new_sid

    return jsonify({
        "status": "SUCCESS",
        "was_current": is_current,
        "new_thread_id": new_sid,
    })


def _run_assistant_background(
    app,
    user_id: int,
    thread_id: str,
    exec_id: int,
    question: str,
    state_input: dict,
):
    """브라우저 페이지 이동과 무관하게 백그라운드 스레드에서 에이전트를 끝까지 실행하고 DB에 커밋합니다."""
    start_ts = time.time()
    with app.app_context():
        try:
            from modules.agent.supervisor import ship_agent_app

            config = {"configurable": {"thread_id": thread_id}}

            # 👇 [수정] invoke 대신 stream(stream_mode="updates")으로 단계별 진행 상태 실시간 기록
            for event in ship_agent_app.stream(state_input, config=config, stream_mode="updates"):
                for node_name in event.keys():
                    try:
                        step_exec = AgentExecution.query.get(exec_id)
                        if step_exec and step_exec.status_code == "RUNNING":
                            step_exec.current_state = str(node_name)
                            db.session.commit()
                    except Exception:
                        db.session.rollback()

            final_snapshot = ship_agent_app.get_state(config)
            result = dict(final_snapshot.values) if final_snapshot and final_snapshot.values else {}
            elapsed_ms = round((time.time() - start_ts) * 1000, 2)

            user = User.query.get(user_id)
            exec_log = AgentExecution.query.get(exec_id)

            answer_text = result.get("final_answer") or "응답을 생성하지 못했습니다."
            task_type = result.get("task_type") or "knowledge_qa"
            retrieved_docs = result.get("retrieved_docs") or []
            generated_doc = result.get("generated_document")

            # 1) 인용된 RAG 문서 출처 정리
            sources = []
            for d in retrieved_docs:
                meta = d.get("metadata") or {}
                sources.append({
                    "doc_code": meta.get("document_code") or "DOC-RAG",
                    "title": meta.get("document_name") or "기술규격 문서",
                    "page": meta.get("page") or 1,
                    "collection_type": meta.get("collection_type") or "PUBLIC",
                })

            # 2) AI 답변 DB 저장
            chunks_to_save = list(sources)
            if generated_doc and generated_doc.get("files"):
                doc_snapshot = {
                    k: generated_doc.get(k)
                    for k in (
                        "title", "status", "approval_status", "execution_type",
                        "work_order_number", "allocated_lot_no", "assigned_vehicle_code",
                        "wire_item_code", "reserve_wire_kg", "parameters",
                        "welding_params_json", "source_refs_json", "external_env_json",
                        "weather", "vision_inspection", "reviewed_by",
                        "erp_resource_status", "erp_commit", "partner_erp_plan",
                        "instruction_text", "content", "files",
                    )
                    if generated_doc.get(k) is not None
                }
                chunks_to_save.append({
                    "_type": "WORK_ORDER_FILES",
                    "files": generated_doc["files"],
                    "approval_status": generated_doc.get("approval_status", "DRAFT"),
                    "doc": doc_snapshot,
                })

            chunks_to_save.append({
                "_type": "UI_RESULT_BLOCKS",
                "task_type": task_type,
                "extracted_params": result.get("extracted_params") or {},
                "erp_resource_status": result.get("erp_resource_status"),
                "vision_data": result.get("vision_data"),
                "partner_erp_plan": result.get("partner_erp_plan"),
                "weather_data": result.get("weather_data"),
                "tool_trace": result.get("tool_trace") or [],
                "reasoning_summary": result.get("agent_reasoning_summary") or "",
            })

            ai_msg = AgentChatHistory(
                session_id=thread_id,
                user_id=user_id,
                role_code="ASSISTANT",
                content=answer_text,
                retrieval_chunks_json=json.dumps(chunks_to_save, ensure_ascii=False, default=str) if chunks_to_save else None,
            )
            db.session.add(ai_msg)

            # 3) 작업지시서 초안(DRAFT) 생성 시 WORK_ORDERS(READY) & WORK_INSTRUCTIONS(DRAFT) 선적재
            draft_work_id = None
            if generated_doc and generated_doc.get("approval_status") == "DRAFT":
                params_draft = generated_doc.get("parameters") or {}
                erp_commit_draft = generated_doc.get("erp_commit") or {}
                is_standalone_draft = bool(generated_doc.get("is_standalone_po"))
                first_blk_d = ErpBlockMaster.query.first()
                block_id_draft = params_draft.get("block_id") or ("DIRECT-PO" if is_standalone_draft else (first_blk_d.block_id if first_blk_d else "UNASSIGNED"))
                base_wo_no = (
                    generated_doc.get("work_order_number")
                    or erp_commit_draft.get("work_order_number")
                    or f"WO-DRAFT-{block_id_draft}"
                )
                draft_wo_no = f"{base_wo_no}-{uuid.uuid4().hex[:4].upper()}"

                raw_thick = str(params_draft.get("thickness", "0" if is_standalone_draft else "20")).upper().replace("T", "").replace("MM", "").strip()
                thickness_val = float(raw_thick) if raw_thick.replace(".", "", 1).isdigit() else (0.0 if is_standalone_draft else 20.0)

                draft_wo = WorkOrder(
                    work_no=draft_wo_no,
                    work_title=generated_doc.get("title") or f"[DRAFT] {block_id_draft} 자동용접 작업지시",
                    work_description=f"AI 어시스턴트 기안 ({user.full_name if user else '담당자'} 요청 · 관리자 결재 대기)",
                    ship_code=params_draft.get("ship_code") or ("N/A" if is_standalone_draft else (first_blk_d.ship_code if first_blk_d else "UNASSIGNED")),
                    block_code=block_id_draft,
                    work_region=params_draft.get("branch_code") or (user.erp_branch_code if user else None) or _get_default_branch_code(user),
                    work_zone=params_draft.get("work_zone") or (user.default_workspace if user else None) or "ZONE-A-BAY1",
                    process_code=params_draft.get("process") or ("N/A" if is_standalone_draft else "FCAW"),
                    material_grade=params_draft.get("material") or ("N/A" if is_standalone_draft else "AH36"),
                    thickness_mm=thickness_val,
                    weld_position=params_draft.get("weld_position") or ("N/A" if is_standalone_draft else "1G"),
                    status_code="READY",
                    requested_by=user_id,
                )
                db.session.add(draft_wo)
                db.session.flush()
                draft_work_id = draft_wo.work_id

                # 외주(OUTSOURCED_B2B) 초안인 경우: 관리자 사전 검토용 발주 주문서(PO) 초안 PDF 함께 생성
                is_draft_outsourced = (
                    generated_doc.get("execution_type") == "OUTSOURCED_B2B"
                    or bool(((generated_doc.get("partner_erp_plan") or {}).get("selected_partner") or {}).get("partner_id"))
                )
                already_has_po_pdf = bool((generated_doc.get("files") or {}).get("download_po_pdf_url"))
                if is_draft_outsourced and not already_has_po_pdf:
                    try:
                        from modules.agent.tools.document_tools import export_outsourcing_po_pdf
                        draft_po_payload = dict(generated_doc)
                        draft_po_payload["work_order_number"] = draft_wo_no
                        draft_po_payload["approval_status"] = "DRAFT"
                        po_draft_raw = export_outsourcing_po_pdf.invoke({"document_data": draft_po_payload})
                        po_draft_res = json.loads(po_draft_raw) if isinstance(po_draft_raw, str) else {}
                        if po_draft_res.get("status") == "SUCCESS":
                            files_dict = generated_doc.setdefault("files", {})
                            files_dict["po_pdf_path"] = po_draft_res.get("po_pdf_path")
                            files_dict["download_po_pdf_url"] = po_draft_res.get("download_po_pdf_url")
                            for c in chunks_to_save:
                                if isinstance(c, dict) and c.get("_type") == "WORK_ORDER_FILES":
                                    c["files"] = files_dict
                                    if isinstance(c.get("doc"), dict):
                                        c["doc"]["files"] = files_dict
                            ai_msg.retrieval_chunks_json = json.dumps(chunks_to_save, ensure_ascii=False, default=str)
                    except Exception as po_draft_err:
                        print(f"⚠️ [PO 초안 PDF 생성 참고]: {po_draft_err}")

                draft_inst = WorkInstruction(
                    work_id=draft_wo.work_id,
                    revision_no="Rev.1",
                    instruction_text=generated_doc.get("instruction_text") or generated_doc.get("content") or "작업지침 본문",
                    welding_params_json=json.dumps(generated_doc.get("welding_params_json") or params_draft, ensure_ascii=False, default=str),
                    source_refs_json=json.dumps(generated_doc.get("source_refs_json") or [], ensure_ascii=False, default=str),
                    external_env_json=json.dumps({
                        "weather": generated_doc.get("external_env_json") or {},
                        "files": generated_doc.get("files") or {},
                        "_draft_payload": generated_doc,
                        "_source_session_id": thread_id,
                    }, ensure_ascii=False, default=str),
                    approval_status="DRAFT",
                    approved_by=None,
                )
                db.session.add(draft_inst)
                db.session.flush()
                generated_doc["work_id"] = draft_wo.work_id
                generated_doc["instruction_id"] = draft_inst.instruction_id

                # [수정] 실제로 초안(draft_wo)이 생성되어 승인 대기(DRAFT) 상태일 때만 결재 요청 메일 발송
                if draft_wo and draft_wo_no and generated_doc.get("approval_status") == "DRAFT":
                    try:
                        from modules.agent.tools.gmail_mcp_client import send_approval_request_email

                        draft_files = generated_doc.get("files") or {}
                        draft_attachments = [
                            p for p in [draft_files.get("pdf_path"), draft_files.get("po_pdf_path")] if p
                        ]
                        requester_str = f"{user.full_name} ({user.erp_employee_code or user.username})"

                        send_approval_request_email(
                            work_order_number=draft_wo_no,
                            title=draft_wo.work_title,
                            requester_label=requester_str,
                            block_id=block_id_draft,
                            branch_name=draft_wo.work_region,
                            work_zone=draft_wo.work_zone,
                            execution_type=generated_doc.get("execution_type") or "IN_HOUSE",
                            summary_text=generated_doc.get("instruction_text") or generated_doc.get("content") or "",
                            pdf_paths=draft_attachments,
                        )
                    except Exception as notify_err:
                        print(f"⚠️ [결재 상신 알림 메일(비차단)]: {notify_err}")   

                # 채팅창 결재 버튼(pending_doc)과 DB 스냅샷에도 work_id / instruction_id 동기화
                for c in chunks_to_save:
                    if isinstance(c, dict) and c.get("_type") == "WORK_ORDER_FILES" and isinstance(c.get("doc"), dict):
                        c["doc"]["work_id"] = draft_wo.work_id
                        c["doc"]["instruction_id"] = draft_inst.instruction_id
                ai_msg.retrieval_chunks_json = json.dumps(chunks_to_save, ensure_ascii=False, default=str)

                draft_inst.external_env_json = json.dumps({
                    "weather": generated_doc.get("external_env_json") or {},
                    "files": generated_doc.get("files") or {},
                    "_draft_payload": generated_doc,
                    "_source_session_id": thread_id,
                }, ensure_ascii=False, default=str)

            # 4) AGENT_EXECUTIONS 상태를 RUNNING -> SUCCESS로 갱신
            intent_map = {
                "doc_generation": "DOC_GENERATION",
                "knowledge_qa": "WPS_LOOKUP",
                "vision_inspection_qa": "MCP_TOOL",
                "erp_resource_qa": "MCP_TOOL",
                "hr_worker_qa": "MCP_TOOL",
                "hr_safety_qa": "MCP_TOOL",
                "scm_partner_qa": "MCP_TOOL",
            }
            if exec_log:
                exec_log.work_id = draft_work_id
                exec_log.intent_type = intent_map.get(task_type, "QA")
                exec_log.current_state = f"COMPLETED_{task_type.upper()}"
                exec_log.selected_tools_json = json.dumps(
                    {"task_type": task_type, "rag_count": len(sources)}, ensure_ascii=False
                )
                exec_log.execution_time_ms = elapsed_ms
                exec_log.status_code = "SUCCESS"
                exec_log.error_message = None

            db.session.commit()
            print(f"✅ [Background Agent 완료] session={thread_id}, task={task_type}, 소요={elapsed_ms}ms")

        except Exception as e:
            db.session.rollback()
            traceback.print_exc()
            try:
                exec_log = AgentExecution.query.get(exec_id)
                if exec_log:
                    exec_log.status_code = "FAILED"
                    exec_log.current_state = "ERROR"
                    exec_log.error_message = f"{type(e).__name__}: {str(e)}"
                err_msg = AgentChatHistory(
                    session_id=thread_id,
                    user_id=user_id,
                    role_code="ASSISTANT",
                    content=f"❌ 에이전트 처리 중 오류가 발생했습니다: {type(e).__name__} - {str(e)}",
                    retrieval_chunks_json=None,
                )
                db.session.add(err_msg)
                db.session.commit()
            except Exception:
                db.session.rollback()
        finally:
            db.session.remove()


@service_bp.route("/api/assistant/ask", methods=["POST"])
@login_required
def ask_assistant_agent():
    """사용자 질문을 DB에 기록하고 백그라운드 스레드에서 에이전트를 비동기 실행합니다."""
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or data.get("message") or "").strip()
        page_ctx = data.get("page_context") or {}
        if not question:
            return jsonify({"status": "ERROR", "message": "질문이나 작업지시 내용을 입력해 주세요."}), 400

        user = User.query.get_or_404(session["user_id"])
        if data.get("new_session"):
            thread_id = f"wf_{user.username}_{uuid.uuid4().hex[:8]}"
        else:
            thread_id = (data.get("session_id") or "").strip() or session.get("assistant_thread_id") or f"ast_{user.username}_{uuid.uuid4().hex[:6]}"
        session["assistant_thread_id"] = thread_id

        # 이미 같은 대화방에 실행 중인 작업이 있으면 중복 실행 방지
        if _check_session_running(user.user_id, thread_id):
            return jsonify({
                "status": "RUNNING",
                "session_id": thread_id,
                "message": "이전 요청을 백그라운드에서 처리 중입니다.",
            }), 202

        # 1) 사용자 질문 및 RUNNING 상태 실행 로그 즉시 커밋
        user_msg = AgentChatHistory(
            session_id=thread_id,
            user_id=user.user_id,
            role_code="USER",
            content=question,
            retrieval_chunks_json=None,
        )
        db.session.add(user_msg)

        exec_log = AgentExecution(
            session_id=thread_id,
            user_id=user.user_id,
            work_id=None,
            intent_type="QA",
            current_state="RUNNING_AGENT",
            selected_tools_json=None,
            execution_time_ms=0.0,
            status_code="RUNNING",
            error_message=None,
        )
        db.session.add(exec_log)
        db.session.commit()

        ctx_prefix = ""
        if isinstance(page_ctx, dict) and page_ctx:
            ctx_parts = []
            if page_ctx.get("page_title"):
                ctx_parts.append(f"현재화면={page_ctx.get('page_title')}")
            sel = page_ctx.get("selected_work") or {}
            if isinstance(sel, dict) and sel.get("block_id"):
                ctx_parts.append(f"선택대상={sel.get('branch_code','')}/{sel.get('ship_code','')}/{sel.get('block_id','')}")
            if page_ctx.get("summary"):
                ctx_parts.append(f"화면요약={page_ctx.get('summary')}")
            if ctx_parts:
                ctx_prefix = "[현재 화면 컨텍스트: " + " | ".join(ctx_parts) + "]\n"

        enriched_query = f"{ctx_prefix}{question}" if ctx_prefix else question
        ctx_branch = (
            (page_ctx.get("selected_work") or {}).get("branch_code")
            or page_ctx.get("branch_code")
            or user.erp_branch_code
            or _get_default_branch_code(user)
        )

        state_input = {
            "messages": [HumanMessage(content=enriched_query)],
            "user_query": enriched_query,
            "user_id": str(user.user_id),
            "user_name": f"{user.full_name} ({user.erp_employee_code or user.username})",
            "user_role": user.role,
            "user_type": user.user_type,
            "can_access_erp": user.can_access_erp or "N",
            "can_access_internal_rag": user.can_access_internal_rag or "N",
            "can_control_robot": user.can_control_robot or "N",
            "default_workspace": user.default_workspace or "ZONE-A-BAY1",
            "branch_code": ctx_branch,
            "hitl_approved": False,
            "generated_document": None,
            "weather_data": None,
            "vision_data": None,
            "partner_erp_plan": None,
            "team2_plan": None,
        }

        app_obj = current_app._get_current_object()
        worker = threading.Thread(
            target=_run_assistant_background,
            args=(app_obj, user.user_id, thread_id, exec_log.execution_id, question, state_input),
            daemon=True,
        )
        worker.start()

        return jsonify({
            "status": "RUNNING",
            "session_id": thread_id,
            "execution_id": exec_log.execution_id,
        }), 202

    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({
            "status": "ERROR",
            "message": f"요청 접수 중 오류 발생: {type(e).__name__} - {str(e)}"
        }), 500

STEP_STATUS_LABELS = {
    "RUNNING_AGENT": "요청 내용과 기존 문서 중복 여부를 확인하고 있습니다.",
    "top_supervisor_init": "작업 조건과 세션 상태를 초기화하고 있습니다.",
    "team1_analysis": "야드 자재 재고·작업자 자격·설비 가용 상태를 점검하고 있습니다.",
    "team2_search": "용접 규격서(RAG)·현장 기상·비전 정보·협력사 자원을 조회하고 있습니다.",
    "team2_knowledge": "용접 규격서(RAG)·현장 기상·비전 정보·협력사 자원을 조회하고 있습니다.",
    "team3_docgen": "안전 기준 검증 후 통합 실행안과 문서를 조립하고 있습니다.",
}


@service_bp.route("/api/assistant/status/<session_id>", methods=["GET"])
@login_required
def get_assistant_execution_status(session_id):
    """백그라운드 에이전트 실행 상태 및 완료 결과를 조회하는 경량 폴링 API"""
    user = User.query.get_or_404(session["user_id"])

    if _check_session_running(user.user_id, session_id):
        running_exec = (
            AgentExecution.query
            .filter_by(user_id=user.user_id, session_id=session_id, status_code="RUNNING")
            .order_by(AgentExecution.execution_id.desc())
            .first()
        )
        raw_state = (running_exec.current_state if running_exec else "") or "RUNNING_AGENT"
        step_msg = STEP_STATUS_LABELS.get(raw_state, "🤖 AI 어시스턴트가 에이전트 협업 작업을 처리 중입니다...")
        return jsonify({
            "status": "RUNNING",
            "session_id": session_id,
            "current_state": raw_state,
            "step_message": step_msg,
        })

    latest_exec = (
        AgentExecution.query
        .filter_by(user_id=user.user_id, session_id=session_id)
        .order_by(AgentExecution.execution_id.desc())
        .first()
    )
    if not latest_exec:
        return jsonify({"status": "IDLE", "session_id": session_id})

    last_ai = (
        AgentChatHistory.query
        .filter_by(user_id=user.user_id, session_id=session_id, role_code="ASSISTANT")
        .order_by(AgentChatHistory.message_id.desc())
        .first()
    )

    sources = []
    saved_files = None
    saved_status = "DRAFT"
    saved_doc = None
    result_blocks = {}
    if last_ai and last_ai.retrieval_chunks_json:
        raw_list = _parse_json_or_literal(last_ai.retrieval_chunks_json)
        for item in raw_list:
            if isinstance(item, dict) and item.get("_type") == "WORK_ORDER_FILES":
                saved_files = item.get("files")
                saved_status = item.get("approval_status", "DRAFT")
                saved_doc = item.get("doc")
            elif isinstance(item, dict) and item.get("_type") == "UI_RESULT_BLOCKS":
                result_blocks = item
            else:
                sources.append(item)

    if saved_doc and saved_files and not saved_doc.get("files"):
        saved_doc["files"] = saved_files

    task_type = "knowledge_qa"
    if latest_exec.selected_tools_json:
        try:
            meta = json.loads(_read_clob(latest_exec.selected_tools_json))
            task_type = meta.get("task_type") or task_type
        except Exception:
            pass

    return jsonify({
        "status": latest_exec.status_code,
        "session_id": session_id,
        "answer": _read_clob(last_ai.content) if last_ai else (latest_exec.error_message or "응답이 없습니다."),
        "task_type": task_type,
        "execution_time_ms": latest_exec.execution_time_ms or 0.0,
        "sources": sources,
        "files": saved_files,
        "approval_status": saved_status,
        "generated_document": saved_doc,
        "result_blocks": result_blocks,
    })


@service_bp.route("/api/assistant/approve", methods=["POST"])
@login_required
def approve_work_instruction():
    """생성된 작업지시서 초안(DRAFT)을 관리자가 최종 승인(APPROVED)하여 ERP 및 DB에 커밋"""
    user = User.query.get_or_404(session["user_id"])
    if user.role not in ("ADMIN", "MANAGER"):
        return jsonify({"status": "ERROR", "message": "작업지시 최종 승인은 관리자(ADMIN/MANAGER)만 가능합니다."}), 403

    payload = request.get_json(silent=True) or {}
    action = payload.get("action", "APPROVE")
    doc = payload.get("doc") or {}

    work_id = payload.get("work_id") or doc.get("work_id")
    instruction_id = payload.get("instruction_id") or doc.get("instruction_id")

    target_wo = WorkOrder.query.get(work_id) if work_id else None
    target_inst = WorkInstruction.query.get(instruction_id) if instruction_id else (
        WorkInstruction.query.filter_by(work_id=target_wo.work_id).order_by(WorkInstruction.instruction_id.desc()).first()
        if target_wo else None
    )

    source_session_id = None
    if target_inst and target_inst.external_env_json:
        raw_env = _read_clob(target_inst.external_env_json).strip()
        env_meta = None
        if raw_env:
            try:
                env_meta = json.loads(raw_env)
            except Exception:
                try:
                    env_meta = ast.literal_eval(raw_env)
                except Exception as e:
                    print(f"⚠️ [_draft_payload 복원 경고]: {e}")
        if isinstance(env_meta, dict):
            saved_payload = env_meta.get("_draft_payload")
            if isinstance(saved_payload, dict) and not doc.get("welding_params_json"):
                doc = saved_payload
            source_session_id = env_meta.get("_source_session_id")

    # 2차 안전장치: external_env_json에서 복원 못 했을 경우 해당 세션의 AI 메시지 청크에서 복원
    if not doc.get("welding_params_json") and source_session_id:
        last_draft_ai = (
            AgentChatHistory.query
            .filter_by(session_id=source_session_id, role_code="ASSISTANT")
            .order_by(AgentChatHistory.message_id.desc())
            .first()
        )
        if last_draft_ai:
            for item in _parse_json_or_literal(last_draft_ai.retrieval_chunks_json):
                if isinstance(item, dict) and item.get("_type") == "WORK_ORDER_FILES" and isinstance(item.get("doc"), dict):
                    doc = item["doc"]
                    break

    params = doc.get("parameters") or {}
    partner_plan = doc.get("partner_erp_plan") or {}
    is_outsourced = (
        doc.get("execution_type") == "OUTSOURCED_B2B"
        or bool((partner_plan.get("selected_partner") or {}).get("partner_id"))
        or bool((doc.get("erp_commit") or {}).get("is_outsourced"))
    )

    if not target_wo and doc.get("work_order_number"):
        target_wo = WorkOrder.query.filter(
            WorkOrder.work_no.like(f"{doc.get('work_order_number')}%")
        ).first()
    if target_wo and not target_inst:
        target_inst = (
            WorkInstruction.query.filter_by(work_id=target_wo.work_id)
            .order_by(WorkInstruction.instruction_id.desc())
            .first()
        )

    if action == "REJECT":
        if target_inst:
            target_inst.approval_status = "REJECTED"
            target_inst.approved_by = user.user_id
        if target_wo:
            target_wo.status_code = "SUSPENDED"
            target_wo.work_title = str(target_wo.work_title or "").replace("[DRAFT]", "[REJECTED]")

        # 👇 이 블록은 그대로 유지하세요! (채팅창 새로고침 시 결재 박스 재노출 차단)
        reject_sid = source_session_id or session.get("assistant_thread_id")
        if reject_sid:
            last_ai_msg = (
                AgentChatHistory.query
                .filter_by(session_id=reject_sid, role_code="ASSISTANT")
                .order_by(AgentChatHistory.message_id.desc())
                .first()
            )
            if last_ai_msg:
                existing_chunks = _parse_json_or_literal(last_ai_msg.retrieval_chunks_json)
                for c in existing_chunks:
                    if isinstance(c, dict) and c.get("_type") == "WORK_ORDER_FILES":
                        c["approval_status"] = "REJECTED"
                        c["doc"] = None
                last_ai_msg.retrieval_chunks_json = json.dumps(existing_chunks, ensure_ascii=False, default=str)
                last_ai_msg.content = (
                    _read_clob(last_ai_msg.content)
                    + f"\n\n> ❌ **[요청 반려 완료 (`STATUS: REJECTED`)]** {user.full_name} 관리자에 의해 요청이 반려(Rollback)되었습니다."
                )

        db.session.commit()
        return jsonify({
            "status": "REJECTED",
            "message": f"❌ [요청 반려] {user.full_name} 관리자에 의해 요청이 반려 처리되었습니다."
        })

    try:
        from modules.agent.tools.erp_tools import commit_erp_work_order
        from modules.agent.tools.document_tools import export_work_order_file, export_outsourcing_po_pdf, log_inspection_history
        from modules.agent.tools.gmail_mcp_client import send_work_order_dispatch_email

        selected_partner = (partner_plan.get("selected_partner") or {}) if partner_plan else {}
        erp_commit_draft = doc.get("erp_commit") or {}
        erp_res_status = doc.get("erp_resource_status") or {}
        worker_qual = erp_res_status.get("worker_qualification") or {}

        lot_no = doc.get("allocated_lot_no") or erp_commit_draft.get("allocated_lot_no") or erp_res_status.get("assigned_lot_no") or "LOT-E71T1-202610-01"
        first_blk_a = ErpBlockMaster.query.first()
        block_id = params.get("block_id") or (target_wo.block_code if target_wo else (first_blk_a.block_id if first_blk_a else "UNASSIGNED"))
        worker_id = erp_commit_draft.get("worker_id") or worker_qual.get("worker_id")
        worker_name = erp_commit_draft.get("worker_name") or params.get("worker_name") or worker_qual.get("worker_name") or user.full_name
        vehicle_code = doc.get("assigned_vehicle_code") or erp_commit_draft.get("assigned_vehicle_code") or erp_res_status.get("assigned_vehicle_code")
        wire_code = doc.get("wire_item_code") or erp_res_status.get("selected_wire_code")
        req_wire_kg = doc.get("reserve_wire_kg") or erp_commit_draft.get("reserved_wire_kg") or 15.0
        branch_default_zone = (erp_res_status.get("branch_info") or {}).get("default_zone")
        delivery_zone = (
            params.get("work_zone")
            or branch_default_zone
            or (target_wo.work_zone if target_wo else None)
            or user.default_workspace
            or "ZONE-A-BAY1"
        )
        resolved_partner_id = (
            selected_partner.get("partner_id")
            or erp_commit_draft.get("partner_id")
            or "COMP-HQ"
        )

        commit_res = commit_erp_work_order.invoke({
            "block_id": block_id,
            "material": params.get("material") or (target_wo.material_grade if target_wo else "AH36"),
            "allocated_lot_no": lot_no,
            "worker_name": worker_name,
            "worker_id": worker_id,
            "assigned_vehicle_code": vehicle_code,
            "wire_item_code": wire_code,
            "is_outsourced": is_outsourced,
            "partner_id": resolved_partner_id,
            "reserve_wire_kg": req_wire_kg,
            "delivery_yard_zone": delivery_zone,
        })

        wo_no = commit_res.get("work_order_number") or f"WO-{uuid.uuid4().hex[:8].upper()}"
        approver_label = f"{user.full_name} ({user.erp_employee_code or user.username})"

        if not target_wo and doc.get("work_order_number"):
            target_wo = WorkOrder.query.filter_by(work_no=doc.get("work_order_number")).first()

        requester_user = User.query.get(target_wo.requested_by) if (target_wo and target_wo.requested_by) else user
        reviewer_label = f"{requester_user.full_name} ({requester_user.erp_employee_code or requester_user.username})"

        final_doc_data = dict(doc)
        final_doc_data["work_order_number"] = wo_no
        final_doc_data["erp_commit"] = commit_res
        final_doc_data["status"] = "APPROVED"
        final_doc_data["approval_status"] = "APPROVED"
        final_doc_data["reviewed_by"] = reviewer_label
        final_doc_data["approved_by"] = approver_label
        if not final_doc_data.get("weather") and final_doc_data.get("external_env_json"):
            final_doc_data["weather"] = final_doc_data["external_env_json"]

        is_standalone_po = bool(
            doc.get("is_standalone_po")
            or erp_commit_draft.get("is_standalone_po")
            or (doc.get("welding_params_json") or {}).get("standalone_po")
        )

        if is_standalone_po:
            file_links = {"status": "SUCCESS"}
        else:
            export_raw = export_work_order_file.invoke({
                "document_data": final_doc_data,
                "file_prefix": f"weld_order_{block_id}",
            })
            file_links = json.loads(export_raw) if isinstance(export_raw, str) else {}

        # 외주(OUTSOURCED_B2B) 작업 또는 단독 발주인 경우: 1페이지 공식 발주 주문서(PO) PDF 생성
        po_pdf_path = None
        resolved_po_no = None
        if is_outsourced or is_standalone_po:
            po_raw = export_outsourcing_po_pdf.invoke({
                "document_data": final_doc_data,
            })
            po_res = json.loads(po_raw) if isinstance(po_raw, str) else {}
            if po_res.get("status") == "SUCCESS":
                resolved_po_no = po_res.get("po_number")
                po_pdf_path = po_res.get("po_pdf_path")
                file_links["po_pdf_path"] = po_pdf_path
                file_links["download_po_pdf_url"] = po_res.get("download_po_pdf_url")

        # 해외 공개 Gmail MCP (@gongrzhe/server-gmail-autoauth-mcp) 자동 메일 발송
        gmail_result = None
        if is_outsourced or is_standalone_po:
            attachments_to_send = [
                p for p in [file_links.get("pdf_path"), po_pdf_path] if p
            ]
            partner_display_name = (
                selected_partner.get("partner_name")
                or resolved_partner_id
                or "외주협력사"
            )
            gmail_result = send_work_order_dispatch_email(
                work_order_number=wo_no,
                po_number=resolved_po_no,
                partner_name=partner_display_name,
                block_id=block_id,
                branch_name=params.get("branch_code") or user.erp_branch_code or _get_default_branch_code(user),
                work_zone=delivery_zone,
                process_code=params.get("process") or "-",
                material_grade=params.get("material") or "-",
                thickness_mm=params.get("thickness") or "-",
                worker_name=worker_name,
                approver_label=approver_label,
                pdf_paths=attachments_to_send,
            )

        log_inspection_history.invoke({
            "session_id": wo_no,
            "query": f"HITL 관리자 승인 완료 ({approver_label}): {block_id}",
            "status": "APPROVED",
        })

        raw_thick = str(params.get("thickness", "20")).upper().replace("T", "").replace("MM", "").strip()
        thickness_val = float(raw_thick) if raw_thick.replace(".", "", 1).isdigit() else 20.0
        clean_title = str(doc.get("title") or f"{block_id} 자동용접 작업지시").replace("[DRAFT]", "[APPROVED]")

        if target_wo:
            target_wo.work_no = wo_no
            target_wo.work_title = clean_title
            target_wo.work_description = f"AI 어시스턴트 발행 ({approver_label} 승인)"
            target_wo.work_zone = delivery_zone
            target_wo.status_code = "IN_PROGRESS"
            target_wo.updated_at = datetime.utcnow()
            db.session.flush()
            active_wo = target_wo
        else:
            active_wo = WorkOrder(
                work_no=wo_no,
                work_title=clean_title,
                work_description=f"AI 어시스턴트 발행 ({approver_label} 승인)",
                ship_code=params.get("ship_code") or (first_blk_a.ship_code if first_blk_a else "UNASSIGNED"),
                block_code=block_id,
                work_region=params.get("branch_code") or user.erp_branch_code or _get_default_branch_code(user),
                work_zone=delivery_zone,
                process_code=params.get("process") or "FCAW",
                material_grade=params.get("material") or "AH36",
                thickness_mm=thickness_val,
                weld_position=params.get("weld_position") or "1G",
                status_code="IN_PROGRESS",
                requested_by=user.user_id,
            )
            db.session.add(active_wo)
            db.session.flush()

        if not target_inst:
            target_inst = (
                WorkInstruction.query
                .filter_by(work_id=active_wo.work_id)
                .order_by(WorkInstruction.instruction_id.desc())
                .first()
            )

        if target_inst:
            target_inst.approval_status = "APPROVED"
            target_inst.approved_by = user.user_id
            try:
                prev_env = json.loads(_read_clob(target_inst.external_env_json)) if target_inst.external_env_json else {}
                if not isinstance(prev_env, dict):
                    prev_env = {"weather": prev_env}
            except Exception:
                prev_env = {}
            prev_env["files"] = file_links
            target_inst.external_env_json = json.dumps(prev_env, ensure_ascii=False, default=str)
        else:
            target_inst = WorkInstruction(
                work_id=active_wo.work_id,
                revision_no="Rev.1",
                instruction_text=doc.get("instruction_text") or doc.get("content") or "작업지침 본문",
                welding_params_json=json.dumps(doc.get("welding_params_json") or params, ensure_ascii=False),
                source_refs_json=json.dumps(doc.get("source_refs_json") or [], ensure_ascii=False),
                external_env_json=json.dumps({
                    "weather": doc.get("external_env_json") or {},
                    "files": file_links,
                }, ensure_ascii=False),
                approval_status="APPROVED",
                approved_by=user.user_id,
            )
            db.session.add(target_inst)

        # 기안자의 원래 대화방 말풍선도 APPROVED 상태로 동기화
        target_sid = source_session_id or session.get("assistant_thread_id")
        if target_sid and file_links:
            last_ai_msg = (
                AgentChatHistory.query
                .filter_by(session_id=target_sid, role_code="ASSISTANT")
                .order_by(AgentChatHistory.message_id.desc())
                .first()
            )
            if last_ai_msg:
                existing_chunks = _parse_json_or_literal(last_ai_msg.retrieval_chunks_json)
                existing_chunks = [
                    c for c in existing_chunks
                    if not (isinstance(c, dict) and c.get("_type") == "WORK_ORDER_FILES")
                ]
                existing_chunks.append({
                    "_type": "WORK_ORDER_FILES",
                    "files": file_links,
                    "approval_status": "APPROVED",
                    "doc": None,
                })
                last_ai_msg.retrieval_chunks_json = json.dumps(existing_chunks, ensure_ascii=False, default=str)

                prev_content = _read_clob(last_ai_msg.content)
                draft_wo_no = erp_commit_draft.get("work_order_number") or f"WO-DRAFT-{block_id}"
                updated_content = prev_content.replace(draft_wo_no, wo_no)

                mail_note = ""
                if gmail_result and gmail_result.get("status") == "SUCCESS":
                    to_addr = ", ".join(gmail_result.get("to") or [])
                    mail_note = f" (📧 Gmail MCP 발송 완료: `{to_addr}`, 발주서 `{resolved_po_no}` 첨부)"

                draft_notice_prefix = "> ⚠️ **[관리자 결재 대기 (`APPROVAL_STATUS: DRAFT`)]**"
                committed_notice = (
                    f"> ✅ **[원장 커밋 완료 (`STATUS: COMMITTED`)]** {approver_label} 결재로 "
                    f"공식 작업지시 번호 `{wo_no}`의 `MES_WORK_ORDER_DISPATCH` 배정 및 자재 선점이 확정되었습니다.{mail_note}"
                )

                if draft_notice_prefix in updated_content:
                    updated_content = updated_content.split(draft_notice_prefix)[0].rstrip() + "\n\n" + committed_notice
                elif "[원장 커밋 완료" not in updated_content:
                    updated_content = updated_content + "\n\n" + committed_notice

                last_ai_msg.content = updated_content
        db.session.commit()

        res_msg = f"✅ [최종 승인 완료] {approver_label} 결재로 작업번호 [{wo_no}] ERP 자원 배정 및 지침서 등록이 확정되었습니다!"
        if gmail_result and gmail_result.get("status") == "SUCCESS":
            res_msg += f" (📧 협력사 메일 및 발주서 [{resolved_po_no}] 자동 발송 완료)"

        return jsonify({
            "status": "APPROVED",
            "message": res_msg,
            "files": file_links,
            "gmail_dispatch": gmail_result,
        })
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"status": "ERROR", "message": f"승인 커밋 오류: {str(e)}"}), 500


@service_bp.route("/api/files/download/<date_str>/<filename>", methods=["GET"])
@login_required
def download_generated_work_order(date_str, filename):
    target_dir = PROJECT_ROOT / "generated_docs" / "outputs" / date_str
    target_file = target_dir / filename
    if not target_file.exists():
        abort(404)

    is_pdf = filename.lower().endswith(".pdf")
    return send_from_directory(
        str(target_dir),
        filename,
        as_attachment=not is_pdf,
    )


# =====================================================================
# 2. 공통 유틸리티 (CAD/WORK_PLANS 안전 조회 · 문서 파일 · 진척률)
# =====================================================================
_KST = timezone(timedelta(hours=9))
_NON_HULL_CAD_KEYWORDS = ("propeller", "prop_", "_prop", "kp505", "kp458", "rudder", "rud_", "_rud", "blade", "nozzle", "boss", "hub", "skeg")


def _now_kst() -> datetime:
    return datetime.now(_KST)


def _today_kst_str() -> str:
    return _now_kst().strftime("%Y-%m-%d")


def _to_kst(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).astimezone(_KST)
    return dt.astimezone(_KST)


def _fmt_kst(dt: datetime | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    kst_dt = _to_kst(dt)
    return kst_dt.strftime(fmt) if kst_dt else "-"


def _is_non_hull_cad_file(file_name: str | None, storage_uri: str | None = None) -> bool:
    hay = f"{file_name or ''} {storage_uri or ''}".lower()
    return any(k in hay for k in _NON_HULL_CAD_KEYWORDS)


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _calc_nav_eta(
    origin_lat,
    origin_lng,
    dest_lat,
    dest_lng,
    base_dt: datetime | None = None,
    prep_hours: float = 1.0,
) -> dict:
    """출발지(협력사)와 도착지(야드) DB 좌표를 이용해 도로 굴곡계수(1.28) · 화물차 평균속도(55km/h) 기반 네비게이션 ETA 계산"""
    base_kst = _to_kst(base_dt) or _now_kst()
    straight_km = None
    road_km = None
    drive_hours = 1.0

    try:
        if all(v is not None for v in (origin_lat, origin_lng, dest_lat, dest_lng)):
            straight_km = round(_haversine_km(float(origin_lat), float(origin_lng), float(dest_lat), float(dest_lng)), 1)
            road_km = round(max(1.5, straight_km * 1.28), 1)
            # 화물/중량물 운송 평균 시속 55km/h (최소 이동시간 20분)
            drive_hours = max(20.0 / 60.0, road_km / 55.0)
    except Exception:
        straight_km = None
        road_km = None
        drive_hours = 1.0

    total_hours = float(prep_hours or 0.0) + drive_hours
    eta_kst = base_kst + timedelta(hours=total_hours)
    drive_min = int(round(drive_hours * 60))
    drive_h_part, drive_m_part = divmod(drive_min, 60)
    drive_label = f"{drive_h_part}시간 {drive_m_part}분" if drive_h_part else f"{drive_m_part}분"

    return {
        "straight_km": straight_km,
        "road_km": road_km,
        "drive_minutes": drive_min,
        "drive_label": drive_label,
        "prep_hours": round(float(prep_hours or 0.0), 1),
        "total_hours": round(total_hours, 2),
        "eta_kst": eta_kst,
        "eta_str": eta_kst.strftime("%Y-%m-%d %H:%M"),
    }


def _sync_work_plan_to_erp_block_master(plan, drawing=None) -> bool:
    """WORK_PLANS에 등록된 블록이 ERP_BLOCK_MASTER에 없으면 DB 값 그대로 자동 동기화(UPSERT)하여 작업지시 생성 차단 방지"""
    if not plan or not plan.block_code:
        return False
    try:
        exists = ErpBlockMaster.query.filter_by(block_id=plan.block_code).first()
        if exists:
            return True
        ship_code = (drawing.ship_code if drawing and drawing.ship_code else None)
        if not ship_code and plan.drawing_id:
            dwg = CadDrawing.query.filter_by(drawing_id=plan.drawing_id).first()
            ship_code = dwg.ship_code if dwg else None
        if not ship_code:
            ship_code = f"HN-{plan.drawing_id[:10]}" if plan.drawing_id else "HN-PLAN"

        thick_val = float(plan.thickness_mm) if plan.thickness_mm is not None else 20.0
        thick_str = f"{int(thick_val)}T" if thick_val.is_integer() else f"{thick_val}T"

        new_blk = ErpBlockMaster(
            block_id=plan.block_code[:50],
            ship_code=ship_code[:50],
            block_name=(plan.block_name or plan.block_code)[:100],
            steel_type=(plan.material_grade or "AH36")[:50],
            thickness_mm=thick_str[:20],
            joint_type=(plan.joint_type or "BUTT")[:50],
            process=(plan.process_code or "FCAW")[:50],
            target_dock=(plan.work_zone or "ZONE-A-BAY1")[:50],
            branch_code=(plan.branch_code or "BR-ULSAN-01")[:50],
            weight_ton=plan.est_weight_ton if plan.est_weight_ton is not None else 10.0,
        )
        db.session.add(new_blk)
        db.session.commit()
        return True
    except Exception:
        db.session.rollback()
        return False


def _calc_cert_days_left(expiry_str: str | None) -> int | None:
    if not expiry_str:
        return None
    s = str(expiry_str).strip()[:10]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            dt = datetime.strptime(s, fmt).date()
            return (dt - _now_kst().date()).days
        except ValueError:
            continue
    return None


def _format_cert_dday(days: int | None) -> tuple[str, str]:
    """만료일 D-Day 표시 문구 및 상태 코드 반환 (D--683 같은 이중 마이너스 방지)"""
    if days is None:
        return "-", "NORMAL"
    if days < 0:
        return f"만료됨 ({abs(days)}일 경과)", "EXPIRED"
    if days == 0:
        return "오늘 만료", "EXPIRED"
    if days <= 90:
        return f"D-{days} (임박)", "EXPIRING"
    return f"정상 (D-{days})", "NORMAL"


def _resolve_doc_abs_path(storage_uri: str | None) -> Path | None:
    uri = (storage_uri or "").strip()
    if not uri or uri.startswith(("http://", "https://", "s3://")) or uri == "ARCHIVED_HISTORY":
        return None
    p = Path(uri)
    abs_p = p if p.is_absolute() else (PROJECT_ROOT / uri)
    try:
        resolved = abs_p.resolve()
        if PROJECT_ROOT.resolve() in resolved.parents or resolved == PROJECT_ROOT.resolve():
            return resolved
    except Exception:
        return None
    return None


def _load_env_meta(inst) -> dict:
    if not inst or not inst.external_env_json:
        return {}
    raw_str = _read_clob(inst.external_env_json).strip()
    if not raw_str:
        return {}
    try:
        meta = json.loads(raw_str)
    except Exception:
        try:
            meta = ast.literal_eval(raw_str)
        except Exception:
            meta = {}
    return meta if isinstance(meta, dict) else {}


def _extract_inst_pdf_url(inst) -> str:
    """물리 PDF가 있으면 해당 경로를, 없으면 DB 원장 기반 공식 작업지시서 뷰 URL을 반환해 클릭 불가를 원천 차단"""
    if not inst:
        return ""
    meta = _load_env_meta(inst)
    files = meta.get("files") or (meta.get("_draft_payload") or {}).get("files") or {}
    raw_url = files.get("download_pdf_url") or ""
    if raw_url:
        parts = raw_url.strip("/").split("/")
        if len(parts) >= 2:
            date_str, filename = parts[-2], parts[-1]
            if (PROJECT_ROOT / "generated_docs" / "outputs" / date_str / filename).exists():
                return raw_url if raw_url.startswith("/service/") else f"/service{raw_url}"
    if getattr(inst, "instruction_id", None):
        return f"/service/work-instruction/{inst.instruction_id}/view"
    return ""


def _extract_inst_po_pdf_url(inst) -> str:
    if not inst:
        return ""
    meta = _load_env_meta(inst)
    files = meta.get("files") or (meta.get("_draft_payload") or {}).get("files") or {}
    raw_url = files.get("download_po_pdf_url") or ""
    if not raw_url:
        return ""
    parts = raw_url.strip("/").split("/")
    if len(parts) >= 2:
        date_str, filename = parts[-2], parts[-1]
        if not (PROJECT_ROOT / "generated_docs" / "outputs" / date_str / filename).exists():
            return ""
    return raw_url if raw_url.startswith("/service/") else f"/service{raw_url}"


def _is_plan_scheduled_today(plan, today_str: str) -> bool:
    """WORK_PLANS의 계획 시작/종료일(또는 생성일 KST)을 기준으로 오늘 작업 예정 블록인지 판별"""
    s_date = (plan.planned_start_date or "").strip()[:10]
    e_date = (plan.planned_end_date or "").strip()[:10]
    if s_date and e_date:
        return s_date <= today_str <= e_date
    if s_date:
        return s_date == today_str
    if e_date:
        return today_str <= e_date
    if plan.created_at:
        return _fmt_kst(plan.created_at, "%Y-%m-%d") == today_str
    return False


def _safe_query_work_plans(branch_code: str | None = None, states: list[str] | None = None, today_only: bool = False):
    """CAD_DRAWINGS / WORK_PLANS DDL 미적용 DB에서도 안전하게 빈 목록 반환"""
    try:
        with db.session.begin_nested():
            q = WorkPlan.query
            if states:
                q = q.filter(WorkPlan.plan_state.in_(states))
            else:
                q = q.filter(WorkPlan.plan_state.in_(["PENDING", "IN_PROGRESS"]))
            if branch_code:
                q = q.filter(WorkPlan.branch_code == branch_code)
            rows = q.order_by(WorkPlan.priority_rank.asc(), WorkPlan.created_at.asc()).all()
            if today_only:
                today_str = _today_kst_str()
                rows = [p for p in rows if _is_plan_scheduled_today(p, today_str)]
            return rows
    except Exception:
        return []


def _safe_query_cad_drawings(limit: int = 10):
    try:
        with db.session.begin_nested():
            return CadDrawing.query.order_by(CadDrawing.created_at.desc()).limit(limit).all()
    except Exception:
        return []


def _safe_get_wo_progress_map(work_ids: list[int]) -> dict[int, dict]:
    """WORK_ORDERS.PROGRESS_PCT / PLAN_ID 존재 시 읽고, 미적용 DB면 안전 fallback"""
    if not work_ids:
        return {}
    try:
        with db.session.begin_nested():
            placeholders = ", ".join(f":id_{i}" for i in range(len(work_ids)))
            params = {f"id_{i}": int(wid) for i, wid in enumerate(work_ids)}
            rows = db.session.execute(
                text(f"SELECT WORK_ID, PLAN_ID, PROGRESS_PCT FROM WORK_ORDERS WHERE WORK_ID IN ({placeholders})"),
                params,
            ).fetchall()
            return {
                int(r[0]): {"plan_id": r[1], "progress_pct": int(r[2]) if r[2] is not None else None}
                for r in rows
            }
    except Exception:
        return {}


def _safe_update_wo_progress(work_id: int, progress_pct: int) -> bool:
    try:
        with db.session.begin_nested():
            db.session.execute(
                text("UPDATE WORK_ORDERS SET PROGRESS_PCT = :pct WHERE WORK_ID = :wid"),
                {"pct": int(progress_pct), "wid": int(work_id)},
            )
        return True
    except Exception:
        return False


def _resolve_progress_pct(wo, latest_inst, prog_map: dict) -> tuple[int, list[dict]]:
    """DB 컬럼(PROGRESS_PCT) 및 지침서 메타(_progress_meta)를 통합해 정확한 진척률·이력 반환"""
    meta = _load_env_meta(latest_inst)
    pmeta = meta.get("_progress_meta") if isinstance(meta.get("_progress_meta"), dict) else {}
    timeline = pmeta.get("timeline") if isinstance(pmeta.get("timeline"), list) else []

    db_pct = prog_map.get(wo.work_id, {}).get("progress_pct")
    if db_pct is not None:
        return int(db_pct), timeline
    if "progress_pct" in pmeta and pmeta["progress_pct"] is not None:
        return int(pmeta["progress_pct"]), timeline
    if wo.status_code == "COMPLETED":
        return 100, timeline
    if wo.status_code == "IN_PROGRESS":
        return 40, timeline
    return 0, timeline


# =====================================================================
# 3. [W-01a ~ W-01e] 통합 업무 (동적 단일 페이지 AI + ERP 허브)
# =====================================================================
@service_bp.route("/work-instruction/<int:instruction_id>/view", methods=["GET"])
@login_required
def view_work_instruction_doc(instruction_id: int):
    """물리 PDF 파일이 없거나 다른 PC에서 생성된 경우에도 DB 원장으로 공식 작업지시서를 즉시 열람"""
    inst = WorkInstruction.query.get_or_404(instruction_id)
    wo = WorkOrder.query.get_or_404(inst.work_id)
    branch = ErpBranch.query.filter_by(branch_code=wo.work_region).first()
    dispatch_info = MesWorkOrderDispatch.query.filter_by(wo_number=wo.work_no).first()
    welding_params = {}
    if inst.welding_params_json:
        try:
            parsed = json.loads(_read_clob(inst.welding_params_json))
            if isinstance(parsed, dict):
                welding_params = parsed
        except Exception:
            welding_params = {}
    return render_template(
        "service/work_instruction_view.html",
        wo=wo,
        inst=inst,
        branch_name=branch.branch_name if branch else wo.work_region,
        dispatch_info=dispatch_info,
        welding_params=welding_params,
        instruction_text=_read_clob(inst.instruction_text),
        issued_at_kst=_fmt_kst(wo.requested_at),
    )


@service_bp.route("/agent/plan-by-work/<int:work_id>", methods=["GET"])
@login_required
def agent_plan_by_work(work_id: int):
    """session_id가 없는 승인 대기 작업지시서(DRAFT)도 DB 원장 기준으로 즉시 검토/승인 화면에 연결"""
    user = User.query.get_or_404(session["user_id"])
    wo = WorkOrder.query.get_or_404(work_id)
    inst = (
        WorkInstruction.query
        .filter_by(work_id=wo.work_id)
        .order_by(WorkInstruction.instruction_id.desc())
        .first()
    )
    meta = _load_env_meta(inst)
    src_sid = meta.get("_source_session_id")
    if src_sid:
        has_msg = AgentChatHistory.query.filter_by(session_id=src_sid, role_code="ASSISTANT").first()
        if has_msg:
            return redirect(url_for("service.agent_plan_page", session_id=src_sid))

    saved_doc = meta.get("_draft_payload") if isinstance(meta.get("_draft_payload"), dict) else {}
    saved_doc = dict(saved_doc)
    saved_doc["work_id"] = wo.work_id
    saved_doc["instruction_id"] = inst.instruction_id if inst else None
    saved_doc["work_order_number"] = wo.work_no
    saved_doc["title"] = wo.work_title
    saved_doc["approval_status"] = inst.approval_status if inst else "DRAFT"
    params = dict(saved_doc.get("parameters") or {})
    params.setdefault("block_id", wo.block_code)
    params.setdefault("ship_code", wo.ship_code)
    params.setdefault("material", wo.material_grade)
    params.setdefault("thickness", f"{wo.thickness_mm}mm")
    params.setdefault("process", wo.process_code)
    params.setdefault("branch_code", wo.work_region)
    params.setdefault("work_zone", wo.work_zone)
    saved_doc["parameters"] = params
    pdf_url = _extract_inst_pdf_url(inst)
    if pdf_url:
        saved_doc.setdefault("files", {})["download_pdf_url"] = pdf_url

    answer = _read_clob(inst.instruction_text) if inst else (wo.work_description or "작업지시서 초안 내용")
    quick_workers = ErpWorkerCert.query.filter_by(status="VALID").order_by(ErpWorkerCert.worker_name.asc()).limit(8).all()
    quick_blocks = ErpBlockMaster.query.order_by(ErpBlockMaster.block_id.asc()).limit(8).all()

    return render_template(
        "service/agent_plan.html",
        me=user,
        session_id=f"work_{wo.work_id}",
        answer=answer,
        doc=saved_doc,
        ui_blocks={},
        sources=[],
        quick_workers=quick_workers,
        quick_blocks=quick_blocks,
        cancel_url=url_for("service.workspace_page"),
        can_approve=(user.role in ("ADMIN", "MANAGER")),
    )


@service_bp.route("/workspace", methods=["GET"])
@login_required
def workspace_page():
    user = User.query.get_or_404(session["user_id"])
    branches = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).all()
    first_block = ErpBlockMaster.query.order_by(ErpBlockMaster.block_id.asc()).first()
    first_branch = branches[0] if branches else None
    sample_placeholder = (
        f"업무 요청 입력 (Enter 전송 · Shift+Enter 줄바꿈) — 예: {first_branch.branch_name} {first_block.block_id} 자재와 용접 기준 확인해줘"
        if (first_branch and first_block)
        else "업무 요청 입력 (Enter 전송 · Shift+Enter 줄바꿈) — 선박·블록·자재·규격·작업지시를 자유롭게 입력하세요"
    )
    initial_prompt = (request.args.get("prompt") or "").strip()

    return render_template(
        "service/workspace.html",
        me=user,
        branches=branches,
        can_erp=(user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER")),
        can_internal_rag=(user.can_access_internal_rag == "Y" or user.role in ("ADMIN", "MANAGER")),
        can_issue_wo=(user.role in ("ADMIN", "MANAGER")),
        sample_placeholder=sample_placeholder,
        initial_prompt=initial_prompt,
    )


@service_bp.route("/api/workspace/summary", methods=["GET"])
@login_required
def api_workspace_summary():
    """W-01a 종합 현황: 한국시간(KST) 기준 오늘 작업 예정 블록 · 승인 대기 직결 · 야드별 카드 · 부족 자원"""
    db.session.expire_all()
    user = User.query.get_or_404(session["user_id"])
    can_erp = (user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER"))
    today_kst = _today_kst_str()

    branches_raw = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).all()
    branch_map = {b.branch_code: b.branch_name for b in branches_raw}
    all_orders = WorkOrder.query.order_by(WorkOrder.work_id.desc()).all()
    prog_map = _safe_get_wo_progress_map([o.work_id for o in all_orders])

    # 1) WORK_PLANS 조회 & ERP_BLOCK_MASTER 자동 동기화 + 오늘(KST) 예정 블록 엄격 분리
    all_active_plans = _safe_query_work_plans(states=["PENDING", "IN_PROGRESS"], today_only=False)
    for p in all_active_plans:
        _sync_work_plan_to_erp_block_master(p)

    today_plans = [p for p in all_active_plans if _is_plan_scheduled_today(p, today_kst)]
    other_date_plans_cnt = max(0, len(all_active_plans) - len(today_plans))

    wo_by_no = {o.work_no: o for o in all_orders}
    wo_by_blk = {}
    for o in all_orders:
        if o.block_code not in wo_by_blk and o.status_code in ("READY", "IN_PROGRESS"):
            wo_by_blk[o.block_code] = o

    active_queue = []
    if all_active_plans:
        # DB에 작업 계획(WORK_PLANS)이 존재하면 반드시 오늘(KST) 계획된 블록만 표시
        for p in today_plans[:8]:
            linked_wo = wo_by_no.get(p.assigned_wo_number) or wo_by_blk.get(p.block_code)
            active_queue.append({
                "plan_id": p.plan_id,
                "priority_rank": p.priority_rank,
                "block_code": p.block_code,
                "block_name": p.block_name,
                "branch_code": p.branch_code,
                "branch_name": branch_map.get(p.branch_code, p.branch_code),
                "material_grade": p.material_grade,
                "thickness_mm": float(p.thickness_mm) if p.thickness_mm is not None else 0.0,
                "process_code": p.process_code,
                "state": p.plan_state,
                "planned_start_date": p.planned_start_date or today_kst,
                "planned_end_date": p.planned_end_date or today_kst,
                "wo_no": p.assigned_wo_number or (linked_wo.work_no if linked_wo else ""),
                "work_id": linked_wo.work_id if linked_wo else None,
            })
    else:
        # WORK_PLANS 자체가 비어 있으면 오늘(KST) 생성된 활성 작업지시서만 표시
        today_orders = [
            o for o in all_orders
            if o.status_code in ("READY", "IN_PROGRESS")
            and _fmt_kst(o.requested_at, "%Y-%m-%d") == today_kst
        ]
        other_date_plans_cnt = sum(
            1 for o in all_orders
            if o.status_code in ("READY", "IN_PROGRESS")
            and _fmt_kst(o.requested_at, "%Y-%m-%d") != today_kst
        )
        for idx, wo in enumerate(today_orders[:8], 1):
            active_queue.append({
                "plan_id": prog_map.get(wo.work_id, {}).get("plan_id") or "",
                "priority_rank": idx,
                "block_code": wo.block_code,
                "block_name": wo.work_title,
                "ship_code": wo.ship_code,
                "branch_code": wo.work_region,
                "branch_name": branch_map.get(wo.work_region, wo.work_region),
                "material_grade": wo.material_grade,
                "thickness_mm": float(wo.thickness_mm) if wo.thickness_mm is not None else 0.0,
                "process_code": wo.process_code,
                "state": wo.status_code,
                "planned_start_date": _fmt_kst(wo.requested_at, "%Y-%m-%d"),
                "planned_end_date": _fmt_kst(wo.requested_at, "%Y-%m-%d"),
                "wo_no": wo.work_no,
                "work_id": wo.work_id,
            })

    all_blocks = ErpBlockMaster.query.order_by(ErpBlockMaster.branch_code.asc(), ErpBlockMaster.block_id.asc()).all()

    # 2) 야드별 주의 자원 (권한자)
    low_materials, expiring_workers = [], []
    all_mats, all_equips = [], []
    comp_map = {c.company_code: c.company_name for c in ErpCompany.query.all()}
    if can_erp:
        all_mats = ErpBomInventory.query.all()
        all_equips = ErpVehiclesEquipment.query.all()
        for m in all_mats:
            avail = float(m.total_qty or 0) - float(m.reserved_qty or 0)
            if avail <= 50.0:
                low_materials.append({
                    "item_code": m.item_code,
                    "item_name": m.item_name,
                    "supplier_name": comp_map.get(m.supplier_code, m.supplier_code),
                    "branch_code": m.branch_code,
                    "branch_name": branch_map.get(m.branch_code, m.branch_code),
                    "available_qty": avail,
                    "unit": m.unit,
                })
        for w in ErpWorkerCert.query.order_by(ErpWorkerCert.cert_expiry_date.asc()).all():
            days = _calc_cert_days_left(w.cert_expiry_date)
            label, level = _format_cert_dday(days)
            if level in ("EXPIRED", "EXPIRING"):
                expiring_workers.append({
                    "worker_id": w.worker_id,
                    "worker_name": w.worker_name,
                    "cert_code": w.cert_code,
                    "weld_process": w.weld_process,
                    "cert_expiry_date": w.cert_expiry_date,
                    "days_left": days,
                    "dday_label": label,
                    "level": level,
                    "branch_code": w.branch_code,
                    "branch_name": branch_map.get(w.branch_code, w.branch_code),
                })

    # 3) 야드별 운영 요약 카드 (DB ERP_BRANCHES 기반 동적 생성)
    yard_cards = []
    for b in branches_raw:
        b_blocks = [x for x in all_blocks if x.branch_code == b.branch_code]
        b_orders = [o for o in all_orders if o.work_region == b.branch_code and o.status_code in ("READY", "IN_PROGRESS")]
        b_low = [m for m in low_materials if m["branch_code"] == b.branch_code]
        b_eq = [e for e in all_equips if e.branch_code == b.branch_code]
        yard_cards.append({
            "branch_code": b.branch_code,
            "branch_name": b.branch_name,
            "region_name": b.region_name,
            "camera_code": b.camera_code or "-",
            "block_count": len(b_blocks),
            "active_order_count": len(b_orders),
            "low_mat_count": len(b_low),
            "equip_count": len(b_eq),
            "primary_block": b_blocks[0].block_id if b_blocks else "-",
            "primary_ship": b_blocks[0].ship_code if b_blocks else "-",
        })

    # 4) 결재 대기(DRAFT) 목록 직결 및 최신 문서 현황
    draft_insts = (
        WorkInstruction.query
        .filter_by(approval_status="DRAFT")
        .order_by(WorkInstruction.instruction_id.desc())
        .all()
    )
    pending_docs = []
    for inst in draft_insts:
        wo = WorkOrder.query.get(inst.work_id)
        if not wo:
            continue
        meta = _load_env_meta(inst)
        sid = meta.get("_source_session_id") or ""
        review_url = f"/service/agent/plan/{sid}" if sid else f"/service/agent/plan-by-work/{wo.work_id}"
        pending_docs.append({
            "work_id": wo.work_id,
            "instruction_id": inst.instruction_id,
            "work_no": wo.work_no,
            "ship_code": wo.ship_code,
            "block_code": wo.block_code,
            "branch_name": branch_map.get(wo.work_region, wo.work_region),
            "revision_no": inst.revision_no,
            "created_at_kst": _fmt_kst(inst.created_at),
            "review_url": review_url,
            "pdf_url": _extract_inst_pdf_url(inst),
        })

    recent_docs = []
    for wo in all_orders[:5]:
        inst = WorkInstruction.query.filter_by(work_id=wo.work_id).order_by(WorkInstruction.instruction_id.desc()).first()
        meta = _load_env_meta(inst)
        sid = meta.get("_source_session_id") or ""
        is_draft = (inst is not None and inst.approval_status == "DRAFT")
        recent_docs.append({
            "work_id": wo.work_id,
            "work_no": wo.work_no,
            "ship_code": wo.ship_code,
            "block_code": wo.block_code,
            "branch_name": branch_map.get(wo.work_region, wo.work_region),
            "status_code": wo.status_code,
            "revision_no": inst.revision_no if inst else "Rev.1",
            "approval_status": inst.approval_status if inst else "DRAFT",
            "requested_at_kst": _fmt_kst(wo.requested_at),
            "pdf_url": _extract_inst_pdf_url(inst),
            "review_url": (f"/service/agent/plan/{sid}" if sid else f"/service/agent/plan-by-work/{wo.work_id}") if is_draft else f"/service/work-status?work_id={wo.work_id}",
        })

    # 5) DB 기반 동적 자주 쓰는 업무
    quick_prompts = []
    if all_blocks:
        fb = all_blocks[0]
        quick_prompts.append({
            "label": f"{fb.steel_type} {fb.thickness_mm} 용접 기준",
            "prompt": f"{fb.steel_type} {fb.thickness_mm} {fb.process} 용접 예열 및 시공 기준 알려줘",
        })
        quick_prompts.append({
            "label": f"{fb.block_id} 자재 · 인력",
            "prompt": f"{fb.block_id} 블록 자재 재고와 작업자 확인해줘",
        })
    if expiring_workers and can_erp:
        quick_prompts.append({
            "label": "자격 만료 안내 발송",
            "prompt": "용접 자격 만료 및 임박 작업자 갱신 안내 메일 써줘",
        })
    if low_materials and can_erp:
        quick_prompts.append({
            "label": "부족 자재 긴급 발주",
            "prompt": "부족한 자재 발주 메일 써줘",
        })

    robots_cnt = RobotExecution.query.count()

    return jsonify({
        "status": "SUCCESS",
        "can_erp": can_erp,
        "today_kst": today_kst,
        "other_date_plans_cnt": other_date_plans_cnt,
        "kpi": {
            "active_orders": sum(1 for o in all_orders if o.status_code in ("READY", "IN_PROGRESS")),
            "completed_orders": sum(1 for o in all_orders if o.status_code == "COMPLETED"),
            "active_queue_cnt": len(active_queue),
            "low_materials_cnt": len(low_materials),
            "expiring_workers_cnt": len(expiring_workers),
            "pending_approvals": len(pending_docs),
            "robots_cnt": robots_cnt,
            "yards_cnt": len(branches_raw),
        },
        "yard_cards": yard_cards,
        "active_queue": active_queue,
        "low_materials": low_materials[:6],
        "expiring_workers": expiring_workers[:6],
        "pending_approvals": len(pending_docs),
        "pending_docs": pending_docs,
        "recent_docs": recent_docs,
        "quick_prompts": quick_prompts[:4],
    })


@service_bp.route("/api/workspace/qa-context", methods=["GET"])
@login_required
def api_workspace_qa_context():
    """W-01b: 권한별 열람 가능한 공통 기술문서 / 사내 규정 / 선박 도면(PDF만) 분리 반환"""
    user = User.query.get_or_404(session["user_id"])
    is_admin = user.role in ("ADMIN", "MANAGER")
    can_internal = (user.can_access_internal_rag == "Y") or is_admin

    pub_rows = (
        DocumentMetadata.query
        .filter(DocumentMetadata.is_active == "Y", DocumentMetadata.collection_type == "PUBLIC")
        .order_by(DocumentMetadata.document_id.desc())
        .limit(8)
        .all()
    )
    int_rows = []
    cad_pdf_rows = []
    if can_internal:
        int_rows = (
            DocumentMetadata.query
            .filter(
                DocumentMetadata.is_active == "Y",
                DocumentMetadata.collection_type == "INTERNAL",
                DocumentMetadata.category_code != "CAD_DRAWING",
            )
            .order_by(DocumentMetadata.document_id.desc())
            .limit(8)
            .all()
        )
        cad_all = (
            DocumentMetadata.query
            .filter(
                DocumentMetadata.is_active == "Y",
                DocumentMetadata.category_code == "CAD_DRAWING",
            )
            .order_by(DocumentMetadata.document_id.desc())
            .all()
        )
        for c in cad_all:
            if (c.original_file_name or "").lower().endswith(".pdf"):
                cad_pdf_rows.append(c)
                if len(cad_pdf_rows) >= 6:
                    break

    def _ser(d):
        return {
            "document_id": d.document_id,
            "document_code": d.document_code,
            "document_title": _clean_doc_title(d.document_title, d.original_file_name),
            "original_file_name": d.original_file_name,
            "collection_type": d.collection_type,
            "category_code": d.category_code,
        }

    return jsonify({
        "status": "SUCCESS",
        "can_internal": can_internal,
        "public_docs": [_ser(d) for d in pub_rows],
        "internal_docs": [_ser(d) for d in int_rows],
        "cad_pdf_docs": [_ser(d) for d in cad_pdf_rows],
    })


@service_bp.route("/api/workspace/route-command", methods=["POST"])
@login_required
def api_workspace_route_command():
    """
    통합 업무 자연어 명령 스마트 라우터:
    - 특정 야드 ERP 요청 -> 해당 야드 ERP 화면으로 즉시 전환
    - 메일 작성 / 발주 메일 요청 -> DB 조회 기반 수정 가능한 메일 미리보기 + 발송 버튼만 즉시 표시
    - 오늘 업무 조회 요청 -> 오늘 활성 작업 목록 표시 (없으면 생성 버튼 안내)
    - 그 외(용접 규격 질의, 작업지시서 생성 등) -> 기존 LangGraph Agent(ship_agent_app)로 전달
    """
    user = User.query.get_or_404(session["user_id"])
    can_erp = (user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER"))
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"action": "ERROR", "message": "요청 내용을 입력해 주세요."}), 400

    ql = q.lower()
    branches = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).all()
    branch_map = {b.branch_code: b.branch_name for b in branches}

    matched_branch = None
    for b in branches:
        bname = (b.branch_name or "").lower()
        reg = (b.region_name or "").lower()
        bcode = (b.branch_code or "").lower()
        if bcode in ql or (reg and reg in ql) or any(tok and tok in ql for tok in bname.split() if len(tok) >= 2):
            matched_branch = b
            break

    # 0) 특정 야드 CCTV / 카메라 / 현장 영상 보기 요청 감지
    is_cam_req = any(k in ql for k in ("cctv", "카메라", "현장 영상", "현장 보", "현장보", "로봇캠", "robotcam"))
    if is_cam_req and not any(k in q for k in ("작업지시서", "작업 지시서")):
        target_br = matched_branch or (branches[0] if branches else None)
        br_blocks = [b for b in ErpBlockMaster.query.all() if target_br and b.branch_code == target_br.branch_code]
        return jsonify({
            "action": "SHOW_VISION",
            "branch_code": target_br.branch_code if target_br else "",
            "branch_name": target_br.branch_name if target_br else "",
            "block_code": br_blocks[0].block_id if br_blocks else "",
        })

    # 1) 메일 / 발주 요청 감지 ("발주" 단독 입력도 즉시 발주 메일 미리보기로 연결)
    is_mail_req = any(k in q for k in ("메일", "이메일", "발주", "안내 보내", "갱신 안내"))
    if is_mail_req and not any(k in q for k in ("작업지시서", "작업 지시서", "용접 기준", "예열")):
        if not can_erp:
            return jsonify({"action": "FORBIDDEN", "message": "메일 및 발주 요청은 담당자 권한이 필요합니다."}), 403

        is_po = any(k in q for k in ("발주", "자재", "와이어", "강판", "가스"))
        if is_po:
            mats = ErpBomInventory.query.all()
            if matched_branch:
                mats = [m for m in mats if m.branch_code == matched_branch.branch_code] or mats
            low_mats = [m for m in mats if (float(m.total_qty or 0) - float(m.reserved_qty or 0)) <= 50.0]
            target_mats = low_mats or mats[:1]
            comp_map = {c.company_code: c.company_name for c in ErpCompany.query.all()}
            if target_mats:
                m0 = target_mats[0]
                sup_name = comp_map.get(m0.supplier_code, m0.supplier_code)
                b_name = branch_map.get(m0.branch_code, m0.branch_code)
                lines = "\n".join(
                    f"- {m.item_name} ({m.item_code}) | 현재 가용: {float(m.total_qty or 0) - float(m.reserved_qty or 0)}{m.unit} | 야드: {branch_map.get(m.branch_code, m.branch_code)}"
                    for m in target_mats[:5]
                )
                return jsonify({
                    "action": "SHOW_DRAFT",
                    "draft": {
                        "type": "PO",
                        "recipient": f"{sup_name} 영업담당자",
                        "subject": f"[긴급 발주 요청] {m0.item_name} 외 {len(target_mats)-1}건" if len(target_mats) > 1 else f"[긴급 발주 요청] {m0.item_name} ({m0.item_code})",
                        "body": f"안녕하세요, {sup_name} 담당자님.\n{b_name} 현장 공정용 자재 재고가 부족하여 아래와 같이 긴급 발주를 요청드립니다.\n\n{lines}\n\n가능한 납품 일정을 회신해 주시기 바랍니다.",
                    },
                })

        workers = ErpWorkerCert.query.order_by(ErpWorkerCert.cert_expiry_date.asc()).all()
        named_worker = next((w for w in workers if w.worker_name and w.worker_name in q), None)
        if named_worker:
            return jsonify({
                "action": "SHOW_DRAFT",
                "draft": {
                    "type": "EMAIL",
                    "recipient": f"{named_worker.worker_name} ({named_worker.worker_id})",
                    "subject": f"[업무 안내] {named_worker.worker_name}님 확인 요청",
                    "body": f"안녕하세요, {named_worker.worker_name}님.\n보유 자격({named_worker.cert_code} / 만료일 {named_worker.cert_expiry_date}) 및 현장 투입 일정 관련하여 안내드립니다.\n\n확인 후 회신 부탁드립니다.",
                },
            })

        alert_workers = []
        for w in workers:
            days = _calc_cert_days_left(w.cert_expiry_date)
            dday_label, level = _format_cert_dday(days)
            if level in ("EXPIRED", "EXPIRING"):
                alert_workers.append((w, dday_label))
        if alert_workers:
            recips = ", ".join(f"{w.worker_name}({w.worker_id})" for w, _ in alert_workers)
            lines = "\n".join(
                f"- {w.worker_name} ({w.cert_code} / {w.weld_process}, 만료일 {w.cert_expiry_date}, {lbl})"
                for w, lbl in alert_workers
            )
            return jsonify({
                "action": "SHOW_DRAFT",
                "draft": {
                    "type": "EMAIL",
                    "recipient": recips,
                    "subject": f"[일괄 안내] 용접 자격 만료 및 갱신 요청 ({len(alert_workers)}명)",
                    "body": f"안녕하세요. 운영지원팀입니다.\n아래 작업자분들의 용접 자격 만료일이 도래하거나 경과하여 갱신 안내를 드립니다.\n\n{lines}\n\n갱신 완료 후 자격증 사본을 담당 팀장에게 제출해 주시기 바랍니다.",
                },
            })
        return jsonify({
            "action": "SHOW_DRAFT",
            "draft": {
                "type": "EMAIL",
                "recipient": "",
                "subject": "[업무 안내]",
                "body": f"요청 내용: {q}\n\n내용을 수정하신 뒤 아래 [승인 후 발송] 버튼을 눌러 주세요.",
            },
        })

    # 2) 특정 야드 ERP / 자재 / 인력 / 장비 조회 요청 감지
    is_erp_req = any(k in ql for k in ("erp", "재고", "자재", "인력", "작업자", "장비", "로봇 현황"))
    if is_erp_req and not any(k in q for k in ("작업지시", "발행", "생성", "예열", "규격", "기준")):
        if not can_erp:
            return jsonify({"action": "FORBIDDEN", "message": "야드 자재·인력 조회 권한이 없는 계정입니다."}), 403
        subtab = "materials"
        if any(k in q for k in ("인력", "작업자", "자격", "면허")):
            subtab = "workers"
        elif any(k in q for k in ("장비", "차량", "로봇", "크레인")):
            subtab = "equipment"
        elif any(k in q for k in ("협력사", "외주", "b2b")):
            subtab = "b2b"
        return jsonify({
            "action": "SHOW_ERP",
            "branch_code": matched_branch.branch_code if matched_branch else "ALL",
            "subtab": subtab,
        })

    # 3) 오늘 업무 / 할 일 조회 요청 감지
    if any(k in q for k in ("오늘 업무", "오늘 할일", "오늘 할 일", "오늘 작업", "당일 업무", "진행 중인 작업")):
        orders_q = WorkOrder.query.filter(WorkOrder.status_code.in_(["READY", "IN_PROGRESS"]))
        if matched_branch:
            orders_q = orders_q.filter(WorkOrder.work_region == matched_branch.branch_code)
        orders = orders_q.order_by(WorkOrder.work_id.desc()).all()
        items = [
            {
                "work_id": o.work_id,
                "work_no": o.work_no,
                "work_title": o.work_title,
                "ship_code": o.ship_code,
                "block_code": o.block_code,
                "branch_code": o.work_region,
                "branch_name": branch_map.get(o.work_region, o.work_region),
                "process_code": o.process_code,
                "material_grade": o.material_grade,
                "thickness_mm": float(o.thickness_mm) if o.thickness_mm is not None else 0.0,
                "status_code": o.status_code,
            }
            for o in orders
        ]
        return jsonify({
            "action": "SHOW_TODAY_WORK",
            "branch_name": matched_branch.branch_name if matched_branch else "전체 야드",
            "branch_code": matched_branch.branch_code if matched_branch else "",
            "orders": items,
        })

    return jsonify({"action": "USE_AGENT"})


@service_bp.route("/api/workspace/erp-yard", methods=["GET"])
@login_required
def api_workspace_erp_yard():
    """W-01c: 전사/야드별 전문 ERP 조회 (자재 BOM · 작업자 자격 · 장비/로봇 · 협력사 B2B)"""
    db.session.expire_all()
    user = User.query.get_or_404(session["user_id"])
    if not (user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER")):
        return jsonify({"status": "FORBIDDEN", "message": "자원(ERP) 조회 권한이 없는 계정입니다."}), 403

    branch_code = (request.args.get("branch_code") or "ALL").strip()
    branches = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).all()
    branch_map = {b.branch_code: b.branch_name for b in branches}
    comp_map = {c.company_code: c.company_name for c in ErpCompany.query.all()}

    mat_q = ErpBomInventory.query
    wrk_q = ErpWorkerCert.query
    eq_q = ErpVehiclesEquipment.query
    if branch_code and branch_code != "ALL":
        mat_q = mat_q.filter_by(branch_code=branch_code)
        wrk_q = wrk_q.filter_by(branch_code=branch_code)
        eq_q = eq_q.filter_by(branch_code=branch_code)

    mats_raw = mat_q.order_by(ErpBomInventory.branch_code.asc(), ErpBomInventory.item_code.asc()).all()
    workers_raw = wrk_q.order_by(ErpWorkerCert.cert_expiry_date.asc()).all()
    equips_raw = eq_q.order_by(ErpVehiclesEquipment.branch_code.asc(), ErpVehiclesEquipment.asset_code.asc()).all()
    worker_name_map = {w.worker_id: w.worker_name for w in ErpWorkerCert.query.all()}

    open_po_codes = set()
    open_po_eta_map = {}
    branch_obj_map = {b.branch_code: b for b in branches}
    partner_loc_map = {}
    lead_hours_map = {}
    b2b_materials, b2b_workers, b2b_equipment = [], [], []
    try:
        for loc in B2BCompanyLocation.query.all():
            partner_loc_map[loc.partner_id] = loc
        for dm in B2BDaehanMaterial.query.all():
            lead_hours_map[dm.hq_item_code] = float(dm.lead_time_hours or 1.0)
        open_pos = B2BSalesOrder.query.filter(B2BSalesOrder.order_status.in_(["PENDING", "CONFIRMED", "SHIPPED"])).order_by(B2BSalesOrder.created_at.desc()).all()
        open_po_codes = {po.ordered_ref_code for po in open_pos}
        for po in open_pos:
            if po.ordered_ref_code in open_po_eta_map:
                continue
            ploc = partner_loc_map.get(po.partner_id)
            wo_match = WorkOrder.query.filter_by(work_no=po.hq_wo_number).first()
            dest_br = branch_obj_map.get(wo_match.work_region) if wo_match else (branches[0] if branches else None)
            nav = _calc_nav_eta(
                ploc.latitude if ploc else None,
                ploc.longitude if ploc else None,
                dest_br.latitude if dest_br else None,
                dest_br.longitude if dest_br else None,
                base_dt=po.created_at,
                prep_hours=lead_hours_map.get(po.ordered_ref_code, 1.0),
            )
            open_po_eta_map[po.ordered_ref_code] = {
                "eta_str": nav["eta_str"],
                "road_km": nav["road_km"],
                "drive_label": nav["drive_label"],
                "partner_name": comp_map.get(po.partner_id, po.partner_id),
            }
        for m in B2BDaehanMaterial.query.all():
            b2b_materials.append({
                "partner_name": comp_map.get(m.partner_id, m.partner_id),
                "hq_item_code": m.hq_item_code,
                "item_name": m.item_name,
                "available_qty": float(m.available_qty or 0),
                "unit": m.unit,
                "unit_price_krw": int(m.unit_price_krw or 0),
                "lead_time_hours": m.lead_time_hours,
            })
        for w in B2BDonghaeWorker.query.all():
            b2b_workers.append({
                "partner_name": comp_map.get(w.partner_id, w.partner_id),
                "worker_name": w.worker_name,
                "specialty_role": w.specialty_role,
                "cert_code": w.cert_code,
                "weld_process": w.weld_process,
                "current_region": w.current_region,
                "daily_rate_krw": int(w.daily_rate_krw or 0),
            })
        for e in B2BHankookEquipment.query.all():
            b2b_equipment.append({
                "partner_name": comp_map.get(e.partner_id, e.partner_id),
                "asset_name": e.asset_name,
                "asset_category": e.asset_category,
                "current_region": e.current_region,
                "max_load_ton": float(e.max_load_ton or 0),
                "hourly_rental_krw": int(e.hourly_rental_krw or 0),
            })
    except Exception:
        pass

    materials = []
    for m in mats_raw:
        avail = float(m.total_qty or 0) - float(m.reserved_qty or 0)
        materials.append({
            "item_code": m.item_code,
            "item_name": m.item_name,
            "material_category": m.material_category,
            "lot_no": m.lot_no,
            "supplier_name": comp_map.get(m.supplier_code, m.supplier_code),
            "branch_code": m.branch_code,
            "branch_name": branch_map.get(m.branch_code, m.branch_code),
            "total_qty": float(m.total_qty or 0),
            "reserved_qty": float(m.reserved_qty or 0),
            "available_qty": avail,
            "unit": m.unit,
            "is_low": avail <= 50.0,
            "has_open_po": m.item_code in open_po_codes,
            "open_po_eta": open_po_eta_map.get(m.item_code),
        })

    workers = []
    for w in workers_raw:
        days = _calc_cert_days_left(w.cert_expiry_date)
        dday_label, level = _format_cert_dday(days)
        workers.append({
            "worker_id": w.worker_id,
            "worker_name": w.worker_name,
            "employment_type": w.employment_type,
            "company_name": comp_map.get(w.company_code, w.company_code),
            "branch_code": w.branch_code,
            "branch_name": branch_map.get(w.branch_code, w.branch_code),
            "department_name": w.department_name or "-",
            "role": w.role,
            "phone": w.phone or "-",
            "cert_code": w.cert_code,
            "weld_process": w.weld_process,
            "cert_expiry_date": w.cert_expiry_date,
            "days_left": days,
            "dday_label": dday_label,
            "level": level,
            "status": w.status,
        })

    equipment = []
    for eq in equips_raw:
        equipment.append({
            "asset_code": eq.asset_code,
            "asset_name": eq.asset_name,
            "asset_category": eq.asset_category,
            "company_name": comp_map.get(eq.company_code, eq.company_code),
            "branch_code": eq.branch_code,
            "branch_name": branch_map.get(eq.branch_code, eq.branch_code),
            "current_zone": eq.current_zone or "-",
            "max_load_ton": float(eq.max_load_ton or 0),
            "driver_name": worker_name_map.get(eq.driver_worker_id, eq.driver_worker_id or "-"),
            "operation_status": eq.operation_status,
        })

    return jsonify({
        "status": "SUCCESS",
        "branch_code": branch_code,
        "branch_name": "전체 야드" if branch_code == "ALL" else branch_map.get(branch_code, branch_code),
        "kpi": {
            "material_count": len(materials),
            "low_material_count": sum(1 for m in materials if m["is_low"]),
            "worker_count": len(workers),
            "alert_worker_count": sum(1 for w in workers if w["level"] in ("EXPIRED", "EXPIRING")),
            "equipment_count": len(equipment),
            "standby_equipment_count": sum(1 for e in equipment if e["operation_status"] == "STANDBY"),
            "b2b_count": len(b2b_materials) + len(b2b_workers) + len(b2b_equipment),
        },
        "materials": materials,
        "workers": workers,
        "equipment": equipment,
        "b2b": {
            "materials": b2b_materials,
            "workers": b2b_workers,
            "equipment": b2b_equipment,
        },
    })


@service_bp.route("/api/workspace/draft-center", methods=["GET"])
@login_required
def api_workspace_draft_center():
    """W-01d: 메일 / 발주계획서 / 승인 대기 초안 전용 센터 (야드 ERP 조회와 완전 분리)"""
    db.session.expire_all()
    user = User.query.get_or_404(session["user_id"])
    if not (user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER")):
        return jsonify({"status": "FORBIDDEN", "message": "메일/문서 초안 권한이 없는 계정입니다."}), 403

    branch_map = {b.branch_code: b.branch_name for b in ErpBranch.query.all()}
    comp_map = {c.company_code: c.company_name for c in ErpCompany.query.all()}

    # 1) 만료·임박 자격 작업자 목록 (개별/일괄 메일 발송 대상)
    alert_workers = []
    for w in ErpWorkerCert.query.order_by(ErpWorkerCert.cert_expiry_date.asc()).all():
        days = _calc_cert_days_left(w.cert_expiry_date)
        dday_label, level = _format_cert_dday(days)
        if level in ("EXPIRED", "EXPIRING"):
            alert_workers.append({
                "worker_id": w.worker_id,
                "worker_name": w.worker_name,
                "company_name": comp_map.get(w.company_code, w.company_code),
                "branch_name": branch_map.get(w.branch_code, w.branch_code),
                "cert_code": w.cert_code,
                "weld_process": w.weld_process,
                "cert_expiry_date": w.cert_expiry_date,
                "dday_label": dday_label,
                "level": level,
            })

    # 2) 부족 자재 목록 (발주 초안 대상)
    open_po_codes = set()
    try:
        open_po_codes = {
            po.ordered_ref_code
            for po in B2BSalesOrder.query.filter(B2BSalesOrder.order_status.in_(["PENDING", "CONFIRMED"])).all()
        }
    except Exception:
        pass

    low_materials = []
    for m in ErpBomInventory.query.all():
        avail = float(m.total_qty or 0) - float(m.reserved_qty or 0)
        if avail <= 50.0:
            low_materials.append({
                "item_code": m.item_code,
                "item_name": m.item_name,
                "branch_name": branch_map.get(m.branch_code, m.branch_code),
                "supplier_name": comp_map.get(m.supplier_code, m.supplier_code),
                "available_qty": avail,
                "unit": m.unit,
                "has_open_po": m.item_code in open_po_codes,
            })

    # 3) 승인 대기 작업지시서 초안(DRAFT)
    draft_insts = (
        WorkInstruction.query
        .filter_by(approval_status="DRAFT")
        .order_by(WorkInstruction.instruction_id.desc())
        .limit(10)
        .all()
    )
    pending_docs = []
    for inst in draft_insts:
        wo = WorkOrder.query.get(inst.work_id)
        if not wo:
            continue
        meta = _load_env_meta(inst)
        pending_docs.append({
            "work_id": wo.work_id,
            "instruction_id": inst.instruction_id,
            "work_no": wo.work_no,
            "work_title": wo.work_title,
            "ship_code": wo.ship_code,
            "block_code": wo.block_code,
            "branch_name": branch_map.get(wo.work_region, wo.work_region),
            "revision_no": inst.revision_no,
            "pdf_url": _extract_inst_pdf_url(inst),
            "session_id": meta.get("_source_session_id") or "",
        })

    # 4) 최근 발송/처리 이력
    sent_rows = (
        AgentExecution.query
        .filter_by(intent_type="HITL_DRAFT_SEND")
        .order_by(AgentExecution.execution_id.desc())
        .limit(8)
        .all()
    )
    sent_history = []
    for r in sent_rows:
        info = {}
        if r.selected_tools_json:
            try:
                info = json.loads(_read_clob(r.selected_tools_json))
            except Exception:
                info = {}
        sent_history.append({
            "execution_id": r.execution_id,
            "draft_type": info.get("draft_type") or "EMAIL",
            "recipient": info.get("recipient") or "-",
            "subject": info.get("subject") or "-",
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "-",
        })

    return jsonify({
        "status": "SUCCESS",
        "alert_workers": alert_workers,
        "low_materials": low_materials,
        "pending_docs": pending_docs,
        "sent_history": sent_history,
    })


@service_bp.route("/api/workspace/send-draft", methods=["POST"])
@login_required
def api_workspace_send_draft():
    """W-01d HITL: 사용자 검토·수정 완료된 메일/발주계획서 초안 승인 발송"""
    user = User.query.get_or_404(session["user_id"])
    if not (user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER")):
        return jsonify({"status": "FORBIDDEN", "message": "발송 권한이 없습니다."}), 403

    data = request.get_json(silent=True) or {}
    draft_type = (data.get("draft_type") or "EMAIL").upper()
    recipient = (data.get("recipient") or "").strip()
    subject = (data.get("subject") or "").strip()
    body = (data.get("body") or "").strip()
    if not recipient or not subject or not body:
        return jsonify({"status": "ERROR", "message": "수신자, 제목, 본문을 모두 입력해 주세요."}), 400

    # [AGENT_TEAM_HOOK] 추후 Agent 팀원의 단독 Gmail MCP / B2B 발주 발송 함수 연결 지점
    exec_log = AgentExecution(
        session_id=f"hitl_draft_{uuid.uuid4().hex[:8]}",
        user_id=user.user_id,
        intent_type="HITL_DRAFT_SEND",
        current_state=f"SENT_{draft_type}",
        selected_tools_json=json.dumps(
            {"draft_type": draft_type, "recipient": recipient, "subject": subject, "body_preview": body[:200]},
            ensure_ascii=False,
        ),
        status_code="SUCCESS",
    )
    db.session.add(exec_log)
    db.session.commit()
    return jsonify({"status": "SUCCESS", "message": f"[{subject}] 승인 및 발송 요청이 정상 접수되었습니다."})


@service_bp.route("/api/workspace/check-duplicate", methods=["POST"])
@login_required
def api_workspace_check_duplicate():
    """W-01e: DB 작업 계획(WORK_PLANS) 기반 대상 블록만 조회 + ERP_BLOCK_MASTER 자동 동기화 + 당일(KST) 중복 검증"""
    db.session.expire_all()
    data = request.get_json(silent=True) or {}
    branch_code = (data.get("branch_code") or "").strip()
    block_code = (data.get("block_code") or "").strip()
    today_kst = _today_kst_str()

    # 1) DB 작업 계획(WORK_PLANS) 우선 조회 및 ERP_BLOCK_MASTER 자동 동기화
    all_plans = _safe_query_work_plans(states=["PENDING", "IN_PROGRESS"], today_only=False)
    for p in all_plans:
        _sync_work_plan_to_erp_block_master(p)

    today_plans = [p for p in all_plans if _is_plan_scheduled_today(p, today_kst)]
    target_plans = today_plans if today_plans else all_plans

    erp_block_map = {b.block_id: b for b in ErpBlockMaster.query.all()}
    dwg_map = {}
    try:
        dwg_map = {d.drawing_id: d for d in CadDrawing.query.all()}
    except Exception:
        dwg_map = {}

    block_options = []
    block_dict_map = {}

    if target_plans:
        for p in target_plans:
            if p.block_code in block_dict_map:
                continue
            eb = erp_block_map.get(p.block_code)
            dwg = dwg_map.get(p.drawing_id)
            ship_cd = (eb.ship_code if eb else None) or (dwg.ship_code if dwg else f"HN-{p.drawing_id[:8]}")
            thick_val = float(p.thickness_mm) if p.thickness_mm is not None else 20.0
            thick_str = eb.thickness_mm if eb else (f"{int(thick_val)}T" if thick_val.is_integer() else f"{thick_val}T")
            item = {
                "block_id": p.block_code,
                "ship_code": ship_cd,
                "branch_code": p.branch_code,
                "block_name": p.block_name or p.block_code,
                "steel_type": p.material_grade,
                "thickness_mm": thick_str,
                "process": p.process_code,
                "joint_type": p.joint_type,
                "weight_ton": float(p.est_weight_ton or 0.0),
                "planned_date": p.planned_start_date or today_kst,
                "is_today": _is_plan_scheduled_today(p, today_kst),
            }
            block_options.append(item)
            block_dict_map[p.block_code] = item
    else:
        # WORK_PLANS가 아예 비어 있는 경우: 활성 작업지시서 블록 우선, 없으면 ERP_BLOCK_MASTER
        active_wos = WorkOrder.query.filter(WorkOrder.status_code.in_(["READY", "IN_PROGRESS"])).order_by(WorkOrder.work_id.desc()).all()
        if active_wos:
            for wo in active_wos:
                if wo.block_code in block_dict_map:
                    continue
                eb = erp_block_map.get(wo.block_code)
                item = {
                    "block_id": wo.block_code,
                    "ship_code": wo.ship_code,
                    "branch_code": wo.work_region,
                    "block_name": eb.block_name if eb else wo.work_title,
                    "steel_type": wo.material_grade,
                    "thickness_mm": f"{wo.thickness_mm}mm",
                    "process": wo.process_code,
                    "joint_type": eb.joint_type if eb else "BUTT",
                    "weight_ton": float(eb.weight_ton or 0.0) if eb else 0.0,
                    "planned_date": _fmt_kst(wo.requested_at, "%Y-%m-%d"),
                    "is_today": _fmt_kst(wo.requested_at, "%Y-%m-%d") == today_kst,
                }
                block_options.append(item)
                block_dict_map[wo.block_code] = item
        else:
            for eb in ErpBlockMaster.query.order_by(ErpBlockMaster.branch_code.asc(), ErpBlockMaster.block_id.asc()).all():
                item = {
                    "block_id": eb.block_id,
                    "ship_code": eb.ship_code,
                    "branch_code": eb.branch_code,
                    "block_name": eb.block_name,
                    "steel_type": eb.steel_type,
                    "thickness_mm": eb.thickness_mm,
                    "process": eb.process,
                    "joint_type": eb.joint_type,
                    "weight_ton": float(eb.weight_ton or 0.0),
                    "planned_date": today_kst,
                    "is_today": True,
                }
                block_options.append(item)
                block_dict_map[eb.block_id] = item

    block = None
    if block_code and block_code in block_dict_map:
        block = block_dict_map[block_code]
    elif block_code and block_code in erp_block_map:
        eb = erp_block_map[block_code]
        block = {
            "block_id": eb.block_id,
            "ship_code": eb.ship_code,
            "branch_code": eb.branch_code,
            "block_name": eb.block_name,
            "steel_type": eb.steel_type,
            "thickness_mm": eb.thickness_mm,
            "process": eb.process,
            "joint_type": eb.joint_type,
            "weight_ton": float(eb.weight_ton or 0.0),
            "planned_date": today_kst,
            "is_today": True,
        }
    if not block and branch_code:
        block = next((b for b in block_options if b["branch_code"] == branch_code), None)
    if not block and block_options:
        block = block_options[0]

    if not block:
        return jsonify({
            "status": "SUCCESS",
            "today_kst": today_kst,
            "exists": False,
            "block": None,
            "block_options": [],
            "existing_wo": None,
            "ship_plan_exists": False,
            "vision": None,
        })

    target_branch = branch_code or block["branch_code"]
    branch = ErpBranch.query.filter_by(branch_code=target_branch).first()

    existing_wo = (
        WorkOrder.query
        .filter(
            WorkOrder.block_code == block["block_id"],
            WorkOrder.status_code.in_(["READY", "IN_PROGRESS"]),
        )
        .order_by(WorkOrder.work_id.desc())
        .first()
    )
    existing_wo_info = None
    if existing_wo:
        latest_inst = (
            WorkInstruction.query
            .filter_by(work_id=existing_wo.work_id)
            .order_by(WorkInstruction.instruction_id.desc())
            .first()
        )
        req_date_kst = _fmt_kst(existing_wo.requested_at, "%Y-%m-%d")
        existing_wo_info = {
            "work_id": existing_wo.work_id,
            "work_no": existing_wo.work_no,
            "status_code": existing_wo.status_code,
            "revision_no": latest_inst.revision_no if latest_inst else "Rev.1",
            "requested_at_kst": _fmt_kst(existing_wo.requested_at),
            "is_today_kst": (req_date_kst == today_kst),
            "pdf_url": _extract_inst_pdf_url(latest_inst),
        }

    ship_wo = (
        WorkOrder.query
        .filter(WorkOrder.ship_code == block["ship_code"], WorkOrder.status_code.in_(["READY", "IN_PROGRESS"]))
        .order_by(WorkOrder.work_id.desc())
        .first()
    )

    latest_va = None
    if branch and branch.camera_code:
        latest_va = (
            VisionAnalysis.query
            .filter_by(camera_code=branch.camera_code)
            .order_by(VisionAnalysis.analysis_id.desc())
            .first()
        )
    if not latest_va:
        latest_va = VisionAnalysis.query.order_by(VisionAnalysis.analysis_id.desc()).first()

    vision_summary = None
    if latest_va:
        dets = VisionDetection.query.filter_by(analysis_id=latest_va.analysis_id).all()
        ocr_vals = [d.ocr_text for d in dets if d.ocr_text]
        vision_summary = {
            "camera_code": latest_va.camera_code,
            "image_uri": latest_va.visualized_image_uri or latest_va.original_image_uri,
            "ocr_text": ocr_vals[0] if ocr_vals else "",
            "detected_objects": [d.class_name for d in dets],
            "welding_state": latest_va.welding_state,
            "captured_at": _fmt_kst(latest_va.created_at),
        }

    return jsonify({
        "status": "SUCCESS",
        "today_kst": today_kst,
        "plan_based": bool(target_plans),
        "exists": existing_wo_info is not None,
        "ship_plan_exists": ship_wo is not None,
        "ship_code": block["ship_code"],
        "branch": {
            "branch_code": target_branch,
            "branch_name": branch.branch_name if branch else target_branch,
        },
        "block": block,
        "block_options": block_options,
        "existing_wo": existing_wo_info,
        "vision": vision_summary,
    })


# =====================================================================
# 4. [SA-01 / SA-02] 에이전트 분석 & 통합 실행안 (요청 시 자동 진입)
# =====================================================================
@service_bp.route("/agent/analyze", methods=["GET"])
@login_required
def agent_analyze_page():
    user = User.query.get_or_404(session["user_id"])
    prompt = (request.args.get("prompt") or "").strip()
    mode = (request.args.get("mode") or "NEW").upper()
    branch_code = (request.args.get("branch_code") or "").strip()
    block_code = (request.args.get("block_code") or "").strip()
    work_id = request.args.get("work_id", type=int)
    return_to = (request.args.get("from") or "workspace").strip().lower()
    cancel_url = url_for("vision.index", branch=branch_code) if return_to == "vision" else url_for("service.workspace_page")

    if not prompt and not block_code and not work_id:
        flash("작업 준비 요청을 입력해 주세요.", "info")
        return redirect(cancel_url)

    # 작업 계획(WORK_PLANS) 블록이면 ERP_BLOCK_MASTER에 자동 동기화 후 분석 진행
    if block_code:
        try:
            wp = WorkPlan.query.filter_by(block_code=block_code).first()
            if wp:
                _sync_work_plan_to_erp_block_master(wp)
        except Exception:
            pass

    if return_to == "vision" and prompt:
        work_keywords = ("작업", "용접", "블록", "선박", "지시", "준비", "자재", "인력", "로봇", "크레인", "도면", "예열", "검토", "개정", "발주", "배정", "fcaw", "saw", "gmaw")
        if not any(k in prompt.lower() for k in work_keywords):
            flash("현장 관제에서는 용접·블록·작업지시 등 현장 업무 요청만 입력할 수 있습니다.", "warning")
            return redirect(cancel_url)

    existing_wo = WorkOrder.query.get(work_id) if work_id else None
    if not existing_wo and block_code:
        existing_wo = (
            WorkOrder.query
            .filter(WorkOrder.block_code == block_code, WorkOrder.status_code.in_(["READY", "IN_PROGRESS"]))
            .order_by(WorkOrder.work_id.desc())
            .first()
        )
    existing_inst = (
        WorkInstruction.query.filter_by(work_id=existing_wo.work_id).order_by(WorkInstruction.instruction_id.desc()).first()
        if existing_wo else None
    )
    existing_dispatch = (
        MesWorkOrderDispatch.query.filter_by(wo_number=existing_wo.work_no).first()
        if existing_wo else None
    )

    # REVISE 모드인데 사용자가 변경 조건을 아직 입력하지 않았다면, 자동 실행하지 않고 조건 변경 전용 채팅창을 표시
    revise_waiting_input = (mode == "REVISE" and not prompt)

    if not prompt and not revise_waiting_input:
        prompt = f"{branch_code} {block_code} 용접 작업지시서와 취급 계획을 발행하기 위해 규격, 자재, 인력, 로봇, 기상을 통합 분석해줘"
    elif mode == "REVISE" and prompt and existing_wo:
        prompt = (
            f"[기존 작업지시서 조건 변경·개정 요청] 대상: {existing_wo.work_no} "
            f"(선박 {existing_wo.ship_code} / 블록 {existing_wo.block_code} / 현재 버전 {existing_inst.revision_no if existing_inst else 'Rev.1'})\n"
            f"변경 요청 사항: {prompt}\n"
            f"위 변경 조건만 반영하여 개정 작업지시서 초안을 산출해줘."
        )

    return render_template(
        "service/agent_analyze.html",
        me=user,
        prompt=prompt,
        mode=mode,
        branch_code=branch_code or (existing_wo.work_region if existing_wo else ""),
        block_code=block_code or (existing_wo.block_code if existing_wo else ""),
        work_id=work_id or (existing_wo.work_id if existing_wo else None),
        return_to=return_to,
        cancel_url=cancel_url,
        revise_waiting_input=revise_waiting_input,
        existing_wo=existing_wo,
        existing_inst=existing_inst,
        existing_dispatch=existing_dispatch,
    )


@service_bp.route("/agent/plan/<session_id>", methods=["GET"])
@login_required
def agent_plan_page(session_id: str):
    user = User.query.get_or_404(session["user_id"])
    last_ai = (
        AgentChatHistory.query
        .filter_by(user_id=user.user_id, session_id=session_id, role_code="ASSISTANT")
        .order_by(AgentChatHistory.message_id.desc())
        .first()
    )
    if not last_ai:
        flash("분석 결과를 찾을 수 없습니다. 다시 요청해 주세요.", "warning")
        return redirect(url_for("service.workspace_page"))

    answer = _read_clob(last_ai.content)
    saved_doc, ui_blocks, sources = {}, {}, []
    if last_ai.retrieval_chunks_json:
        for item in _parse_json_or_literal(last_ai.retrieval_chunks_json):
            if isinstance(item, dict) and item.get("_type") == "WORK_ORDER_FILES":
                saved_doc = item.get("doc") or {}
                if item.get("files") and not saved_doc.get("files"):
                    saved_doc["files"] = item.get("files")
            elif isinstance(item, dict) and item.get("_type") == "UI_RESULT_BLOCKS":
                ui_blocks = item
            else:
                sources.append(item)

    quick_workers = []
    quick_blocks = []
    try:
        quick_workers = (
            ErpWorkerCert.query
            .filter_by(status="VALID")
            .order_by(ErpWorkerCert.worker_name.asc())
            .limit(8)
            .all()
        )
        quick_blocks = (
            ErpBlockMaster.query
            .order_by(ErpBlockMaster.block_id.asc())
            .limit(8)
            .all()
        )
    except Exception:
        pass

    return render_template(
        "service/agent_plan.html",
        me=user,
        session_id=session_id,
        answer=answer,
        doc=saved_doc,
        ui_blocks=ui_blocks,
        sources=sources,
        quick_workers=quick_workers,
        quick_blocks=quick_blocks,
        cancel_url=url_for("vision.index") if request.args.get("from") == "vision" else url_for("service.workspace_page"),
        can_approve=(user.role in ("ADMIN", "MANAGER")),
    )


# =====================================================================
# 5. [SA-03] 작업 현황 (다일 진척률 · 상태 수정 · 확인 기반 완료 · 자동 활성화)
# =====================================================================
@service_bp.route("/work-status", methods=["GET"])
@login_required
def work_status_page():
    db.session.expire_all()
    user = User.query.get_or_404(session["user_id"])
    is_manager = user.role in ("ADMIN", "MANAGER")
    status_filter = (request.args.get("status") or "ALL").strip().upper()

    wo_q = WorkOrder.query
    if not is_manager:
        wo_q = wo_q.filter_by(requested_by=user.user_id)
    all_orders_raw = wo_q.order_by(WorkOrder.work_id.desc()).all()

    kpi = {
        "total": len(all_orders_raw),
        "in_progress": sum(1 for o in all_orders_raw if o.status_code == "IN_PROGRESS"),
        "ready": sum(1 for o in all_orders_raw if o.status_code == "READY"),
        "completed": sum(1 for o in all_orders_raw if o.status_code == "COMPLETED"),
    }

    orders_raw = [
        o for o in all_orders_raw
        if status_filter == "ALL" or o.status_code == status_filter
    ]

    prog_map = _safe_get_wo_progress_map([wo.work_id for wo in orders_raw])
    branches_all = ErpBranch.query.all()
    branch_map = {b.branch_code: b.branch_name for b in branches_all}
    branch_obj_map = {b.branch_code: b for b in branches_all}
    comp_map = {c.company_code: c.company_name for c in ErpCompany.query.all()}
    selected_id = request.args.get("work_id", type=int) or (orders_raw[0].work_id if orders_raw else None)

    all_plans_by_block = {}
    for p in _safe_query_work_plans(states=["PENDING", "IN_PROGRESS", "COMPLETED"], today_only=False):
        all_plans_by_block[p.block_code] = p

    orders_data, selected_detail = [], None
    for wo in orders_raw:
        insts = (
            WorkInstruction.query
            .filter_by(work_id=wo.work_id)
            .order_by(WorkInstruction.instruction_id.desc())
            .all()
        )
        latest_inst = insts[0] if insts else None
        pct, progress_timeline = _resolve_progress_pct(wo, latest_inst, prog_map)

        # 작업 지시 시간(KST) 및 예상 납기 시간 / 완료 시 실제 납기 시간
        issued_kst_dt = _to_kst(wo.requested_at) or _now_kst()
        issued_at_kst = issued_kst_dt.strftime("%Y-%m-%d %H:%M")
        linked_plan = all_plans_by_block.get(wo.block_code)
        if wo.status_code == "COMPLETED":
            expected_due_kst = _fmt_kst(wo.updated_at or wo.requested_at)
        elif linked_plan and linked_plan.planned_end_date:
            expected_due_kst = f"{linked_plan.planned_end_date} 18:00 (예상)"
        else:
            est_days = int(linked_plan.est_duration_days or 1) if linked_plan else 1
            expected_due_kst = (issued_kst_dt + timedelta(days=est_days)).strftime("%Y-%m-%d %H:%M") + " (예상)"

        item = {
            "wo": wo,
            "branch_name": branch_map.get(wo.work_region, wo.work_region),
            "progress_pct": pct,
            "progress_timeline": progress_timeline,
            "plan_id": prog_map.get(wo.work_id, {}).get("plan_id") or "",
            "latest_inst": latest_inst,
            "revisions": insts,
            "pdf_url": _extract_inst_pdf_url(latest_inst),
            "po_pdf_url": _extract_inst_po_pdf_url(latest_inst),
            "issued_at_kst": issued_at_kst,
            "expected_due_kst": expected_due_kst,
        }
        orders_data.append(item)
        if wo.work_id == selected_id:
            selected_detail = item

    if not selected_detail and orders_data:
        selected_detail = orders_data[0]

    queued_plans = _safe_query_work_plans(states=["PENDING"], today_only=False)
    dispatch_info, robot_info = None, None
    cad_doc_info = None
    yard_po_docs = []
    partner_tabs = []
    all_timeline_events = []
    partner_nav_cards = []

    if selected_detail:
        wo = selected_detail["wo"]
        yard_code = wo.work_region
        dest_branch = branch_obj_map.get(yard_code)
        dispatch_info = MesWorkOrderDispatch.query.filter_by(wo_number=wo.work_no).first()
        robot_info = RobotExecution.query.filter_by(work_id=wo.work_id).first()

        # 1) 실제 선박 선체 CAD 도면 선택 (프로펠러 등 비선체 부품 제외 + 호선/야드 매칭 우선)
        if is_manager or user.can_access_internal_rag == "Y":
            cad_rows_raw = (
                DocumentMetadata.query
                .filter(DocumentMetadata.category_code == "CAD_DRAWING", DocumentMetadata.is_active == "Y")
                .order_by(DocumentMetadata.document_id.desc())
                .all()
            )
            cad_rows = [c for c in cad_rows_raw if not _is_non_hull_cad_file(c.original_file_name, c.storage_uri)]

            chosen_cad = None
            # 1순위: CAD_DRAWINGS에서 ship_code 또는 branch_code 매칭되는 실제 선체 도면
            try:
                dwg = (
                    CadDrawing.query.filter_by(ship_code=wo.ship_code).order_by(CadDrawing.created_at.desc()).first()
                    or CadDrawing.query.filter_by(branch_code=yard_code).order_by(CadDrawing.created_at.desc()).first()
                )
                if dwg:
                    for c in cad_rows:
                        if (dwg.file_sha256 and c.file_hash_sha256 == dwg.file_sha256) or (
                            dwg.dxf_file_path and Path(dwg.dxf_file_path).name == c.original_file_name
                        ):
                            chosen_cad = c
                            break
            except Exception:
                pass

            # 2순위: 실제 선체 CAD (.dxf / .igs / .iges) 우선, 그 다음 선체 .pdf
            if not chosen_cad:
                for ext_pref in (".dxf", ".igs", ".iges", ".pdf"):
                    for c in cad_rows:
                        if Path(c.original_file_name or "").suffix.lower() == ext_pref:
                            chosen_cad = c
                            break
                    if chosen_cad:
                        break
            if not chosen_cad and cad_rows:
                chosen_cad = cad_rows[0]

            if chosen_cad:
                cad_doc_info = {
                    "document_id": chosen_cad.document_id,
                    "title": _clean_doc_title(chosen_cad.document_title, chosen_cad.original_file_name),
                    "file_name": chosen_cad.original_file_name,
                    "ext": Path(chosen_cad.original_file_name or "").suffix.lstrip(".").upper(),
                }

        # 2) 해당 야드의 실제 발주 계획서 PDF만 수집 (일반 작업지시서와 절대 혼동 금지, KST 표기)
        yard_orders = WorkOrder.query.filter_by(work_region=yard_code).order_by(WorkOrder.work_id.desc()).all()
        yard_wo_nos = {o.work_no for o in yard_orders}
        seen_po_urls = set()

        for y_wo in yard_orders:
            y_insts = (
                WorkInstruction.query
                .filter_by(work_id=y_wo.work_id)
                .order_by(WorkInstruction.instruction_id.desc())
                .all()
            )
            for y_inst in y_insts:
                purl = _extract_inst_po_pdf_url(y_inst)
                wo_url = _extract_inst_pdf_url(y_inst)
                if purl and purl != wo_url and purl not in seen_po_urls:
                    seen_po_urls.add(purl)
                    yard_po_docs.append({
                        "title": f"{y_wo.work_no} ({y_wo.block_code}) 발주 계획서",
                        "pdf_url": purl,
                        "revision_no": y_inst.revision_no,
                        "created_at": _fmt_kst(y_inst.created_at),
                        "is_current": (y_wo.work_id == wo.work_id),
                    })

        # 3) 협력사 발주 납기 및 처리 이력 (엄격히 해당 야드 작업번호 소속 B2B 발주만 조회 + 좌표 기반 네비 ETA 계산)
        b2b_rows = []
        partner_loc_map = {}
        lead_hours_map = {}
        try:
            for loc in B2BCompanyLocation.query.all():
                partner_loc_map[loc.partner_id] = loc
            for m in B2BDaehanMaterial.query.all():
                lead_hours_map[m.hq_item_code] = float(m.lead_time_hours or 1.5)
            all_b2b = B2BSalesOrder.query.order_by(B2BSalesOrder.created_at.desc()).all()
            for r in all_b2b:
                if r.hq_wo_number == wo.work_no or r.hq_wo_number in yard_wo_nos:
                    b2b_rows.append(r)
        except Exception:
            b2b_rows = []

        partner_groups: dict[str, dict] = {}
        for po in b2b_rows:
            pid = po.partner_id
            pname = comp_map.get(pid, pid)
            ploc = partner_loc_map.get(pid)
            if pid not in partner_groups:
                partner_groups[pid] = {
                    "partner_id": pid,
                    "partner_name": pname,
                    "order_count": 0,
                    "events": [],
                }
            partner_groups[pid]["order_count"] += 1

            base_kst = _to_kst(po.created_at) or _now_kst()
            dt_str = base_kst.strftime("%Y-%m-%d %H:%M")
            prep_h = lead_hours_map.get(po.ordered_ref_code, 1.0)
            nav = _calc_nav_eta(
                ploc.latitude if ploc else None,
                ploc.longitude if ploc else None,
                dest_branch.latitude if dest_branch else None,
                dest_branch.longitude if dest_branch else None,
                base_dt=po.created_at,
                prep_hours=prep_h,
            )
            type_ko = "자재 발주" if po.order_type == "WELDING_MATERIAL" else ("인력 파견" if po.order_type == "OUTSOURCED_WORKER" else "장비 배차")
            is_done = po.order_status in ("DELIVERED", "COMPLETED")
            dist_desc = (
                f"예상 도로거리 {nav['road_km']}km · 예상 이동 {nav['drive_label']} (출하준비 {nav['prep_hours']}h 포함)"
                if nav["road_km"] is not None
                else f"예상 이동 {nav['drive_label']} (출하준비 {nav['prep_hours']}h 포함)"
            )

            partner_nav_cards.append({
                "b2b_order_no": po.b2b_order_no,
                "partner_name": pname,
                "type_ko": type_ko,
                "ref_code": po.ordered_ref_code,
                "qty": float(po.order_qty or 1),
                "origin_addr": ploc.address if (ploc and ploc.address) else pname,
                "dest_addr": dest_branch.address if (dest_branch and dest_branch.address) else branch_map.get(yard_code, yard_code),
                "road_km": nav["road_km"],
                "drive_label": nav["drive_label"],
                "ordered_at_kst": dt_str,
                "eta_str": nav["eta_str"],
                "is_done": is_done,
                "order_status": po.order_status,
            })

            ev1 = {
                "time": dt_str,
                "title": f"[{po.b2b_order_no}] {type_ko} 접수",
                "detail": f"대상 작업 {po.hq_wo_number} · 품목/코드 {po.ordered_ref_code} (수량 {po.order_qty}) · 납품 구역 {po.delivery_yard_zone}",
                "state": "done",
                "partner_name": pname,
            }
            ev2 = {
                "time": (base_kst + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M"),
                "title": f"{pname} 수주 확인 및 출하 이동 ({po.order_status})",
                "detail": f"배정 Lot/식별번호: {po.allocated_lot_or_id} · {dist_desc}",
                "state": "done" if po.order_status in ("CONFIRMED", "SHIPPED", "DELIVERED", "COMPLETED") else "active",
                "partner_name": pname,
            }
            ev3 = {
                "time": nav["eta_str"] + (" (납기 완료)" if is_done else " (예상 도착)"),
                "title": "야드 도착 및 현장 입고 검수" + (" 완료" if is_done else " 예정"),
                "detail": f"도착 야드: {branch_map.get(yard_code, yard_code)} · {po.delivery_yard_zone}",
                "state": "done" if is_done else "pending",
                "partner_name": pname,
            }

            partner_groups[pid]["events"].extend([ev1, ev2, ev3])
            all_timeline_events.extend([ev1, ev2, ev3])

        # 작업지시서 발행/개정 이력 (KST)
        for rev in reversed(selected_detail["revisions"]):
            r_time = _fmt_kst(rev.created_at)
            all_timeline_events.append({
                "time": r_time,
                "title": f"작업지시서 {rev.revision_no} ({'승인 완료' if rev.approval_status == 'APPROVED' else '승인 대기'})",
                "detail": f"작업번호 {wo.work_no} · {wo.ship_code} / {wo.block_code}",
                "state": "done" if rev.approval_status == "APPROVED" else "active",
                "partner_name": "본사 승인",
            })

        all_timeline_events.sort(key=lambda x: x["time"])
        partner_tabs = list(partner_groups.values())

    return render_template(
        "service/work_status.html",
        me=user,
        is_manager=is_manager,
        kpi=kpi,
        status_filter=status_filter,
        orders=orders_data,
        selected=selected_detail,
        queued_plans=queued_plans,
        dispatch_info=dispatch_info,
        robot_info=robot_info,
        cad_doc_info=cad_doc_info,
        yard_po_docs=yard_po_docs,
        partner_tabs=partner_tabs,
        partner_nav_cards=partner_nav_cards,
        all_timeline_events=all_timeline_events,
    )


@service_bp.route("/api/work-status/progress", methods=["POST"])
@login_required
def api_update_work_progress():
    """다일 공정 진척률·상태 수정 (100% 완료 후에도 되돌리기/재조정 가능, 완료 시 다음 블록 자동 활성화)"""
    user = User.query.get_or_404(session["user_id"])
    if user.role not in ("ADMIN", "MANAGER"):
        return jsonify({"status": "FORBIDDEN", "message": "진척률 제어는 담당자/관리자만 가능합니다."}), 403

    data = request.get_json(silent=True) or {}
    work_id = data.get("work_id")
    progress_pct = max(0, min(100, int(data.get("progress_pct", 0))))
    requested_status = (data.get("status_code") or "").strip().upper()
    note = (data.get("note") or "").strip()
    completion_type = (data.get("completion_type") or "").strip().upper()

    wo = WorkOrder.query.get_or_404(work_id)

    # 상태 결정 (명시적 상태가 오면 반영하고, 진척률과 정합성 맞춤)
    if requested_status in ("READY", "IN_PROGRESS", "COMPLETED", "SUSPENDED"):
        wo.status_code = requested_status
        if requested_status == "COMPLETED":
            progress_pct = 100
        elif requested_status == "READY" and progress_pct == 100:
            progress_pct = 0
        elif requested_status == "IN_PROGRESS" and progress_pct == 100:
            progress_pct = 90
    else:
        if progress_pct >= 100:
            wo.status_code = "COMPLETED"
        elif progress_pct == 0:
            wo.status_code = "READY"
        else:
            wo.status_code = "IN_PROGRESS"

    wo.updated_at = datetime.utcnow()

    # 1) WORK_ORDERS.PROGRESS_PCT 컬럼이 있으면 저장 (SAVEPOINT로 안전 처리)
    _safe_update_wo_progress(wo.work_id, progress_pct)

    # 2) 최신 WORK_INSTRUCTIONS의 EXTERNAL_ENV_JSON에도 진척률·이력 영구 기록 (미마이그레이션 DB에서도 100% 유지)
    latest_inst = (
        WorkInstruction.query
        .filter_by(work_id=wo.work_id)
        .order_by(WorkInstruction.instruction_id.desc())
        .first()
    )
    if latest_inst:
        meta = _load_env_meta(latest_inst)
        pmeta = meta.get("_progress_meta") if isinstance(meta.get("_progress_meta"), dict) else {}
        timeline = pmeta.get("timeline") if isinstance(pmeta.get("timeline"), list) else []
        tag = "조기 완료" if completion_type == "EARLY" else ("정상 완료" if completion_type == "NORMAL" else "진척 수정")
        timeline.insert(0, {
            "at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "actor": user.full_name,
            "tag": tag,
            "progress_pct": progress_pct,
            "status_code": wo.status_code,
            "note": note or f"진척률 {progress_pct}% ({wo.status_code}) 반영",
        })
        pmeta["progress_pct"] = progress_pct
        pmeta["timeline"] = timeline[:20]
        meta["_progress_meta"] = pmeta
        latest_inst.external_env_json = json.dumps(meta, ensure_ascii=False, default=str)

    # 3) WORK_PLANS 연동 (완료 시 다음 순위 활성화, 되돌리기 시 IN_PROGRESS 복원)
    activated_next_block = None
    try:
        with db.session.begin_nested():
            plan = WorkPlan.query.filter(
                or_(WorkPlan.assigned_wo_number == wo.work_no, WorkPlan.block_code == wo.block_code)
            ).first()
            if plan:
                plan.progress_pct = progress_pct
                if wo.status_code == "COMPLETED":
                    plan.plan_state = "COMPLETED"
                    next_plan = (
                        WorkPlan.query
                        .filter(
                            WorkPlan.drawing_id == plan.drawing_id,
                            WorkPlan.plan_state == "PENDING",
                            WorkPlan.priority_rank > plan.priority_rank,
                        )
                        .order_by(WorkPlan.priority_rank.asc())
                        .first()
                    )
                    if next_plan:
                        next_plan.plan_state = "IN_PROGRESS"
                        activated_next_block = next_plan.block_code
                elif wo.status_code == "READY":
                    plan.plan_state = "PENDING"
                else:
                    plan.plan_state = "IN_PROGRESS"
    except Exception:
        pass

    db.session.commit()
    return jsonify({
        "status": "SUCCESS",
        "work_id": wo.work_id,
        "progress_pct": progress_pct,
        "status_code": wo.status_code,
        "activated_next_block": activated_next_block,
    })


@service_bp.route("/api/work-status/<int:work_id>/robot-payload", methods=["GET"])
@login_required
def api_work_robot_payload(work_id: int):
    """2차 프로젝트(ROS2 / MoveIt2 / VLM / VLA / Manipulator) 연동용 표준 JSON 계약"""
    wo = WorkOrder.query.get_or_404(work_id)
    inst = WorkInstruction.query.filter_by(work_id=wo.work_id).order_by(WorkInstruction.instruction_id.desc()).first()
    insp = VisionInspection.query.filter_by(work_id=wo.work_id).order_by(VisionInspection.inspection_id.desc()).first()
    det = VisionDetection.query.get(insp.detection_id) if insp else None
    robot = RobotExecution.query.filter_by(work_id=wo.work_id).first()

    welding_params = {}
    if inst and inst.welding_params_json:
        try:
            welding_params = json.loads(_read_clob(inst.welding_params_json))
        except Exception:
            welding_params = {}

    return jsonify({
        "schema_version": "phase2-ros2-vla-1.0",
        "work_order": {
            "work_id": wo.work_id,
            "work_no": wo.work_no,
            "ship_code": wo.ship_code,
            "block_code": wo.block_code,
            "branch_code": wo.work_region,
            "work_zone": wo.work_zone,
            "process_code": wo.process_code,
            "material_grade": wo.material_grade,
            "thickness_mm": float(wo.thickness_mm) if wo.thickness_mm is not None else 0.0,
            "weld_position": wo.weld_position,
            "status_code": wo.status_code,
        },
        "instruction": {
            "instruction_id": inst.instruction_id if inst else None,
            "revision_no": inst.revision_no if inst else None,
            "approval_status": inst.approval_status if inst else None,
            "welding_params": welding_params,
        },
        "vision_trajectory": {
            "inspection_id": insp.inspection_id if insp else None,
            "ocr_text": insp.ocr_text if insp else None,
            "start_point": json.loads(insp.start_point_json) if (insp and insp.start_point_json) else None,
            "end_point": json.loads(insp.end_point_json) if (insp and insp.end_point_json) else None,
            "centerline_waypoints": json.loads(_read_clob(det.centerline_waypoints_json)) if (det and det.centerline_waypoints_json) else [],
            "safety_clearance": (insp.safety_clearance_yn == "Y") if insp else False,
        },
        "robot_telemetry": {
            "equipment_code": robot.equipment_code if robot else None,
            "robot_status": robot.robot_status if robot else "UNASSIGNED",
            "ros2_node_status": robot.ros2_node_status if robot else "DISCONNECTED",
            "emergency_stop": (robot.emergency_stop_yn == "Y") if robot else False,
        },
    })


# =====================================================================
# 6. [SR-01a / SR-01b / SR-01c] 자료실 · DB 동기화 · PDF 적재 · CAD AI 큐
# =====================================================================
def _group_cad_documents(cad_rows: list[DocumentMetadata]) -> list[dict]:
    """같은 선박·도면 모델의 여러 파일 형식(PDF/DXF/IGS/STEP 등)을 하나의 카드로 묶어 반환"""
    from modules.knowledge.cad_parser import detect_vessel_context

    groups: dict[str, dict] = {}
    for d in cad_rows:
        if _is_non_hull_cad_file(d.original_file_name, d.storage_uri):
            continue
        abs_p = _resolve_doc_abs_path(d.storage_uri)
        p_obj = abs_p or Path(d.storage_uri or d.original_file_name or "")
        ext = Path(d.original_file_name or "").suffix.lstrip(".").upper() or "FILE"
        stem = Path(d.original_file_name or "").stem
        vessel_name, vessel_desc = detect_vessel_context(p_obj)
        vessel_desc = vessel_desc.replace("·프로펠러(KP505)·러더", "").replace("·프로펠러(KP458)·러더", "")

        # 같은 파일명(확장자만 다름) 또는 같은 선박 모델 폴더 단위로 묶음
        norm_key = vessel_name.strip().lower() if vessel_name and vessel_name != "선박 도면" else stem.strip().lower()
        if norm_key not in groups:
            groups[norm_key] = {
                "group_title": vessel_name if (vessel_name and vessel_name != "선박 도면") else _clean_doc_title(d.document_title, d.original_file_name),
                "group_desc": vessel_desc,
                "primary_doc_id": d.document_id,
                "pdf_doc_id": None,
                "dxf_doc_id": None,
                "formats": [],
                "files": [],
                "is_active": d.is_active,
                "ingest_status": d.ingest_status,
                "created_at": d.created_at.strftime("%Y-%m-%d") if d.created_at else "-",
            }

        g = groups[norm_key]
        if ext not in g["formats"]:
            g["formats"].append(ext)
        if ext == "PDF" and not g["pdf_doc_id"]:
            g["pdf_doc_id"] = d.document_id
        if ext == "DXF" and not g["dxf_doc_id"]:
            g["dxf_doc_id"] = d.document_id
        # PDF가 없더라도 같은 경로에 동명 .pdf가 있으면 미리보기 우선 지정
        if not g["pdf_doc_id"] and abs_p and abs_p.with_suffix(".pdf").is_file():
            g["pdf_doc_id"] = d.document_id

        g["files"].append({
            "document_id": d.document_id,
            "document_code": d.document_code,
            "title": _clean_doc_title(d.document_title, d.original_file_name),
            "file_name": d.original_file_name,
            "ext": ext,
            "is_active": d.is_active,
        })

    result = list(groups.values())
    for g in result:
        # 미리보기 우선순위: 실제 선체 DXF 2D 청사진 > 선체 PDF > 첫 번째 파일
        g["preview_doc_id"] = g["dxf_doc_id"] or g["pdf_doc_id"] or g["primary_doc_id"]
        g["has_pdf"] = bool(g["pdf_doc_id"] and not g["dxf_doc_id"])
    return result


@service_bp.route("/library", methods=["GET"])
@login_required
def library_page():
    db.session.expire_all()
    user = User.query.get_or_404(session["user_id"])
    is_admin = user.role in ("ADMIN", "MANAGER")
    can_internal = (user.can_access_internal_rag == "Y") or is_admin

    # 사내 보안 탭 제거 -> 공통 기술문서 / 사내 규정 / 선박 도면(CAD) 3개 깔끔 구성
    allowed_tabs = ["public"]
    if can_internal:
        allowed_tabs.append("internal")
    if is_admin or can_internal:
        allowed_tabs.append("cad")

    tab = (request.args.get("tab") or "public").strip().lower()
    if tab not in allowed_tabs:
        tab = allowed_tabs[0]
    q = (request.args.get("q") or "").strip()
    page = max(1, request.args.get("page", 1, type=int))
    per_page = 10

    base_q = DocumentMetadata.query if is_admin else DocumentMetadata.query.filter(DocumentMetadata.is_active == "Y")
    all_cad_rows_raw = (
        DocumentMetadata.query
        .filter(DocumentMetadata.category_code == "CAD_DRAWING", DocumentMetadata.is_active == "Y")
        .order_by(DocumentMetadata.document_id.desc())
        .all()
    )
    all_cad_rows = [r for r in all_cad_rows_raw if not _is_non_hull_cad_file(r.original_file_name, r.storage_uri)]
    cad_groups_all = _group_cad_documents(all_cad_rows)

    tab_counts = {
        "public": base_q.filter(DocumentMetadata.collection_type == "PUBLIC").count(),
        "internal": base_q.filter(
            DocumentMetadata.collection_type == "INTERNAL",
            DocumentMetadata.category_code != "CAD_DRAWING",
        ).count() if can_internal else 0,
        "cad": len(cad_groups_all) if (is_admin or can_internal) else 0,
    }

    docs = []
    cad_groups = []
    if tab == "cad":
        filtered = cad_groups_all
        if q:
            ql = q.lower()
            filtered = [
                g for g in cad_groups_all
                if ql in g["group_title"].lower()
                or any(ql in (f["file_name"] or "").lower() or ql in (f["title"] or "").lower() for f in g["files"])
            ]
        total = len(filtered)
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = min(page, total_pages)
        cad_groups = filtered[(page - 1) * per_page : page * per_page]
    else:
        doc_q = base_q
        if tab == "public":
            doc_q = doc_q.filter(DocumentMetadata.collection_type == "PUBLIC")
        else:
            doc_q = doc_q.filter(
                DocumentMetadata.collection_type == "INTERNAL",
                DocumentMetadata.category_code != "CAD_DRAWING",
            )
        if q:
            like_q = f"%{q}%"
            doc_q = doc_q.filter(or_(
                DocumentMetadata.document_title.ilike(like_q),
                DocumentMetadata.document_code.ilike(like_q),
                DocumentMetadata.original_file_name.ilike(like_q),
            ))
        total = doc_q.count()
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = min(page, total_pages)
        docs_raw = (
            doc_q.order_by(DocumentMetadata.document_id.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        docs = [
            {
                "document_id": d.document_id,
                "document_code": d.document_code,
                "document_title": _clean_doc_title(d.document_title, d.original_file_name),
                "collection_type": d.collection_type,
                "category_code": d.category_code,
                "original_file_name": d.original_file_name,
                "ingest_status": d.ingest_status,
                "is_active": d.is_active,
                "created_at": d.created_at.strftime("%Y-%m-%d") if d.created_at else "-",
            }
            for d in docs_raw
        ]

    return render_template(
        "service/library.html",
        me=user,
        is_admin=is_admin,
        allowed_tabs=allowed_tabs,
        tab_counts=tab_counts,
        tab=tab,
        q=q,
        docs=docs,
        cad_groups=cad_groups,
        page=page,
        total_pages=total_pages,
        total_count=total,
        page_items=_build_page_items(page, total_pages),
    )



def _extract_iges_wireframe(igs_path: Path, max_segments: int = 1200) -> dict:
    """
    순수 파이썬 IGES(.igs/.iges) 3D 와이어프레임 추출기 (브라우저 Canvas 3D 뷰어용).
    Entity 110(Line), 126(B-Spline Curve), 128(B-Spline Surface)에서 3D 선분을 샘플링합니다.
    """
    try:
        raw_lines = igs_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return {"segments": [], "entity_count": 0}

    # Parameter Data(col 73 == 'P')을 엔티티 포인터(col 65..72) 기준으로 결합
    param_chunks: dict[str, list[str]] = {}
    for ln in raw_lines:
        if len(ln) >= 73 and ln[72] == "P":
            ptr = ln[64:72].strip()
            body = ln[:64].strip()
            if ptr:
                param_chunks.setdefault(ptr, []).append(body)

    segments: list[list[float]] = []
    entity_count = 0

    def _to_float(tok: str) -> float | None:
        try:
            return float(tok.replace("D", "E").replace("d", "e"))
        except Exception:
            return None

    for ptr, parts in param_chunks.items():
        if len(segments) >= max_segments:
            break
        joined = "".join(parts)
        if ";" in joined:
            joined = joined.split(";", 1)[0]
        tokens = [t.strip() for t in joined.split(",") if t.strip() != ""]
        if not tokens:
            continue
        etype = tokens[0]
        if etype == "110" and len(tokens) >= 7:
            vals = [_to_float(tokens[i]) for i in range(1, 7)]
            if all(v is not None for v in vals):
                segments.append([vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]])
                entity_count += 1
        elif etype == "126" and len(tokens) >= 12:
            # 126, K, M, prop1..4, knots(K+M+2), weights(K+1), ctrl_pts(3*(K+1))
            try:
                k_val = int(float(tokens[1]))
                m_val = int(float(tokens[2]))
            except Exception:
                continue
            if k_val < 1 or k_val > 500 or m_val < 0 or m_val > 20:
                continue
            n_pts = k_val + 1
            knot_cnt = k_val + m_val + 2
            start_idx = 7 + knot_cnt + n_pts
            if start_idx + n_pts * 3 > len(tokens):
                continue
            pts = []
            step = max(1, n_pts // 16)
            for idx in range(0, n_pts, step):
                base = start_idx + idx * 3
                x = _to_float(tokens[base])
                y = _to_float(tokens[base + 1])
                z = _to_float(tokens[base + 2])
                if x is not None and y is not None and z is not None:
                    pts.append((x, y, z))
            # 끝점 포함
            if (n_pts - 1) % step != 0:
                base = start_idx + (n_pts - 1) * 3
                x = _to_float(tokens[base])
                y = _to_float(tokens[base + 1])
                z = _to_float(tokens[base + 2])
                if x is not None and y is not None and z is not None:
                    pts.append((x, y, z))
            for i in range(len(pts) - 1):
                if len(segments) >= max_segments:
                    break
                p1, p2 = pts[i], pts[i + 1]
                segments.append([p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]])
            if pts:
                entity_count += 1
        elif etype == "128" and len(tokens) >= 20:
            # 128, K1, K2, M1, M2, prop1..5, knots1(K1+M1+2), knots2(K2+M2+2), weights((K1+1)*(K2+1)), ctrl_pts
            try:
                k1 = int(float(tokens[1]))
                k2 = int(float(tokens[2]))
                m1 = int(float(tokens[3]))
                m2 = int(float(tokens[4]))
            except Exception:
                continue
            if k1 < 1 or k2 < 1 or k1 > 120 or k2 > 120:
                continue
            n1, n2 = k1 + 1, k2 + 1
            start_idx = 10 + (k1 + m1 + 2) + (k2 + m2 + 2) + (n1 * n2)
            if start_idx + n1 * n2 * 3 > len(tokens):
                continue
            # 외곽 경계 u=0, u=k1, v=0, v=k2 만 추출해 선체 윤곽선 표현
            grid = {}
            for j in range(n2):
                for i in range(n1):
                    if i in (0, k1) or j in (0, k2) or (i % max(1, n1 // 4) == 0 and j % max(1, n2 // 4) == 0):
                        base = start_idx + (j * n1 + i) * 3
                        x = _to_float(tokens[base])
                        y = _to_float(tokens[base + 1])
                        z = _to_float(tokens[base + 2])
                        if x is not None and y is not None and z is not None:
                            grid[(i, j)] = (x, y, z)
            for j in (0, k2):
                prev = None
                for i in range(n1):
                    cur = grid.get((i, j))
                    if cur and prev and len(segments) < max_segments:
                        segments.append([prev[0], prev[1], prev[2], cur[0], cur[1], cur[2]])
                    if cur:
                        prev = cur
            for i in (0, k1):
                prev = None
                for j in range(n2):
                    cur = grid.get((i, j))
                    if cur and prev and len(segments) < max_segments:
                        segments.append([prev[0], prev[1], prev[2], cur[0], cur[1], cur[2]])
                    if cur:
                        prev = cur
            entity_count += 1

    if not segments:
        return {"segments": [], "entity_count": entity_count}

    # 좌표 정규화 (-1.0 ~ 1.0)
    xs = [s[0] for s in segments] + [s[3] for s in segments]
    ys = [s[1] for s in segments] + [s[4] for s in segments]
    zs = [s[2] for s in segments] + [s[5] for s in segments]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-6)
    scale = 1.8 / span

    norm_segs = [
        [
            round((s[0] - cx) * scale, 4),
            round((s[1] - cy) * scale, 4),
            round((s[2] - cz) * scale, 4),
            round((s[3] - cx) * scale, 4),
            round((s[4] - cy) * scale, 4),
            round((s[5] - cz) * scale, 4),
        ]
        for s in segments
    ]
    return {
        "segments": norm_segs,
        "entity_count": entity_count,
        "raw_bbox": {
            "x": round(max(xs) - min(xs), 2),
            "y": round(max(ys) - min(ys), 2),
            "z": round(max(zs) - min(zs), 2),
        },
    }


@service_bp.route("/api/cad/preview/<int:doc_id>", methods=["GET"])
@login_required
def cad_preview_api(doc_id: int):
    """선박 도면(CAD) 브라우저 미리보기 API (PDF 인라인 / DXF 2D 도면 / 3D 제원 요약)"""
    user = User.query.get_or_404(session["user_id"])
    if not (user.role in ("ADMIN", "MANAGER") or user.can_access_internal_rag == "Y"):
        return jsonify({"status": "FORBIDDEN", "message": "열람 권한이 없습니다."}), 403

    doc = DocumentMetadata.query.filter_by(document_id=doc_id).first_or_404()
    abs_p = _resolve_doc_abs_path(doc.storage_uri)
    title = _clean_doc_title(doc.document_title, doc.original_file_name)
    if not abs_p or not abs_p.exists():
        return jsonify({
            "status": "SUCCESS",
            "title": title,
            "file_name": doc.original_file_name,
            "mode": "missing",
            "pdf_url": "",
            "svg": "",
            "summary": {"format": "파일 미존재"},
            "download_url": "",
        })

    from modules.knowledge.cad_parser import extract_cad_summary, render_dxf_to_svg

    ext = abs_p.suffix.lower()
    pdf_url = ""
    svg_html = ""
    mode = "spec"
    summary = {}

    if ext == ".pdf":
        mode = "pdf"
        pdf_url = f"/api/document/file/{doc.document_id}"
    elif abs_p.with_suffix(".pdf").is_file() and not _is_non_hull_cad_file(abs_p.with_suffix(".pdf").name, str(abs_p.with_suffix(".pdf"))):
        mode = "pdf"
        pdf_url = f"/api/document/file/{doc.document_id}?preview_pdf=1"
    elif ext == ".dxf":
        mode = "svg"
        try:
            svg_html = render_dxf_to_svg(abs_p, cache_dir=PROJECT_ROOT / "data" / "rag_cache" / "cad_svg")
        except Exception:
            svg_html = ""
        try:
            summary = extract_cad_summary(abs_p)
        except Exception:
            summary = {"format": "DXF 2D 도면"}
    elif ext in (".igs", ".iges"):
        wireframe_3d = _extract_iges_wireframe(abs_p)
        mode = "iges3d" if wireframe_3d.get("segments") else "spec"
        try:
            summary = extract_cad_summary(abs_p)
        except Exception:
            summary = {"format": "IGES 3D CAD"}
    else:
        mode = "spec"
        wireframe_3d = None
        try:
            summary = extract_cad_summary(abs_p)
        except Exception:
            summary = {"format": ext.lstrip(".").upper()}

    if "wireframe_3d" not in locals():
        wireframe_3d = None

    return jsonify({
        "status": "SUCCESS",
        "title": title,
        "file_name": doc.original_file_name,
        "mode": mode,
        "pdf_url": pdf_url,
        "svg": svg_html,
        "wireframe_3d": wireframe_3d,
        "summary": summary,
        "download_url": f"/api/document/file/{doc.document_id}?download=1",
    })


@service_bp.route("/api/library/sync-db", methods=["POST"])
@login_required
def api_library_sync_db():
    """자료실 폴더(raw_pdfs / internal_pdfs / cad_drawings) ↔ DB(DOCUMENT_METADATA) 통합 동기화"""
    import hashlib
    user = User.query.get_or_404(session["user_id"])
    if user.role not in ("ADMIN", "MANAGER"):
        return jsonify({"status": "FORBIDDEN", "message": "DB 동기화는 관리자/담당자만 실행할 수 있습니다."}), 403

    try:
        synced_pdf, synced_cad, deactivated = 0, 0, 0

        # 1) 디스크에서 삭제된 문서 비활성화 처리 (DB 이력 보존)
        for row in DocumentMetadata.query.all():
            abs_p = _resolve_doc_abs_path(row.storage_uri)
            is_non_hull = (row.category_code == "CAD_DRAWING" and _is_non_hull_cad_file(row.original_file_name, row.storage_uri))
            if abs_p is None or not abs_p.exists() or is_non_hull:
                if row.is_active == "Y":
                    row.is_active = "N"
                    deactivated += 1

        # 2) PDF 폴더 동기화 (raw_pdfs -> PUBLIC, internal_pdfs -> INTERNAL)
        pdf_specs = [
            (PROJECT_ROOT / "data" / "raw_pdfs", "PUBLIC", "col_public_weld_std"),
            (PROJECT_ROOT / "data" / "internal_pdfs", "INTERNAL", "col_internal_ship_sop"),
        ]
        for folder, col_type, vec_col in pdf_specs:
            if not folder.exists():
                continue
            for p in sorted(folder.rglob("*.pdf")):
                if not p.is_file():
                    continue
                fhash = hashlib.sha256(p.read_bytes()).hexdigest()
                rel_uri = p.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()[:500]
                existing = DocumentMetadata.query.filter_by(file_hash_sha256=fhash, collection_type=col_type).first()
                if existing:
                    existing.storage_uri = rel_uri
                    existing.is_active = "Y"
                    existing.ingest_status = "COMPLETED"
                else:
                    prefix = "DOC-PUB" if col_type == "PUBLIC" else "DOC-INT"
                    cat = "SAFETY" if any(k in p.name.lower() for k in ("안전", "kosha", "osha", "화기")) else "WELDING_SOP"
                    code = f"{prefix}-{fhash[:10].upper()}"
                    if DocumentMetadata.query.filter_by(document_code=code).first():
                        code = f"{prefix}-{uuid.uuid4().hex[:10].upper()}"
                    db.session.add(DocumentMetadata(
                        document_code=code,
                        document_title=p.stem.replace("_", " ")[:200],
                        collection_type=col_type,
                        category_code=cat,
                        original_file_name=p.name[:255],
                        storage_uri=rel_uri,
                        file_hash_sha256=fhash,
                        vector_collection=vec_col,
                        ingest_status="COMPLETED",
                        is_active="Y",
                        uploaded_by=user.user_id,
                    ))
                    synced_pdf += 1

        # 3) CAD 폴더 동기화 (data/cad_drawings)
        cad_dir = PROJECT_ROOT / "data" / "cad_drawings"
        supported_cad = {".dxf", ".pdf", ".igs", ".iges", ".stp", ".step", ".stl", ".dwg"}
        if cad_dir.exists():
            for p in sorted(cad_dir.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in supported_cad:
                    continue
                if any(ex in {pt.lower() for pt in p.parts} for ex in ("github_pygem", "__macosx")):
                    continue
                if _is_non_hull_cad_file(p.name, str(p)):
                    continue
                fhash = hashlib.sha256(p.read_bytes()).hexdigest()
                rel_uri = p.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()[:500]
                existing = DocumentMetadata.query.filter_by(file_hash_sha256=fhash, category_code="CAD_DRAWING").first()
                if existing:
                    existing.storage_uri = rel_uri
                    existing.is_active = "Y"
                else:
                    code = f"CAD-INT-{fhash[:10].upper()}"
                    if DocumentMetadata.query.filter_by(document_code=code).first():
                        code = f"CAD-INT-{uuid.uuid4().hex[:10].upper()}"
                    db.session.add(DocumentMetadata(
                        document_code=code,
                        document_title=f"{p.stem} ({p.suffix.lstrip('.').upper()})"[:200],
                        collection_type="INTERNAL",
                        category_code="CAD_DRAWING",
                        original_file_name=p.name[:255],
                        storage_uri=rel_uri,
                        file_hash_sha256=fhash,
                        vector_collection="col_internal_ship_sop",
                        ingest_status="COMPLETED",
                        is_active="Y",
                        uploaded_by=user.user_id,
                    ))
                    synced_cad += 1

        db.session.commit()
        return jsonify({
            "status": "SUCCESS",
            "message": f"DB 동기화 완료 — 신규 PDF {synced_pdf}건, 신규 CAD {synced_cad}건 등록, 미존재 비활성 {deactivated}건 처리됨.",
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "ERROR", "message": f"DB 동기화 중 오류: {str(e)}"}), 500


@service_bp.route("/library/upload-pdf", methods=["GET"])
@login_required
def library_upload_pdf_page():
    user = User.query.get_or_404(session["user_id"])
    if user.role not in ("ADMIN", "MANAGER"):
        flash("문서 등록은 담당자(관리자/팀장)만 이용할 수 있습니다.", "warning")
        return redirect(url_for("service.library_page"))
    return render_template("service/library_upload_pdf.html", me=user)


@service_bp.route("/library/upload-cad", methods=["GET"])
@login_required
def library_upload_cad_page():
    user = User.query.get_or_404(session["user_id"])
    if user.role not in ("ADMIN", "MANAGER"):
        flash("선박 도면 등록은 담당자만 이용할 수 있습니다.", "warning")
        return redirect(url_for("service.library_page"))
    branches = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).all()
    branch_map = {b.branch_code: b.branch_name for b in branches}
    existing_drawings = _safe_query_cad_drawings(limit=20)
    return render_template(
        "service/library_upload_cad.html",
        me=user,
        branches=branches,
        branch_map=branch_map,
        existing_drawings=existing_drawings,
    )


@service_bp.route("/api/library/cad-upload", methods=["POST"])
@login_required
def api_library_cad_upload():
    """
    선박 CAD 도면(DXF / PDF) 등록 API:
    - SHA-256 해시로 DOCUMENT_METADATA 및 CAD_DRAWINGS 중복 여부를 엄격히 검사
    - 이미 등록된 동일 파일이면 즉시 409 거절 (중복 적재 차단)
    - 작업 순서 산출/질의는 통합 업무(AI Agent)에서 수행
    """
    import hashlib
    from werkzeug.utils import secure_filename

    user = User.query.get_or_404(session["user_id"])
    if user.role not in ("ADMIN", "MANAGER"):
        return jsonify({"status": "FORBIDDEN", "message": "권한이 없습니다."}), 403

    dxf_file = request.files.get("dxf_file")
    pdf_file = request.files.get("pdf_file")
    branch_code = (request.form.get("branch_code") or "").strip()
    ship_code = (request.form.get("ship_code") or "").strip()
    work_zone = (request.form.get("work_zone") or "ZONE-A-BAY1").strip()
    custom_title = (request.form.get("drawing_title") or "").strip()

    if not dxf_file or not dxf_file.filename:
        return jsonify({"status": "ERROR", "message": "등록할 CAD 도면 파일(DXF)을 선택해 주세요."}), 400
    if not branch_code:
        return jsonify({"status": "ERROR", "message": "배치 야드를 선택해 주세요."}), 400

    dxf_bytes = dxf_file.read()
    if not dxf_bytes:
        return jsonify({"status": "ERROR", "message": "빈 파일은 등록할 수 없습니다."}), 400

    sha256 = hashlib.sha256(dxf_bytes).hexdigest()

    # 1) DOCUMENT_METADATA 중복 검사 (이미 data/cad_drawings 등에 적재된 동일 파일 차단)
    dup_meta = DocumentMetadata.query.filter_by(file_hash_sha256=sha256).first()
    if dup_meta:
        return jsonify({
            "status": "DUPLICATE",
            "message": f"이미 등록된 동일한 CAD 도면 파일입니다. ({dup_meta.original_file_name} / {dup_meta.document_code}) 중복 업로드는 거절됩니다.",
        }), 409

    # 2) CAD_DRAWINGS 중복 검사
    try:
        dup_cad = CadDrawing.query.filter_by(file_sha256=sha256).first()
        if dup_cad:
            return jsonify({
                "status": "DUPLICATE",
                "message": f"이미 등록된 동일한 CAD 도면입니다. (도면번호: {dup_cad.drawing_id}, 호선: {dup_cad.ship_code}) 중복 업로드는 거절됩니다.",
            }), 409
    except Exception:
        db.session.rollback()

    # 3) PDF 첨부 시 PDF도 중복 검사
    pdf_bytes = None
    if pdf_file and pdf_file.filename:
        pdf_bytes = pdf_file.read()
        if pdf_bytes:
            pdf_sha = hashlib.sha256(pdf_bytes).hexdigest()
            dup_pdf = DocumentMetadata.query.filter_by(file_hash_sha256=pdf_sha).first()
            if dup_pdf:
                return jsonify({
                    "status": "DUPLICATE",
                    "message": f"첨부한 도면 PDF가 이미 등록되어 있습니다. ({dup_pdf.original_file_name} / {dup_pdf.document_code})",
                }), 409

    # 4) 신규 파일 저장
    upload_dir = PROJECT_ROOT / "data" / "cad_drawings" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_dxf = secure_filename(dxf_file.filename) or f"cad_{uuid.uuid4().hex[:6]}.dxf"
    dxf_path = upload_dir / f"{uuid.uuid4().hex[:6]}_{safe_dxf}"
    dxf_path.write_bytes(dxf_bytes)
    dxf_rel = dxf_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()[:500]

    pdf_rel = None
    if pdf_bytes and pdf_file and pdf_file.filename:
        safe_pdf = secure_filename(pdf_file.filename) or f"cad_{uuid.uuid4().hex[:6]}.pdf"
        pdf_path = upload_dir / f"{uuid.uuid4().hex[:6]}_{safe_pdf}"
        pdf_path.write_bytes(pdf_bytes)
        pdf_rel = pdf_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()[:500]

    drawing_title = (custom_title or Path(safe_dxf).stem)[:200]
    resolved_ship_code = (ship_code or f"HN-{_re.sub(r'[^A-Za-z0-9]', '', Path(safe_dxf).stem)[:12].upper() or 'SHIP01'}")[:30]
    drawing_id = f"DWG-{uuid.uuid4().hex[:10].upper()}"
    doc_code = f"CAD-INT-{sha256[:10].upper()}"
    if DocumentMetadata.query.filter_by(document_code=doc_code).first():
        doc_code = f"CAD-INT-{uuid.uuid4().hex[:10].upper()}"

    try:
        # DOCUMENT_METADATA 등록
        db.session.add(DocumentMetadata(
            document_code=doc_code,
            document_title=f"{drawing_title} (DXF)"[:200],
            collection_type="INTERNAL",
            category_code="CAD_DRAWING",
            original_file_name=dxf_file.filename[:255],
            storage_uri=dxf_rel,
            file_hash_sha256=sha256,
            vector_collection="col_internal_ship_sop",
            ingest_status="COMPLETED",
            is_active="Y",
            uploaded_by=user.user_id,
        ))
        # CAD_DRAWINGS 등록 (테이블 존재 시)
        try:
            with db.session.begin_nested():
                db.session.add(CadDrawing(
                    drawing_id=drawing_id,
                    ship_code=resolved_ship_code,
                    drawing_title=drawing_title,
                    dxf_file_path=dxf_rel,
                    pdf_file_path=pdf_rel,
                    file_sha256=sha256,
                    revision_no=0,
                    branch_code=branch_code,
                    work_zone=work_zone,
                    plan_status="ACTIVE",
                    uploaded_by=user.username,
                ))
        except Exception:
            pass

        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "ERROR", "message": f"도면 등록 중 오류가 발생했습니다: {str(e)}"}), 500

    return jsonify({
        "status": "SUCCESS",
        "drawing_id": drawing_id,
        "ship_code": resolved_ship_code,
        "message": f"[{drawing_title}] 선박 도면이 등록되었습니다. 작업 순서나 작업지시 준비는 통합 업무에서 요청해 주세요.",
    })


@service_bp.route("/api/cad/file/<int:doc_id>", methods=["GET"])
@login_required
def cad_raw_file(doc_id: int):
    user = User.query.get_or_404(session["user_id"])
    if not (user.role in ("ADMIN", "MANAGER") or user.can_access_internal_rag == "Y"):
        abort(403)
    doc = DocumentMetadata.query.filter_by(document_id=doc_id, category_code="CAD_DRAWING").first_or_404()
    abs_p = _resolve_doc_abs_path(doc.storage_uri)
    if not abs_p or not abs_p.exists():
        abort(404)
    if request.args.get("preview_pdf") == "1" and abs_p.with_suffix(".pdf").exists():
        return send_file(abs_p.with_suffix(".pdf"), mimetype="application/pdf", as_attachment=False)
    as_dl = request.args.get("download", "0") == "1"
    if not as_dl and abs_p.suffix.lower() == ".pdf":
        return send_file(abs_p, mimetype="application/pdf", as_attachment=False)
    return send_file(abs_p, as_attachment=as_dl, download_name=doc.original_file_name)


# =====================================================================
# 7. [C-01a ~ C-04] FAQ / QNA 목록 · 상세 · 작성 · 수정 · 삭제
# =====================================================================
FAQ_CATEGORY = "FAQ"
QNA_CATEGORIES = ("일반문의", "계정/권한", "자재/장비", "현장기술", "커뮤니티")


@service_bp.route("/faq", methods=["GET"])
@login_required
def faq_board():
    user = User.query.get_or_404(session["user_id"])
    tab = (request.args.get("tab") or "faq").strip().lower()
    if tab not in ("faq", "qna"):
        tab = "faq"
    q = (request.args.get("q") or "").strip()
    page = max(1, request.args.get("page", 1, type=int))
    per_page = 10

    query = CommunityBoard.query.filter(CommunityBoard.parent_id.is_(None))
    if tab == "faq":
        query = query.filter(CommunityBoard.category == FAQ_CATEGORY)
    else:
        query = query.filter(CommunityBoard.category != FAQ_CATEGORY)

    if q:
        like_q = f"%{q}%"
        query = query.filter(or_(
            CommunityBoard.title.ilike(like_q),
            CommunityBoard.content.ilike(like_q),
            CommunityBoard.writer_display.ilike(like_q),
        ))

    total_count = query.count()
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    page = min(page, total_pages)
    posts = (
        query.order_by(CommunityBoard.board_id.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    reply_counts = dict(
        db.session.query(CommunityBoard.parent_id, func.count(CommunityBoard.board_id))
        .filter(CommunityBoard.parent_id.isnot(None))
        .group_by(CommunityBoard.parent_id)
        .all()
    )

    return render_template(
        "community/faq_board.html",
        me=user,
        tab=tab,
        q=q,
        posts=posts,
        reply_counts=reply_counts,
        page=page,
        total_pages=total_pages,
        total_count=total_count,
        page_items=_build_page_items(page, total_pages),
    )


@service_bp.route("/faq/<int:board_id>", methods=["GET"])
@login_required
def faq_detail(board_id):
    user = User.query.get_or_404(session["user_id"])
    post = CommunityBoard.query.filter_by(board_id=board_id, parent_id=None).first_or_404()
    post.view_count = (post.view_count or 0) + 1
    db.session.commit()

    is_faq = (post.category == FAQ_CATEGORY)
    replies = [] if is_faq else (
        CommunityBoard.query
        .filter_by(parent_id=post.board_id)
        .order_by(CommunityBoard.board_id.asc())
        .all()
    )
    is_community = (post.category == "커뮤니티")
    can_reply = True if is_community else (user.role in ("ADMIN", "MANAGER"))
    can_edit = (user.role == "ADMIN") if is_faq else (post.writer_id == user.user_id)
    can_delete = (user.role == "ADMIN") or (post.writer_id == user.user_id)

    return render_template(
        "community/faq_detail.html",
        me=user,
        post=post,
        replies=replies,
        is_faq=is_faq,
        can_reply=can_reply,
        can_edit=can_edit,
        can_delete=can_delete,
    )


@service_bp.route("/faq/write", methods=["GET", "POST"])
@login_required
def faq_write():
    user = User.query.get_or_404(session["user_id"])
    if request.method == "GET":
        mode = (request.args.get("tab") or "qna").strip().lower()
        if mode == "faq" and user.role != "ADMIN":
            mode = "qna"
        return render_template(
            "community/faq_write.html",
            me=user,
            mode=mode,
            categories=QNA_CATEGORIES,
        )

    mode = (request.form.get("mode") or "qna").strip().lower()
    category = FAQ_CATEGORY if (mode == "faq" and user.role == "ADMIN") else (request.form.get("category") or "일반문의").strip()
    title = (request.form.get("title") or "").strip()
    content = (request.form.get("content") or "").strip()

    if not title or not content:
        flash("제목과 내용을 모두 입력해 주세요.", "warning")
        return redirect(url_for("service.faq_write", tab=mode))

    new_post = CommunityBoard(
        parent_id=None,
        category=category,
        title=title,
        content=content,
        writer_id=user.user_id,
        writer_display=user.display_name or user.full_name,
        is_anonymous="N",
        view_count=0,
    )
    db.session.add(new_post)
    db.session.commit()
    return redirect(url_for("service.faq_detail", board_id=new_post.board_id))


@service_bp.route("/faq/edit/<int:board_id>", methods=["POST"])
@login_required
def faq_edit(board_id):
    user = User.query.get_or_404(session["user_id"])
    post = CommunityBoard.query.filter_by(board_id=board_id, parent_id=None).first_or_404()
    is_faq = (post.category == FAQ_CATEGORY)
    if (is_faq and user.role != "ADMIN") or (not is_faq and post.writer_id != user.user_id):
        flash("수정 권한이 없습니다.", "danger")
        return redirect(url_for("service.faq_detail", board_id=board_id))

    title = (request.form.get("title") or "").strip()
    content = (request.form.get("content") or "").strip()
    if title and content:
        post.title = title
        post.content = content
        post.updated_at = datetime.utcnow()
        db.session.commit()
        flash("수정되었습니다.", "success")
    return redirect(url_for("service.faq_detail", board_id=board_id))


@service_bp.route("/faq/reply/<int:board_id>", methods=["POST"])
@login_required
def faq_reply(board_id):
    parent_post = CommunityBoard.query.filter_by(board_id=board_id, parent_id=None).first_or_404()
    user = User.query.get_or_404(session["user_id"])
    if parent_post.category == FAQ_CATEGORY:
        return redirect(url_for("service.faq_detail", board_id=board_id))

    if parent_post.category != "커뮤니티" and user.role not in ("ADMIN", "MANAGER"):
        flash("문의 답변은 담당자 또는 관리자만 작성할 수 있습니다.", "danger")
        return redirect(url_for("service.faq_detail", board_id=board_id))

    content = (request.form.get("reply_content") or "").strip()
    if not content:
        flash("답변 내용을 입력해 주세요.", "warning")
        return redirect(url_for("service.faq_detail", board_id=board_id))

    db.session.add(CommunityBoard(
        parent_id=parent_post.board_id,
        category=parent_post.category,
        title=None,
        content=content,
        writer_id=user.user_id,
        writer_display=user.display_name or user.full_name,
        is_anonymous="N",
    ))
    db.session.commit()
    return redirect(url_for("service.faq_detail", board_id=board_id))


@service_bp.route("/faq/delete/<int:board_id>", methods=["POST"])
@login_required
def faq_delete(board_id):
    item = CommunityBoard.query.get_or_404(board_id)
    if item.writer_id != session["user_id"] and session.get("role") != "ADMIN":
        flash("삭제 권한이 없습니다.", "danger")
        return redirect(url_for("service.faq_board"))

    if item.parent_id is not None:
        pid = item.parent_id
        db.session.delete(item)
        db.session.commit()
        return redirect(url_for("service.faq_detail", board_id=pid))

    tab = "faq" if item.category == FAQ_CATEGORY else "qna"
    CommunityBoard.query.filter_by(parent_id=item.board_id).delete()
    db.session.delete(item)
    db.session.commit()
    return redirect(url_for("service.faq_board", tab=tab))


# =====================================================================
# 8. 구버전 URL 호환 리다이렉트
# =====================================================================
@service_bp.route("/vision")
@login_required
def vision_page():
    return redirect(url_for("vision.index"))


@service_bp.route("/erp", methods=["GET"])
@login_required
def erp_board_page():
    return redirect(url_for("service.workspace_page"))


@service_bp.route("/yards", methods=["GET"])
@login_required
def yards_page():
    return redirect(url_for("vision.index"))


@service_bp.route("/history", methods=["GET"])
@login_required
def history_page():
    return redirect(url_for("service.work_status_page"))
