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
from datetime import datetime
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
def _calc_cert_days_left(expiry_str: str | None) -> int | None:
    if not expiry_str:
        return None
    s = str(expiry_str).strip()[:10]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            dt = datetime.strptime(s, fmt).date()
            return (dt - datetime.now().date()).days
        except ValueError:
            continue
    return None


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


def _extract_inst_pdf_url(inst) -> str:
    if not inst or not inst.external_env_json:
        return ""
    try:
        raw_str = _read_clob(inst.external_env_json).strip()
        try:
            meta = json.loads(raw_str)
        except Exception:
            meta = ast.literal_eval(raw_str)
        if not isinstance(meta, dict):
            return ""
        files = meta.get("files") or (meta.get("_draft_payload") or {}).get("files") or {}
        raw_url = files.get("download_pdf_url") or ""
        if not raw_url:
            return ""
        parts = raw_url.strip("/").split("/")
        if len(parts) >= 2:
            date_str, filename = parts[-2], parts[-1]
            if not (PROJECT_ROOT / "generated_docs" / "outputs" / date_str / filename).exists():
                return ""
        return raw_url if raw_url.startswith("/service/") else f"/service{raw_url}"
    except Exception:
        return ""


def _extract_inst_po_pdf_url(inst) -> str:
    if not inst or not inst.external_env_json:
        return ""
    try:
        raw_str = _read_clob(inst.external_env_json).strip()
        try:
            meta = json.loads(raw_str)
        except Exception:
            meta = ast.literal_eval(raw_str)
        if not isinstance(meta, dict):
            return ""
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
    except Exception:
        return ""


def _safe_query_work_plans(branch_code: str | None = None, states: list[str] | None = None):
    """CAD_DRAWINGS / WORK_PLANS DDL 적용 전에도 안전하게 빈 목록 반환"""
    try:
        q = WorkPlan.query
        if states:
            q = q.filter(WorkPlan.plan_state.in_(states))
        else:
            q = q.filter(WorkPlan.plan_state.in_(["PENDING", "IN_PROGRESS"]))
        if branch_code:
            q = q.filter(WorkPlan.branch_code == branch_code)
        return q.order_by(WorkPlan.priority_rank.asc(), WorkPlan.created_at.asc()).all()
    except Exception:
        db.session.rollback()
        return []


def _safe_get_wo_progress_map(work_ids: list[int]) -> dict[int, dict]:
    """WORK_ORDERS.PROGRESS_PCT / PLAN_ID 컬럼 존재 시 읽고, 미적용 DB면 안전 fallback"""
    if not work_ids:
        return {}
    try:
        rows = db.session.execute(
            text("SELECT WORK_ID, PLAN_ID, PROGRESS_PCT FROM WORK_ORDERS WHERE WORK_ID IN :ids"),
            {"ids": tuple(work_ids)},
        ).fetchall()
        return {int(r[0]): {"plan_id": r[1], "progress_pct": int(r[2] or 0)} for r in rows}
    except Exception:
        db.session.rollback()
        return {}


def _safe_update_wo_progress(work_id: int, progress_pct: int):
    try:
        db.session.execute(
            text("UPDATE WORK_ORDERS SET PROGRESS_PCT = :pct WHERE WORK_ID = :wid"),
            {"pct": progress_pct, "wid": work_id},
        )
    except Exception:
        db.session.rollback()


# =====================================================================
# 3. [W-01a ~ W-01d] 통합 업무 (동적 단일 페이지 AI 워크스페이스)
# =====================================================================
@service_bp.route("/workspace", methods=["GET"])
@login_required
def workspace_page():
    user = User.query.get_or_404(session["user_id"])
    branches = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).all()
    first_block = ErpBlockMaster.query.order_by(ErpBlockMaster.block_id.asc()).first()
    first_branch = branches[0] if branches else None
    sample_placeholder = (
        f"예: {first_branch.branch_name} {first_block.block_id} 오늘 작업 상태나 필요한 자재를 확인해줘"
        if (first_branch and first_block)
        else "예: 선박 번호나 블록 번호를 입력해 규정, 자재, 현장 상태, 작업 지시를 요청해 주세요"
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
    """W-01a 초기 진입 요약: 당일 활성 공정 큐 · 야드 주의 자원 · RDB 결재/문서 현황"""
    user = User.query.get_or_404(session["user_id"])
    can_erp = (user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER"))
    branch_map = {b.branch_code: b.branch_name for b in ErpBranch.query.all()}

    plans = _safe_query_work_plans()
    active_queue = []
    if plans:
        for p in plans[:8]:
            active_queue.append({
                "plan_id": p.plan_id,
                "priority_rank": p.priority_rank,
                "block_code": p.block_code,
                "branch_code": p.branch_code,
                "branch_name": branch_map.get(p.branch_code, p.branch_code),
                "process_code": p.process_code,
                "state": p.plan_state,
                "progress_pct": int(p.progress_pct or 0),
                "wo_no": p.assigned_wo_number or "",
            })
    else:
        orders = (
            WorkOrder.query
            .filter(WorkOrder.status_code.in_(["READY", "IN_PROGRESS"]))
            .order_by(WorkOrder.work_id.desc())
            .limit(8)
            .all()
        )
        prog_map = _safe_get_wo_progress_map([o.work_id for o in orders])
        for idx, wo in enumerate(orders, 1):
            p_info = prog_map.get(wo.work_id, {})
            pct = p_info.get("progress_pct")
            if pct is None:
                pct = 40 if wo.status_code == "IN_PROGRESS" else 0
            active_queue.append({
                "plan_id": p_info.get("plan_id") or "",
                "priority_rank": idx,
                "block_code": wo.block_code,
                "branch_code": wo.work_region,
                "branch_name": branch_map.get(wo.work_region, wo.work_region),
                "process_code": wo.process_code,
                "state": wo.status_code,
                "progress_pct": pct,
                "wo_no": wo.work_no,
            })

    low_materials, expiring_workers = [], []
    if can_erp:
        for m in ErpBomInventory.query.all():
            avail = float(m.total_qty or 0) - float(m.reserved_qty or 0)
            if avail <= 50.0:
                low_materials.append({
                    "item_code": m.item_code,
                    "item_name": m.item_name,
                    "branch_code": m.branch_code,
                    "branch_name": branch_map.get(m.branch_code, m.branch_code),
                    "available_qty": avail,
                    "unit": m.unit,
                })
        for w in ErpWorkerCert.query.all():
            days = _calc_cert_days_left(w.cert_expiry_date)
            if days is not None and days <= 90:
                expiring_workers.append({
                    "worker_id": w.worker_id,
                    "worker_name": w.worker_name,
                    "cert_code": w.cert_code,
                    "days_left": days,
                    "branch_code": w.branch_code,
                    "branch_name": branch_map.get(w.branch_code, w.branch_code),
                })

    pending_count = WorkInstruction.query.filter_by(approval_status="DRAFT").count()
    latest_orders = WorkOrder.query.order_by(WorkOrder.work_id.desc()).limit(4).all()
    recent_docs = []
    for wo in latest_orders:
        inst = (
            WorkInstruction.query
            .filter_by(work_id=wo.work_id)
            .order_by(WorkInstruction.instruction_id.desc())
            .first()
        )
        recent_docs.append({
            "work_id": wo.work_id,
            "work_no": wo.work_no,
            "ship_code": wo.ship_code,
            "block_code": wo.block_code,
            "revision_no": inst.revision_no if inst else "Rev.1",
            "approval_status": inst.approval_status if inst else "DRAFT",
        })

    return jsonify({
        "status": "SUCCESS",
        "can_erp": can_erp,
        "active_queue": active_queue,
        "low_materials": low_materials[:6],
        "expiring_workers": expiring_workers[:6],
        "pending_approvals": pending_count,
        "recent_docs": recent_docs,
    })


@service_bp.route("/api/workspace/erp-yard", methods=["GET"])
@login_required
def api_workspace_erp_yard():
    """W-01c: 선택 야드 자재 재고(미결 발주 중복 체크 포함) 및 자격 인력 조회"""
    user = User.query.get_or_404(session["user_id"])
    if not (user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER")):
        return jsonify({"status": "FORBIDDEN", "message": "자원(ERP) 조회 권한이 없는 계정입니다."}), 403

    branch_code = (request.args.get("branch_code") or "").strip()
    block_code = (request.args.get("block_code") or "").strip()
    if block_code and not branch_code:
        blk = ErpBlockMaster.query.filter_by(block_id=block_code).first()
        if blk:
            branch_code = blk.branch_code
    if not branch_code:
        branch_code = _get_default_branch_code(user)

    branch = ErpBranch.query.filter_by(branch_code=branch_code).first()
    mats_raw = ErpBomInventory.query.filter_by(branch_code=branch_code).order_by(ErpBomInventory.item_code.asc()).all()
    workers_raw = ErpWorkerCert.query.filter_by(branch_code=branch_code).order_by(ErpWorkerCert.cert_expiry_date.asc()).all()

    open_po_codes = set()
    try:
        open_pos = B2BSalesOrder.query.filter(B2BSalesOrder.order_status.in_(["PENDING", "CONFIRMED"])).all()
        open_po_codes = {po.ordered_ref_code for po in open_pos}
    except Exception:
        pass

    materials = []
    for m in mats_raw:
        avail = float(m.total_qty or 0) - float(m.reserved_qty or 0)
        materials.append({
            "item_code": m.item_code,
            "item_name": m.item_name,
            "available_qty": avail,
            "total_qty": float(m.total_qty or 0),
            "reserved_qty": float(m.reserved_qty or 0),
            "unit": m.unit,
            "is_low": avail <= 50.0,
            "has_open_po": m.item_code in open_po_codes,
        })

    workers = []
    for w in workers_raw:
        days = _calc_cert_days_left(w.cert_expiry_date)
        workers.append({
            "worker_id": w.worker_id,
            "worker_name": w.worker_name,
            "cert_code": w.cert_code,
            "weld_process": w.weld_process,
            "cert_expiry_date": w.cert_expiry_date,
            "days_left": days,
            "is_expiring": (days is not None and days <= 90),
        })

    return jsonify({
        "status": "SUCCESS",
        "branch_code": branch_code,
        "branch_name": branch.branch_name if branch else branch_code,
        "materials": materials,
        "workers": workers,
    })


@service_bp.route("/api/workspace/send-draft", methods=["POST"])
@login_required
def api_workspace_send_draft():
    """W-01c HITL: 사용자 검토·수정 완료된 메일/발주계획서 초안 승인 발송"""
    user = User.query.get_or_404(session["user_id"])
    if not (user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER")):
        return jsonify({"status": "FORBIDDEN", "message": "발송 권한이 없습니다."}), 403

    data = request.get_json(silent=True) or {}
    draft_type = (data.get("draft_type") or "EMAIL").upper()
    recipient = (data.get("recipient") or "").strip()
    subject = (data.get("subject") or "").strip()
    body = (data.get("body") or "").strip()
    if not subject or not body:
        return jsonify({"status": "ERROR", "message": "제목과 내용을 입력해 주세요."}), 400

    # [AGENT_TEAM_HOOK] 추후 Agent 팀원의 단독 Gmail MCP / B2B 발주 발송 함수 연결 지점
    exec_log = AgentExecution(
        session_id=f"hitl_draft_{uuid.uuid4().hex[:8]}",
        user_id=user.user_id,
        intent_type="HITL_DRAFT_SEND",
        current_state=f"SENT_{draft_type}",
        selected_tools_json=json.dumps(
            {"draft_type": draft_type, "recipient": recipient, "subject": subject},
            ensure_ascii=False,
        ),
        status_code="SUCCESS",
    )
    db.session.add(exec_log)
    db.session.commit()
    return jsonify({"status": "SUCCESS", "message": f"[{subject}] 승인 및 발송 요청이 접수되었습니다."})


@service_bp.route("/api/workspace/check-duplicate", methods=["POST"])
@login_required
def api_workspace_check_duplicate():
    """W-01d: 현장 Vision 스냅샷 + 당일 동일 블록 작업지시서 및 선박 중량물 계획서 중복 검증"""
    data = request.get_json(silent=True) or {}
    branch_code = (data.get("branch_code") or "").strip()
    block_code = (data.get("block_code") or "").strip()

    all_blocks = ErpBlockMaster.query.order_by(ErpBlockMaster.branch_code.asc(), ErpBlockMaster.block_id.asc()).all()
    block = None
    if block_code:
        block = next((b for b in all_blocks if b.block_id == block_code), None)
    if not block and branch_code:
        block = next((b for b in all_blocks if b.branch_code == branch_code), None)
    if not block and all_blocks:
        block = all_blocks[0]

    block_options = [
        {"block_id": b.block_id, "ship_code": b.ship_code, "branch_code": b.branch_code, "block_name": b.block_name}
        for b in all_blocks
    ]

    if not block:
        return jsonify({
            "status": "SUCCESS",
            "exists": False,
            "block": None,
            "block_options": block_options,
            "existing_wo": None,
            "ship_plan_exists": False,
            "vision": None,
        })

    target_branch = branch_code or block.branch_code
    branch = ErpBranch.query.filter_by(branch_code=target_branch).first()

    existing_wo = (
        WorkOrder.query
        .filter(
            WorkOrder.block_code == block.block_id,
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
        prog_map = _safe_get_wo_progress_map([existing_wo.work_id])
        existing_wo_info = {
            "work_id": existing_wo.work_id,
            "work_no": existing_wo.work_no,
            "status_code": existing_wo.status_code,
            "progress_pct": prog_map.get(existing_wo.work_id, {}).get("progress_pct", 0),
            "revision_no": latest_inst.revision_no if latest_inst else "Rev.1",
            "pdf_url": _extract_inst_pdf_url(latest_inst),
        }

    # 선박 단위 중량물 취급 계획서 존재 여부(동일 호선의 기존 지시서/계획서 확인)
    ship_wo = (
        WorkOrder.query
        .filter(WorkOrder.ship_code == block.ship_code, WorkOrder.status_code.in_(["READY", "IN_PROGRESS"]))
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
        }

    return jsonify({
        "status": "SUCCESS",
        "exists": existing_wo_info is not None,
        "ship_plan_exists": ship_wo is not None,
        "ship_code": block.ship_code,
        "branch": {
            "branch_code": target_branch,
            "branch_name": branch.branch_name if branch else target_branch,
        },
        "block": {
            "block_id": block.block_id,
            "ship_code": block.ship_code,
            "block_name": block.block_name,
            "steel_type": block.steel_type,
            "thickness_mm": block.thickness_mm,
            "process": block.process,
            "joint_type": block.joint_type,
        },
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

    if not prompt and not block_code:
        flash("통합 업무 또는 현장 관제에서 작업 준비 요청을 통해 진입해 주세요.", "info")
        return redirect(url_for("service.workspace_page"))

    if not prompt:
        if mode == "REVISE":
            prompt = f"{branch_code} {block_code} 기존 작업지시서의 일정과 작업 조건을 개정(Rev)하기 위해 자재, 인력, 기상, 용접 기준을 다시 분석해줘"
        else:
            prompt = f"{branch_code} {block_code} 용접 작업지시서와 취급 계획을 발행하기 위해 규격, 자재, 인력, 로봇, 기상을 통합 분석해줘"

    return render_template(
        "service/agent_analyze.html",
        me=user,
        prompt=prompt,
        mode=mode,
        branch_code=branch_code,
        block_code=block_code,
        work_id=work_id,
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

    return render_template(
        "service/agent_plan.html",
        me=user,
        session_id=session_id,
        answer=answer,
        doc=saved_doc,
        ui_blocks=ui_blocks,
        sources=sources,
        can_approve=(user.role in ("ADMIN", "MANAGER")),
    )


# =====================================================================
# 5. [SA-03] 작업 현황 (다일 진척률 · 지연 개정 · 다음 블록 자동 활성화)
# =====================================================================
@service_bp.route("/work-status", methods=["GET"])
@login_required
def work_status_page():
    user = User.query.get_or_404(session["user_id"])
    is_manager = user.role in ("ADMIN", "MANAGER")

    wo_q = WorkOrder.query
    if not is_manager:
        wo_q = wo_q.filter_by(requested_by=user.user_id)
    orders_raw = wo_q.order_by(WorkOrder.work_id.desc()).all()

    prog_map = _safe_get_wo_progress_map([wo.work_id for wo in orders_raw])
    branch_map = {b.branch_code: b.branch_name for b in ErpBranch.query.all()}
    selected_id = request.args.get("work_id", type=int) or (orders_raw[0].work_id if orders_raw else None)

    orders_data, selected_detail = [], None
    for wo in orders_raw:
        insts = (
            WorkInstruction.query
            .filter_by(work_id=wo.work_id)
            .order_by(WorkInstruction.instruction_id.desc())
            .all()
        )
        latest_inst = insts[0] if insts else None
        p_info = prog_map.get(wo.work_id, {})
        pct = p_info.get("progress_pct")
        if pct is None:
            pct = 100 if wo.status_code == "COMPLETED" else (40 if wo.status_code == "IN_PROGRESS" else 0)
        item = {
            "wo": wo,
            "branch_name": branch_map.get(wo.work_region, wo.work_region),
            "progress_pct": pct,
            "plan_id": p_info.get("plan_id") or "",
            "latest_inst": latest_inst,
            "revisions": insts,
            "pdf_url": _extract_inst_pdf_url(latest_inst),
            "po_pdf_url": _extract_inst_po_pdf_url(latest_inst),
        }
        orders_data.append(item)
        if wo.work_id == selected_id:
            selected_detail = item

    queued_plans = _safe_query_work_plans(states=["PENDING"])
    b2b_deliveries, dispatch_info, robot_info, cad_doc_id = [], None, None, None
    if selected_detail:
        wo = selected_detail["wo"]
        dispatch_info = MesWorkOrderDispatch.query.filter_by(wo_number=wo.work_no).first()
        robot_info = RobotExecution.query.filter_by(work_id=wo.work_id).first()
        try:
            b2b_deliveries = B2BSalesOrder.query.filter_by(hq_wo_number=wo.work_no).all()
        except Exception:
            b2b_deliveries = []
        if is_manager or user.can_access_internal_rag == "Y":
            cad_doc = (
                DocumentMetadata.query
                .filter(DocumentMetadata.category_code == "CAD_DRAWING", DocumentMetadata.is_active == "Y")
                .order_by(DocumentMetadata.document_id.desc())
                .first()
            )
            cad_doc_id = cad_doc.document_id if cad_doc else None

    return render_template(
        "service/work_status.html",
        me=user,
        is_manager=is_manager,
        orders=orders_data,
        selected=selected_detail,
        queued_plans=queued_plans,
        dispatch_info=dispatch_info,
        robot_info=robot_info,
        b2b_deliveries=b2b_deliveries,
        cad_doc_id=cad_doc_id,
    )


@service_bp.route("/api/work-status/progress", methods=["POST"])
@login_required
def api_update_work_progress():
    """다일 공정 진척률 저장 및 100% 완료 시 다음 PRIORITY_RANK(P2) 작업 자동 활성화"""
    user = User.query.get_or_404(session["user_id"])
    if user.role not in ("ADMIN", "MANAGER"):
        return jsonify({"status": "FORBIDDEN", "message": "진척률 제어는 담당자/관리자만 가능합니다."}), 403

    data = request.get_json(silent=True) or {}
    work_id = data.get("work_id")
    progress_pct = max(0, min(100, int(data.get("progress_pct", 0))))
    wo = WorkOrder.query.get_or_404(work_id)

    _safe_update_wo_progress(wo.work_id, progress_pct)
    if progress_pct >= 100:
        wo.status_code = "COMPLETED"
    elif progress_pct > 0:
        wo.status_code = "IN_PROGRESS"

    activated_next_block = None
    try:
        plan = WorkPlan.query.filter(
            or_(WorkPlan.assigned_wo_number == wo.work_no, WorkPlan.block_code == wo.block_code)
        ).first()
        if plan:
            plan.progress_pct = progress_pct
            if progress_pct >= 100:
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
            elif progress_pct > 0:
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
    inst = (
        WorkInstruction.query
        .filter_by(work_id=wo.work_id)
        .order_by(WorkInstruction.instruction_id.desc())
        .first()
    )
    insp = (
        VisionInspection.query
        .filter_by(work_id=wo.work_id)
        .order_by(VisionInspection.inspection_id.desc())
        .first()
    )
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
# 6. [SR-01a / SR-01b / SR-01c] 자료실 · PDF 적재 · CAD 공정 큐 등록
# =====================================================================
@service_bp.route("/library", methods=["GET"])
@login_required
def library_page():
    user = User.query.get_or_404(session["user_id"])
    is_admin = user.role in ("ADMIN", "MANAGER")
    can_internal = (user.can_access_internal_rag == "Y") or is_admin

    allowed_tabs = ["public"]
    if can_internal:
        allowed_tabs.append("internal")
    if is_admin:
        allowed_tabs.extend(["security", "cad"])

    tab = (request.args.get("tab") or "public").strip().lower()
    if tab not in allowed_tabs:
        tab = allowed_tabs[0]
    q = (request.args.get("q") or "").strip()
    page = max(1, request.args.get("page", 1, type=int))
    per_page = 10

    doc_q = DocumentMetadata.query
    if not is_admin:
        doc_q = doc_q.filter(DocumentMetadata.is_active == "Y")

    if tab == "public":
        doc_q = doc_q.filter(DocumentMetadata.collection_type == "PUBLIC")
    elif tab == "internal":
        doc_q = doc_q.filter(
            DocumentMetadata.collection_type == "INTERNAL",
            DocumentMetadata.category_code != "CAD_DRAWING",
            DocumentMetadata.category_code != "SECURITY",
        )
    elif tab == "security":
        doc_q = doc_q.filter(
            DocumentMetadata.collection_type == "INTERNAL",
            DocumentMetadata.category_code == "SECURITY",
        )
    elif tab == "cad":
        doc_q = doc_q.filter(DocumentMetadata.category_code == "CAD_DRAWING")

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
        tab=tab,
        q=q,
        docs=docs,
        page=page,
        total_pages=total_pages,
        total_count=total,
        page_items=_build_page_items(page, total_pages),
    )


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
        flash("CAD 도면 및 공정 큐 등록은 담당자만 이용할 수 있습니다.", "warning")
        return redirect(url_for("service.library_page"))
    branches = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).all()
    return render_template("service/library_upload_cad.html", me=user, branches=branches)


@service_bp.route("/api/library/cad-parse", methods=["POST"])
@login_required
def api_library_cad_parse():
    """SR-01c Gate 1: DXF/PDF 업로드 후 SHA-256 해시 + cad_parser.py 표제란·블록 파싱"""
    import hashlib
    from werkzeug.utils import secure_filename
    from modules.knowledge.cad_parser import parse_dxf_metadata, detect_vessel_context

    user = User.query.get_or_404(session["user_id"])
    if user.role not in ("ADMIN", "MANAGER"):
        return jsonify({"status": "FORBIDDEN", "message": "권한이 없습니다."}), 403

    dxf_file = request.files.get("dxf_file")
    pdf_file = request.files.get("pdf_file")
    if not dxf_file or not dxf_file.filename:
        return jsonify({"status": "ERROR", "message": "DXF 도면 파일을 선택해 주세요."}), 400

    upload_dir = PROJECT_ROOT / "data" / "cad_drawings" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_dxf = secure_filename(dxf_file.filename) or f"cad_{uuid.uuid4().hex[:6]}.dxf"
    dxf_path = upload_dir / f"{uuid.uuid4().hex[:6]}_{safe_dxf}"
    dxf_file.save(str(dxf_path))

    pdf_rel_path = ""
    if pdf_file and pdf_file.filename:
        safe_pdf = secure_filename(pdf_file.filename) or f"cad_{uuid.uuid4().hex[:6]}.pdf"
        pdf_path = upload_dir / f"{uuid.uuid4().hex[:6]}_{safe_pdf}"
        pdf_file.save(str(pdf_path))
        pdf_rel_path = str(pdf_path.relative_to(PROJECT_ROOT))

    sha256 = hashlib.sha256(dxf_path.read_bytes()).hexdigest()
    meta = parse_dxf_metadata(dxf_path)
    vessel_name, _ = detect_vessel_context(dxf_path)

    raw_blocks = meta.get("blocks") or meta.get("layers") or [Path(safe_dxf).stem]
    bbox = meta.get("bbox") or {}
    base_len = round(float(bbox.get("width") or 10000.0), 1)
    if base_len <= 0:
        base_len = 10000.0

    # [AGENT_TEAM_HOOK] 추후 LLM 기반 공정 순서(PLAN_RATIONALE) 및 일정 산출 연결 지점
    parsed_plans = []
    for idx, bname in enumerate(raw_blocks[:6], 1):
        est_len = round(max(1500.0, base_len / max(1, min(len(raw_blocks), 3))), 1)
        est_wire = round(max(5.0, (est_len / 1000.0) * 3.0), 1)
        parsed_plans.append({
            "priority_rank": idx,
            "block_code": str(bname).upper().replace(" ", "-")[:30],
            "block_name": f"{vessel_name} - {bname}"[:140],
            "material_grade": "AH36",
            "thickness_mm": 12.0,
            "process_code": "FCAW",
            "joint_type": "BUTT",
            "wire_mat_code": "WIRE-CSF71T",
            "est_weld_length_mm": est_len,
            "est_weight_ton": 15.0,
            "est_wire_kg": est_wire,
            "est_duration_days": max(1, min(3, int(est_len // 7000) + 1)),
        })

    return jsonify({
        "status": "SUCCESS",
        "dxf_file_path": str(dxf_path.relative_to(PROJECT_ROOT)),
        "pdf_file_path": pdf_rel_path,
        "file_sha256": sha256,
        "drawing_title": Path(safe_dxf).stem,
        "vessel_name": vessel_name,
        "plans": parsed_plans,
    })


@service_bp.route("/api/library/cad-save", methods=["POST"])
@login_required
def api_library_cad_save():
    """SR-01c Gate 2: CAD_DRAWINGS + WORK_PLANS 동시 적재 (STAGING 임시저장 또는 ACTIVE 확정)"""
    user = User.query.get_or_404(session["user_id"])
    if user.role not in ("ADMIN", "MANAGER"):
        return jsonify({"status": "FORBIDDEN", "message": "권한이 없습니다."}), 403

    data = request.get_json(silent=True) or {}
    confirm = bool(data.get("confirm"))
    plan_status = "ACTIVE" if confirm else "STAGING"
    branch_code = (data.get("branch_code") or "").strip()
    work_zone = (data.get("work_zone") or "ZONE-A-BAY1").strip()
    ship_code = (data.get("ship_code") or "").strip()
    plans_in = data.get("plans") or []

    if not branch_code or not ship_code:
        return jsonify({"status": "ERROR", "message": "선박 호선과 배치 야드를 입력해 주세요."}), 400
    if not plans_in:
        return jsonify({"status": "ERROR", "message": "등록할 블록 공정 항목이 없습니다."}), 400

    drawing_id = f"DWG-{uuid.uuid4().hex[:10].upper()}"
    try:
        dwg = CadDrawing(
            drawing_id=drawing_id,
            ship_code=ship_code,
            drawing_title=(data.get("drawing_title") or ship_code)[:200],
            dxf_file_path=(data.get("dxf_file_path") or "")[:500],
            pdf_file_path=(data.get("pdf_file_path") or "")[:500] or None,
            file_sha256=(data.get("file_sha256") or uuid.uuid4().hex)[:64],
            revision_no=int(data.get("revision_no") or 0),
            branch_code=branch_code,
            work_zone=work_zone,
            plan_status=plan_status,
            uploaded_by=user.username,
        )
        db.session.add(dwg)

        for idx, row in enumerate(plans_in, 1):
            rank = int(row.get("priority_rank") or idx)
            state = "PENDING" if confirm else "STAGING"
            db.session.add(WorkPlan(
                plan_id=f"PLAN-{uuid.uuid4().hex[:10].upper()}",
                drawing_id=drawing_id,
                branch_code=branch_code,
                work_zone=work_zone,
                priority_rank=rank,
                block_code=str(row.get("block_code") or f"BLK-{rank}").strip()[:40],
                block_name=str(row.get("block_name") or f"{ship_code} 블록 {rank}").strip()[:150],
                material_grade=str(row.get("material_grade") or "AH36").strip()[:20],
                thickness_mm=float(row.get("thickness_mm") or 12.0),
                process_code=str(row.get("process_code") or "FCAW").strip()[:20],
                joint_type=str(row.get("joint_type") or "BUTT").strip()[:20],
                wire_mat_code=str(row.get("wire_mat_code") or "WIRE-CSF71T").strip()[:40],
                est_weld_length_mm=float(row.get("est_weld_length_mm") or 10000.0),
                est_weight_ton=float(row.get("est_weight_ton") or 10.0),
                est_wire_kg=float(row.get("est_wire_kg") or 20.0),
                est_duration_days=int(row.get("est_duration_days") or 1),
                plan_state=state,
            ))

        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({
            "status": "ERROR",
            "message": f"DB 저장 실패 (CAD_DRAWINGS / WORK_PLANS 테이블 생성 여부를 확인하세요): {str(e)}",
        }), 500

    return jsonify({
        "status": "SUCCESS",
        "drawing_id": drawing_id,
        "plan_status": plan_status,
        "message": "공정 큐가 확정(ACTIVE)되어 통합 업무와 현장 관제에 반영되었습니다." if confirm else "임시 저장(STAGING)되었습니다.",
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
        return send_file(abs_p.with_suffix(".pdf"), as_attachment=False)
    as_dl = request.args.get("download", "0") == "1"
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
# 8. 구버전 URL 호환 리다이렉트 (기존 링크 깨짐 방지)
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
