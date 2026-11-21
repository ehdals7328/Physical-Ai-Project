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
from sqlalchemy import func, or_
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
        return redirect(url_for("service.erp_board_page", prompt=prompt))
    return redirect(url_for("service.erp_board_page"))


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
        thread_id = session.get("assistant_thread_id") or f"ast_{user.username}_{uuid.uuid4().hex[:6]}"
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
    "RUNNING_AGENT": "🧭 질의 의도 및 작업 컨텍스트 분석 중...",
    "team1_analysis": "🏭 본사 SHIP_ERP 재고·자격·설비 진단 및 제원 검증 중...",
    "team2_knowledge": "📚 사내 WPS 규정·현장 기상·비전 관제·협력사 데이터 수집 중...",
    "team3_docgen": "🛡️ 안전·자격 거버넌스 인터록 검사 및 결과서 작성 중...",
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
    if last_ai and last_ai.retrieval_chunks_json:
        raw_list = _parse_json_or_literal(last_ai.retrieval_chunks_json)
        for item in raw_list:
            if isinstance(item, dict) and item.get("_type") == "WORK_ORDER_FILES":
                saved_files = item.get("files")
                saved_status = item.get("approval_status", "DRAFT")
                saved_doc = item.get("doc")
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
# 2. 고객지원 FAQ (1:1 공식 문의 & 현장 커뮤니티) (/service/faq)
# =====================================================================
@service_bp.route("/faq", methods=["GET"])
@login_required
def faq_board():
    user = User.query.get_or_404(session["user_id"])
    tab = (request.args.get("tab") or "inquiry").strip().lower()
    if tab not in ("inquiry", "community"):
        tab = "inquiry"
    q = (request.args.get("q") or "").strip()
    cat_filter = (request.args.get("category") or "ALL").strip()
    page = max(1, request.args.get("page", 1, type=int))
    per_page = 10

    query = CommunityBoard.query.filter(CommunityBoard.parent_id.is_(None))
    if tab == "inquiry":
        query = query.filter(CommunityBoard.category.in_(INQUIRY_CATEGORIES))
    else:
        query = query.filter(~CommunityBoard.category.in_(INQUIRY_CATEGORIES))

    if cat_filter != "ALL":
        query = query.filter(CommunityBoard.category == cat_filter)

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

    inquiry_total = CommunityBoard.query.filter(
        CommunityBoard.parent_id.is_(None),
        CommunityBoard.category.in_(INQUIRY_CATEGORIES),
    ).count()
    community_total = CommunityBoard.query.filter(
        CommunityBoard.parent_id.is_(None),
        ~CommunityBoard.category.in_(INQUIRY_CATEGORIES),
    ).count()

    return render_template(
        "community/faq_board.html",
        me=user,
        tab=tab,
        q=q,
        selected_category=cat_filter,
        posts=posts,
        reply_counts=reply_counts,
        page=page,
        total_pages=total_pages,
        total_count=total_count,
        page_items=_build_page_items(page, total_pages),
        inquiry_total=inquiry_total,
        community_total=community_total,
    )


@service_bp.route("/faq/<int:board_id>", methods=["GET"])
@login_required
def faq_detail(board_id):
    user = User.query.get_or_404(session["user_id"])
    post = CommunityBoard.query.filter_by(board_id=board_id, parent_id=None).first_or_404()
    post.view_count = (post.view_count or 0) + 1
    db.session.commit()

    replies = (
        CommunityBoard.query
        .filter_by(parent_id=post.board_id)
        .order_by(CommunityBoard.board_id.asc())
        .all()
    )

    is_inquiry = post.category in INQUIRY_CATEGORIES
    can_reply = (user.role in ("ADMIN", "MANAGER")) if is_inquiry else True

    return render_template(
        "community/faq_detail.html",
        me=user,
        post=post,
        replies=replies,
        is_inquiry=is_inquiry,
        can_reply=can_reply,
    )


@service_bp.route("/faq/write", methods=["POST"])
@login_required
def faq_write():
    tab = (request.form.get("tab") or "inquiry").strip().lower()
    category = (request.form.get("category") or ("공식문의" if tab == "inquiry" else "자유소통")).strip()
    title = (request.form.get("title") or "").strip()
    content = (request.form.get("content") or "").strip()
    # 익명은 커뮤니티 탭에서만 허용
    is_anon = "Y" if (tab == "community" and request.form.get("is_anonymous") == "Y") else "N"

    if not title or not content:
        flash("제목과 내용을 모두 입력해 주세요.", "warning")
        return redirect(url_for("service.faq_board", tab=tab))

    user = User.query.get_or_404(session["user_id"])
    display_writer = "익명" if is_anon == "Y" else f"{user.display_name or user.full_name}"

    new_post = CommunityBoard(
        parent_id=None,
        category=category,
        title=title,
        content=content,
        writer_id=user.user_id,
        writer_display=display_writer,
        is_anonymous=is_anon,
        view_count=0,
    )
    db.session.add(new_post)
    db.session.commit()

    flash("게시글이 정상적으로 등록되었습니다.", "success")
    return redirect(url_for("service.faq_detail", board_id=new_post.board_id))


@service_bp.route("/faq/reply/<int:board_id>", methods=["POST"])
@login_required
def faq_reply(board_id):
    parent_post = CommunityBoard.query.filter_by(board_id=board_id, parent_id=None).first_or_404()
    user = User.query.get_or_404(session["user_id"])
    is_inquiry = parent_post.category in INQUIRY_CATEGORIES

    if is_inquiry and user.role not in ("ADMIN", "MANAGER"):
        flash("1:1 공식 문의 답변은 팀장(MANAGER) 또는 관리자(ADMIN)만 작성할 수 있습니다.", "danger")
        return redirect(url_for("service.faq_detail", board_id=board_id))

    content = (request.form.get("reply_content") or "").strip()
    if not content:
        flash("답변 또는 댓글 내용을 입력해 주세요.", "warning")
        return redirect(url_for("service.faq_detail", board_id=board_id))

    is_anon = "Y" if (not is_inquiry and request.form.get("reply_anonymous") == "Y") else "N"
    display_writer = "익명" if is_anon == "Y" else f"{user.display_name or user.full_name}"

    reply = CommunityBoard(
        parent_id=parent_post.board_id,
        category=parent_post.category,
        title=None,
        content=content,
        writer_id=user.user_id,
        writer_display=display_writer,
        is_anonymous=is_anon,
    )
    db.session.add(reply)
    db.session.commit()

    flash("답변이 등록되었습니다." if is_inquiry else "댓글이 등록되었습니다.", "success")
    return redirect(url_for("service.faq_detail", board_id=board_id))


@service_bp.route("/faq/delete/<int:board_id>", methods=["POST"])
@login_required
def faq_delete(board_id):
    item = CommunityBoard.query.get_or_404(board_id)
    if item.writer_id != session["user_id"] and session.get("role") != "ADMIN":
        flash("본인이 작성한 글만 삭제할 수 있습니다.", "danger")
        return redirect(url_for("service.faq_board"))

    if item.parent_id is not None:
        parent_id = item.parent_id
        db.session.delete(item)
        db.session.commit()
        flash("댓글이 삭제되었습니다.", "secondary")
        return redirect(url_for("service.faq_detail", board_id=parent_id))

    is_inq = item.category in INQUIRY_CATEGORIES
    CommunityBoard.query.filter_by(parent_id=item.board_id).delete()
    db.session.delete(item)
    db.session.commit()
    flash("게시글이 삭제되었습니다.", "secondary")
    return redirect(url_for("service.faq_board", tab="inquiry" if is_inq else "community"))


# =====================================================================
# 3. 비전 서비스 페이지 라우트
# =====================================================================
@service_bp.route("/vision")
@login_required
def vision_page():
    return redirect(url_for("vision.index"))


# =====================================================================
# 4. 작업이력 서비스 페이지 라우트
# =====================================================================
def _extract_inst_pdf_url(inst) -> str:
    """WorkInstruction의 external_env_json에서 PDF 다운로드 URL을 추출하되, 실제 파일 존재 시에만 반환"""
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
    """WorkInstruction의 external_env_json에서 외주 발주 주문서(PO) PDF URL 추출"""
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

@service_bp.route("/history", methods=["GET"])
@login_required
def history_page():
    """작업이력 및 ERP 자원관리 대시보드"""
    user = User.query.get_or_404(session["user_id"])

    my_orders_raw = (
        WorkOrder.query
        .filter_by(requested_by=user.user_id)
        .order_by(WorkOrder.work_id.desc())
        .all()
    )
    my_orders = []
    for wo in my_orders_raw:
        inst = (
            WorkInstruction.query
            .filter_by(work_id=wo.work_id)
            .order_by(WorkInstruction.instruction_id.desc())
            .first()
        )
        my_orders.append({
            "wo": wo,
            "inst": inst,
            "pdf_url": _extract_inst_pdf_url(inst),
            "po_pdf_url": _extract_inst_po_pdf_url(inst),
        })

    all_orders_raw = WorkOrder.query.order_by(WorkOrder.work_id.desc()).all()
    user_map = {u.user_id: u.full_name for u in User.query.all()}
    all_orders = []
    for wo in all_orders_raw:
        inst = (
            WorkInstruction.query
            .filter_by(work_id=wo.work_id)
            .order_by(WorkInstruction.instruction_id.desc())
            .first()
        )
        all_orders.append({
            "wo": wo,
            "requester_name": user_map.get(wo.requested_by, "시스템"),
            "inst": inst,
            "pdf_url": _extract_inst_pdf_url(inst),
            "po_pdf_url": _extract_inst_po_pdf_url(inst),  # 👈 이 1줄 추가
        })

    erp_bom, erp_workers, erp_equipments, erp_dispatches = [], [], [], []
    if user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER"):
        erp_bom = ErpBomInventory.query.order_by(ErpBomInventory.item_code.asc()).all()
        erp_workers = ErpWorkerCert.query.order_by(ErpWorkerCert.worker_id.asc()).all()
        erp_equipments = ErpVehiclesEquipment.query.order_by(ErpVehiclesEquipment.asset_code.asc()).all()
        erp_dispatches = MesWorkOrderDispatch.query.order_by(MesWorkOrderDispatch.created_at.desc()).all()

    pending_approvals = []
    if user.role in ("ADMIN", "MANAGER"):
        for item in all_orders:
            wo = item.get("wo")
            inst = item.get("inst")
            if (
                inst
                and inst.approval_status == "DRAFT"
                and wo
                and wo.status_code not in ("SUSPENDED", "COMPLETED", "CANCELLED")
            ):
                pending_approvals.append(item)

    tab = (request.args.get("tab") or "vision").strip().lower()
    if tab not in ("vision", "orders", "agent"):
        tab = "vision"
    branch_filter = (request.args.get("branch") or "ALL").strip()
    q = (request.args.get("q") or "").strip()
    page = max(1, request.args.get("page", 1, type=int))
    per_page = 10

    branches = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).all()
    branch_name_map = {b.branch_code: b.branch_name for b in branches}

    # 1) 야드별 용접선 탐지 로그 (VISION_INSPECTIONS + VISION_DETECTIONS + WORK_ORDERS)
    vis_q = (
        db.session.query(VisionInspection, VisionDetection, WorkOrder, VisionAnalysis)
        .outerjoin(VisionDetection, VisionInspection.detection_id == VisionDetection.detection_id)
        .outerjoin(WorkOrder, VisionInspection.work_id == WorkOrder.work_id)
        .outerjoin(VisionAnalysis, VisionInspection.analysis_id == VisionAnalysis.analysis_id)
    )
    if branch_filter != "ALL":
        vis_q = vis_q.filter(WorkOrder.work_region == branch_filter)
    if q:
        like_q = f"%{q}%"
        vis_q = vis_q.filter(or_(
            WorkOrder.ship_code.ilike(like_q),
            WorkOrder.block_code.ilike(like_q),
            WorkOrder.work_no.ilike(like_q),
            VisionInspection.ocr_text.ilike(like_q),
        ))

    vision_total = vis_q.count()
    vision_total_pages = max(1, (vision_total + per_page - 1) // per_page)
    vis_page = min(page, vision_total_pages) if tab == "vision" else 1
    vis_rows = (
        vis_q.order_by(VisionInspection.inspection_id.desc())
        .offset((vis_page - 1) * per_page)
        .limit(per_page)
        .all()
    )
    vision_logs = []
    for insp, det, wo, ana in vis_rows:
        reg = wo.work_region if wo else "-"
        vision_logs.append({
            "inspection_id": insp.inspection_id,
            "branch_code": reg,
            "branch_name": branch_name_map.get(reg, reg),
            "camera_code": ana.camera_code if ana else "-",
            "ship_code": wo.ship_code if wo else "-",
            "block_code": wo.block_code if wo else "-",
            "work_no": wo.work_no if wo else "-",
            "ocr_text": insp.ocr_text or (det.ocr_text if det else "-"),
            "tracker_id": det.tracker_id if det else None,
            "class_name": det.class_name if det else "weld_joint",
            "confidence": float(det.confidence) if (det and det.confidence is not None) else None,
            "start_point": insp.start_point_json or "-",
            "end_point": insp.end_point_json or "-",
            "safety_yn": insp.safety_clearance_yn,
            "status": insp.inspection_status,
            "created_at": insp.created_at.strftime("%Y-%m-%d %H:%M") if insp.created_at else "-",
        })

    # 2) AI 업무 실행 이력 (AGENT_EXECUTIONS)
    ag_q = AgentExecution.query
    if user.role not in ("ADMIN", "MANAGER"):
        ag_q = ag_q.filter_by(user_id=user.user_id)
    agent_total = ag_q.count()
    agent_total_pages = max(1, (agent_total + per_page - 1) // per_page)
    ag_page = min(page, agent_total_pages) if tab == "agent" else 1
    agent_execs = (
        ag_q.order_by(AgentExecution.execution_id.desc())
        .offset((ag_page - 1) * per_page)
        .limit(per_page)
        .all()
    )
    agent_logs = [
        {
            "execution_id": a.execution_id,
            "session_id": a.session_id,
            "requester_name": user_map.get(a.user_id, str(a.user_id)),
            "intent_type": a.intent_type,
            "current_state": a.current_state or "-",
            "status_code": a.status_code,
            "execution_time_ms": float(a.execution_time_ms or 0),
            "error_message": _read_clob(a.error_message) if a.error_message else "",
            "created_at": a.created_at.strftime("%Y-%m-%d %H:%M") if a.created_at else "-",
        }
        for a in agent_execs
    ]

    # 3) 작업지시서 필터링
    filtered_orders = all_orders
    if branch_filter != "ALL":
        filtered_orders = [o for o in filtered_orders if o["wo"].work_region == branch_filter]
    if q:
        ql = q.lower()
        filtered_orders = [
            o for o in filtered_orders
            if ql in (o["wo"].ship_code or "").lower()
            or ql in (o["wo"].block_code or "").lower()
            or ql in (o["wo"].work_no or "").lower()
            or ql in (o["wo"].work_title or "").lower()
        ]
    order_total = len(filtered_orders)
    order_total_pages = max(1, (order_total + per_page - 1) // per_page)
    ord_page = min(page, order_total_pages) if tab == "orders" else 1
    paged_orders = filtered_orders[(ord_page - 1) * per_page : ord_page * per_page]

    active_total_pages = {"vision": vision_total_pages, "orders": order_total_pages, "agent": agent_total_pages}[tab]
    active_page = {"vision": vis_page, "orders": ord_page, "agent": ag_page}[tab]

    return render_template(
        "service/history.html",
        me=user,
        tab=tab,
        q=q,
        branch_filter=branch_filter,
        branches=branches,
        branch_name_map=branch_name_map,
        vision_logs=vision_logs,
        vision_total=vision_total,
        paged_orders=paged_orders,
        order_total=order_total,
        agent_logs=agent_logs,
        agent_total=agent_total,
        page=active_page,
        total_pages=active_total_pages,
        page_items=_build_page_items(active_page, active_total_pages),
        pending_approvals=pending_approvals,
    )


@service_bp.route("/api/history/robot-control", methods=["POST"])
@login_required
def control_robot_status():
    """로봇 제어 권한(CAN_CONTROL_ROBOT='Y') 보유자가 로봇 상태/비상정지(E-STOP)를 즉시 제어"""
    user = User.query.get_or_404(session["user_id"])
    if user.can_control_robot != "Y" and user.role != "ADMIN":
        return jsonify({
            "status": "ERROR",
            "message": "로봇 물리 제어 권한(CAN_CONTROL_ROBOT=Y)이 없습니다."
        }), 403

    data = request.get_json(silent=True) or {}
    robot_id = data.get("robot_exec_id")
    command = data.get("command")

    robot = RobotExecution.query.get_or_404(robot_id)

    if command == "ESTOP":
        robot.robot_status = "ERROR"
        robot.emergency_stop_yn = "Y"
        msg = f"🚨 [{robot.robot_name}] 비상정지(E-STOP)가 발동되었습니다!"
    elif command == "RESUME_IDLE":
        robot.robot_status = "IDLE"
        robot.emergency_stop_yn = "N"
        msg = f"✅ [{robot.robot_name}] 비상정지가 해제되고 대기(IDLE) 상태로 전환되었습니다."
    elif command == "START_WELD":
        robot.robot_status = "WELDING"
        robot.emergency_stop_yn = "N"
        msg = f"🔥 [{robot.robot_name}] 자동 용접(WELDING) 구동을 시작합니다."
    elif command == "PAUSE":
        robot.robot_status = "PAUSED"
        msg = f"⏸️ [{robot.robot_name}] 작업이 일시 정지(PAUSED)되었습니다."
    else:
        return jsonify({"status": "ERROR", "message": "잘못된 제어 명령입니다."}), 400

    robot.last_heartbeat_at = datetime.utcnow()
    robot.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({"status": "SUCCESS", "message": msg})


@service_bp.route("/api/history/order-status", methods=["POST"])
@login_required
def update_work_order_status():
    """작업오더 진행 상태(READY -> IN_PROGRESS -> COMPLETED) 변경 및 ERP 배정 원장(MES_WORK_ORDER_DISPATCH) 동기화"""
    from sqlalchemy import text
    from modules.agent.tools.erp_tools import erp_engine

    data = request.get_json(silent=True) or {}
    work_id = data.get("work_id")
    next_status = str(data.get("status_code") or "").strip().upper()

    wo = WorkOrder.query.get_or_404(work_id)
    wo.status_code = next_status
    wo.updated_at = datetime.utcnow()
    db.session.commit()

    # ERP 배정 원장(MES_WORK_ORDER_DISPATCH) 상태 동기화
    if erp_engine and wo.work_no:
        try:
            with erp_engine.begin() as conn:
                if next_status == "COMPLETED":
                    # 1) 해당 작업지시 번호의 배정 상태를 COMPLETED로 변경 (작업자 투입 해제)
                    conn.execute(
                        text("""
                            UPDATE MES_WORK_ORDER_DISPATCH
                            SET status = 'COMPLETED'
                            WHERE wo_number = :wo_no
                              AND UPPER(status) = 'COMMITTED'
                        """),
                        {"wo_no": wo.work_no},
                    )
                    # 2) 동일 블록(예: B-305)에 대해 웹 WORK_ORDERS 상 활성(READY/IN_PROGRESS) 오더가 0건이면
                    #    과거 시드 데이터 등 잔여 COMMITTED 건도 함께 COMPLETED로 마감
                    active_block_wo_cnt = (
                        WorkOrder.query
                        .filter(
                            WorkOrder.block_code == wo.block_code,
                            WorkOrder.status_code.in_(["READY", "IN_PROGRESS"]),
                        )
                        .count()
                    )
                    if wo.block_code and active_block_wo_cnt == 0:
                        conn.execute(
                            text("""
                                UPDATE MES_WORK_ORDER_DISPATCH
                                SET status = 'COMPLETED'
                                WHERE block_id = :b_id
                                  AND UPPER(status) = 'COMMITTED'
                            """),
                            {"b_id": wo.block_code},
                        )

                elif next_status in ("SUSPENDED", "CANCELLED"):
                    conn.execute(
                        text("""
                            UPDATE MES_WORK_ORDER_DISPATCH
                            SET status = 'ROLLED_BACK'
                            WHERE wo_number = :wo_no
                              AND UPPER(status) = 'COMMITTED'
                        """),
                        {"wo_no": wo.work_no},
                    )

                elif next_status == "IN_PROGRESS":
                    conn.execute(
                        text("""
                            UPDATE MES_WORK_ORDER_DISPATCH
                            SET status = 'COMMITTED'
                            WHERE wo_number = :wo_no
                              AND UPPER(status) IN ('COMPLETED', 'ROLLED_BACK')
                        """),
                        {"wo_no": wo.work_no},
                    )
        except Exception as e:
            print(f"⚠️ [MES_WORK_ORDER_DISPATCH 상태 동기화 경고]: {e}")

    return jsonify({"status": "SUCCESS", "message": f"작업 [{wo.work_no}] 상태가 {next_status}로 변경되었습니다."})


# =====================================================================
# 5. 조선소 야드 · 선박 설계도(CAD) 현황 & 야드 대시보드 (/service/yards)
#    - 하드코딩 0%: 모든 지부, 선박/블록, 진행률, CAD, 자재, 인력 DB 동적 조회
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


def _serialize_cad_doc(doc: DocumentMetadata) -> dict:
    from modules.knowledge.cad_parser import detect_vessel_context

    abs_p = _resolve_doc_abs_path(doc.storage_uri)
    exists = bool(abs_p and abs_p.exists() and abs_p.is_file())
    ext = Path(doc.original_file_name or "").suffix.lower()
    vessel_name, vessel_desc = detect_vessel_context(abs_p or Path(doc.storage_uri or doc.original_file_name or ""))

    if ext == ".dxf":
        preview_mode, ext_badge = "svg", "2D CAD (DXF)"
    elif ext == ".pdf":
        preview_mode, ext_badge = "pdf", "도면 PDF"
    elif ext in (".igs", ".iges", ".stp", ".step", ".stl"):
        preview_mode, ext_badge = "spec", f"3D 모델 ({ext.lstrip('.').upper()})"
    else:
        preview_mode, ext_badge = "spec", f"CAD ({ext.lstrip('.').upper() or 'FILE'})"

    size_kb = round(abs_p.stat().st_size / 1024.0, 1) if exists and abs_p else 0.0
    return {
        "document_id": doc.document_id,
        "document_code": doc.document_code,
        "document_title": _clean_doc_title(doc.document_title, doc.original_file_name),
        "original_file_name": doc.original_file_name,
        "collection_type": doc.collection_type,
        "storage_uri": doc.storage_uri,
        "extension": ext,
        "ext_badge": ext_badge,
        "preview_mode": preview_mode,
        "vessel_name": vessel_name,
        "vessel_desc": vessel_desc,
        "file_exists": exists,
        "size_kb": size_kb,
        "created_at": doc.created_at.strftime("%Y-%m-%d") if doc.created_at else "-",
    }


def _get_active_cad_docs() -> list[dict]:
    rows = (
        DocumentMetadata.query
        .filter(
            DocumentMetadata.category_code == "CAD_DRAWING",
            DocumentMetadata.is_active == "Y",
        )
        .order_by(DocumentMetadata.document_id.desc())
        .all()
    )
    items = [_serialize_cad_doc(r) for r in rows]
    priority = {"svg": 0, "pdf": 1, "spec": 2}
    items.sort(key=lambda x: (0 if x["file_exists"] else 1, priority.get(x["preview_mode"], 9), x["document_id"]))
    return items


@service_bp.route("/yards", methods=["GET"])
@login_required
def yards_page():
    user = User.query.get_or_404(session["user_id"])
    can_see_partners = (user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER"))

    branches = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).all()
    all_blocks = ErpBlockMaster.query.order_by(ErpBlockMaster.ship_code.asc(), ErpBlockMaster.block_id.asc()).all()
    all_orders = WorkOrder.query.order_by(WorkOrder.work_id.desc()).all()
    all_workers = ErpWorkerCert.query.all()
    all_bom = ErpBomInventory.query.all()
    cad_docs = _get_active_cad_docs()

    cad_dir = PROJECT_ROOT / "data" / "cad_drawings"
    local_cad_count = 0
    if cad_dir.exists():
        local_cad_count = sum(
            1 for p in cad_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".dxf", ".pdf", ".igs", ".iges", ".stp", ".step", ".stl", ".dwg"}
            and "github_pygem" not in [part.lower() for part in p.parts]
        )

    yard_cards = []
    for br in branches:
        br_blocks = [b for b in all_blocks if b.branch_code == br.branch_code]
        br_orders = [o for o in all_orders if o.work_region == br.branch_code]
        br_workers = [w for w in all_workers if w.branch_code == br.branch_code]
        br_bom = [m for m in all_bom if m.branch_code == br.branch_code]

        ships = sorted({b.ship_code for b in br_blocks if b.ship_code} | {o.ship_code for o in br_orders if o.ship_code})
        total_o = len(br_orders)
        done_o = sum(1 for o in br_orders if o.status_code == "COMPLETED")
        in_prog_o = sum(1 for o in br_orders if o.status_code == "IN_PROGRESS")
        ready_o = sum(1 for o in br_orders if o.status_code == "READY")
        progress_pct = int(round(((done_o + 0.5 * in_prog_o) / total_o) * 100)) if total_o > 0 else 0
        low_stock_cnt = sum(1 for m in br_bom if (float(m.total_qty or 0) - float(m.reserved_qty or 0)) <= 10)

        yard_cards.append({
            "branch_code": br.branch_code,
            "branch_name": br.branch_name,
            "region_name": br.region_name,
            "branch_type": br.branch_type,
            "address": br.address or "-",
            "camera_code": br.camera_code or f"CAM-{br.branch_code}",
            "ships": ships,
            "block_count": len(br_blocks),
            "order_count": total_o,
            "ready_count": ready_o,
            "in_progress_count": in_prog_o,
            "completed_count": done_o,
            "progress_pct": progress_pct,
            "worker_count": len(br_workers),
            "material_count": len(br_bom),
            "low_stock_count": low_stock_cnt,
        })

    selected_branch = request.args.get("branch", "").strip()
    if not selected_branch and yard_cards:
        selected_branch = yard_cards[0]["branch_code"]

    q = (request.args.get("q") or "").strip()
    branch_filter = (request.args.get("branch_filter") or "ALL").strip() or "ALL"
    branch_map = {b.branch_code: b for b in branches}

    active_work_cards = []
    for idx, blk in enumerate(all_blocks):
        br = branch_map.get(blk.branch_code)
        b_name = br.branch_name if br else blk.branch_code
        br_orders = [o for o in all_orders if o.work_region == blk.branch_code]
        stage = _infer_yard_stage(br, br_orders) if br else "WORKING"
        matched_cad = _match_cad_for_block(blk, cad_docs, idx)
        matched_wo = next((o for o in all_orders if o.block_code == blk.block_id), None)

        card = {
            "branch_code": blk.branch_code,
            "branch_name": b_name,
            "region_name": br.region_name if br else "-",
            "ship_code": blk.ship_code,
            "block_id": blk.block_id,
            "block_name": blk.block_name,
            "steel_type": blk.steel_type,
            "thickness_mm": str(blk.thickness_mm),
            "joint_type": blk.joint_type,
            "process": blk.process,
            "target_dock": blk.target_dock,
            "weight_ton": float(blk.weight_ton) if blk.weight_ton is not None else None,
            "stage": stage,
            "stage_label": "운반 중" if stage == "TRANSPORT" else "작업 중",
            "work_no": matched_wo.work_no if matched_wo else "작업 준비",
            "cad_doc_id": matched_cad["document_id"] if matched_cad else None,
            "cad_title": matched_cad["document_title"] if matched_cad else f"{blk.ship_code} 블록 도면",
            "cad_file": matched_cad["original_file_name"] if matched_cad else "-",
        }
        if branch_filter != "ALL" and card["branch_code"] != branch_filter:
            continue
        if q:
            ql = q.lower()
            if not (
                ql in card["ship_code"].lower()
                or ql in card["block_id"].lower()
                or ql in card["block_name"].lower()
                or ql in card["branch_name"].lower()
                or ql in card["branch_code"].lower()
            ):
                continue
        active_work_cards.append(card)

    # 스마트 페이지네이션 (3열 그리드 기준 페이지당 9개 · 10페이지 초과 시 ... 생략)
    page = max(1, request.args.get("page", 1, type=int))
    per_page = 9
    total_cards = len(active_work_cards)
    total_pages = max(1, (total_cards + per_page - 1) // per_page)
    page = min(page, total_pages)
    paged_work_cards = active_work_cards[(page - 1) * per_page : page * per_page]

    return render_template(
        "service/yards.html",
        me=user,
        can_see_partners=can_see_partners,
        yard_cards=yard_cards,
        active_work_cards=paged_work_cards,
        total_cards=total_cards,
        page=page,
        total_pages=total_pages,
        page_items=_build_page_items(page, total_pages),
        branch_filter=branch_filter,
        cad_docs=cad_docs,
        local_cad_count=local_cad_count,
        selected_branch=selected_branch,
        q=q,
    )


@service_bp.route("/api/yards/<branch_code>", methods=["GET"])
@login_required
def get_yard_dashboard_data(branch_code: str):
    br = ErpBranch.query.filter_by(branch_code=branch_code).first()
    if not br:
        return jsonify({"status": "ERROR", "message": "해당 야드 정보를 찾을 수 없습니다."}), 404

    blocks = ErpBlockMaster.query.filter_by(branch_code=branch_code).order_by(ErpBlockMaster.block_id.asc()).all()
    block_map = {b.block_id: b for b in blocks}
    orders = WorkOrder.query.filter_by(work_region=branch_code).order_by(WorkOrder.work_id.desc()).all()
    bom_rows = ErpBomInventory.query.filter_by(branch_code=branch_code).order_by(ErpBomInventory.item_code.asc()).all()
    worker_rows = ErpWorkerCert.query.filter_by(branch_code=branch_code).order_by(ErpWorkerCert.worker_id.asc()).all()
    equip_rows = ErpVehiclesEquipment.query.filter_by(branch_code=branch_code).order_by(ErpVehiclesEquipment.asset_code.asc()).all()

    order_ids = {o.work_id for o in orders}
    robots = []
    for rb in RobotExecution.query.all():
        loc = (rb.current_location_desc or "")
        if (rb.work_id in order_ids) or (br.region_name and br.region_name in (rb.robot_name or "") + loc):
            robots.append({
                "equipment_code": rb.equipment_code,
                "robot_name": rb.robot_name,
                "robot_status": rb.robot_status,
                "location": rb.current_location_desc or "-",
                "battery_percent": int(rb.battery_percent or 0),
                "emergency_stop_yn": rb.emergency_stop_yn,
            })

    weather_info = None
    for o in orders:
        inst = WorkInstruction.query.filter_by(work_id=o.work_id).order_by(WorkInstruction.instruction_id.desc()).first()
        if inst and inst.external_env_json:
            try:
                env_data = json.loads(_read_clob(inst.external_env_json))
                if isinstance(env_data, dict):
                    cand = env_data.get("weather") if isinstance(env_data.get("weather"), dict) else env_data
                    if "temperature_c" in cand or "weather" in cand:
                        weather_info = cand
                        break
            except Exception:
                pass

    waiting_blocks, active_blocks, completed_blocks = [], [], []
    seen_order_blocks = set()

    for o in orders:
        bm = block_map.get(o.block_code)
        seen_order_blocks.add(o.block_code)
        entry = {
            "work_id": o.work_id,
            "work_no": o.work_no,
            "work_title": o.work_title,
            "ship_code": o.ship_code,
            "block_code": o.block_code,
            "block_name": bm.block_name if bm else o.block_code,
            "steel_type": bm.steel_type if bm else o.material_grade,
            "thickness_mm": str(bm.thickness_mm) if bm else f"{float(o.thickness_mm):.0f}mm",
            "joint_type": bm.joint_type if bm else o.weld_position,
            "process": o.process_code,
            "target_dock": bm.target_dock if bm else o.work_zone,
            "weight_ton": float(bm.weight_ton) if (bm and bm.weight_ton is not None) else None,
            "status_code": o.status_code,
        }
        if o.status_code == "COMPLETED":
            completed_blocks.append(entry)
        elif o.status_code == "IN_PROGRESS":
            active_blocks.append(entry)
        else:
            waiting_blocks.append(entry)

    for bm in blocks:
        if bm.block_id not in seen_order_blocks:
            waiting_blocks.append({
                "work_id": None,
                "work_no": "지시 대기",
                "work_title": f"{bm.block_name} 작업 준비",
                "ship_code": bm.ship_code,
                "block_code": bm.block_id,
                "block_name": bm.block_name,
                "steel_type": bm.steel_type,
                "thickness_mm": str(bm.thickness_mm),
                "joint_type": bm.joint_type,
                "process": bm.process,
                "target_dock": bm.target_dock,
                "weight_ton": float(bm.weight_ton) if bm.weight_ton is not None else None,
                "status_code": "READY",
            })

    ships = sorted({b.ship_code for b in blocks if b.ship_code} | {o.ship_code for o in orders if o.ship_code})
    total_cnt = len(waiting_blocks) + len(active_blocks) + len(completed_blocks)
    progress_pct = int(round(((len(completed_blocks) + 0.5 * len(active_blocks)) / total_cnt) * 100)) if total_cnt > 0 else 0

    materials = []
    for m in bom_rows:
        tot = float(m.total_qty or 0)
        res = float(m.reserved_qty or 0)
        avail = max(tot - res, 0.0)
        is_low = (avail <= 10) or (tot > 0 and (avail / tot) <= 0.25)
        materials.append({
            "item_code": m.item_code,
            "item_name": m.item_name,
            "category": m.material_category,
            "lot_no": m.lot_no,
            "total_qty": tot,
            "reserved_qty": res,
            "available_qty": avail,
            "unit": m.unit,
            "is_low": is_low,
        })

    workers = []
    for w in worker_rows:
        d_left = _calc_cert_days_left(w.cert_expiry_date)
        workers.append({
            "worker_id": w.worker_id,
            "worker_name": w.worker_name,
            "department_name": w.department_name or "-",
            "role": w.role,
            "cert_code": w.cert_code,
            "weld_process": w.weld_process,
            "cert_expiry_date": w.cert_expiry_date,
            "days_left": d_left,
            "expiring_soon": (d_left is not None and d_left <= 60),
            "status": w.status,
        })

    equipments = [
        {
            "asset_code": e.asset_code,
            "asset_name": e.asset_name,
            "asset_category": e.asset_category,
            "current_zone": e.current_zone or "-",
            "max_load_ton": float(e.max_load_ton) if e.max_load_ton is not None else None,
            "operation_status": e.operation_status,
        }
        for e in equip_rows
    ]

    work_zones = []
    if br.work_zones_json:
        try:
            parsed_z = json.loads(br.work_zones_json)
            if isinstance(parsed_z, list):
                work_zones = parsed_z
        except Exception:
            pass

    return jsonify({
        "status": "SUCCESS",
        "branch": {
            "branch_code": br.branch_code,
            "branch_name": br.branch_name,
            "region_name": br.region_name,
            "branch_type": br.branch_type,
            "address": br.address or "-",
            "camera_code": br.camera_code or f"CAM-{br.branch_code}",
            "work_zones": work_zones,
        },
        "ships": ships,
        "progress_pct": progress_pct,
        "weather": weather_info,
        "stages": {
            "waiting": waiting_blocks,
            "active": active_blocks,
            "completed": completed_blocks,
        },
        "materials": materials,
        "workers": workers,
        "equipments": equipments,
        "robots": robots,
        "cad_docs": _get_active_cad_docs(),
    })


@service_bp.route("/api/cad/preview/<int:doc_id>", methods=["GET"])
@login_required
def cad_preview_api(doc_id: int):
    doc = DocumentMetadata.query.filter_by(document_id=doc_id, category_code="CAD_DRAWING").first()
    if not doc:
        return jsonify({"status": "ERROR", "message": "해당 설계 도면을 찾을 수 없습니다."}), 404

    from modules.knowledge.cad_parser import extract_cad_summary, render_dxf_to_svg

    info = _serialize_cad_doc(doc)
    abs_p = _resolve_doc_abs_path(doc.storage_uri)
    if not abs_p or not abs_p.exists():
        return jsonify({
            "status": "SUCCESS",
            "doc": info,
            "summary": {"format": "파일 경로 미존재", "vessel_name": info["vessel_name"]},
            "svg": "",
            "pdf_url": "",
        })

    summary = extract_cad_summary(abs_p)
    svg_html = ""
    pdf_url = ""
    ext = abs_p.suffix.lower()
    sibling_pdf = abs_p.with_suffix(".pdf")
    if ext == ".pdf":
        pdf_url = url_for("service.cad_raw_file", doc_id=doc.document_id)
    elif ext in (".dxf", ".dwg") and sibling_pdf.exists():
        # 같은 이름의 공식 A0 도면 PDF가 있으면 선명한 원본 도면 뷰어(확대/축소 지원)로 표시
        pdf_url = url_for("service.cad_raw_file", doc_id=doc.document_id, preview_pdf="1")
    elif ext == ".dxf":
        svg_html = render_dxf_to_svg(abs_p, cache_dir=PROJECT_ROOT / "data" / "rag_cache" / "cad_svg")

    return jsonify({
        "status": "SUCCESS",
        "doc": info,
        "summary": summary,
        "svg": svg_html,
        "pdf_url": pdf_url,
    })


@service_bp.route("/api/cad/file/<int:doc_id>", methods=["GET"])
@login_required
def cad_raw_file(doc_id: int):
    doc = DocumentMetadata.query.filter_by(document_id=doc_id, category_code="CAD_DRAWING").first_or_404()
    abs_p = _resolve_doc_abs_path(doc.storage_uri)
    if not abs_p or not abs_p.exists():
        abort(404)
    if request.args.get("preview_pdf") == "1" and abs_p.with_suffix(".pdf").exists():
        return send_file(abs_p.with_suffix(".pdf"), as_attachment=False)
    as_dl = request.args.get("download", "0") == "1"
    return send_file(abs_p, as_attachment=as_dl, download_name=doc.original_file_name)


# =====================================================================
# 6. 자재 · 인력 · 발주 관리 보드 (/service/erp)
# =====================================================================
@service_bp.route("/erp", methods=["GET"])
@login_required
def erp_board_page():
    db.session.expire_all()
    user = User.query.get_or_404(session["user_id"])
    can_access_erp = (user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER"))
    selected_branch = request.args.get("branch", "ALL").strip() or "ALL"

    branches = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).all()
    branch_name_map = {b.branch_code: b.branch_name for b in branches}

    materials, workers, equipments, robots, dispatches = [], [], [], [], []
    partner_materials, partner_workers, partner_equipments, partner_orders = [], [], [], []
    partner_error = ""

    if can_access_erp:
        for m in ErpBomInventory.query.order_by(ErpBomInventory.branch_code.asc(), ErpBomInventory.item_code.asc()).all():
            tot = float(m.total_qty or 0)
            res = float(m.reserved_qty or 0)
            avail = max(tot - res, 0.0)
            is_low = (avail <= 10) or (tot > 0 and (avail / tot) <= 0.25)
            materials.append({
                "item_code": m.item_code,
                "item_name": m.item_name,
                "material_category": m.material_category,
                "lot_no": m.lot_no,
                "supplier_code": m.supplier_code,
                "branch_code": m.branch_code,
                "branch_name": branch_name_map.get(m.branch_code, m.branch_code),
                "total_qty": tot,
                "reserved_qty": res,
                "available_qty": avail,
                "unit": m.unit,
                "is_low": is_low,
            })

        for w in ErpWorkerCert.query.order_by(ErpWorkerCert.cert_expiry_date.asc(), ErpWorkerCert.worker_id.asc()).all():
            d_left = _calc_cert_days_left(w.cert_expiry_date)
            workers.append({
                "worker_id": w.worker_id,
                "worker_name": w.worker_name,
                "employment_type": w.employment_type,
                "company_code": w.company_code,
                "branch_code": w.branch_code,
                "branch_name": branch_name_map.get(w.branch_code, w.branch_code),
                "department_name": w.department_name or "-",
                "role": w.role,
                "phone": w.phone or "-",
                "cert_code": w.cert_code,
                "weld_process": w.weld_process,
                "cert_expiry_date": w.cert_expiry_date,
                "days_left": d_left,
                "expiring_soon": (d_left is not None and d_left <= 60),
                "is_expired": (d_left is not None and d_left < 0),
                "status": w.status,
            })

        for e in ErpVehiclesEquipment.query.order_by(ErpVehiclesEquipment.branch_code.asc(), ErpVehiclesEquipment.asset_code.asc()).all():
            equipments.append({
                "asset_code": e.asset_code,
                "asset_name": e.asset_name,
                "asset_category": e.asset_category,
                "branch_code": e.branch_code,
                "branch_name": branch_name_map.get(e.branch_code, e.branch_code),
                "current_zone": e.current_zone or "-",
                "max_load_ton": float(e.max_load_ton) if e.max_load_ton is not None else None,
                "operation_status": e.operation_status,
            })

        robots = RobotExecution.query.order_by(RobotExecution.robot_exec_id.asc()).all()
        dispatches = MesWorkOrderDispatch.query.order_by(MesWorkOrderDispatch.created_at.desc()).limit(20).all()

        try:
            partner_materials = B2BDaehanMaterial.query.all()
            partner_workers = B2BDonghaeWorker.query.all()
            partner_equipments = B2BHankookEquipment.query.all()
            partner_orders = B2BSalesOrder.query.order_by(B2BSalesOrder.created_at.desc()).limit(15).all()
        except Exception as exc:
            partner_error = f"협력사 실시간 연계 DB를 잠시 불러오지 못했습니다: {exc}"

    pending_approvals = []
    if user.role in ("ADMIN", "MANAGER"):
        user_map = {u.user_id: u.full_name for u in User.query.all()}
        for wo in WorkOrder.query.order_by(WorkOrder.work_id.desc()).all():
            inst = WorkInstruction.query.filter_by(work_id=wo.work_id).order_by(WorkInstruction.instruction_id.desc()).first()
            if (
                inst
                and inst.approval_status == "DRAFT"
                and wo.status_code not in ("SUSPENDED", "COMPLETED", "CANCELLED")
            ):
                pending_approvals.append({
                    "wo": wo,
                    "inst": inst,
                    "requester_name": user_map.get(wo.requested_by, "현장 요청"),
                    "pdf_url": _extract_inst_pdf_url(inst),
                    "po_pdf_url": _extract_inst_po_pdf_url(inst),
                })

    low_stock_count = sum(1 for m in materials if m["is_low"])
    expiring_workers = [w for w in workers if w["expiring_soon"]]
    partner_resource_count = len(partner_materials) + len(partner_workers) + len(partner_equipments)

    # 야드 필터 적용 (대시보드 요약 수치)
    if selected_branch != "ALL":
        materials = [m for m in materials if m["branch_code"] == selected_branch]
        workers = [w for w in workers if w["branch_code"] == selected_branch]
        equipments = [e for e in equipments if e["branch_code"] == selected_branch]
        low_stock_count = sum(1 for m in materials if m["is_low"])
        expiring_workers = [w for w in workers if w["expiring_soon"]]

    # 로봇·장비 상태 분포 (대기 / 사용 중 / 점검)
    eq_standby = sum(1 for e in equipments if e["operation_status"] == "STANDBY")
    eq_in_use = sum(1 for e in equipments if e["operation_status"] == "IN_USE")
    eq_maint = sum(1 for e in equipments if e["operation_status"] not in ("STANDBY", "IN_USE"))

    # 선박별 공정 진행률
    ship_progress = []
    all_wos = WorkOrder.query.order_by(WorkOrder.work_id.desc()).all()
    if selected_branch != "ALL":
        all_wos = [o for o in all_wos if o.work_region == selected_branch]
    ships_seen = sorted({o.ship_code for o in all_wos if o.ship_code})
    for sc in ships_seen:
        swos = [o for o in all_wos if o.ship_code == sc]
        done_c = sum(1 for o in swos if o.status_code == "COMPLETED")
        prog_c = sum(1 for o in swos if o.status_code == "IN_PROGRESS")
        pct = int(round(((done_c + 0.55 * prog_c) / max(len(swos), 1)) * 100))
        ship_progress.append({
            "ship_code": sc,
            "region": branch_name_map.get(swos[0].work_region, swos[0].work_region),
            "total": len(swos),
            "in_progress": prog_c,
            "completed": done_c,
            "pct": pct,
        })

    low_materials = [m for m in materials if m["is_low"]]
    if not low_materials:
        low_materials = sorted(materials, key=lambda x: x["available_qty"])[:4]

    return render_template(
        "service/erp_board.html",
        me=user,
        can_access_erp=can_access_erp,
        branches=branches,
        selected_branch=selected_branch,
        materials=materials,
        low_materials=low_materials[:5],
        workers=workers,
        expiring_workers=expiring_workers[:5],
        equipments=equipments,
        robots=robots,
        dispatches=dispatches,
        partner_materials=partner_materials,
        partner_workers=partner_workers,
        partner_equipments=partner_equipments,
        partner_orders=partner_orders,
        partner_error=partner_error,
        pending_approvals=pending_approvals,
        ship_progress=ship_progress,
        eq_stats={"standby": eq_standby, "in_use": eq_in_use, "maint": eq_maint, "total": len(equipments)},
        initial_drawer=request.args.get("drawer", ""),
        initial_prompt=request.args.get("prompt", ""),
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        kpi={
            "total_materials": len(materials),
            "low_stock_count": low_stock_count,
            "total_workers": len(workers),
            "expiring_worker_count": len(expiring_workers),
            "total_equipments": len(equipments),
            "standby_equipments": eq_standby,
            "partner_resource_count": partner_resource_count,
            "pending_approval_count": len(pending_approvals),
        },
    )


@service_bp.route("/api/erp/resources", methods=["GET"])
@login_required
def api_erp_resources():
    """ERP 대량 자원 검색 · 필터 · 스마트 페이지네이션 API (실시간 DB 동기화 지원)"""
    db.session.expire_all()
    user = User.query.get_or_404(session["user_id"])
    if not (user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER")):
        return jsonify({"status": "ERROR", "message": "ERP 조회 권한이 없습니다."}), 403

    res_type = (request.args.get("type") or "materials").strip().lower()
    q = (request.args.get("q") or "").strip().lower()
    branch = (request.args.get("branch") or "ALL").strip()
    filter_flag = (request.args.get("filter") or "ALL").strip().upper()
    page = max(1, request.args.get("page", 1, type=int))
    per_page = 8

    branches = {b.branch_code: b.branch_name for b in ErpBranch.query.all()}
    items = []

    if res_type == "materials":
        for m in ErpBomInventory.query.order_by(ErpBomInventory.branch_code.asc(), ErpBomInventory.item_code.asc()).all():
            tot = float(m.total_qty or 0)
            res = float(m.reserved_qty or 0)
            avail = max(tot - res, 0.0)
            is_low = (avail <= 10) or (tot > 0 and (avail / tot) <= 0.25)
            row = {
                "code": m.item_code,
                "name": m.item_name,
                "category": m.material_category,
                "lot_no": m.lot_no,
                "branch_code": m.branch_code,
                "branch_name": branches.get(m.branch_code, m.branch_code),
                "available_qty": avail,
                "total_qty": tot,
                "unit": m.unit,
                "is_low": is_low,
            }
            if branch != "ALL" and row["branch_code"] != branch:
                continue
            if filter_flag == "LOW" and not is_low:
                continue
            if filter_flag in ("FILLER_WIRE", "STEEL_PLATE", "SHIELDING_GAS") and row["category"] != filter_flag:
                continue
            if q and not (q in row["code"].lower() or q in row["name"].lower() or q in row["lot_no"].lower()):
                continue
            items.append(row)

    elif res_type == "workers":
        for w in ErpWorkerCert.query.order_by(ErpWorkerCert.cert_expiry_date.asc()).all():
            d_left = _calc_cert_days_left(w.cert_expiry_date)
            expiring = (d_left is not None and d_left <= 60)
            expired = (d_left is not None and d_left < 0)
            row = {
                "code": w.worker_id,
                "name": w.worker_name,
                "employment_type": w.employment_type,
                "company_code": w.company_code,
                "branch_code": w.branch_code,
                "branch_name": branches.get(w.branch_code, w.branch_code),
                "department": w.department_name or "-",
                "role": w.role,
                "cert_code": w.cert_code,
                "weld_process": w.weld_process,
                "expiry": w.cert_expiry_date,
                "days_left": d_left,
                "expiring": expiring,
                "expired": expired,
                "status": w.status,
            }
            if branch != "ALL" and row["branch_code"] != branch:
                continue
            if filter_flag == "EXPIRING" and not expiring:
                continue
            if filter_flag in ("REGULAR", "SUBCONTRACT", "FREELANCER") and row["employment_type"] != filter_flag:
                continue
            if q and not (q in row["code"].lower() or q in row["name"].lower() or q in row["cert_code"].lower()):
                continue
            items.append(row)

    elif res_type == "equipment":
        for e in ErpVehiclesEquipment.query.order_by(ErpVehiclesEquipment.branch_code.asc(), ErpVehiclesEquipment.asset_code.asc()).all():
            row = {
                "code": e.asset_code,
                "name": e.asset_name,
                "category": e.asset_category,
                "branch_code": e.branch_code,
                "branch_name": branches.get(e.branch_code, e.branch_code),
                "zone": e.current_zone or "-",
                "max_load_ton": float(e.max_load_ton) if e.max_load_ton is not None else None,
                "status": e.operation_status,
            }
            if branch != "ALL" and row["branch_code"] != branch:
                continue
            if filter_flag in ("STANDBY", "IN_USE", "MAINTENANCE") and row["status"] != filter_flag:
                continue
            if q and not (q in row["code"].lower() or q in row["name"].lower() or q in row["category"].lower()):
                continue
            items.append(row)

    elif res_type == "partners":
        try:
            for pm in B2BDaehanMaterial.query.all():
                items.append({
                    "kind": "자재",
                    "partner_id": pm.partner_id,
                    "code": pm.hq_item_code,
                    "name": pm.item_name,
                    "detail": f"가용 {int(pm.available_qty or 0)} {pm.unit} · 납기 {pm.lead_time_hours}시간",
                    "region": "협력사 공급",
                })
            for pw in B2BDonghaeWorker.query.all():
                items.append({
                    "kind": "외부인력",
                    "partner_id": pw.partner_id,
                    "code": pw.ext_worker_id,
                    "name": f"{pw.worker_name} ({pw.cert_code})",
                    "detail": f"{pw.weld_process} · {pw.dispatch_status}",
                    "region": pw.current_region or "-",
                })
            for pe in B2BHankookEquipment.query.all():
                items.append({
                    "kind": "장비/차량",
                    "partner_id": pe.partner_id,
                    "code": pe.ext_asset_code,
                    "name": pe.asset_name,
                    "detail": f"{pe.asset_category} · {pe.availability_status}",
                    "region": pe.current_region or "-",
                })
            if q:
                items = [it for it in items if q in it["code"].lower() or q in it["name"].lower() or q in it["partner_id"].lower()]
        except Exception as exc:
            return jsonify({"status": "ERROR", "message": f"협력사 B2B DB 연결 오류: {exc}"}), 500

    total_count = len(items)
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    page = min(page, total_pages)
    paged = items[(page - 1) * per_page : page * per_page]

    return jsonify({
        "status": "SUCCESS",
        "type": res_type,
        "items": paged,
        "total_count": total_count,
        "page": page,
        "total_pages": total_pages,
        "page_items": _build_page_items(page, total_pages),
    })


# =====================================================================
# 7. ERP 자원별 전용 상세 조회 페이지 (/service/erp/<resource_type>)
#    - /service/erp/materials  (전체 자재·재고 조회)
#    - /service/erp/workers    (사내·외부 작업자 및 자격증)
#    - /service/erp/equipment  (로봇·장비·차량 현황)
#    - /service/erp/partners   (협력사 B2B 가용 자원)
# =====================================================================
@service_bp.route("/erp/materials", defaults={"resource_type": "materials"}, methods=["GET"])
@service_bp.route("/erp/workers", defaults={"resource_type": "workers"}, methods=["GET"])
@service_bp.route("/erp/equipment", defaults={"resource_type": "equipment"}, methods=["GET"])
@service_bp.route("/erp/partners", defaults={"resource_type": "partners"}, methods=["GET"])
@login_required
def erp_resource_page(resource_type: str):
    db.session.expire_all()
    user = User.query.get_or_404(session["user_id"])
    can_access_erp = (user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER"))

    selected_branch = (request.args.get("branch") or "ALL").strip() or "ALL"
    q = (request.args.get("q") or "").strip()
    filter_flag = (request.args.get("filter") or "ALL").strip().upper()
    page = max(1, request.args.get("page", 1, type=int))
    per_page = 12

    branches = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).all()
    branch_name_map = {b.branch_code: b.branch_name for b in branches}

    meta_map = {
        "materials": {
            "title": "전체 자재 · 재고 조회",
            "desc": "전 야드 강판·용접와이어·차폐가스 재고와 Lot 번호를 검색하고 부족 품목은 AI에게 즉시 발주 요청할 수 있습니다.",
            "filters": [("ALL", "전체 품목"), ("LOW", "부족 주의만"), ("FILLER_WIRE", "용접와이어"), ("STEEL_PLATE", "선체강판"), ("SHIELDING_GAS", "차폐가스")],
        },
        "workers": {
            "title": "사내 · 외부 작업자 및 자격증",
            "desc": "직영·협력사·프리랜서 작업자의 용접 자격증 유효기간을 조회하고 만료 임박 대상자에게 갱신 안내 메일을 보낼 수 있습니다.",
            "filters": [("ALL", "전체 인력"), ("EXPIRING", "만료·임박만"), ("REGULAR", "직영 직원"), ("SUBCONTRACT", "협력사"), ("FREELANCER", "프리랜서")],
        },
        "equipment": {
            "title": "로봇 · 장비 · 차량 현황",
            "desc": "전 야드 용접로봇, 트랜스포터, 지게차, 크레인의 실시간 운용 상태(대기/사용 중/점검)를 확인합니다.",
            "filters": [("ALL", "전체 상태"), ("STANDBY", "대기"), ("IN_USE", "사용 중"), ("MAINTENANCE", "점검 중")],
        },
        "partners": {
            "title": "협력사 B2B 가용 자원",
            "desc": "계약 중인 외부 협력사(PostgreSQL B2B 뷰)의 가용 자재, 파견 가능 인력, 임대 장비를 실시간 조회합니다.",
            "filters": [("ALL", "전체 자원"), ("MATERIAL", "외부 자재"), ("WORKER", "외부 인력"), ("EQUIPMENT", "외부 장비·차량")],
        },
    }
    page_meta = meta_map.get(resource_type, meta_map["materials"])

    items = []
    partner_error = ""
    ql = q.lower()

    if can_access_erp:
        if resource_type == "materials":
            for m in ErpBomInventory.query.order_by(ErpBomInventory.branch_code.asc(), ErpBomInventory.item_code.asc()).all():
                tot = float(m.total_qty or 0)
                res = float(m.reserved_qty or 0)
                avail = max(tot - res, 0.0)
                is_low = (avail <= 10) or (tot > 0 and (avail / tot) <= 0.25)
                row = {
                    "code": m.item_code,
                    "name": m.item_name,
                    "category": m.material_category,
                    "lot_no": m.lot_no,
                    "supplier_code": m.supplier_code,
                    "branch_code": m.branch_code,
                    "branch_name": branch_name_map.get(m.branch_code, m.branch_code),
                    "total_qty": tot,
                    "reserved_qty": res,
                    "available_qty": avail,
                    "unit": m.unit,
                    "is_low": is_low,
                }
                if selected_branch != "ALL" and row["branch_code"] != selected_branch:
                    continue
                if filter_flag == "LOW" and not is_low:
                    continue
                if filter_flag in ("FILLER_WIRE", "STEEL_PLATE", "SHIELDING_GAS") and row["category"] != filter_flag:
                    continue
                if ql and not (ql in row["code"].lower() or ql in row["name"].lower() or ql in row["lot_no"].lower()):
                    continue
                items.append(row)

        elif resource_type == "workers":
            for w in ErpWorkerCert.query.order_by(ErpWorkerCert.cert_expiry_date.asc(), ErpWorkerCert.worker_id.asc()).all():
                d_left = _calc_cert_days_left(w.cert_expiry_date)
                expiring = (d_left is not None and d_left <= 60)
                expired = (d_left is not None and d_left < 0)
                row = {
                    "code": w.worker_id,
                    "name": w.worker_name,
                    "employment_type": w.employment_type,
                    "company_code": w.company_code,
                    "branch_code": w.branch_code,
                    "branch_name": branch_name_map.get(w.branch_code, w.branch_code),
                    "department": w.department_name or "-",
                    "role": w.role,
                    "phone": w.phone or "-",
                    "cert_code": w.cert_code,
                    "weld_process": w.weld_process,
                    "expiry": w.cert_expiry_date,
                    "days_left": d_left,
                    "expiring": expiring,
                    "expired": expired,
                    "status": w.status,
                }
                if selected_branch != "ALL" and row["branch_code"] != selected_branch:
                    continue
                if filter_flag == "EXPIRING" and not expiring:
                    continue
                if filter_flag in ("REGULAR", "SUBCONTRACT", "FREELANCER") and row["employment_type"] != filter_flag:
                    continue
                if ql and not (ql in row["code"].lower() or ql in row["name"].lower() or ql in row["cert_code"].lower() or ql in row["department"].lower()):
                    continue
                items.append(row)

        elif resource_type == "equipment":
            for e in ErpVehiclesEquipment.query.order_by(ErpVehiclesEquipment.branch_code.asc(), ErpVehiclesEquipment.asset_code.asc()).all():
                row = {
                    "code": e.asset_code,
                    "name": e.asset_name,
                    "category": e.asset_category,
                    "company_code": e.company_code,
                    "branch_code": e.branch_code,
                    "branch_name": branch_name_map.get(e.branch_code, e.branch_code),
                    "zone": e.current_zone or "-",
                    "max_load_ton": float(e.max_load_ton) if e.max_load_ton is not None else None,
                    "status": e.operation_status,
                }
                if selected_branch != "ALL" and row["branch_code"] != selected_branch:
                    continue
                if filter_flag in ("STANDBY", "IN_USE", "MAINTENANCE") and row["status"] != filter_flag:
                    continue
                if ql and not (ql in row["code"].lower() or ql in row["name"].lower() or ql in row["category"].lower() or ql in row["zone"].lower()):
                    continue
                items.append(row)

        elif resource_type == "partners":
            try:
                if filter_flag in ("ALL", "MATERIAL"):
                    for pm in B2BDaehanMaterial.query.all():
                        items.append({
                            "kind": "외부 자재",
                            "kind_code": "MATERIAL",
                            "partner_id": pm.partner_id,
                            "code": pm.hq_item_code,
                            "name": pm.item_name,
                            "detail": f"가용 {int(pm.available_qty or 0)} {pm.unit} · 납기 {pm.lead_time_hours}시간 · Lot {pm.supplier_lot_no}",
                            "region": "협력사 직출",
                        })
                if filter_flag in ("ALL", "WORKER"):
                    for pw in B2BDonghaeWorker.query.all():
                        items.append({
                            "kind": "외부 인력",
                            "kind_code": "WORKER",
                            "partner_id": pw.partner_id,
                            "code": pw.ext_worker_id,
                            "name": f"{pw.worker_name} ({pw.cert_code})",
                            "detail": f"{pw.specialty_role} · {pw.weld_process} · 만료 {pw.cert_expiry_date}",
                            "region": pw.current_region or "-",
                        })
                if filter_flag in ("ALL", "EQUIPMENT"):
                    for pe in B2BHankookEquipment.query.all():
                        items.append({
                            "kind": "외부 장비·차량",
                            "kind_code": "EQUIPMENT",
                            "partner_id": pe.partner_id,
                            "code": pe.ext_asset_code,
                            "name": pe.asset_name,
                            "detail": f"{pe.asset_category} · 최대 {pe.max_load_ton or '-'}톤 · {pe.availability_status}",
                            "region": pe.current_region or "-",
                        })
                if ql:
                    items = [
                        it for it in items
                        if ql in it["code"].lower() or ql in it["name"].lower() or ql in it["partner_id"].lower() or ql in it["detail"].lower()
                    ]
            except Exception as exc:
                partner_error = f"협력사 B2B DB 연결 중 오류가 발생했습니다: {exc}"

    total_count = len(items)
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    page = min(page, total_pages)
    paged_items = items[(page - 1) * per_page : page * per_page]

    return render_template(
        "service/erp_resources.html",
        me=user,
        can_access_erp=can_access_erp,
        resource_type=resource_type,
        page_meta=page_meta,
        branches=branches,
        selected_branch=selected_branch,
        q=q,
        filter_flag=filter_flag,
        items=paged_items,
        total_count=total_count,
        page=page,
        total_pages=total_pages,
        page_items=_build_page_items(page, total_pages),
        partner_error=partner_error,
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
