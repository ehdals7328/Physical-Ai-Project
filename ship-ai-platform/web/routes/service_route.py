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
    current_app,
)
from sqlalchemy import func
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
    ErpBranch,
    ErpBlockMaster,
    ErpBomInventory,
    ErpWorkerCert,
    ErpVehiclesEquipment,
    MesWorkOrderDispatch,
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
            result = ship_agent_app.invoke(state_input, config=config)
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
                block_id_draft = params_draft.get("block_id") or ("DIRECT-PO" if is_standalone_draft else "BLK-E102")
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
                    ship_code=params_draft.get("ship_code") or ("N/A" if is_standalone_draft else "HN-2026-301"),
                    block_code=block_id_draft,
                    work_region=params_draft.get("branch_code") or (user.erp_branch_code if user else None) or "BR-ULSAN-01",
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
        question = data.get("question", "").strip()
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

        state_input = {
            "messages": [HumanMessage(content=question)],
            "user_query": question,
            "user_id": str(user.user_id),
            "user_name": f"{user.full_name} ({user.erp_employee_code or user.username})",
            "user_role": user.role,
            "user_type": user.user_type,
            "can_access_erp": user.can_access_erp or "N",
            "can_access_internal_rag": user.can_access_internal_rag or "N",
            "can_control_robot": user.can_control_robot or "N",
            "default_workspace": user.default_workspace or "ZONE-A-BAY1",
            "branch_code": user.erp_branch_code or "BR-ULSAN-01",
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


@service_bp.route("/api/assistant/status/<session_id>", methods=["GET"])
@login_required
def get_assistant_execution_status(session_id):
    """백그라운드 에이전트 실행 상태 및 완료 결과를 조회하는 경량 폴링 API"""
    user = User.query.get_or_404(session["user_id"])

    if _check_session_running(user.user_id, session_id):
        return jsonify({
            "status": "RUNNING",
            "session_id": session_id,
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
        block_id = params.get("block_id") or (target_wo.block_code if target_wo else "BLK-E102")
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
                branch_name=params.get("branch_code") or user.erp_branch_code or "BR-ULSAN-01",
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
                ship_code=params.get("ship_code") or "HN-2026-301",
                block_code=block_id,
                work_region=params.get("branch_code") or user.erp_branch_code or "BR-ULSAN-01",
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
# 2. DC/루리웹 스타일 사내 커뮤니티 & FAQ 게시판 (/service/faq)
# =====================================================================
def _get_board_list_context(category_filter: str, current_uid: int):
    query = CommunityBoard.query.filter(CommunityBoard.parent_id.is_(None))
    if category_filter != "ALL":
        query = query.filter_by(category=category_filter)

    posts = query.order_by(CommunityBoard.board_id.desc()).all()

    reply_counts = dict(
        db.session.query(CommunityBoard.parent_id, func.count(CommunityBoard.board_id))
        .filter(CommunityBoard.parent_id.isnot(None))
        .group_by(CommunityBoard.parent_id)
        .all()
    )

    my_posts = (
        CommunityBoard.query
        .filter(CommunityBoard.parent_id.is_(None), CommunityBoard.writer_id == current_uid)
        .order_by(CommunityBoard.board_id.desc())
        .all()
    )

    return posts, reply_counts, my_posts


@service_bp.route("/faq", methods=["GET"])
@login_required
def faq_board():
    category_filter = request.args.get("category", "ALL")
    posts, reply_counts, my_posts = _get_board_list_context(category_filter, session["user_id"])
    return render_template(
        "community/faq_board.html",
        posts=posts,
        reply_counts=reply_counts,
        my_posts=my_posts,
        selected_category=category_filter,
    )


@service_bp.route("/faq/<int:board_id>", methods=["GET"])
@login_required
def faq_detail(board_id):
    post = CommunityBoard.query.filter_by(board_id=board_id, parent_id=None).first_or_404()
    post.view_count = (post.view_count or 0) + 1
    db.session.commit()

    replies = (
        CommunityBoard.query
        .filter_by(parent_id=post.board_id)
        .order_by(CommunityBoard.board_id.asc())
        .all()
    )

    category_filter = request.args.get("category", "ALL")
    posts, reply_counts, my_posts = _get_board_list_context(category_filter, session["user_id"])

    return render_template(
        "community/faq_detail.html",
        post=post,
        replies=replies,
        posts=posts,
        reply_counts=reply_counts,
        my_posts=my_posts,
        selected_category=category_filter,
    )


@service_bp.route("/faq/write", methods=["POST"])
@login_required
def faq_write():
    category = request.form.get("category", "현장기술").strip()
    title = request.form.get("title", "").strip()
    content = request.form.get("content", "").strip()
    is_anon = "Y" if request.form.get("is_anonymous") == "Y" else "N"

    user = User.query.get_or_404(session["user_id"])
    display_writer = "ㅇㅇ(익명)" if is_anon == "Y" else f"{user.full_name}({user.department_name or user.role})"

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

    flash("게시글이 등록되었습니다.", "success")
    return redirect(url_for("service.faq_detail", board_id=new_post.board_id))


@service_bp.route("/faq/reply/<int:board_id>", methods=["POST"])
@login_required
def faq_reply(board_id):
    parent_post = CommunityBoard.query.get_or_404(board_id)
    content = request.form.get("reply_content", "").strip()
    is_anon = "Y" if request.form.get("reply_anonymous") == "Y" else "N"

    user = User.query.get_or_404(session["user_id"])
    display_writer = "ㅇㅇ(익명)" if is_anon == "Y" else f"{user.full_name}({user.department_name or user.role})"

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

    CommunityBoard.query.filter_by(parent_id=item.board_id).delete()
    db.session.delete(item)
    db.session.commit()
    flash("게시글이 삭제되었습니다.", "secondary")
    return redirect(url_for("service.faq_board"))


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

    return render_template(
        "service/history.html",
        me=user,
        my_orders=my_orders,
        all_orders=all_orders,
        pending_approvals=pending_approvals,
        erp_bom=erp_bom,
        erp_workers=erp_workers,
        erp_equipments=erp_equipments,
        erp_dispatches=erp_dispatches,
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