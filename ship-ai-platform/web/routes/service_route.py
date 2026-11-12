# web/routes/service_route.py
import sys
import json
import time
import uuid
import traceback
from pathlib import Path
from functools import wraps
from dotenv import load_dotenv

from flask import Blueprint, render_template, request, session, redirect, url_for, flash, jsonify
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
        return f(*args, **kwargs)
    return decorated_function


# =====================================================================
# 1. AI 통합 어시스턴트 (/service/assistant) - 대화방 목록 & 스마트 빌더 연동
# =====================================================================
def _get_user_session_list(user_id: int, hidden_list: list = None):
    """
    DB에 저장된 대화 세션 목록을 추출하되,
    사용자가 목록에서 지운 세션(ROLE_CODE='SYSTEM', CONTENT='HIDDEN_BY_USER')은 영구 제외!
    (원본 USER/ASSISTANT 대화 데이터는 DB에 100% 그대로 보존됨)
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

    # 1) DB에 영구 숨김 마커(HIDDEN_BY_USER)가 찍힌 세션 ID 수집
    db_hidden_sids = {
        r.session_id for r in rows
        if r.role_code == "SYSTEM" and r.content == "HIDDEN_BY_USER"
    }
    if hidden_list:
        db_hidden_sids.update(hidden_list)

    # 2) 숨기지 않은 세션만 목록으로 구성
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
        if log.retrieval_chunks_json:
            try:
                chunks = json.loads(log.retrieval_chunks_json)
            except Exception:
                chunks = []
        formatted_logs.append({
            "role_code": log.role_code,
            "content": log.content,
            "sources": chunks,
            "created_at": log.created_at.strftime("%H:%M") if log.created_at else "",
        })

    session_list = _get_user_session_list(user.user_id, hidden_sessions)

    # ⭐ 하드코딩 제거: Oracle SHIP_ERP.ERP_BRANCHES 테이블에서 활성 야드 목록을 직접 조회!
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
        branches=active_branches,  # DB 야드 목록 전달
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
    """
    ⭐ X 버튼 클릭 시 대화방 영구 숨김 처리
    (CK_ACH_CHUNKS_JSON 제약조건 통과를 위해 retrieval_chunks_json='[]' 명시)
    """
    data = request.get_json(silent=True) or {}
    target_sid = str(data.get("session_id") or "").strip()
    user = User.query.get_or_404(session["user_id"])

    # 1) 브라우저 세션에도 즉시 등록
    hidden = list(session.get("hidden_assistant_sessions", []))
    if target_sid and target_sid not in hidden:
        hidden.append(target_sid)
        session["hidden_assistant_sessions"] = hidden
        session.modified = True

    # 2) Oracle DB에도 SYSTEM 마커 기록 (재접속/서버 재시작 후에도 영구 숨김 유지)
    if target_sid:
        try:
            exists = AgentChatHistory.query.filter_by(
                session_id=target_sid,
                user_id=user.user_id,
                role_code="SYSTEM",
                content="HIDDEN_BY_USER",
            ).first()
            if not exists:
                hide_marker = AgentChatHistory(
                    session_id=target_sid,
                    user_id=user.user_id,
                    role_code="SYSTEM",
                    content="HIDDEN_BY_USER",
                    retrieval_chunks_json="[]",  # JSON 제약조건 안전 통과
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


@service_bp.route("/api/assistant/ask", methods=["POST"])
@login_required
def ask_assistant_agent():
    """사용자 질문을 받아 modules/agent/supervisor.py의 ship_agent_app을 실행"""
    start_ts = time.time()
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"status": "ERROR", "message": "질문이나 작업지시 내용을 입력해 주세요."}), 400

        user = User.query.get_or_404(session["user_id"])
        thread_id = session.get("assistant_thread_id") or f"ast_{user.username}_{uuid.uuid4().hex[:6]}"
        session["assistant_thread_id"] = thread_id

        # 1) 사용자 질문 DB 기록
        user_msg = AgentChatHistory(
            session_id=thread_id,
            user_id=user.user_id,
            role_code="USER",
            content=question,
            retrieval_chunks_json=None,
        )
        db.session.add(user_msg)
        db.session.commit()

        # 2) LangGraph Supervisor Agent 호출
        from modules.agent.supervisor import ship_agent_app

        state_input = {
            "messages": [HumanMessage(content=question)],
            "user_query": question,
            "user_id": str(user.user_id),
            "user_role": user.role,
            "user_type": user.user_type,
            "can_access_erp": user.can_access_erp or "N",
            "can_access_internal_rag": user.can_access_internal_rag or "N",
            "can_control_robot": user.can_control_robot or "N",
            "default_workspace": user.default_workspace or "ZONE-A-BAY1",
            "branch_code": user.erp_branch_code or "BR-ULSAN-01",
        }
        config = {"configurable": {"thread_id": thread_id}}

        result = ship_agent_app.invoke(state_input, config=config)
        elapsed_ms = round((time.time() - start_ts) * 1000, 2)

        answer_text = result.get("final_answer") or "응답을 생성하지 못했습니다."
        task_type = result.get("task_type") or "knowledge_qa"
        retrieved_docs = result.get("retrieved_docs") or []
        generated_doc = result.get("generated_document")

        # 3) 인용된 RAG 문서 출처 정리
        sources = []
        for d in retrieved_docs:
            meta = d.get("metadata") or {}
            sources.append({
                "doc_code": meta.get("document_code") or "DOC-RAG",
                "title": meta.get("document_name") or "기술규격 문서",
                "page": meta.get("page") or 1,
                "collection_type": meta.get("collection_type") or "PUBLIC",
            })

        # 4) AI 답변 DB 저장
        ai_msg = AgentChatHistory(
            session_id=thread_id,
            user_id=user.user_id,
            role_code="ASSISTANT",
            content=answer_text,
            retrieval_chunks_json=json.dumps(sources, ensure_ascii=False) if sources else None,
        )
        db.session.add(ai_msg)

        # 5) AGENT_EXECUTIONS 로그 저장
        intent_map = {
            "doc_generation": "DOC_GENERATION",
            "knowledge_qa": "WPS_LOOKUP",
            "erp_resource_qa": "MCP_TOOL",
            "hr_worker_qa": "MCP_TOOL",
            "hr_safety_qa": "MCP_TOOL",
            "scm_partner_qa": "MCP_TOOL",
        }
        mapped_intent = intent_map.get(task_type, "QA")

        exec_log = AgentExecution(
            session_id=thread_id,
            user_id=user.user_id,
            work_id=None,
            intent_type=mapped_intent,
            current_state=f"COMPLETED_{task_type.upper()}",
            selected_tools_json=json.dumps({"task_type": task_type, "rag_count": len(sources)}, ensure_ascii=False),
            execution_time_ms=elapsed_ms,
            status_code="SUCCESS",
            error_message=None,
        )
        db.session.add(exec_log)
        db.session.commit()

        return jsonify({
            "status": "SUCCESS",
            "answer": answer_text,
            "task_type": task_type,
            "execution_time_ms": elapsed_ms,
            "sources": sources,
            "generated_document": generated_doc,
        })

    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({
            "status": "ERROR",
            "message": f"에이전트 처리 중 오류 발생: {type(e).__name__} - {str(e)}"
        }), 500


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
    params = doc.get("parameters") or {}
    partner_plan = doc.get("partner_erp_plan") or {}
    is_outsourced = (doc.get("execution_type") == "OUTSOURCED_B2B")

    if action == "REJECT":
        return jsonify({
            "status": "REJECTED",
            "message": "❌ [요청 반려] 관리자에 의해 작업지시서 및 자원 배정이 반려(Rollback)되었습니다."
        })

    try:
        from modules.agent.tools.erp_tools import commit_erp_work_order

        selected_partner = (partner_plan.get("selected_partner") or {}) if partner_plan else {}
        lot_no = doc.get("allocated_lot_no") or "LOT-E71T1-202610-01"
        block_id = params.get("block_id") or "BLK-E102"

        commit_res = commit_erp_work_order.invoke({
            "block_id": block_id,
            "material": params.get("material", "AH36"),
            "allocated_lot_no": lot_no,
            "worker_name": params.get("worker_name", user.full_name),
            "is_outsourced": is_outsourced,
            "partner_id": selected_partner.get("partner_id") or "COMP-HQ",
            "reserve_wire_kg": 15.0,
        })

        wo_no = commit_res.get("work_order_number") or f"WO-{uuid.uuid4().hex[:8].upper()}"

        raw_thick = str(params.get("thickness", "20")).upper().replace("T", "").replace("MM", "").strip()
        thickness_val = float(raw_thick) if raw_thick.replace(".", "", 1).isdigit() else 20.0

        new_wo = WorkOrder(
            work_no=wo_no,
            work_title=doc.get("title") or f"{block_id} 자동용접 작업지시",
            work_description=f"AI 어시스턴트 발행 ({user.full_name} 승인)",
            ship_code=params.get("ship_code") or "HN-2026-301",
            block_code=block_id,
            work_region=user.erp_branch_code or "BR-ULSAN-01",
            work_zone=params.get("work_zone") or user.default_workspace or "ZONE-A-BAY1",
            process_code=params.get("process") or "FCAW",
            material_grade=params.get("material") or "AH36",
            thickness_mm=thickness_val,
            weld_position=params.get("weld_position") or "1G",
            status_code="IN_PROGRESS",
            requested_by=user.user_id,
        )
        db.session.add(new_wo)
        db.session.flush()

        new_inst = WorkInstruction(
            work_id=new_wo.work_id,
            revision_no="Rev.1",
            instruction_text=doc.get("instruction_text") or doc.get("content") or "작업지침 본문",
            welding_params_json=json.dumps(doc.get("welding_params_json") or params, ensure_ascii=False),
            source_refs_json=json.dumps(doc.get("source_refs_json") or [], ensure_ascii=False),
            external_env_json=json.dumps(doc.get("external_env_json") or {}, ensure_ascii=False),
            approval_status="APPROVED",
            approved_by=user.user_id,
        )
        db.session.add(new_inst)
        db.session.commit()

        return jsonify({
            "status": "APPROVED",
            "message": f"✅ [최종 승인 완료] 작업번호 [{wo_no}]로 ERP 자원 배정 및 지침서 등록이 확정되었습니다!"
        })
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"status": "ERROR", "message": f"승인 커밋 오류: {str(e)}"}), 500


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
from datetime import datetime
from web.models import (
    RobotExecution,
    ErpBlockMaster,
    ErpBomInventory,
    ErpWorkerCert,
    ErpVehiclesEquipment,
    MesWorkOrderDispatch,
)

@service_bp.route("/history", methods=["GET"])
@login_required
def history_page():
    """작업이력 및 ERP 자원관리 대시보드 (무거운 로봇 제어는 Vision 페이지로 이관 완료!)"""
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
        my_orders.append({"wo": wo, "inst": inst})

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
        })

    erp_bom, erp_workers, erp_equipments, erp_dispatches = [], [], [], []
    if user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER"):
        erp_bom = ErpBomInventory.query.order_by(ErpBomInventory.item_code.asc()).all()
        erp_workers = ErpWorkerCert.query.order_by(ErpWorkerCert.worker_id.asc()).all()
        erp_equipments = ErpVehiclesEquipment.query.order_by(ErpVehiclesEquipment.asset_code.asc()).all()
        erp_dispatches = MesWorkOrderDispatch.query.order_by(MesWorkOrderDispatch.created_at.desc()).all()

    return render_template(
        "service/history.html",
        me=user,
        my_orders=my_orders,
        all_orders=all_orders,
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
    command = data.get("command")  # 'ESTOP', 'RESUME_IDLE', 'START_WELD', 'PAUSE'

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
    """작업오더 진행 상태(READY -> IN_PROGRESS -> COMPLETED) 변경"""
    data = request.get_json(silent=True) or {}
    work_id = data.get("work_id")
    next_status = data.get("status_code")

    wo = WorkOrder.query.get_or_404(work_id)
    wo.status_code = next_status
    wo.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({"status": "SUCCESS", "message": f"작업 [{wo.work_no}] 상태가 {next_status}로 변경되었습니다."})