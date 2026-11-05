"""화면과 서비스 계층 사이의 HTTP API를 정의합니다."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request, session
from sqlalchemy import func, or_, text
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename

from app.models.schema import (
    db,
    User,
    WorkOrder,
    VisionAnalysis,
    VisionDetection,
    AgentExecution,
    WorkInstruction,
    DocumentMetadata,
    VisionStreamLog,
    HumanApprovalLog,
    ErpCompany,
    ErpBranch,
    ErpWorker,
    ErpWeldingRobot,
    ErpBranchInventory,
    ErpEquipmentMaintenance,
    ErpMaterialTransaction,
)
from app.services.document_service import (
    DuplicateDocumentError,
    delete_internal_document,
    ingest_internal_pdf,
    list_internal_documents,
)
from modules.knowledge.retriever import get_knowledge_store

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)


def login_required(view_function):
    """세션에 로그인한 사용자만 API를 실행하도록 제한합니다."""
    @wraps(view_function)
    def wrapped(*args, **kwargs):
        """요청 세션을 검증한 뒤 원본 API 뷰 함수를 실행합니다."""
        if not session.get("user_id"):
            return jsonify({"status": "unauthorized", "message": "로그인이 필요합니다."}), 401
        return view_function(*args, **kwargs)
    return wrapped


def admin_required(view_function):
    """ADMIN 역할을 가진 사용자만 관리자 API를 실행하도록 제한합니다."""
    @wraps(view_function)
    def wrapped(*args, **kwargs):
        """관리자 역할을 검증한 뒤 원본 API 뷰 함수를 실행합니다."""
        if session.get("role") != "ADMIN":
            return jsonify({"status": "forbidden", "message": "관리자 권한이 필요합니다."}), 403
        return view_function(*args, **kwargs)
    return wrapped


def _count_models(models: dict[str, type]) -> tuple[dict[str, int | None], bool]:
    """모델별 행 수를 조회하고 데이터베이스 연결 실패를 별도로 기록합니다."""
    counts: dict[str, int | None] = {}
    healthy = True
    for label, model in models.items():
        try:
            counts[label] = int(db.session.query(func.count()).select_from(model).scalar() or 0)
        except Exception:
            logger.exception("테이블 카운트 실패: %s", label)
            db.session.rollback()
            counts[label] = None
            healthy = False
    return counts, healthy


def _serialize_work_order(order: WorkOrder) -> dict:
    """작업 요청 ORM 객체를 API 응답용 딕셔너리로 변환합니다."""
    return {
        "work_id": order.work_id,
        "work_name": order.work_name,
        "work_area": order.work_area,
        "block_no": order.block_no,
        "process_type": order.process_type,
        "base_material": order.base_material,
        "material_thickness": order.material_thickness,
        "joint_type": order.joint_type,
        "groove_type": order.groove_type,
        "welding_position": order.welding_position,
        "work_status": order.work_status,
        "created_at": order.created_at.isoformat() if order.created_at else None,
    }


@api_bp.get("/")
def index():
    """개발·통합 테스트용 메인 콘솔 화면을 렌더링합니다."""
    return render_template("index.html")


@api_bp.get("/login")
def login_page():
    """로그인 화면을 렌더링합니다."""
    return render_template("auth.html")


@api_bp.post("/api/auth/login")
def auth_login():
    """아이디와 비밀번호 해시를 검증하고 사용자 세션을 생성합니다."""
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    if not username or not password:
        return jsonify({"status": "error", "message": "아이디와 비밀번호를 입력하세요."}), 400

    user = User.query.filter_by(username=username).first()
    if not user or user.is_active not in ("Y", "YES", "TRUE", True):
        return jsonify({"status": "error", "message": "계정 정보가 올바르지 않거나 비활성 계정입니다."}), 401
    if not check_password_hash(user.password_hash, password):
        return jsonify({"status": "error", "message": "계정 정보가 올바르지 않거나 비활성 계정입니다."}), 401

    session.clear()
    session.permanent = True
    session["user_id"] = user.user_id
    session["username"] = user.username
    session["display_name"] = user.display_name or user.username
    session["role"] = user.role or "USER"
    return jsonify(
        {
            "status": "success",
            "user": {
                "username": user.username,
                "display_name": user.display_name or user.username,
                "role": user.role or "USER",
            },
        }
    )


@api_bp.post("/api/auth/logout")
def auth_logout():
    """현재 사용자 세션을 종료합니다."""
    session.clear()
    return jsonify({"status": "success"})


@api_bp.get("/api/auth/session-check")
def auth_check():
    """현재 세션의 로그인 여부와 최소 사용자 정보를 반환합니다."""
    return jsonify(
        {
            "logged_in": bool(session.get("user_id")),
            "username": session.get("username"),
            "display_name": session.get("display_name"),
            "role": session.get("role"),
        }
    )


@api_bp.get("/api/system/status")
def system_status():
    """두 Oracle DB와 두 RAG 컬렉션의 연결 상태를 반환합니다."""
    user_models = {
        "USERS": User,
        "WORK_ORDERS": WorkOrder,
        "VISION_ANALYSES": VisionAnalysis,
        "VISION_DETECTIONS": VisionDetection,
        "AGENT_EXECUTIONS": AgentExecution,
        "WORK_INSTRUCTIONS": WorkInstruction,
        "DOCUMENT_METADATA": DocumentMetadata,
        "VISION_STREAM_LOGS": VisionStreamLog,
        "HUMAN_APPROVAL_LOGS": HumanApprovalLog,
    }
    erp_models = {
        "ERP_COMPANIES": ErpCompany,
        "ERP_BRANCHES": ErpBranch,
        "ERP_WORKERS": ErpWorker,
        "ERP_WELDING_ROBOTS": ErpWeldingRobot,
        "ERP_BRANCH_INVENTORY": ErpBranchInventory,
        "ERP_EQUIPMENT_MAINTENANCE": ErpEquipmentMaintenance,
        "ERP_MATERIAL_TRANSACTIONS": ErpMaterialTransaction,
    }
    # 기본 상태 확인은 DB별 SELECT 1만 수행해 대시보드 초기 로딩을 가볍게 유지합니다.
    try:
        with db.engine.connect() as connection:
            connection.execute(text("SELECT 1 FROM DUAL"))
        user_ok = True
    except Exception:
        logger.exception("ship_user DB 연결 확인 실패")
        db.session.rollback()
        user_ok = False

    try:
        with db.engines["erp_db"].connect() as connection:
            connection.execute(text("SELECT 1 FROM DUAL"))
        erp_ok = True
    except Exception:
        logger.exception("ship_erp DB 연결 확인 실패")
        db.session.rollback()
        erp_ok = False

    # 상세 행 수는 요청한 경우에만 조회합니다. 일반 화면 로딩에서는 수행하지 않습니다.
    include_counts = request.args.get("include_counts", "false").lower() == "true"
    user_counts = erp_counts = None
    if include_counts:
        user_counts, user_counts_ok = _count_models(user_models)
        erp_counts, erp_counts_ok = _count_models(erp_models)
        user_ok = user_ok and user_counts_ok
        erp_ok = erp_ok and erp_counts_ok

    try:
        rag_counts = get_knowledge_store().counts()
        rag_ok = True
    except Exception:
        logger.exception("ChromaDB 상태 조회 실패")
        rag_counts = {"public": None, "internal": None}
        rag_ok = False

    return jsonify(
        {
            "status": "success",
            "databases": {
                "ship_user": {"connected": user_ok, "table_count": 9, "counts": user_counts},
                "ship_erp": {"connected": erp_ok, "table_count": 7, "counts": erp_counts},
            },
            "rag": {"connected": rag_ok, "counts": rag_counts},
            "integrations": {"agent": "NOT_CONNECTED", "vision": "NOT_CONNECTED", "llm_generation": "NOT_CONNECTED"},
            "checked_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    )


@api_bp.get("/api/admin/internal-docs")
@login_required
@admin_required
def get_internal_docs():
    """관리자에게 활성 사내 PDF의 메타데이터 목록을 반환합니다."""
    docs = list_internal_documents()
    return jsonify(
        {
            "status": "success",
            "documents": [
                {
                    "document_id": item.document_id,
                    "document_name": item.document_name,
                    "category": item.category,
                    "source": item.source,
                    "ingest_status": item.ingest_status,
                    "created_at": item.created_at.strftime("%Y-%m-%d %H:%M") if item.created_at else "",
                }
                for item in docs
            ],
        }
    )


@api_bp.post("/api/admin/internal-docs/upload")
@login_required
@admin_required
def upload_internal_doc():
    """관리자가 업로드한 PDF를 검증하고 사내 RAG 컬렉션에 적재합니다."""
    uploaded_file = request.files.get("pdf_file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"status": "error", "message": "업로드할 PDF 파일을 선택하세요."}), 400

    submitted_name = Path(uploaded_file.filename.replace("\\", "/")).name
    if Path(submitted_name).suffix.lower() != ".pdf":
        return jsonify({"status": "error", "message": "PDF 파일만 업로드할 수 있습니다."}), 400

    original_name = secure_filename(submitted_name) or "uploaded_document.pdf"
    target_name = f"{uuid.uuid4().hex}_{original_name}"
    save_path = Path(current_app.config["INTERNAL_PDF_FOLDER"]) / target_name
    uploaded_file.save(save_path)

    try:
        with save_path.open("rb") as source:
            if source.read(5) != b"%PDF-":
                save_path.unlink(missing_ok=True)
                return jsonify({"status": "error", "message": "유효한 PDF 파일이 아닙니다."}), 400

        metadata = ingest_internal_pdf(
            save_path,
            submitted_name,
            int(session["user_id"]),
        )
        return jsonify(
            {
                "status": "success",
                "message": "사내 문서 색인이 완료되었습니다.",
                "document": {
                    "document_id": metadata.document_id,
                    "document_name": metadata.document_name,
                    "ingest_status": metadata.ingest_status,
                },
            }
        )
    except DuplicateDocumentError as exc:
        save_path.unlink(missing_ok=True)
        return jsonify({"status": "error", "message": str(exc)}), 409
    except ValueError as exc:
        save_path.unlink(missing_ok=True)
        return jsonify({"status": "error", "message": str(exc)}), 422
    except Exception as exc:
        logger.exception("사내 PDF 업로드/색인 실패")
        return jsonify({"status": "error", "message": f"문서 색인에 실패했습니다: {exc}"}), 500


@api_bp.delete("/api/admin/internal-docs/<int:document_id>")
@login_required
@admin_required
def remove_internal_doc(document_id: int):
    """관리자가 지정한 사내 문서와 연결된 벡터 청크를 삭제합니다."""
    try:
        deleted = delete_internal_document(document_id)
    except Exception:
        logger.exception("사내 문서 삭제 실패: document_id=%s", document_id)
        return jsonify({"status": "error", "message": "문서 삭제 중 오류가 발생했습니다."}), 500
    if not deleted:
        return jsonify({"status": "error", "message": "해당 사내 문서를 찾을 수 없습니다."}), 404
    return jsonify({"status": "success", "message": "문서와 검색 인덱스를 삭제했습니다."})


@api_bp.post("/api/rag/ask")
@login_required
def rag_search():
    """선택한 RAG 컬렉션에서 관련 문서 청크와 출처 정보를 검색합니다."""
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    rag_mode = str(data.get("rag_mode", "ALL")).upper()
    try:
        top_k = int(data.get("top_k", current_app.config.get("RAG_DEFAULT_TOP_K", 5)))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "top_k는 정수여야 합니다."}), 400

    if not query:
        return jsonify({"status": "error", "message": "검색어를 입력하세요."}), 400
    if len(query) > 2000:
        return jsonify({"status": "error", "message": "검색어는 2,000자 이내로 입력하세요."}), 400

    try:
        results = get_knowledge_store().search(query, top_k=top_k, rag_mode=rag_mode)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception:
        logger.exception("RAG 검색 실패")
        return jsonify({"status": "error", "message": "RAG 검색에 실패했습니다. 임베딩 모델과 ChromaDB 로그를 확인하세요."}), 503

    return jsonify(
        {
            "status": "success",
            "query": query,
            "rag_mode": rag_mode,
            "result_count": len(results),
            "references": results,
            "generation": {
                "status": "NOT_CONNECTED",
                "message": "현재 단계에서는 문서 검색까지만 수행합니다. LLM 답변 생성은 Agent 연동 단계에서 연결합니다.",
            },
        }
    )


@api_bp.get("/api/erp/branches")
@login_required
def erp_branches():
    """ERP DB에 등록된 사업장 및 야드 목록을 반환합니다."""
    branches = ErpBranch.query.order_by(ErpBranch.branch_id.asc()).all()
    return jsonify(
        {
            "status": "success",
            "branches": [{"branch_id": item.branch_id, "branch_name": item.branch_name} for item in branches],
        }
    )


@api_bp.get("/api/erp/resources")
@login_required
def erp_resources():
    """선택한 사업장의 로봇·기술자·자재 재고를 ERP DB에서 조회합니다."""
    branch_id = request.args.get("branch_id", type=int)
    branch_query = ErpBranch.query.order_by(ErpBranch.branch_id.asc())
    branch = db.session.get(ErpBranch, branch_id) if branch_id else branch_query.first()
    if branch is None:
        return jsonify({"status": "success", "branch": None, "robots": [], "workers": [], "inventory": []})

    robots = ErpWeldingRobot.query.filter_by(branch_id=branch.branch_id).order_by(ErpWeldingRobot.robot_id).all()
    workers = ErpWorker.query.filter_by(branch_id=branch.branch_id).order_by(ErpWorker.worker_id).all()
    inventory = (
        ErpBranchInventory.query.filter(
            or_(
                ErpBranchInventory.branch_id == branch.branch_id,
                (ErpBranchInventory.branch_id.is_(None))
                & (ErpBranchInventory.branch_name == branch.branch_name),
            )
        )
        .order_by(ErpBranchInventory.inventory_id)
        .all()
    )
    return jsonify(
        {
            "status": "success",
            "branch": {"branch_id": branch.branch_id, "branch_name": branch.branch_name},
            "robots": [
                {"robot_id": item.robot_id, "serial": item.robot_serial, "model": item.model_name,
                 "payload_kg": item.payload_kg, "status": item.status}
                for item in robots
            ],
            "workers": [
                {"worker_id": item.worker_id, "name": item.worker_name,
                 "skill_grade": item.skill_grade, "status": item.status}
                for item in workers
            ],
            "inventory": [
                {"inventory_id": item.inventory_id, "material_code": item.material_code,
                 "name": item.material_name, "qty": item.current_qty, "unit": item.unit,
                 "min_safety_qty": item.min_safety_qty,
                 "is_low": (item.current_qty or 0) < (item.min_safety_qty or 0)}
                for item in inventory
            ],
        }
    )


@api_bp.get("/api/work-orders")
@login_required
def get_work_orders():
    """로그인 사용자가 생성한 최근 작업 요청 이력을 반환합니다."""
    orders = (
        WorkOrder.query.filter_by(user_id=session["user_id"])
        .order_by(WorkOrder.work_id.desc())
        .limit(50)
        .all()
    )
    return jsonify({"status": "success", "work_orders": [_serialize_work_order(item) for item in orders]})


@api_bp.post("/api/work-orders")
@login_required
def create_work_order():
    """작업 입력값을 검증해 DB 1에 작업 요청을 저장합니다."""
    data = request.get_json(silent=True) or {}
    try:
        thickness = float(data.get("material_thickness"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "판 두께를 숫자로 입력하세요."}), 400
    if thickness <= 0 or thickness > 500:
        return jsonify({"status": "error", "message": "판 두께는 0 초과 500mm 이하로 입력하세요."}), 400

    work_area = str(data.get("work_area", "")).strip()
    block_no = str(data.get("block_no", "")).strip()
    base_material = str(data.get("base_material", "")).strip()
    if not all((work_area, block_no, base_material)):
        return jsonify({"status": "error", "message": "작업 지부, 블록 번호, 모재 강종은 필수입니다."}), 400

    order = WorkOrder(
        user_id=session["user_id"],
        work_name=f"{work_area} {block_no} 용접 작업",
        request_text=str(data.get("request_text", "")).strip()[:4000],
        block_no=block_no[:50],
        work_area=work_area[:100],
        process_type=str(data.get("process_type", "FCAW")).strip()[:50],
        base_material=base_material[:100],
        material_thickness=thickness,
        joint_type=str(data.get("joint_type", "")).strip()[:50] or None,
        groove_type=str(data.get("groove_type", "")).strip()[:50] or None,
        welding_position=str(data.get("welding_position", "")).strip()[:20] or None,
        additional_input=str(data.get("additional_input", "")).strip()[:4000],
        work_status="PENDING_AGENT",
    )
    db.session.add(order)
    db.session.commit()
    return jsonify(
        {
            "status": "success",
            "message": "작업 요청을 저장했습니다. Agent 연동 전이므로 작업지시서 생성은 실행하지 않았습니다.",
            "work_order": _serialize_work_order(order),
        }
    ), 201


@api_bp.get("/api/integrations/status")
def integration_status():
    """Agent·Vision·LLM 연결 상태를 표시할 현재 통합 계약 상태를 반환합니다."""
    return jsonify(
        {
            "agent": {"status": "NOT_CONNECTED", "interface": "modules.agent.contracts"},
            "vision": {"status": "NOT_CONNECTED", "interface": "modules.vision.contracts"},
            "llm": {"status": "NOT_CONNECTED", "interface": "modules.agent.contracts"},
        }
    )
