from __future__ import annotations
import logging
from datetime import datetime
from flask import Blueprint, jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash
from app.extensions import db
from app.models.ship_user import AppUser, ChatConversation, RagDocument
from app.models.ship_erp import ErpBranch, ErpCompany
from app.services.access_control import (current_user, login_required, admin_required,
    can_access_erp, can_access_internal_rag, is_sys_admin)
from app.services.chat_service import answer_chat
from app.services.document_service import ingest_pdf, deactivate_document, DuplicateDocumentError
from app.services.erp_service import list_branches, branch_resources
from app.services.work_service import create_work_order, list_work_orders

api_bp = Blueprint("api", __name__, url_prefix="/api")
logger = logging.getLogger(__name__)

def _json_error(message: str, code: int = 400):
    return jsonify(status="error", message=message), code


def _branch_in_user_scope(user, branch_code: str) -> bool:
    return bool(is_sys_admin(user) or not user.erp_branch_code or user.erp_branch_code.upper() == branch_code.upper())

@api_bp.post("/auth/login")
def auth_login():
    payload = request.get_json(silent=True) or {}
    login_id = str(payload.get("login_id", "")).strip()
    password = str(payload.get("password", ""))
    if not login_id or not password:
        return _json_error("아이디와 비밀번호를 입력해 주세요.")
    user = AppUser.query.filter_by(login_id=login_id, is_active="Y").first()
    if not user or not check_password_hash(user.password_hash, password):
        return _json_error("아이디 또는 비밀번호가 올바르지 않습니다.", 401)
    session.clear()
    session.permanent = True
    session["user_id"] = int(user.user_id)
    user.last_login_at = datetime.utcnow()
    db.session.commit()
    return jsonify(status="success", user={"login_id": user.login_id, "user_name": user.user_name,
        "role_code": user.role_code, "can_access_erp": can_access_erp(user),
        "can_access_internal_rag": can_access_internal_rag(user)})

@api_bp.post("/auth/logout")
@login_required
def auth_logout():
    session.clear()
    return jsonify(status="success")

@api_bp.get("/auth/me")
@login_required
def auth_me():
    user = current_user()
    return jsonify(user={"user_id": user.user_id, "login_id": user.login_id, "user_name": user.user_name,
        "role_code": user.role_code, "erp_company_code": user.erp_company_code,
        "can_access_erp": can_access_erp(user), "can_access_internal_rag": can_access_internal_rag(user)})

@api_bp.post("/chat")
@login_required
def chat():
    payload = request.get_json(silent=True) or {}
    try:
        result = answer_chat(user=current_user(), query=payload.get("query", ""),
                             conversation_id=payload.get("conversation_id"),
                             requested_mode=payload.get("mode", "AUTO"))
        return jsonify(status="success", **result)
    except PermissionError as exc:
        return _json_error(str(exc), 403)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception:
        db.session.rollback()
        logger.exception("Chat request failed")
        return _json_error("질의 처리 중 오류가 발생했습니다. 서버 로그를 확인해 주세요.", 500)

@api_bp.get("/chat/conversations")
@login_required
def chat_conversations():
    user = current_user()
    rows = ChatConversation.query.filter_by(user_id=user.user_id).order_by(ChatConversation.conversation_id.desc()).limit(30).all()
    return jsonify(conversations=[{"conversation_id": r.conversation_id, "title": r.title,
        "started_at": r.started_at.strftime("%Y-%m-%d %H:%M") if r.started_at else ""} for r in rows])

@api_bp.get("/work-orders")
@login_required
def work_order_list():
    return jsonify(work_orders=list_work_orders(current_user()))

@api_bp.post("/work-orders")
@login_required
def work_order_create():
    payload = request.get_json(silent=True) or {}
    try:
        order = create_work_order(current_user(), payload)
        return jsonify(status="success", message="작업 요청을 등록했습니다.", work_id=order.work_id,
                       work_no=order.work_no), 201
    except PermissionError as exc:
        db.session.rollback()
        return _json_error(str(exc), 403)
    except ValueError as exc:
        db.session.rollback()
        return _json_error(str(exc))
    except Exception:
        db.session.rollback()
        logger.exception("Work order create failed")
        return _json_error("작업 요청 등록에 실패했습니다.", 500)

@api_bp.get("/admin/users")
@login_required
@admin_required
def admin_user_list():
    user = current_user()
    if not is_sys_admin(user):
        return _json_error("시스템 관리자 권한이 필요합니다.", 403)
    rows = AppUser.query.order_by(AppUser.user_id.asc()).limit(500).all()
    return jsonify(users=[{
        "user_id": row.user_id,
        "login_id": row.login_id,
        "user_name": row.user_name,
        "role_code": row.role_code,
        "erp_company_code": row.erp_company_code or "",
        "erp_branch_code": row.erp_branch_code or "",
        "erp_employee_code": row.erp_employee_code or "",
        "can_access_erp": row.can_access_erp,
        "can_access_internal_rag": row.can_access_internal_rag,
        "is_active": row.is_active,
    } for row in rows])


@api_bp.post("/admin/users")
@login_required
@admin_required
def admin_user_create():
    if not is_sys_admin(current_user()):
        return _json_error("시스템 관리자 권한이 필요합니다.", 403)
    payload = request.get_json(silent=True) or {}
    login_id = str(payload.get("login_id", "")).strip()
    user_name = str(payload.get("user_name", "")).strip()
    password = str(payload.get("password", ""))
    role_code = str(payload.get("role_code", "VIEWER")).strip().upper()
    allowed_roles = {"SYS_ADMIN", "COMPANY_ADMIN", "MANAGER", "WELD_OPERATOR", "INSPECTOR", "VIEWER"}
    if not login_id or not user_name or len(password) < 12:
        return _json_error("로그인 ID, 사용자명과 12자 이상의 초기 비밀번호를 입력해 주세요.")
    if role_code not in allowed_roles:
        return _json_error("허용되지 않은 역할 코드입니다.")
    if AppUser.query.filter_by(login_id=login_id).first():
        return _json_error("이미 사용 중인 로그인 ID입니다.", 409)
    row = AppUser(
        login_id=login_id,
        password_hash=generate_password_hash(password),
        user_name=user_name,
        role_code=role_code,
        erp_company_code=(str(payload.get("erp_company_code", "")).strip().upper() or None),
        erp_branch_code=(str(payload.get("erp_branch_code", "")).strip().upper() or None),
        erp_employee_code=(str(payload.get("erp_employee_code", "")).strip().upper() or None),
        can_access_erp="Y" if payload.get("can_access_erp") else "N",
        can_access_internal_rag="Y" if payload.get("can_access_internal_rag") else "N",
        is_active="Y",
    )
    db.session.add(row)
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        logger.exception("Admin user create failed")
        return _json_error("사용자 생성에 실패했습니다. 로그인 ID 중복 여부를 확인하세요.", 409)
    return jsonify(status="success", user_id=row.user_id), 201


@api_bp.patch("/admin/users/<int:user_id>")
@login_required
@admin_required
def admin_user_update(user_id: int):
    actor = current_user()
    if not is_sys_admin(actor):
        return _json_error("시스템 관리자 권한이 필요합니다.", 403)
    row = db.session.get(AppUser, user_id)
    if row is None:
        return _json_error("사용자를 찾을 수 없습니다.", 404)
    payload = request.get_json(silent=True) or {}
    if "role_code" in payload:
        role_code = str(payload["role_code"]).strip().upper()
        if role_code not in {"SYS_ADMIN", "COMPANY_ADMIN", "MANAGER", "WELD_OPERATOR", "INSPECTOR", "VIEWER"}:
            return _json_error("허용되지 않은 역할 코드입니다.")
        if row.role_code == "SYS_ADMIN" and role_code != "SYS_ADMIN":
            active_admins = AppUser.query.filter_by(role_code="SYS_ADMIN", is_active="Y").count()
            if active_admins <= 1:
                return _json_error("마지막 시스템 관리자 계정의 권한은 낮출 수 없습니다.", 409)
        row.role_code = role_code
    for key, column in (("can_access_erp", "can_access_erp"), ("can_access_internal_rag", "can_access_internal_rag")):
        if key in payload:
            setattr(row, column, "Y" if bool(payload[key]) else "N")
    for key, column in (("erp_company_code", "erp_company_code"), ("erp_branch_code", "erp_branch_code"), ("erp_employee_code", "erp_employee_code")):
        if key in payload:
            setattr(row, column, str(payload[key]).strip().upper() or None)
    if "is_active" in payload:
        active = "Y" if bool(payload["is_active"]) else "N"
        if row.user_id == actor.user_id and active == "N":
            return _json_error("현재 로그인한 계정은 비활성화할 수 없습니다.", 409)
        if row.role_code == "SYS_ADMIN" and active == "N":
            active_admins = AppUser.query.filter_by(role_code="SYS_ADMIN", is_active="Y").count()
            if active_admins <= 1:
                return _json_error("마지막 시스템 관리자 계정은 비활성화할 수 없습니다.", 409)
        row.is_active = active
    row.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify(status="success", message="사용자 권한을 변경했습니다.")


@api_bp.get("/documents")
@login_required
def document_list():
    user = current_user()
    query = RagDocument.query.filter_by(document_status="ACTIVE")
    collection_type = request.args.get("collection_type", "ALL").upper()
    if collection_type not in {"ALL", "PUBLIC", "INTERNAL"}:
        return _json_error("collection_type 값이 올바르지 않습니다.")
    if collection_type == "INTERNAL" and not can_access_internal_rag(user):
        return _json_error("사내 RAG 접근 권한이 없습니다.", 403)
    if collection_type == "PUBLIC":
        query = query.filter_by(collection_type="PUBLIC")
    elif collection_type == "INTERNAL":
        query = query.filter_by(collection_type="INTERNAL")
    else:
        query = query.filter(RagDocument.collection_type == "PUBLIC")
        if can_access_internal_rag(user):
            internal = RagDocument.query.filter_by(collection_type="INTERNAL", document_status="ACTIVE")
            if not is_sys_admin(user):
                internal = internal.filter(RagDocument.owner_company_code.in_([user.erp_company_code, "GLOBAL"]))
            rows = query.order_by(RagDocument.document_id.desc()).limit(200).all()
            rows += internal.order_by(RagDocument.document_id.desc()).limit(200).all()
            return jsonify(documents=[_document_json(x) for x in rows])
    if collection_type == "INTERNAL" and not is_sys_admin(user):
        query = query.filter(RagDocument.owner_company_code.in_([user.erp_company_code, "GLOBAL"]))
    rows = query.order_by(RagDocument.document_id.desc()).limit(300).all()
    return jsonify(documents=[_document_json(x) for x in rows])

def _document_json(row):
    return {"document_id": row.document_id, "document_code": row.document_code,
        "document_title": row.document_title, "collection_type": row.collection_type,
        "category_code": row.category_code, "revision_no": row.revision_no,
        "source_organization": row.source_organization, "owner_company_code": row.owner_company_code,
        "ingest_status": row.ingest_status, "created_at": row.created_at.strftime("%Y-%m-%d %H:%M") if row.created_at else ""}

@api_bp.post("/admin/documents/upload")
@login_required
@admin_required
def document_upload():
    user = current_user()
    file = request.files.get("pdf_file")
    if not file or not file.filename:
        return _json_error("PDF 파일을 선택해 주세요.")
    collection_type = (request.form.get("collection_type") or "INTERNAL").upper()
    if collection_type not in {"PUBLIC", "INTERNAL"}:
        return _json_error("컬렉션은 PUBLIC 또는 INTERNAL이어야 합니다.")
    if collection_type == "INTERNAL" and not can_access_internal_rag(user):
        return _json_error("사내 RAG 등록 권한이 없습니다.", 403)
    if collection_type == "PUBLIC" and not is_sys_admin(user):
        return _json_error("공개 표준 문서 등록은 시스템 관리자만 가능합니다.", 403)
    requested_company = (request.form.get("owner_company_code") or "").strip().upper()
    if collection_type == "INTERNAL":
        if is_sys_admin(user):
            owner_company_code = requested_company or "GLOBAL"
        else:
            owner_company_code = user.erp_company_code
            if not owner_company_code:
                return _json_error("계정에 회사 코드가 연결되지 않아 사내 문서를 등록할 수 없습니다.", 403)
    else:
        owner_company_code = "PUBLIC"
    try:
        row = ingest_pdf(file, collection_type=collection_type,
            category_code=request.form.get("category_code", "GENERAL"),
            source=request.form.get("source_organization", "관리자 등록"), uploader_id=user.user_id,
            owner_company_code=owner_company_code, revision_no=request.form.get("revision_no", "UNKNOWN"))
        return jsonify(status="success", message="문서 등록 및 색인이 완료되었습니다.", document=_document_json(row)), 201
    except DuplicateDocumentError as exc:
        return _json_error(str(exc), 409)
    except ValueError as exc:
        return _json_error(str(exc))
    except Exception:
        db.session.rollback()
        logger.exception("Document upload/index failed")
        return _json_error("문서 색인에 실패했습니다. PDF 텍스트 추출·OCR 설정과 서버 로그를 확인해 주세요.", 500)

@api_bp.delete("/admin/documents/<int:document_id>")
@login_required
@admin_required
def document_delete(document_id: int):
    user = current_user()
    row = db.session.get(RagDocument, document_id)
    if not row:
        return _json_error("문서를 찾을 수 없습니다.", 404)
    if row.collection_type == "PUBLIC" and not is_sys_admin(user):
        return _json_error("공개 표준 자료 삭제는 시스템 관리자만 가능합니다.", 403)
    if row.collection_type == "INTERNAL":
        if not can_access_internal_rag(user):
            return _json_error("사내 RAG 접근 권한이 없습니다.", 403)
        if not is_sys_admin(user) and row.owner_company_code != user.erp_company_code:
            return _json_error("본인 회사가 소유한 사내 문서만 삭제할 수 있습니다.", 403)
    try:
        if deactivate_document(document_id):
            return jsonify(status="success", message="문서와 관련 벡터를 비활성화했습니다.")
        return _json_error("삭제할 문서를 찾을 수 없습니다.", 404)
    except Exception:
        db.session.rollback()
        logger.exception("Document delete failed")
        return _json_error("문서 삭제 처리에 실패했습니다.", 500)

@api_bp.get("/erp/branches")
@login_required
def erp_branches():
    user = current_user()
    if not can_access_erp(user):
        return _json_error("ERP 자원 DB 접근 권한이 없습니다.", 403)
    rows = list_branches(user.erp_company_code, is_sys_admin(user))
    if not is_sys_admin(user) and user.erp_branch_code:
        rows = [row for row in rows if row.branch_code.upper() == user.erp_branch_code.upper()]
    branch_rows = []
    for branch in rows:
        company = db.session.get(ErpCompany, branch.company_id)
        branch_rows.append({"branch_code": branch.branch_code, "branch_name": branch.branch_name,
                            "company_name": company.company_name if company else ""})
    return jsonify(branches=branch_rows)

@api_bp.get("/erp/resources")
@login_required
def erp_resources():
    user = current_user()
    if not can_access_erp(user):
        return _json_error("ERP 자원 DB 접근 권한이 없습니다.", 403)
    branch_code = (request.args.get("branch_code") or "").strip()
    if not branch_code:
        rows = list_branches(user.erp_company_code, is_sys_admin(user))
        if not is_sys_admin(user) and user.erp_branch_code:
            rows = [row for row in rows if row.branch_code.upper() == user.erp_branch_code.upper()]
        if not rows:
            return jsonify(resource=None, message="접근 가능한 사업장이 없습니다.")
        branch_code = rows[0].branch_code
    if not _branch_in_user_scope(user, branch_code):
        return _json_error("계정에 지정된 사업장 외의 ERP 자원에는 접근할 수 없습니다.", 403)
    result = branch_resources(branch_code, user.erp_company_code, is_sys_admin(user))
    if result is None:
        return _json_error("사업장을 찾을 수 없거나 접근 권한이 없습니다.", 404)
    return jsonify(resource=result)

@api_bp.post("/erp/availability-check")
@login_required
def erp_availability_check():
    user = current_user()
    if not can_access_erp(user):
        return _json_error("ERP 자원 DB 접근 권한이 없습니다.", 403)
    payload = request.get_json(silent=True) or {}
    branch_code = (payload.get("branch_code") or "").strip()
    if not branch_code:
        return _json_error("사업장을 선택해 주세요.")
    if not _branch_in_user_scope(user, branch_code):
        return _json_error("계정에 지정된 사업장 외의 ERP 자원에는 접근할 수 없습니다.", 403)
    snapshot = branch_resources(branch_code, user.erp_company_code, is_sys_admin(user))
    if snapshot is None:
        return _json_error("사업장을 찾을 수 없거나 접근 권한이 없습니다.", 404)
    try:
        requested_workers = max(0, int(payload.get("required_workers") or 0))
        requested_equipment = max(0, int(payload.get("required_equipment") or 0))
        requested_quantity = max(0.0, float(payload.get("required_quantity") or 0))
        required_thickness = float(payload["required_thickness_mm"]) if payload.get("required_thickness_mm") not in (None, "") else None
    except (TypeError, ValueError):
        return _json_error("인력·장비 수량과 자재/두께는 숫자로 입력해 주세요.")
    required_role_code = (payload.get("required_role_code") or "").strip().upper()
    required_process_code = (payload.get("required_process_code") or "").strip().upper()
    required_position_code = (payload.get("required_position_code") or "").strip().upper()
    required_certification = (payload.get("required_certification") or "").strip().upper()
    required_material_group = (payload.get("required_material_group") or "").strip().upper()
    requested_equipment_type = (payload.get("required_equipment_type") or "").strip().upper()
    material_code = (payload.get("material_code") or "").strip().upper()
    matching_materials = [x for x in snapshot["inventory"] if x["material_code"].upper() == material_code and x["quality_status"] == "APPROVED" and x["warehouse_status"] == "ACTIVE"] if material_code else []
    available_quantity = sum(x["available_quantity"] for x in matching_materials)

    def certification_matches(cert):
        if required_process_code and cert.get("welding_process_code", "").upper() != required_process_code:
            return False
        if required_certification and cert.get("qualification_type", "").upper() != required_certification:
            return False
        if required_material_group and cert.get("material_group", "").upper() not in {required_material_group, "ALL"}:
            return False
        if required_position_code:
            positions = {part.strip() for part in cert.get("position_range", "").upper().replace("/", ",").replace(";", ",").split(",")}
            if required_position_code not in positions and "ALL" not in positions:
                return False
        if required_thickness is not None:
            low = cert.get("thickness_min_mm")
            high = cert.get("thickness_max_mm")
            if low is not None and required_thickness < low:
                return False
            if high is not None and required_thickness > high:
                return False
        return True

    matching_workers = [x for x in snapshot["workers"] if x["is_available"] and
                        (not required_role_code or x["role"].upper() == required_role_code) and
                        (not (required_process_code or required_position_code or required_thickness is not None or required_certification or required_material_group) or
                         any(certification_matches(cert) for cert in x["certifications"]))]
    matching_equipment = [x for x in snapshot["equipment"] if x["is_available"] and
                          (not requested_equipment_type or x["type_code"].upper() == requested_equipment_type)]
    checks = {
        "workers": {"role_code": required_role_code or "ALL", "required": requested_workers, "available": len(matching_workers),
                    "sufficient": len(matching_workers) >= requested_workers},
        "equipment": {"type_code": required_equipment_type or "ALL", "required": requested_equipment, "available": len(matching_equipment),
                      "sufficient": len(matching_equipment) >= requested_equipment},
    }
    if material_code:
        checks["material"] = {"material_code": material_code, "required": requested_quantity,
                               "available": available_quantity, "sufficient": available_quantity >= requested_quantity,
                               "matched_lots": len(matching_materials)}
    if required_process_code or required_position_code or required_thickness is not None or required_certification or required_material_group:
        checks["worker_qualification"] = {
            "process_code": required_process_code or "ALL",
            "position_code": required_position_code or "ALL",
            "required_thickness_mm": required_thickness,
            "certification": required_certification or "ALL",
            "material_group": required_material_group or "ALL",
            "required": requested_workers,
            "available": len(matching_workers),
            "sufficient": len(matching_workers) >= requested_workers,
        }
    return jsonify(status="success", branch_name=snapshot["branch_name"], checks=checks,
                   overall_sufficient=all(v["sufficient"] for v in checks.values()))

@api_bp.get("/system/health")
@login_required
def system_health():
    checks = {}
    user = current_user()
    try:
        with db.engines[None].connect() as connection:
            connection.exec_driver_sql("SELECT 1 FROM DUAL")
        checks["ship_user"] = "OK"
    except Exception as exc:
        checks["ship_user"] = f"ERROR: {type(exc).__name__}"
    if can_access_erp(user):
        try:
            with db.engines["ship_erp"].connect() as connection:
                connection.exec_driver_sql("SELECT 1 FROM DUAL")
            checks["ship_erp"] = "OK"
        except Exception as exc:
            checks["ship_erp"] = f"ERROR: {type(exc).__name__}"
    else:
        checks["ship_erp"] = "NO_PERMISSION"
    try:
        from modules.knowledge.retriever import get_knowledge_store
        store = get_knowledge_store()
        counts = {"standards": int(store.public.count())}
        # 내부 컬렉션은 권한이 있는 계정에게만 count 자체를 노출한다.
        counts["internal"] = int(store.internal.count()) if can_access_internal_rag(user) else None
        checks["vector_db"] = "OK"
    except Exception as exc:
        counts = {}
        checks["vector_db"] = f"ERROR: {type(exc).__name__}"
    from modules.agent.adapter import AgentAdapter
    from modules.vision.adapter import VisionAdapter
    checks["agent"] = AgentAdapter.health()["status"]
    checks["vision"] = VisionAdapter.health()["status"]
    return jsonify(checks=checks, rag_counts=counts)
