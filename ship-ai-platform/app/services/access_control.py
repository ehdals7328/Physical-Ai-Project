from functools import wraps
from flask import jsonify, redirect, request, session, url_for
from app.models.ship_user import AppUser

def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    user = db_get_user(user_id)
    if user is None or user.is_active != "Y":
        session.clear()
        return None
    return user

def db_get_user(user_id):
    # local import avoids import cycles
    from app.extensions import db
    return db.session.get(AppUser, int(user_id))

def is_sys_admin(user):
    return bool(user and user.role_code == "SYS_ADMIN")

def can_access_internal_rag(user):
    return bool(user and (is_sys_admin(user) or user.can_access_internal_rag == "Y"))

def can_access_erp(user):
    return bool(user and (is_sys_admin(user) or user.can_access_erp == "Y"))

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            if request.path.startswith("/api/"):
                return jsonify(status="unauthorized", message="로그인이 필요합니다."), 401
            return redirect(url_for("web.login_page", next=request.path))
        return view(*args, **kwargs)
    return wrapped

def internal_rag_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if not can_access_internal_rag(current_user()):
            return jsonify(status="forbidden", message="사내 RAG 접근 권한이 없습니다."), 403
        return view(*args, **kwargs)
    return wrapped

def erp_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if not can_access_erp(current_user()):
            if request.path.startswith("/api/"):
                return jsonify(status="forbidden", message="ERP 자원 DB 접근 권한이 없습니다."), 403
            return redirect(url_for("web.index"))
        return view(*args, **kwargs)
    return wrapped

def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        user = current_user()
        if not is_sys_admin(user) and user.role_code != "COMPANY_ADMIN":
            return jsonify(status="forbidden", message="관리자 권한이 필요합니다."), 403
        return view(*args, **kwargs)
    return wrapped
