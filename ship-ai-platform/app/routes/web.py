from flask import Blueprint, render_template, redirect, url_for, session
from app.services.access_control import current_user, can_access_erp, can_access_internal_rag, login_required

web_bp = Blueprint("web", __name__)

@web_bp.route("/")
def index():
    if current_user() is None:
        return redirect(url_for("web.login_page"))
    return render_template("dashboard.html", user=current_user())

@web_bp.route("/login")
def login_page():
    if current_user() is not None:
        return redirect(url_for("web.index"))
    return render_template("login.html")

@web_bp.route("/chat")
@login_required
def chat_page():
    user = current_user()
    return render_template("chat.html", user=user, can_internal=can_access_internal_rag(user))

@web_bp.route("/work-orders")
@login_required
def work_orders_page():
    return render_template("work_orders.html", user=current_user())

@web_bp.route("/documents")
@login_required
def documents_page():
    user = current_user()
    can_manage_documents = user.role_code == "SYS_ADMIN" or (user.role_code == "COMPANY_ADMIN" and can_access_internal_rag(user))
    return render_template("documents.html", user=user,
                           can_internal=can_access_internal_rag(user), can_erp=can_access_erp(user),
                           can_manage_documents=can_manage_documents,
                           company_code=user.erp_company_code,
                           can_manage_public_documents=user.role_code == "SYS_ADMIN")

@web_bp.route("/erp")
@login_required
def erp_page():
    user = current_user()
    if not can_access_erp(user):
        return redirect(url_for("web.index"))
    return render_template("erp.html", user=user)

@web_bp.route("/system")
@login_required
def system_page():
    return render_template("system.html", user=current_user())

@web_bp.route("/users")
@login_required
def users_page():
    user = current_user()
    if user.role_code != "SYS_ADMIN":
        return redirect(url_for("web.index"))
    return render_template("users.html", user=user)
