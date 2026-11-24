# web/routes/main_route.py
from flask import Blueprint, render_template
from web.models import ErpBranch

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    primary_vision_branch = ""
    try:
        br = (
            ErpBranch.query
            .filter_by(status_code="ACTIVE")
            .order_by(ErpBranch.branch_code.asc())
            .first()
        )
        if br:
            primary_vision_branch = br.branch_code
    except Exception:
        pass

    return render_template(
        "main/main.html",
        primary_vision_branch=primary_vision_branch,
    )
