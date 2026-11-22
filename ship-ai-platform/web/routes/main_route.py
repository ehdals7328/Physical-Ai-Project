# web/routes/main_route.py
from flask import Blueprint, render_template
from web.models import ErpBlockMaster, ErpBranch

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    sample_hints = []
    primary_vision_branch = ""
    try:
        blocks = (
            ErpBlockMaster.query
            .order_by(ErpBlockMaster.ship_code.asc(), ErpBlockMaster.block_id.asc())
            .limit(4)
            .all()
        )
        seen = []
        for b in blocks:
            if b.ship_code and b.ship_code not in seen:
                seen.append(b.ship_code)
            if b.block_id and b.block_id not in seen:
                seen.append(b.block_id)
        sample_hints = seen[:4]

        br = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).first()
        if br:
            primary_vision_branch = br.branch_code
    except Exception:
        pass

    return render_template(
        "main/main.html",
        sample_hints=sample_hints,
        primary_vision_branch=primary_vision_branch,
    )
