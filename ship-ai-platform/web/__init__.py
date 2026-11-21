# web/__init__.py
from flask import Flask
from config import Config
from web.models import db, ErpBranch
from web.routes import main_bp, auth_bp, service_bp, vision_bp


def _resolve_video_stage(default_video: str | None) -> tuple[str, str]:
    raw = (default_video or "").strip().lower()
    if raw in ("ocr_test.mp4", "video2.mp4") or "ocr" in raw:
        return "TRANSPORT", "운반 중 · 블록 번호 확인"
    return "WORKING", "작업 중 · CCTV / 로봇 시점"


def create_web():
    web = Flask(__name__)
    web.config.from_object(Config)

    db.init_app(web)

    web.register_blueprint(main_bp)
    web.register_blueprint(auth_bp)
    web.register_blueprint(service_bp)
    web.register_blueprint(vision_bp)

    @web.context_processor
    def inject_global_nav():
        nav_branches = []
        try:
            rows = (
                ErpBranch.query
                .filter_by(status_code="ACTIVE")
                .order_by(ErpBranch.branch_code.asc())
                .all()
            )
            for b in rows:
                stage, stage_desc = _resolve_video_stage(b.default_video)
                nav_branches.append({
                    "branch_code": b.branch_code,
                    "branch_name": b.branch_name,
                    "region_name": b.region_name,
                    "stage": stage,
                    "stage_desc": stage_desc,
                })
        except Exception:
            nav_branches = []
        return {"nav_branches": nav_branches}

    return web
