# web/__init__.py
from flask import Flask
from config import Config
from web.models import db, ErpBranch
from web.routes import main_bp, auth_bp, service_bp, vision_bp


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
                nav_branches.append({
                    "branch_code": b.branch_code,
                    "branch_name": b.branch_name,
                    "region_name": b.region_name,
                })
        except Exception:
            nav_branches = []
        return {"nav_branches": nav_branches}

    return web
