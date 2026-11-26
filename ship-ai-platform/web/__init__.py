# web/__init__.py
import os
from flask import Flask
from config import Config
from web.models import db, ErpBranch, DocumentMetadata
from web.routes import main_bp, auth_bp, service_bp, vision_bp


def _preload_db_info(web: Flask) -> None:
    """웹 기동 시 1회 DB 연결 확인 및 기본 정보 조회 (임베딩/파일 동기화 X)"""
    if web.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        return

    with web.app_context():
        try:
            branch_count = ErpBranch.query.filter_by(status_code="ACTIVE").count()
            doc_count = DocumentMetadata.query.filter_by(is_active="Y").count()
            print(f"[Init] DB 초기 로드 완료 (활성 지점 {branch_count}건 / 활성 문서 {doc_count}건)")
        except Exception as e:
            db.session.rollback()
            print(f"[Init] DB 초기 조회 경고: {e}")


def create_web():
    web = Flask(__name__)
    web.config.from_object(Config)

    db.init_app(web)

    web.register_blueprint(main_bp)
    web.register_blueprint(auth_bp)
    web.register_blueprint(service_bp)
    web.register_blueprint(vision_bp)

    _preload_db_info(web)

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