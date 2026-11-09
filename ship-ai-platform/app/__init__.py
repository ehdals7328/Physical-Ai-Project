from __future__ import annotations
import logging
import os
from pathlib import Path
from flask import Flask, jsonify, session
from config import Config
from app.extensions import db

def create_app(config_object=Config):
    app = Flask(__name__, template_folder="templates", static_folder="../static", static_url_path="/static")
    app.config.from_object(config_object)
    for key in ("UPLOAD_FOLDER", "PUBLIC_PDF_FOLDER", "INTERNAL_PDF_FOLDER", "CHROMA_DB_PATH", "RAG_CACHE_FOLDER"):
        Path(app.config[key]).mkdir(parents=True, exist_ok=True)
    db.init_app(app)
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @app.before_request
    def mark_session_permanent():
        if session.get("user_id"):
            session.permanent = True

    @app.errorhandler(413)
    def file_too_large(_error):
        return jsonify(status="error", message=f"업로드 최대 용량은 {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)}MB입니다."), 413

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.exception("Unhandled application error: %s", error)
        return jsonify(status="error", message="서버 내부 오류가 발생했습니다. 서버 로그를 확인하세요."), 500

    from app.routes.web import web_bp
    from app.routes.api import api_bp
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)
    return app
