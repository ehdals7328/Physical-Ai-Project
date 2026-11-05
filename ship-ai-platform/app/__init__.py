"""Flask 애플리케이션 팩토리와 확장 초기화를 담당합니다."""

from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, session

from config import Config
from app.models.schema import db


def create_app(config_object: type[Config] = Config) -> Flask:
    """환경 설정, 데이터베이스 확장, API 라우트와 오류 처리를 초기화합니다."""
    app = Flask(__name__, static_folder="../static", static_url_path="/static")
    app.config.from_object(config_object)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["INTERNAL_PDF_FOLDER"], exist_ok=True)
    os.makedirs(app.config["CHROMA_DB_PATH"], exist_ok=True)
    os.makedirs(app.config["RAG_CACHE_FOLDER"], exist_ok=True)

    db.init_app(app)

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if app.config["SECRET_KEY"] in {
        "local-development-only-change-me",
        "replace-with-a-long-random-secret",
    }:
        app.logger.warning("예제용 SECRET_KEY를 사용 중입니다. 팀 공유/배포 전 .env에서 변경하세요.")

    @app.before_request
    def persist_user_session() -> None:
        """인증된 사용자 세션의 만료 시간을 설정합니다."""
        if session.get("user_id"):
            session.permanent = True

    @app.errorhandler(413)
    def request_too_large(_error):
        """업로드 크기 제한을 초과한 요청을 JSON 오류로 반환합니다."""
        limit_mb = max(1, int(app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)))
        return jsonify({"status": "error", "message": f"업로드 파일은 최대 {limit_mb}MB까지 허용됩니다."}), 413

    @app.errorhandler(500)
    def internal_server_error(_error):
        """예상하지 못한 서버 오류를 일관된 JSON 형식으로 반환합니다."""
        app.logger.exception("처리되지 않은 서버 오류")
        return jsonify({"status": "error", "message": "서버 내부 오류가 발생했습니다. 서버 로그를 확인하세요."}), 500

    from app.routes.api_routes import api_bp

    app.register_blueprint(api_bp)
    return app
