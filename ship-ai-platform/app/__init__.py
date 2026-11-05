"""
[파일 역할] Flask 애플리케이션 팩토리 함수입니다.
           다중 DB(ship_user + ship_erp)를 초기화하고 서버 기동 시 BGE-M3 임베딩 모델을 사전 예열합니다.
"""
import os
from flask import Flask, session
from config import Config
from app.models.schema import db


def create_app():
    """Flask 앱 객체를 생성하고 DB, 폴더, 블루프린트, 임베딩 모델 예열을 수행하는 팩토리 함수입니다."""
    app = Flask(__name__)
    app.config.from_object(Config)

    # 업로드 및 ChromaDB 저장 디렉터리 자동 생성
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CHROMA_DB_PATH'], exist_ok=True)

    # SQLAlchemy 다중 DB(기본 + erp_db 바인딩) 초기화
    db.init_app(app)

    @app.before_request
    def make_session_permanent():
        """요청 시마다 세션 만료 시간을 갱신하는 훅 함수입니다."""
        session.permanent = True

    # 라우트 블루프린트 등록
    from app.routes.api_routes import api_bp
    app.register_blueprint(api_bp)

    # 서버 시작 시 임베딩 모델 사전 로드 (Cold Start 지연 제거)
    if os.getenv("SKIP_EMBEDDER_WARMUP", "0") != "1":
        from modules.knowledge.embedding import warmup_embedder
        warmup_embedder()

    return app