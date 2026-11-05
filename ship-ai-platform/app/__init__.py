# [개선 후] app/__init__.py
import os
from flask import Flask, session
from config import Config
from app.models.schema import db
from modules.knowledge.embedding import warmup_embedder

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CHROMA_DB_PATH'], exist_ok=True)

    db.init_app(app)

    try:
        warmup_embedder()
    except Exception as e:
        app.logger.warning(f"Embedding Warmup Failed: {e}")

    @app.before_request
    def make_session_permanent():
        session.permanent = True

    from app.routes.api_routes import api_bp
    app.register_blueprint(api_bp)

    return app