import os
from flask import Flask, session
from config import Config
from app.models.schema import db

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CHROMA_DB_PATH'], exist_ok=True)

    db.init_app(app)

    @app.before_request
    def make_session_permanent():
        session.permanent = True

    from app.routes.api_routes import api_bp
    app.register_blueprint(api_bp)

    return app
