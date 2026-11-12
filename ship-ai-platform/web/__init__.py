# web/__init__.py
from flask import Flask
from config import Config
from web.models import db
from web.routes import main_bp, auth_bp, service_bp, vision_bp

def create_web():
    web = Flask(__name__)
    web.config.from_object(Config)

    db.init_app(web)

    web.register_blueprint(main_bp)
    web.register_blueprint(auth_bp)
    web.register_blueprint(service_bp)
    web.register_blueprint(vision_bp)

    return web