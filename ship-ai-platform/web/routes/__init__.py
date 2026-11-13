# web/routes/__init__.py
from .main_route import main_bp
from .auth_route import auth_bp
from .service_route import service_bp
from .vision_route import vision_bp

__all__ = ["main_bp", "auth_bp", "service_bp", "vision_bp"]