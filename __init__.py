from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

import config

db = SQLAlchemy()
migrate = Migrate()


def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    # ORM 초기화
    db.init_app(app)
    migrate.init_app(app, db)

    # 모델 등록
    from . import models

    # 블루프린트 등록
    from .views import answer_views, main_views, question_views, auth_views, my_agent_views, vision_views

    app.register_blueprint(main_views.bp)
    app.register_blueprint(question_views.bp)
    app.register_blueprint(answer_views.bp)
    app.register_blueprint(auth_views.bp)
    app.register_blueprint(my_agent_views.bp)
    app.register_blueprint(vision_views.bp)

    return app