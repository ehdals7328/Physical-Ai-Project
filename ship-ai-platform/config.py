import os
from datetime import timedelta
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))   # 절대 경로 위치 지정
load_dotenv(os.path.join(BASE_DIR, '.env'))

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'ship-welding-secret-key-2026')
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'oracle+oracledb://ship_user:ship1234@localhost:1521/?service_name=XEPDB1'   # [수정 필요] 기본설정 강의실 컴
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False                          # SQLAlchemy 객체의 변경 사항을 Flask 수준의 이벤트 시스템으로 실시간 추적할지
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploads')       # pdf 위치
    CHROMA_DB_PATH = os.path.join(BASE_DIR, 'data', 'chromadb')     # chromadb 위치
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')   # 임베딩 모델 BAAI/bge-m3 사용

    # 세션 유지 시간 (30분 비활동 시 자동 로그아웃)
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=30)
