import os
from datetime import timedelta
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))   # 절대 경로 위치 지정
load_dotenv(os.path.join(BASE_DIR, '.env'))

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'ship-welding-secret-key-2026')
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'oracle+oracledb://ship_user:ship1234@localhost:1521/?service_name=XEPDB1'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # [개선] OracleXE Connection Pool 튜닝 옵션
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),        # 기본 연결 유지 수
        "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "10")),   # 순간 폭증 시 허용 커넥션
        "pool_timeout": 30,                                        # 반환 대기 타임아웃(초)
        "pool_recycle": 1800,                                      # 30분마다 만료 세션 재생성
        "pool_pre_ping": True,                                     # 사용 전 커넥션 유효성 ping 검사
    }

    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploads')
    CHROMA_DB_PATH = os.path.join(BASE_DIR, 'data', 'chromadb')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=30)