import os
from datetime import timedelta
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))   # 절대 경로 위치 지정
load_dotenv(os.path.join(BASE_DIR, '.env'))

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'ship-welding-secret-key-2026')
    # 1) 기본 DB (SHIP_USER: 웹 관제, 지침서, 비전, 감사로그)
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'SHIP_USER_DATABASE_URL',
    )
    
    # 2) 전사 자원 바인드 DB (SHIP_ERP: 블록 마스터, 자재 BOM, 작업자 자격, 배차)
    SQLALCHEMY_BINDS = {
      'ship_erp': os.getenv('SHIP_ERP_DATABASE_URL'),  # 본사 ERP (Oracle)
      'partner_pg': os.getenv(
          'PARTNER_PG_DATABASE_URL'
      ),  # 외부 협력사 (PostgreSQL)
    }
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