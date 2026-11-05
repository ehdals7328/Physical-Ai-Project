import os
from datetime import timedelta
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

class Config:
    """플랫폼 환경 설정."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'ship-welding-platform-secret-2026')

    # [DB 1] ship_user
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'oracle+oracledb://ship_user:ship1234@localhost:1521/?service_name=XEPDB1'
    )

    # [DB 2] ship_erp
    SQLALCHEMY_BINDS = {
        'erp_db': os.getenv(
            'ERP_DATABASE_URL',
            'oracle+oracledb://ship_erp:erp1234@localhost:1521/?service_name=XEPDB1'
        )
    }

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
        "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "10")),
        "pool_timeout": 30,
        "pool_recycle": 1800,
        "pool_pre_ping": True,
    }

    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploads')
    INTERNAL_PDF_FOLDER = os.path.join(BASE_DIR, 'data', 'internal_pdfs')
    CHROMA_DB_PATH = os.path.join(BASE_DIR, 'data', 'chromadb')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=120)
