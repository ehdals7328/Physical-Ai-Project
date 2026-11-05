"""프로젝트 환경 변수와 Flask/Oracle/ChromaDB 설정을 관리합니다."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
CHROMA_PATH_ENV = Path(os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "data" / "chromadb"))).expanduser()
CHROMA_PATH = CHROMA_PATH_ENV if CHROMA_PATH_ENV.is_absolute() else (BASE_DIR / CHROMA_PATH_ENV).resolve()


class Config:
    """애플리케이션 전역 설정을 제공합니다."""

    SECRET_KEY = os.getenv("SECRET_KEY", "local-development-only-change-me")

    # 로컬 개발 편의를 위한 기본 연결 문자열입니다. 실제 비밀번호는 .env에 둡니다.
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "oracle+oracledb://ship_user:YOUR_PASSWORD@localhost:1521/?service_name=XEPDB1",
    )
    SQLALCHEMY_BINDS = {
        "erp_db": os.getenv(
            "ERP_DATABASE_URL",
            "oracle+oracledb://ship_erp:YOUR_PASSWORD@localhost:1521/?service_name=XEPDB1",
        )
    }
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_size": int(os.getenv("DB_POOL_SIZE", "5")),
        "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "5")),
        "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "20")),
        "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "1800")),
        "pool_pre_ping": True,
    }

    UPLOAD_FOLDER = str(BASE_DIR / "data" / "uploads")
    INTERNAL_PDF_FOLDER = str(BASE_DIR / "data" / "internal_pdfs")
    CHROMA_DB_PATH = str(CHROMA_PATH)
    RAG_CACHE_FOLDER = str(BASE_DIR / "data" / "rag_cache")

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "").strip() or None
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    RAG_RERANK_ENABLED = os.getenv("RAG_RERANK_ENABLED", "false").lower() == "true"
    RAG_RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
    RAG_CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "8"))
    RAG_DEFAULT_TOP_K = int(os.getenv("RAG_DEFAULT_TOP_K", "5"))
    INTERNAL_UPLOAD_SEMANTIC = os.getenv("INTERNAL_UPLOAD_SEMANTIC", "false").lower() == "true"

    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024
    ALLOWED_UPLOAD_EXTENSIONS = {".pdf"}

    PERMANENT_SESSION_LIFETIME = timedelta(hours=12)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"
    JSON_AS_ASCII = False

    TEMPLATES_AUTO_RELOAD = os.getenv("FLASK_ENV", "development") == "development"
    SEND_FILE_MAX_AGE_DEFAULT = 60 * 60 * 24 * 7
