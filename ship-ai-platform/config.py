
"""
Ship AI Platform - Environment-based Configuration

Oracle 18c XE / Flask / Dual Database / RAG
비밀값과 DB 접속정보를 소스코드에 하드코딩하지 않는다.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv


# ==========================================================
# 1. BASE DIRECTORY & ENVIRONMENT
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

# 프로젝트 루트의 .env 로드
load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    """환경변수 문자열 반환."""
    return os.getenv(name, default).strip()


def _db_url(name: str, default: str = "") -> str:
    """DB 접속 URL 반환."""
    value = _env(name, default)

    if not value:
        raise RuntimeError(
            f"필수 데이터베이스 환경변수 '{name}'이 설정되지 않았습니다."
        )

    return value


def _bool(name: str, default: str = "0") -> bool:
    """환경변수 Boolean 변환."""
    return _env(name, default).lower() in ("1", "true", "yes", "on")


def _int(name: str, default: str) -> int:
    return int(_env(name, default))


def _float(name: str, default: str) -> float:
    return float(_env(name, default))


# ==========================================================
# 2. FLASK CONFIGURATION
# ==========================================================

class Config:

    # ------------------------------------------------------
    # Flask Security
    # ------------------------------------------------------

    SECRET_KEY = _env("SECRET_KEY")

    if not SECRET_KEY:
        raise RuntimeError(
            ".env에 SECRET_KEY를 설정해야 합니다. "
            "Python secrets.token_hex(32)로 생성하세요."
        )

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = _bool("COOKIE_SECURE", "0")

    PERMANENT_SESSION_LIFETIME = timedelta(
        minutes=_int("SESSION_MINUTES", "120")
    )

    # ------------------------------------------------------
    # Upload
    # ------------------------------------------------------

    MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", "100")
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024

    UPLOAD_FOLDER = str(BASE_DIR / "data" / "uploads")
    PUBLIC_PDF_FOLDER = str(BASE_DIR / "data" / "raw_pdfs")
    INTERNAL_PDF_FOLDER = str(BASE_DIR / "data" / "internal_pdfs")

    ALLOWED_UPLOAD_EXTENSIONS = {".pdf"}

    # ------------------------------------------------------
    # Main Database - Ship Process DB
    # ------------------------------------------------------

    SQLALCHEMY_DATABASE_URI = _db_url(
        "SHIP_USER_DATABASE_URL"
    )

    # ------------------------------------------------------
    # ERP Database - Enterprise Resource DB
    # ------------------------------------------------------

    SQLALCHEMY_BINDS = {
        "ship_erp": _db_url(
            "SHIP_ERP_DATABASE_URL"
        )
    }

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ------------------------------------------------------
    # Database Connection Pool
    # ------------------------------------------------------

    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
        "pool_recycle": _int("DB_POOL_RECYCLE", "1800"),
        "pool_size": _int("DB_POOL_SIZE", "5"),
        "max_overflow": _int("DB_MAX_OVERFLOW", "5"),
    }

    # ------------------------------------------------------
    # ChromaDB
    # ------------------------------------------------------

    CHROMA_DB_PATH = str(
        BASE_DIR / _env("CHROMA_DB_PATH", "data/chromadb")
    )

    PUBLIC_COLLECTION = _env(
        "PUBLIC_RAG_COLLECTION",
        "ship_public_knowledge"
    )

    INTERNAL_COLLECTION = _env(
        "INTERNAL_RAG_COLLECTION",
        "ship_internal_knowledge"
    )

    # ------------------------------------------------------
    # RAG Retrieval
    # ------------------------------------------------------

    RAG_TOP_K = _int("RAG_TOP_K", "5")

    RAG_MIN_SIMILARITY = _float(
        "RAG_MIN_SIMILARITY", "0.35"
    )

    RAG_SEMANTIC = _bool("RAG_SEMANTIC", "1")
    RAG_OCR = _bool("RAG_OCR", "0")

    # ------------------------------------------------------
    # Embedding Model
    # ------------------------------------------------------

    EMBEDDING_MODEL = _env(
        "EMBEDDING_MODEL",
        "BAAI/bge-m3"
    )

    EMBEDDING_DEVICE = _env("EMBEDDING_DEVICE", "")

    # ------------------------------------------------------
    # Reranker Model
    # ------------------------------------------------------

    RERANKER_MODEL = _env(
        "RERANKER_MODEL",
        "BAAI/bge-reranker-base"
    )

    RERANKER_ENABLED = _bool(
        "RERANKER_ENABLED", "true"
    )

    # ------------------------------------------------------
    # PDF / OCR
    # ------------------------------------------------------

    TESSERACT_CMD = _env("TESSERACT_CMD")

    RAG_OCR_LANG = _env(
        "RAG_OCR_LANG",
        "kor+eng"
    )

    RAG_CACHE_FOLDER = str(
        BASE_DIR / "data" / "rag_cache"
    )

    # ------------------------------------------------------
    # OpenAI API
    # ------------------------------------------------------

    OPENAI_API_KEY = _env("OPENAI_API_KEY")

    CHAT_MODEL = _env(
        "CHAT_MODEL",
        "gpt-4o-mini"
    )

    INTENT_MODEL = _env(
        "INTENT_MODEL",
        CHAT_MODEL
    )

    DEFAULT_GENERAL_REPLY = (
        "현재 대화형 AI 모델이 설정되지 않았습니다. "
        ".env에 OPENAI_API_KEY를 설정하면 "
        "일반 대화와 기술 질의 답변을 사용할 수 있습니다."
    )

    # ------------------------------------------------------
    # Initial Accounts
    # ------------------------------------------------------

    INITIAL_ADMIN_USERNAME = _env(
        "INITIAL_ADMIN_USERNAME", "admin"
    )

    INITIAL_ADMIN_PASSWORD = _env(
        "INITIAL_ADMIN_PASSWORD"
    )

    INITIAL_USER_USERNAME = _env(
        "INITIAL_USER_USERNAME", "welder"
    )

    INITIAL_USER_PASSWORD = _env(
        "INITIAL_USER_PASSWORD"
    )

    # ------------------------------------------------------
    # Flask Development
    # ------------------------------------------------------

    FLASK_ENV = _env("FLASK_ENV", "development")
    DEBUG = _bool("FLASK_DEBUG", "true")
