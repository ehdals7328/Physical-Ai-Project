"""
[파일 역할] Flask 서버 전역 설정, 다중 Oracle DB(작업관리 DB + ERP DB) 바인딩,
           커넥션 풀(Connection Pool) 최적화, 다중 ChromaDB 컬렉션 설정을 담당합니다.
"""
import os
from datetime import timedelta
from dotenv import load_dotenv

# 프로젝트 루트 절대 경로 지정 및 .env 로드
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))


class Config:
    """통합 플랫폼 핵심 환경 설정 클래스"""

    # 세션 암호화 키
    SECRET_KEY = os.getenv('SECRET_KEY', 'ship-welding-secret-key-2026')

    # =========================================================================
    # 1. 다중 RDB 연결 설정 (Main 공정 DB + ERP 자원관리 DB)
    # =========================================================================
    # [기본 DB] 작업지시, Vision 분석, Agent 실행 이력, 문서 메타데이터 관리 (ship_user)
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'oracle+oracledb://ship_user:ship1234@localhost:1521/?service_name=XEPDB1'
    )

    # [다중 바인딩 DB] 인사(용접기술자), 협력업체, 자동화 용접로봇, 지부별 자재재고 ERP 관리 (ship_erp)
    SQLALCHEMY_BINDS = {
        'erp_db': os.getenv(
            'ERP_DATABASE_URL',
            'oracle+oracledb://ship_erp:erp1234@localhost:1521/?service_name=XEPDB1'
        )
    }

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # [DB 병목 및 ORA-12516 방지] 멀티스레드(Waitress 8스레드) 환경을 위한 커넥션 풀 설정
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_size": 5,           # 기본 유지 커넥션 수
        "max_overflow": 10,       # 폭주 시 최대 추가 생성 가능 커넥션 수
        "pool_timeout": 30,       # 커넥션 대기 타임아웃(초)
        "pool_pre_ping": True,    # 쿼리 실행 전 세션 살아있는지 핑 체크 (끊김 방지)
    }

    # =========================================================================
    # 2. 파일 업로드 및 다중 RAG (ChromaDB) 경로 설정
    # =========================================================================
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploads')
    CHROMA_DB_PATH = os.path.join(BASE_DIR, 'data', 'chromadb')

    # [다중 RAG 컬렉션 분리]
    # 1) 공통 기술/안전/선급법규 컬렉션 (원청 및 협력사 전체 접근 가능)
    CHROMA_GLOBAL_COLLECTION = os.getenv('CHROMA_COLLECTION', 'shipbuilding_docs')
    # 2) 회사별 사내 내규/로봇 정비/계약 보안 컬렉션 (소속 회사 및 직급별 권한 필터링)
    CHROMA_INTERNAL_COLLECTION = os.getenv('CHROMA_INTERNAL_COLLECTION', 'company_internal_docs')

    # 임베딩 모델 설정 (BAAI/bge-m3)
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')

    # 세션 유지 시간 (30분 비활동 시 자동 로그아웃)
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=30)