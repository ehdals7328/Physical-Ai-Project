"""
scripts/init_ship_databases.py
작업/공정/비전 관리 DB (ship_user) 테이블 자동 생성 및 초기 관리자 계정 시드 스크립트.
집이나 로컬 환경에서 명령어 한 줄로 스키마를 초기화할 수 있습니다.
"""
import os
import sys
from pathlib import Path
from werkzeug.security import generate_password_hash

# 프로젝트 루트 경로 등록
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app
from app.models.schema import (
    db, User, WorkOrder, VisionAnalysis, VisionDetection,
    AgentExecution, WorkInstruction, DocumentMetadata,
    VisionStreamLog, HumanApprovalLog
)

def init_ship_user_db():
    print("=" * 70)
    print("🚀 [ship_user DB] 테이블 초기화 및 관리자 시드 데이터 생성 시작")
    print("=" * 70)

    app = create_app()
    with app.app_context():
        # 1. ship_user 바인딩 테이블 생성
        print("1. Oracle 테이블 자동 생성 (create_all)...")
        db.create_all()
        print("   -> USERS, WORK_ORDERS, VISION_ANALYSES, VISION_DETECTIONS,")
        print("      AGENT_EXECUTIONS, WORK_INSTRUCTIONS, DOCUMENT_METADATA,")
        print("      VISION_STREAM_LOGS, HUMAN_APPROVAL_LOGS 생성 완료.")

        # 2. 초기 테스트/관리자 계정 시드
        print("\n2. 기본 관리자 및 현장 작업자 계정 점검...")
        default_users = [
            ("admin", "admin1234", "총괄 품질관리자", "admin@shipyard.com", "ADMIN"),
            ("welder_kim", "ship1234", "김용접 기장", "kim@shipyard.com", "USER"),
            ("robot_op", "ship1234", "이로봇 오퍼레이터", "lee@shipyard.com", "USER"),
        ]

        added_cnt = 0
        for username, pwd, display_name, email, role in default_users:
            if not User.query.filter_by(username=username).first():
                user = User(
                    username=username,
                    password_hash=generate_password_hash(pwd),
                    display_name=display_name,
                    email=email,
                    role=role,
                    is_active=True
                )
                db.session.add(user)
                added_cnt += 1

        db.session.commit()
        print(f"   -> 기본 사용자 {added_cnt}명 추가 완료.")
        print(f"   -> 현재 등록된 USERS 수: {User.query.count()}명")
        print("=" * 70)
        print("✅ [ship_user DB] 초기화 성공!")
        print("=" * 70)

if __name__ == "__main__":
    init_ship_user_db()