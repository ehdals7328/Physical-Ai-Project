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
    VisionStreamLog, HumanApprovalLog,
    ErpCompany, ErpBranch, ErpWorker, ErpWeldingRobot,
    ErpBranchInventory, ErpEquipmentMaintenance, ErpMaterialTransaction,
)

def init_ship_databases():
    """DB 1·DB 2의 테이블을 생성하고 초기 관리자·사용자 계정을 등록합니다."""
    print("=" * 70)
    print("🚀 [ship_user / ship_erp DB] 테이블 초기화 및 관리자 시드 데이터 생성 시작")
    print("=" * 70)

    app = create_app()
    with app.app_context():
        # 1. ship_user 바인딩 테이블 생성
        print("1. Oracle 테이블 자동 생성 (create_all)...")
        db.create_all()
        print("   -> USERS, WORK_ORDERS, VISION_ANALYSES, VISION_DETECTIONS,")
        print("      AGENT_EXECUTIONS, WORK_INSTRUCTIONS, DOCUMENT_METADATA,")
        print("      VISION_STREAM_LOGS, HUMAN_APPROVAL_LOGS 생성 완료.")
        print("   -> [ship_erp DB] ERP_COMPANIES, ERP_BRANCHES, ERP_WORKERS,")
        print("      ERP_WELDING_ROBOTS, ERP_BRANCH_INVENTORY,")
        print("      ERP_EQUIPMENT_MAINTENANCE, ERP_MATERIAL_TRANSACTIONS 생성 완료.")

        # 2. 초기 테스트/관리자 계정 시드
        print("\n2. 기본 관리자 및 현장 작업자 계정 점검...")
        default_users = [
            (os.getenv("INITIAL_ADMIN_USERNAME", "admin"), os.getenv("INITIAL_ADMIN_PASSWORD", "admin1234"), "총괄 품질관리자", "admin@shipyard.local", "ADMIN"),
            (os.getenv("INITIAL_USER_USERNAME", "welder_operator"), os.getenv("INITIAL_USER_PASSWORD", "welder1234"), "현장 테스트 사용자", "operator@shipyard.local", "USER"),
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
                    is_active="Y"
                )
                db.session.add(user)
                added_cnt += 1

        db.session.commit()
        print(f"   -> 기본 사용자 {added_cnt}명 추가 완료.")
        print(f"   -> 현재 등록된 USERS 수: {User.query.count()}명")
        print("=" * 70)
        print("✅ [ship_user / ship_erp DB] 테이블 초기화 성공!")
        print("=" * 70)

if __name__ == "__main__":
    init_ship_databases()