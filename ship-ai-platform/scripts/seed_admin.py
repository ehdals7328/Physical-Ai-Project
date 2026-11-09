"""초기 시스템 관리자 계정을 생성한다. 자격정보는 .env에서 받는다."""
import os
import sys
from pathlib import Path
from werkzeug.security import generate_password_hash
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import create_app
from app.extensions import db
from app.models.ship_user import AppUser

def main():
    login_id = os.getenv("INITIAL_ADMIN_USERNAME", "admin").strip()
    password = os.getenv("INITIAL_ADMIN_PASSWORD", "")
    if not password or len(password) < 12:
        raise SystemExit("INITIAL_ADMIN_PASSWORD를 .env에 12자 이상으로 설정하세요.")
    app = create_app()
    with app.app_context():
        existing = AppUser.query.filter_by(login_id=login_id).first()
        if existing:
            print(f"관리자 계정이 이미 존재합니다: {login_id}")
            return
        user = AppUser(login_id=login_id, password_hash=generate_password_hash(password), user_name="시스템 관리자",
                       role_code="SYS_ADMIN", can_access_erp="Y", can_access_internal_rag="Y", is_active="Y")
        db.session.add(user)
        db.session.commit()
        print(f"시스템 관리자 생성 완료: {login_id}")

if __name__ == "__main__":
    main()
