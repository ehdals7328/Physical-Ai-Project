"""기존 APP_USERS 계정의 ERP/사내 RAG 권한을 관리한다.
사용 예: python scripts/set_user_permissions.py --login operator01 --erp Y --internal-rag N --company PARTNER_01
"""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import create_app
from app.extensions import db
from app.models.ship_user import AppUser

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--login", required=True)
    parser.add_argument("--erp", choices=["Y", "N"], required=True)
    parser.add_argument("--internal-rag", choices=["Y", "N"], required=True)
    parser.add_argument("--company", default=None)
    parser.add_argument("--role", default=None)
    args = parser.parse_args()
    app = create_app()
    with app.app_context():
        user = AppUser.query.filter_by(login_id=args.login).first()
        if not user:
            raise SystemExit(f"사용자를 찾을 수 없습니다: {args.login}")
        user.can_access_erp = args.erp
        user.can_access_internal_rag = args.internal_rag
        if args.company is not None:
            user.erp_company_code = args.company
        if args.role:
            user.role_code = args.role
        db.session.commit()
        print(f"updated {user.login_id}: role={user.role_code}, company={user.erp_company_code}, ERP={user.can_access_erp}, INTERNAL_RAG={user.can_access_internal_rag}")

if __name__ == "__main__":
    main()
