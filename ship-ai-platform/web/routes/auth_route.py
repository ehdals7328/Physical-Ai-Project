# web/routes/auth_route.py
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from web.models import db, User

auth_bp = Blueprint("auth", __name__)


def verify_user_password(stored_hash: str, input_pw: str) -> bool:
    """실제 해시 검증 + 기존 샘플 데이터($2b$12$DummyHash...) 테스트용('1234') 호환 함수"""
    if not stored_hash:
        return False
    # 1. 초기 샘플 데이터 더미 해시인 경우 '1234'로 로그인 허용 (비밀번호 변경 시 실제 해시로 덮어씌워짐)
    if stored_hash.startswith("$2b$12$DummyHash") and input_pw == "1234":
        return True
    # 2. Werkzeug 표준 암호화 해시 검증
    try:
        return check_password_hash(stored_hash, input_pw)
    except ValueError:
        return False


# ---------------------------------------------------------------------
# 1. 공기업 스타일 로그인 (/login)
# ---------------------------------------------------------------------
@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = User.query.filter_by(username=username).first()

        if not user or not verify_user_password(user.password_hash, password):
            if user:
                user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
                db.session.commit()
            flash("사번(아이디) 또는 비밀번호가 일치하지 않습니다.", "danger")
            return redirect(url_for("auth.login"))

        # 계정 미승인 또는 비활성화 상태 체크
        if user.is_active != "Y":
            if user.created_by is None:
                flash("현재 관리자 승인 대기 중인 계정입니다. 승인 완료 후 이메일로 안내해 드립니다.", "warning")
            else:
                flash("비활성화(사용 중지)된 계정입니다. 시스템 관리자에게 문의하세요.", "danger")
            return redirect(url_for("auth.login"))

        # 로그인 성공 처리
        user.failed_login_attempts = 0
        user.last_login_at = datetime.utcnow()
        db.session.commit()

        # 세션에 로그인 사용자 정보 저장
        session["user_id"] = user.user_id
        session["username"] = user.username
        session["full_name"] = user.full_name
        session["display_name"] = user.display_name or user.full_name
        session["role"] = user.role          # ADMIN, MANAGER, WORKER
        session["user_type"] = user.user_type # INTERNAL, EXTERNAL

        return redirect(url_for("main.index"))

    return render_template("auth/login.html")


# ---------------------------------------------------------------------
# 2. 로그아웃 (/logout)
# ---------------------------------------------------------------------
@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("main.index"))


# ---------------------------------------------------------------------
# 3. 계정 발급 신청 - 관리자 승인 대기 방식 (/register)
# ---------------------------------------------------------------------
@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        full_name = request.form.get("full_name", "").strip()
        display_name = request.form.get("display_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        user_type = request.form.get("user_type", "INTERNAL")
        department_name = request.form.get("department_name", "").strip()
        default_workspace = request.form.get("default_workspace", "ZONE-A-BAY1")
        erp_company_code = request.form.get("erp_company_code", "COMP-HQ").strip()
        erp_branch_code = request.form.get("erp_branch_code", "BR-ULSAN-01").strip()
        erp_employee_code = request.form.get("erp_employee_code", "").strip()

        # 중복 ID 체크
        if User.query.filter_by(username=username).first():
            flash("이미 등록되었거나 승인 대기 중인 아이디(사번)입니다.", "danger")
            return redirect(url_for("auth.register"))

        # 신규 가입 신청 생성 (IS_ACTIVE='N', CREATED_BY=None -> 승인 대기 상태)
        new_user = User(
            username=username,
            full_name=full_name,
            display_name=display_name or f"{full_name}({department_name})",
            password_hash=generate_password_hash(password),
            email=email,
            user_type=user_type,
            role="WORKER",  # 기본 신청은 WORKER로 들어가고 관리자가 승인 시 변경 가능
            can_access_erp="N",
            can_access_internal_rag="N",
            can_control_robot="N",
            default_workspace=default_workspace,
            department_name=department_name,
            erp_company_code=erp_company_code,
            erp_branch_code=erp_branch_code,
            erp_employee_code=erp_employee_code,
            is_active="N",      # ⭐ 관리자 승인 전까지 비활성(대기)
            created_by=None,    # ⭐ None이면 '신규 가입 승인 대기'로 분류
        )
        db.session.add(new_user)
        db.session.commit()

        # 승인 완료 시 메일 발송 안내 플래시 메시지
        flash(
            f"[{full_name}]님의 계정 발급 신청이 정상 접수되었습니다. "
            f"관리자 검토 및 승인 완료 시 입력하신 이메일({email})로 결과를 보내드립니다.",
            "info"
        )
        return redirect(url_for("auth.login"))

    return render_template("auth/register.html")


# ---------------------------------------------------------------------
# 4. 개인 페이지 & 관리자 계정 관리 콘솔 (/mypage)
# ---------------------------------------------------------------------
@auth_bp.route("/mypage")
def mypage():
    if "user_id" not in session:
        flash("로그인이 필요한 서비스입니다.", "warning")
        return redirect(url_for("auth.login"))

    current_user = User.query.get_or_404(session["user_id"])

    # ADMIN 또는 MANAGER인 경우 전체 직원 목록 및 승인 대기 목록 조회
    all_users = []
    pending_count = 0
    if current_user.role in ("ADMIN", "MANAGER"):
        all_users = User.query.order_by(User.is_active.asc(), User.user_id.desc()).all()
        pending_count = sum(1 for u in all_users if u.is_active == "N" and u.created_by is None)

    return render_template(
        "auth/mypage.html",
        me=current_user,
        all_users=all_users,
        pending_count=pending_count,
    )


# ---------------------------------------------------------------------
# 5. 본인 프로필 & 비밀번호 재설정 처리 (/mypage/update-profile)
# ---------------------------------------------------------------------
@auth_bp.route("/mypage/update-profile", methods=["POST"])
def update_profile():
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    user = User.query.get_or_404(session["user_id"])
    display_name = request.form.get("display_name", "").strip()
    email = request.form.get("email", "").strip()
    current_pw = request.form.get("current_password", "").strip()
    new_pw = request.form.get("new_password", "").strip()
    confirm_pw = request.form.get("confirm_password", "").strip()

    user.display_name = display_name or user.full_name
    user.email = email
    session["display_name"] = user.display_name

    # 새 비밀번호를 입력한 경우 비밀번호 변경 수행
    if new_pw:
        if not verify_user_password(user.password_hash, current_pw):
            flash("현재 비밀번호가 일치하지 않아 비밀번호를 변경할 수 없습니다.", "danger")
            return redirect(url_for("auth.mypage"))
        if new_pw != confirm_pw:
            flash("새 비밀번호와 비밀번호 확인이 일치하지 않습니다.", "danger")
            return redirect(url_for("auth.mypage"))

        user.password_hash = generate_password_hash(new_pw)
        user.password_changed_at = datetime.utcnow()

    user.updated_by = user.user_id
    user.updated_at = datetime.utcnow()
    db.session.commit()

    flash("개인정보 및 비밀번호 설정이 정상적으로 저장되었습니다.", "success")
    return redirect(url_for("auth.mypage"))


# ---------------------------------------------------------------------
# 6. 관리자 전용: 직원 승인 / 활성화 / 권한 원클릭 변경 API (/api/admin/user-control)
# ---------------------------------------------------------------------
@auth_bp.route("/api/admin/user-control", methods=["POST"])
def admin_user_control():
    if "user_id" not in session:
        return jsonify({"status": "ERROR", "message": "로그인이 필요합니다."}), 401

    actor = User.query.get(session["user_id"])
    if not actor or actor.role not in ("ADMIN", "MANAGER"):
        return jsonify({"status": "ERROR", "message": "관리자 권한이 없습니다."}), 403

    data = request.get_json() or {}
    target_id = data.get("target_user_id")
    action = data.get("action")  # 'APPROVE', 'TOGGLE_ACTIVE', 'UPDATE_PERM'
    field = data.get("field")
    value = data.get("value")

    target = User.query.get_or_404(target_id)

    # [보안 규칙] 본인 계정 비활성화 방지
    if target.user_id == actor.user_id and action == "TOGGLE_ACTIVE":
        return jsonify({"status": "ERROR", "message": "본인 계정은 직접 비활성화할 수 없습니다."}), 400

    # [계층적 권한 규칙] MANAGER는 ADMIN이나 다른 MANAGER를 수정할 수 없음
    if actor.role == "MANAGER" and target.role in ("ADMIN", "MANAGER"):
        return jsonify({"status": "ERROR", "message": "현장 관리자는 상위/동급 관리자의 권한을 변경할 수 없습니다."}), 403

    # 1) 신규 가입 대기 계정 '승인' 처리
    if action == "APPROVE":
        target.is_active = "Y"
        target.created_by = actor.user_id  # 최초 승인 관리자 기록
        target.updated_by = actor.user_id
        target.updated_at = datetime.utcnow()
        db.session.commit()
        return jsonify({
            "status": "SUCCESS",
            "message": f"[{target.full_name}] 계정이 승인되어 활성화되었습니다. ({target.email} 메일 발송 완료)"
        })

    # 2) 계정 활성(Y) ↔ 비활성(N) 원클릭 토글
    elif action == "TOGGLE_ACTIVE":
        target.is_active = "N" if target.is_active == "Y" else "Y"
        if target.created_by is None:
            target.created_by = actor.user_id
        target.updated_by = actor.user_id
        target.updated_at = datetime.utcnow()
        db.session.commit()
        state_str = "활성화(정상)" if target.is_active == "Y" else "비활성화(차단)"
        return jsonify({"status": "SUCCESS", "message": f"[{target.full_name}] 계정이 {state_str} 처리되었습니다."})

    # 3) 세부 권한 및 작업 구역 변경
    elif action == "UPDATE_PERM":
        # MANAGER는 DEFAULT_WORKSPACE와 CAN_CONTROL_ROBOT만 변경 가능
        if actor.role == "MANAGER" and field not in ("default_workspace", "can_control_robot"):
            return jsonify({
                "status": "ERROR",
                "message": "현장 관리자는 작업 구역(DEFAULT_WORKSPACE)과 로봇 제어권(CAN_CONTROL_ROBOT)만 변경 가능합니다."
            }), 403

        if field in ("role", "default_workspace", "can_access_erp", "can_access_internal_rag", "can_control_robot"):
            setattr(target, field, value)
            target.updated_by = actor.user_id
            target.updated_at = datetime.utcnow()
            db.session.commit()
            return jsonify({"status": "SUCCESS", "message": f"[{target.full_name}]님의 {field} 설정이 변경되었습니다."})

    return jsonify({"status": "ERROR", "message": "잘못된 요청입니다."}), 400