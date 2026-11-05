"""
[파일 역할] 웹 프론트엔드와 다중 DB(`ship_user`, `ship_erp`), 다중 RAG, Vision 모듈, AI-Agent를 하나로 연결하는
           핵심 백엔드 컨트롤러입니다.
  - 사용자 직책/소속 회사별 접근 제어(RBAC) 및 컨텍스트 자동 주입
  - 지부별 로봇·기술자·자재 재고 실시간 ERP 조회 및 배정/차감 API
  - Vision 검출 객체 Bulk Insert (시계열/다건 병목 방지) 및 AI-Agent 자동 연동
"""
import os
import json
import uuid
from datetime import datetime
from functools import wraps
from flask import Blueprint, request, jsonify, render_template, session, redirect, url_for, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_core.messages import HumanMessage

from app.models.schema import (
    db, User, WorkOrder, VisionAnalysis, VisionDetection,
    AgentExecution, WorkInstruction, DocumentMetadata,
    ERPCompany, ERPBranch, ERPWorker, ERPWeldingRobot, ERPBranchInventory
)
from modules.knowledge.retriever import KnowledgeStore
from modules.vision.detector import analyze_welding_frame
from modules.agent.supervisor import ship_agent_app
from modules.agent.tools.retriever_tools import (
    query_branch_erp_resources,
    reserve_erp_resources_transaction,
)

api_bp = Blueprint('api', __name__)


# =============================================================================
# [보안 및 권한 데코레이터]
# =============================================================================
def login_required_page(f):
    """비로그인 사용자가 보호된 웹 페이지에 접근할 때 로그인 화면으로 리다이렉트하는 데코레이터입니다."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('api.login_page', next=request.path))
        return f(*args, **kwargs)
    return decorated


def login_required_api(f):
    """비로그인 상태에서 API를 호출할 때 401 에러를 반환하는 데코레이터입니다."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'status': 'unauthorized', 'message': '로그인이 필요한 서비스입니다.'}), 401
        return f(*args, **kwargs)
    return decorated


def _get_current_user_context() -> dict:
    """현재 로그인 세션에서 사용자의 소속 회사(원청/협력사), 담당 지부, 직책 정보를 꺼내 반환하는 헬퍼 함수입니다."""
    return {
        "user_id": session.get("user_id"),
        "username": session.get("username", "guest"),
        "display_name": session.get("display_name", "사용자"),
        "role": session.get("role", "WORKER"),
        "company_name": session.get("company_name", "HD한국조선해양"),
        "company_type": session.get("company_type", "MAIN"),
        "branch_name": session.get("branch_name", "울산 1야드"),
        "department": session.get("department", "선체생산부"),
    }


# =============================================================================
# [페이지 라우트] (기존 템플릿 호환 유지)
# =============================================================================
@api_bp.route('/')
def home_page():
    """메인 홈 대시보드 화면을 렌더링합니다."""
    return render_template('home.html')


@api_bp.route('/login')
def login_page():
    """로그인 및 계정 생성 화면을 렌더링합니다."""
    return render_template('auth.html')


@api_bp.route('/rag-chat')
@login_required_page
def rag_chat_page():
    """다중 RAG 기술 규격 및 사내 내규 질의응답 채팅 화면을 렌더링합니다."""
    return render_template('rag_chat.html')


@api_bp.route('/qna')
@login_required_page
def qna_page():
    """QnA 게시판/이력 화면을 렌더링합니다."""
    return render_template('qna.html')


@api_bp.route('/work-analysis')
@login_required_page
def work_analysis_page():
    """Vision 이미지 분석 및 AI-Agent 작업지시서 발행 워크스페이스 화면을 렌더링합니다."""
    return render_template('work_analysis.html')


# =============================================================================
# [인증 및 회사/지부/직책 세션 관리 API]
# =============================================================================
@api_bp.route('/api/auth/register', methods=['POST'])
def register():
    """신규 직원 또는 협력사 계정을 생성하고 소속 회사·담당 지부·직책 권한을 부여하는 API입니다."""
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    email = data.get('email', '').strip()
    display_name = data.get('display_name', '').strip() or username
    role = data.get('role', 'WORKER').strip().upper()
    company_name = data.get('company_name', 'HD한국조선해양').strip()
    company_type = data.get('company_type', 'MAIN').strip().upper()
    branch_name = data.get('branch_name', '울산 1야드').strip()
    department = data.get('department', '선체생산부').strip()

    if not username or not password:
        return jsonify({'status': 'error', 'message': '아이디와 비밀번호를 모두 입력하세요.'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'status': 'error', 'message': '이미 존재하는 아이디입니다.'}), 400

    user = User(
        username=username,
        password_hash=generate_password_hash(password),
        email=email,
        display_name=display_name,
        role=role,
        company_name=company_name,
        company_type=company_type,
        branch_name=branch_name,
        department=department,
        is_active=True
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({
        'status': 'success',
        'message': f'[{company_name} / {branch_name}] {display_name}({role}) 계정 생성 완료!'
    })


@api_bp.route('/api/auth/login', methods=['POST'])
def login():
    """사용자 로그인을 검증하고 세션에 회사·지부·직책 컨텍스트를 저장하는 API입니다."""
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'status': 'error', 'message': '아이디 또는 비밀번호가 일치하지 않습니다.'}), 401

    if not user.is_active:
        return jsonify({'status': 'error', 'message': '비활성화된 계정입니다.'}), 403

    session.permanent = True
    session['user_id'] = user.user_id
    session['username'] = user.username
    session['display_name'] = user.display_name
    session['role'] = user.role or 'WORKER'
    session['company_name'] = getattr(user, 'company_name', 'HD한국조선해양') or 'HD한국조선해양'
    session['company_type'] = getattr(user, 'company_type', 'MAIN') or 'MAIN'
    session['branch_name'] = getattr(user, 'branch_name', '울산 1야드') or '울산 1야드'
    session['department'] = getattr(user, 'department', '선체생산부') or '선체생산부'
    session['conversation_id'] = f"CONV_{uuid.uuid4().hex[:10].upper()}"

    ctx = _get_current_user_context()
    return jsonify({
        'status': 'success',
        'message': f"[{ctx['company_name']} - {ctx['branch_name']}] {ctx['display_name']}({ctx['role']})님 환영합니다!",
        'user': ctx
    })


@api_bp.route('/api/auth/logout', methods=['POST'])
def logout():
    """현재 세션을 초기화하고 로그아웃 처리하는 API입니다."""
    session.clear()
    return jsonify({'status': 'success', 'message': '로그아웃 되었습니다.'})


@api_bp.route('/api/auth/session-check', methods=['GET'])
def session_check():
    """프론트엔드에서 현재 로그인 상태와 소속 지부/권한을 확인할 때 호출하는 API입니다."""
    if 'user_id' in session:
        ctx = _get_current_user_context()
        ctx['logged_in'] = True
        return jsonify(ctx)
    return jsonify({'logged_in': False}), 401


# =============================================================================
# [신규 기능: 조선소 ERP 다중 DB (`ship_erp`) 실시간 관제 & 자원 관리 API]
# =============================================================================
@api_bp.route('/api/erp/dashboard', methods=['GET'])
@login_required_api
def erp_dashboard():
    """
    로그인한 사용자의 직책과 소속에 맞춰 지부별 가용 로봇, 용접 기술자(원청/협력사),
    실시간 자재 재고 현황을 `ship_erp` DB에서 조회하여 반환하는 통합 ERP 관제 API입니다.
    - 사장/임원(EXEC/ADMIN): 전체 지부 통합 조회 가능
    - 야드 관리자(MANAGER/WORKER): 요청 지부 또는 본인 관할 지부 조회
    - 협력사(PARTNER): 본인 협력사 소속 인력 및 투입 지부 현황 중심 조회
    """
    user_ctx = _get_current_user_context()
    requested_branch = request.args.get('branch_name', '').strip() or user_ctx['branch_name']

    # 전체 지부 목록 조회
    branches = ERPBranch.query.all()
    branch_names = [b.branch_name for b in branches] or ['울산 1야드', '울산 2야드', '거제 1도크']

    # 선택된 지부의 로봇, 인력, 자재 상세 조회
    erp_snapshot = query_branch_erp_resources(
        branch_name=requested_branch,
        company_type=user_ctx['company_type']
    )

    # 협력사 계정이 조회할 경우 타 협력사의 민감 인력 정보는 마스킹/필터링 (RBAC 보안 가드레일)
    if user_ctx['company_type'] == 'PARTNER':
        my_comp = user_ctx['company_name']
        erp_snapshot['all_workers'] = [
            w for w in erp_snapshot.get('all_workers', [])
            if w.get('company_name') == my_comp
        ]
        erp_snapshot['available_workers'] = [
            w for w in erp_snapshot.get('available_workers', [])
            if w.get('company_name') == my_comp
        ]

    return jsonify({
        'status': 'success',
        'user_context': user_ctx,
        'branches': branch_names,
        'selected_branch': requested_branch,
        'erp_data': erp_snapshot
    })


@api_bp.route('/api/erp/reserve', methods=['POST'])
@login_required_api
def erp_reserve_resources():
    """
    관리자가 작업지시서를 승인(HITL)하거나 수동 배정할 때,
    `ship_erp` DB의 로봇/작업자 상태를 변경하고 자재 재고를 차감하는 API입니다.
    """
    data = request.get_json() or {}
    result = reserve_erp_resources_transaction(
        robot_serial=data.get('robot_serial'),
        worker_emp_no=data.get('worker_emp_no'),
        material_code=data.get('material_code'),
        use_qty=float(data.get('use_qty', 1.0)),
        target_block=data.get('block_no', 'B-101')
    )
    status_code = 200 if result.get('status') == 'success' else 400
    return jsonify(result), status_code


# =============================================================================
# [기능 1: 다중 RAG 질의 & AI-Agent 연동 API]
# =============================================================================
@api_bp.route('/api/rag/ask', methods=['POST'])
@login_required_api
def rag_ask():
    """
    사용자의 질문을 받아 로그인 세션 컨텍스트(소속 회사, 담당 지부, 권한)와
    해당 지부의 실시간 ERP 자원 현황을 `ShipAgentState`에 자동 주입하여 AI-Agent를 실행하고,
    실행 로그를 `AGENT_EXECUTIONS` 테이블에 기록하는 API입니다.
    """
    data = request.get_json() or {}
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'status': 'error', 'message': '질문을 입력하세요.'}), 400

    user_ctx = _get_current_user_context()
    branch_name = data.get('branch_name', '').strip() or user_ctx['branch_name']
    started_at = datetime.utcnow()
    session_id = session.get('conversation_id', f"SESSION_{uuid.uuid4().hex[:8]}")

    # 1. Agent 실행 전 DB에 RUNNING 상태 선(先)기록
    exec_log = AgentExecution(
        user_id=user_ctx['user_id'],
        conversation_id=session_id,
        request_type='RAG_QA',
        agent_name='ShipSupervisorAgent',
        user_query=query,
        execution_status='RUNNING',
        current_node='supervisor',
        started_at=started_at,
    )
    db.session.add(exec_log)
    db.session.commit()

    # 2. 해당 지부의 ERP 가용 자원(로봇/인력/자재) 스냅샷 조회하여 Agent에 함께 전달
    erp_snapshot = query_branch_erp_resources(branch_name=branch_name, company_type=user_ctx['company_type'])

    # 3. ShipAgentState 구성 (카테고리 필터 제거 -> 빈 리스트 고정, 지부/ERP 정보 자동 주입)
    city_for_weather = erp_snapshot.get('location_city', '울산')
    state_input = {
        "user_query": query,
        "messages": [HumanMessage(content=query)],
        "user_context": user_ctx,
        "erp_data": erp_snapshot,
        "vision_data": None,
        "task_type": None,
        "target_categories": [],  # 카테고리 반복 루프 제거 (전체 단일 고속 검색)
        "need_weather": False,
        "extracted_params": {
            "location": city_for_weather,
            "branch_name": branch_name,
            "company_name": user_ctx['company_name'],
        },
        "is_param_complete": None,
        "missing_params": [],
        "search_status": "idle",
        "search_queries": [],
        "retrieved_docs": [],
        "weather_data": None,
        "generated_document": None,
        "final_answer": None,
        "audit_saved": False,
    }

    config = {"configurable": {"thread_id": session_id}}

    try:
        result = ship_agent_app.invoke(state_input, config=config)
        completed_at = datetime.utcnow()

        response_answer = result.get('final_answer') or "질의 처리가 완료되었습니다."
        raw_docs = result.get('retrieved_docs', [])

        # 프론트엔드 출처 카드 렌더링용 메타데이터 정규화
        formatted_references = []
        for idx, doc in enumerate(raw_docs, start=1):
            meta = doc.get("metadata", {}) or {}
            chunk_id = meta.get("doc_id") or meta.get("id") or doc.get("id") or f"CHROMA_{idx:03d}"
            doc_name = meta.get("document_name") or doc.get("document_name") or "사내 기술/안전 규격서"
            page_val = meta.get("page") or meta.get("page_start") or 1
            page_str = f"p.{page_val}"
            source_val = meta.get("source") or doc.get("source") or page_str
            rev_val = meta.get("revision") or "Rev.1"
            sim = meta.get("similarity") or meta.get("similarity_score") or doc.get("similarity_score") or 0.85
            sim_str = f"{float(sim):.2f}"
            txt = doc.get("content") or doc.get("chunk_text") or ""

            formatted_references.append({
                "id": chunk_id,
                "doc_id": chunk_id,
                "document_id": chunk_id,
                "category": doc.get("category", "공통규격"),
                "document_name": doc_name,
                "title": doc_name,
                "source": source_val,
                "page": page_val,
                "page_str": page_str,
                "revision": rev_val,
                "similarity_score": sim_str,
                "chunk_text": txt,
                "metadata": meta,
            })

        # 4. Agent 실행 완료 상태 DB 업데이트
        exec_log.execution_status = 'COMPLETED'
        exec_log.current_node = '__end__'
        exec_log.response_text = response_answer
        exec_log.output_data = json.dumps({"reference_count": len(formatted_references)}, ensure_ascii=False)
        exec_log.completed_at = completed_at
        db.session.commit()

        return jsonify({
            'status': 'success',
            'answer': response_answer,
            'references': formatted_references,
            'task_type': result.get('task_type'),
            'weather_data': result.get('weather_data'),
            'erp_summary': {
                'branch_name': branch_name,
                'idle_robot_count': len(erp_snapshot.get('available_robots', [])),
                'available_worker_count': len(erp_snapshot.get('available_workers', [])),
            }
        })

    except Exception as exc:
        exec_log.execution_status = 'FAILED'
        exec_log.error_message = str(exc)
        exec_log.completed_at = datetime.utcnow()
        db.session.commit()
        return jsonify({'status': 'error', 'message': f'에이전트 처리 중 오류 발생: {exc}'}), 500


# =============================================================================
# [기능 2: Vision 이미지 분석 + ERP 자원 매칭 + 작업지시서 자동 생성 API]
# =============================================================================
@api_bp.route('/api/work/analyze', methods=['POST'])
@login_required_api
def work_analyze():
    """
    현장 용접 이미지 및 작업 제원을 받아:
    1) `WORK_ORDERS` 테이블에 작업 요청 생성
    2) Vision 모듈(`analyze_welding_frame`) 실행 후 `VISION_ANALYSES` 기록
    3) 검출된 다수의 BBox/궤적 좌표를 `db.session.bulk_save_objects()`로 초고속 배치 저장
    4) `ship_erp` DB의 가용 로봇·기술자·자재 현황과 Vision 검출 결과를 `ShipAgentState`에 주입하여 작업지시서 발행
    5) 결과를 `WORK_INSTRUCTIONS` 테이블에 영구 저장하는 End-to-End 파이프라인 API입니다.
    """
    user_ctx = _get_current_user_context()
    data = request.form.to_dict() if request.form else (request.get_json() or {})

    branch_name = data.get('work_area', '').strip() or user_ctx['branch_name']
    block_no = data.get('block_no', 'B-104').strip()
    base_material = data.get('base_material', 'AH36').strip()
    thickness = float(data.get('material_thickness') or 25.0)
    process_type = data.get('process_type', 'FCAW').strip()
    joint_type = data.get('joint_type', '맞대기(Butt)').strip()
    welding_pos = data.get('welding_position', '1G(아래보기)').strip()
    simulate_obstacle = str(data.get('simulate_obstacle', 'false')).lower() in ('true', '1', 'yes')

    # 1. WORK_ORDERS 저장
    work_order = WorkOrder(
        user_id=user_ctx['user_id'],
        work_name=f"[{branch_name} / {block_no}] {base_material} {thickness}mm {process_type} 용접",
        request_text=data.get('additional_input', ''),
        block_no=block_no,
        work_area=branch_name,
        process_type=process_type,
        base_material=base_material,
        material_thickness=thickness,
        joint_type=joint_type,
        groove_type=data.get('groove_type', 'V-Groove'),
        welding_position=welding_pos,
        work_status='ANALYZING',
    )
    db.session.add(work_order)
    db.session.commit()

    # 2. Vision 프레임 분석 실행 및 메타데이터 저장
    v_out = analyze_welding_frame(image_path="", include_defect=simulate_obstacle)
    v_analysis = VisionAnalysis(
        work_id=work_order.work_id,
        image_path=data.get('image_path', 'sample_weld_frame.jpg'),
        model_name=v_out.get('model_name', 'YOLOv11s-ShipWeld'),
        model_version=v_out.get('model_version', 'v2.4'),
        analysis_status=v_out.get('analysis_status', 'SUCCESS'),
        image_width=v_out.get('image_width', 640),
        image_height=v_out.get('image_height', 360),
        result_json=json.dumps(v_out.get('detections', []), ensure_ascii=False),
        started_at=v_out.get('started_at', datetime.utcnow()),
        completed_at=v_out.get('completed_at', datetime.utcnow()),
    )
    db.session.add(v_analysis)
    db.session.commit()

    # 3. [시계열/다건 병목 방지] Vision 검출 객체 리스트를 Bulk Insert로 일괄 저장
    detection_objects = []
    root_gap_mm = 2.1
    obstacles = []
    for det in v_out.get('detections', []):
        geom = det.get('geometry_data', {}) or {}
        if 'root_gap_mm' in geom:
            root_gap_mm = geom['root_gap_mm']
        if det.get('class_name') == 'CLAMP_OBSTACLE':
            obstacles.append(det.get('label_ko', '클램프 간섭물'))

        detection_objects.append(
            VisionDetection(
                vision_id=v_analysis.vision_id,
                class_name=det.get('class_name'),
                confidence=det.get('confidence'),
                coordinate_type=det.get('coordinate_type', 'PIXEL'),
                x_min=det.get('x_min'),
                y_min=det.get('y_min'),
                x_max=det.get('x_max'),
                y_max=det.get('y_max'),
                geometry_data=json.dumps(geom, ensure_ascii=False),
            )
        )
    if detection_objects:
        db.session.bulk_save_objects(detection_objects)
        db.session.commit()

    # 4. ERP DB(`ship_erp`)에서 해당 지부의 가용 로봇·작업자·자재 재고 조회
    erp_snapshot = query_branch_erp_resources(branch_name=branch_name, company_type=user_ctx['company_type'])
    recommended_robot = (erp_snapshot.get('available_robots') or [None])[0]
    recommended_worker = (erp_snapshot.get('available_workers') or [None])[0]

    # 5. Agent 호출을 위한 종합 프롬프트 및 State 구성
    composite_query = (
        f"{branch_name} {block_no} 블록에서 {base_material} 모재(두께 {thickness}mm) "
        f"{joint_type} {process_type}({welding_pos}) 용접 작업지시서를 작성해줘. "
        f"[비전 계측 루트간격: {root_gap_mm}mm, 탐지된 장애물: {', '.join(obstacles) if obstacles else '없음'}] "
        f"[현재 지부 가용 로봇: {recommended_robot['robot_name'] if recommended_robot else '없음'}, "
        f"투입 가능 기술자: {recommended_worker['name'] if recommended_worker else '없음'}]"
    )

    session_id = session.get('conversation_id', f"WORK_{work_order.work_id}")
    state_input = {
        "user_query": composite_query,
        "messages": [HumanMessage(content=composite_query)],
        "user_context": user_ctx,
        "vision_data": v_out,
        "erp_data": erp_snapshot,
        "task_type": "doc_generation",
        "target_categories": [],
        "need_weather": True,
        "extracted_params": {
            "location": erp_snapshot.get("location_city", "울산"),
            "branch_name": branch_name,
            "block_no": block_no,
            "base_material": base_material,
            "material_thickness_mm": thickness,
            "process_type": process_type,
            "joint_type": joint_type,
            "welding_position": welding_pos,
            "vision_root_gap_mm": root_gap_mm,
            "vision_obstacles": obstacles,
            "recommended_robot": recommended_robot,
            "recommended_worker": recommended_worker,
        },
        "search_status": "idle",
        "retrieved_docs": [],
        "weather_data": None,
        "generated_document": None,
        "final_answer": None,
    }

    exec_log = AgentExecution(
        user_id=user_ctx['user_id'],
        work_id=work_order.work_id,
        vision_id=v_analysis.vision_id,
        conversation_id=session_id,
        request_type='WORK_INSTRUCTION_GEN',
        agent_name='ShipSupervisorAgent',
        user_query=composite_query,
        execution_status='RUNNING',
        started_at=datetime.utcnow(),
    )
    db.session.add(exec_log)
    db.session.commit()

    result = ship_agent_app.invoke(state_input, config={"configurable": {"thread_id": session_id}})
    instruction_md = result.get('final_answer') or "작업지시서 초안이 생성되었습니다."

    exec_log.execution_status = 'COMPLETED'
    exec_log.response_text = instruction_md
    exec_log.completed_at = datetime.utcnow()

    # 6. WORK_INSTRUCTIONS 테이블에 최종 작업지시서 및 ERP 추천 자원 저장
    work_inst = WorkInstruction(
        work_id=work_order.work_id,
        execution_id=exec_log.execution_id,
        version_no=1,
        instruction_text=instruction_md,
        instruction_data=json.dumps(state_input["extracted_params"], ensure_ascii=False),
        source_references=json.dumps(result.get("retrieved_docs", []), ensure_ascii=False),
        approval_status='WAITING',
        instruction_status='FINAL',
    )
    work_order.work_status = 'COMPLETED'
    db.session.add(work_inst)
    db.session.commit()

    return jsonify({
        'status': 'success',
        'work_id': work_order.work_id,
        'vision_id': v_analysis.vision_id,
        'instruction_id': work_inst.instruction_id,
        'vision_result': v_out,
        'erp_matching': {
            'branch_name': branch_name,
            'recommended_robot': recommended_robot,
            'recommended_worker': recommended_worker,
            'inventory': erp_snapshot.get('inventory', []),
        },
        'weather_data': result.get('weather_data'),
        'instruction_text': instruction_md,
        'references': result.get('retrieved_docs', []),
    })