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
    AgentExecution, WorkInstruction, DocumentMetadata
)
from modules.vision.detector import analyze_welding_frame
from modules.agent.supervisor import ship_agent_app

api_bp = Blueprint('api', __name__)


# [보안 데코레이터 1] 비로그인 시 웹 페이지 접근 차단
def login_required_page(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('api.login_page', next=request.path))
        return f(*args, **kwargs)
    return decorated


# [보안 데코레이터 2] 비로그인 시 API 작동 금지
def login_required_api(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'status': 'unauthorized', 'message': '로그인이 필요한 서비스입니다.'}), 401
        return f(*args, **kwargs)
    return decorated


def _extract_approval_and_doc_status(result: dict) -> tuple[str, str]:
    """
    에이전트 실행 결과(State)에서 실제 안전 판정 상태와 문서 발행 상태를 추출합니다.
    (하드코딩된 'PASS' 대신 실제 판정 결과를 DB에 반영)
    """
    weather = result.get("weather_data") or {}
    safety_eval = weather.get("safety_evaluation") or {}
    safety_status = safety_eval.get("status")

    vision = result.get("vision_data") or {}
    has_vision_obstacle = any(
        "OBSTACLE" in str(det.get("class_name", "")).upper() or "DEFECT" in str(det.get("class_name", "")).upper()
        for det in (vision.get("detections") or [])
    )
    missing = result.get("missing_params") or []

    # 1. 비전 결함 또는 기상 전면 중지인 경우
    if safety_status == "STOP" or has_vision_obstacle:
        return "STOP", "HOLD"

    # 2. 필수 제원 누락 또는 기상 조건부 승인인 경우
    if missing or safety_status == "CONDITIONAL_PASS" or weather.get("status") in ("NEED_LOCATION", "NOT_FOUND", "ERROR"):
        return "CONDITIONAL_PASS", "DRAFT"

    # 3. 안전 평가가 명시적으로 PASS인 경우
    if safety_status == "PASS":
        return "PASS", "FINAL"

    # 4. 일반 기술 질의 등 별도 안전 판정이 없는 경우
    return "PASS", "FINAL"


def _format_rag_references(raw_docs: list) -> list:
    """프론트엔드 UI 렌더링을 위한 참조 문서 메타데이터 정규화"""
    formatted_references = []
    for idx, doc in enumerate(raw_docs, start=1):
        meta = doc.get("metadata", {}) or {}

        chunk_id = (
            meta.get("doc_id")
            or meta.get("id")
            or doc.get("doc_id")
            or doc.get("id")
            or f"CHROMA_{idx:03d}"
        )
        doc_name = (
            meta.get("document_name")
            or meta.get("title")
            or doc.get("document_name")
            or "사내 기술/안전 규격서"
        )
        page_val = meta.get("page") or meta.get("page_start") or meta.get("page_number") or meta.get("page_no") or 1
        page_str = f"p.{page_val}"
        source_val = meta.get("source") or doc.get("source") or page_str
        rev_val = meta.get("revision") or meta.get("rev") or "UNKNOWN"

        sim = meta.get("similarity") or meta.get("similarity_score") or doc.get("similarity_score")
        sim_str = f"{float(sim):.2f}" if sim is not None else "0.00"
        txt = doc.get("content") or doc.get("chunk_text") or meta.get("chunk_text") or ""

        enriched_meta = dict(meta)
        enriched_meta.update({
            "doc_id": chunk_id,
            "id": chunk_id,
            "document_id": chunk_id,
            "document_name": doc_name,
            "title": doc_name,
            "source": source_val,
            "page": page_val,
            "page_start": page_val,
            "page_number": page_val,
            "page_no": page_val,
            "page_str": page_str,
            "revision": rev_val,
            "similarity_score": sim_str,
            "chunk_text": txt
        })

        formatted_references.append({
            "id": chunk_id,
            "doc_id": chunk_id,
            "document_id": chunk_id,
            "category": doc.get("category", "규격"),
            "document_name": doc_name,
            "title": doc_name,
            "source": source_val,
            "page": page_val,
            "page_start": page_val,
            "page_number": page_val,
            "page_no": page_val,
            "page_str": page_str,
            "revision": rev_val,
            "rev": rev_val,
            "similarity_score": sim_str,
            "score": sim_str,
            "chunk_text": txt,
            "content": txt,
            "text": txt,
            "metadata": enriched_meta
        })
    return formatted_references


# ================= [페이지 라우트] =================
@api_bp.route('/')
def home_page():
    return render_template('home.html')


@api_bp.route('/login')
def login_page():
    return render_template('auth.html')


@api_bp.route('/rag-chat')
@login_required_page
def rag_chat_page():
    return render_template('rag_chat.html')


@api_bp.route('/qna')
@login_required_page
def qna_page():
    return render_template('qna.html')


@api_bp.route('/work-analysis')
@login_required_page
def work_analysis_page():
    return render_template('work_analysis.html')


# ================= [인증 API] =================
@api_bp.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    email = data.get('email', '').strip()
    display_name = data.get('display_name', '').strip() or username
    role = data.get('role', 'USER')

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
        is_active=True
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({'status': 'success', 'message': f'계정 [{username}] 생성 완료!'})


@api_bp.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'status': 'error', 'message': '아이디 또는 비밀번호가 일치하지 않습니다.'}), 401

    if not user.is_active or str(user.is_active).upper() not in ('TRUE', '1', 'Y'):
        return jsonify({'status': 'error', 'message': '비활성화된 계정입니다.'}), 403

    session.permanent = True
    session['user_id'] = user.user_id
    session['username'] = user.username
    session['display_name'] = user.display_name
    session['role'] = user.role
    session['conversation_id'] = f"CONV_{uuid.uuid4().hex[:10].upper()}"
    session['extracted_params'] = {}

    return jsonify({
        'status': 'success',
        'message': f'{user.display_name}({user.role})님 환영합니다!',
        'user': {'user_id': user.user_id, 'username': user.username, 'role': user.role}
    })


@api_bp.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'status': 'success', 'message': '로그아웃 되었습니다.'})


@api_bp.route('/api/auth/session-check', methods=['GET'])
def session_check():
    if 'user_id' in session:
        return jsonify({
            'logged_in': True,
            'user_id': session['user_id'],
            'username': session['username'],
            'display_name': session.get('display_name'),
            'role': session.get('role')
        })
    return jsonify({'logged_in': False}), 401


# ================= [기능 1: RAG 질의 API (멀티턴 문맥 유지)] =================
@api_bp.route('/api/rag/ask', methods=['POST'])
@login_required_api
def rag_ask():
    data = request.get_json() or {}
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'status': 'error', 'message': '질문을 입력하세요.'}), 400

    started_at = datetime.now()
    session_id = session.get('conversation_id')
    if not session_id:
        session_id = f"CONV_{uuid.uuid4().hex[:10].upper()}"
        session['conversation_id'] = session_id

    config = {"configurable": {"thread_id": session_id}}
    prev_params = dict(session.get('extracted_params') or {})
    # 🌟 직전 턴이 제원 누락 역질문 상태('no')였는지 여부만 가져옴 (task_type은 항상 None으로 시작!)
    prev_complete = session.get('is_param_complete')

    state_input = {
        "user_query": query,
        "messages": [HumanMessage(content=query)],
        "extracted_params": prev_params,

        # 🌟 사용자분이 수정하신 대로 task_type은 무조건 None! (항상 Team 1을 거치도록 보장)
        "task_type": None,
        "target_categories": [],
        "need_weather": False,
        "is_param_complete": prev_complete,  # 👈 직전 턴 역질문 대기 여부만 전달

        "search_status": "idle",
        "search_queries": [],
        "retrieved_docs": [],
        "weather_data": None,
        "vision_data": None,
        "missing_params": [],
        "generated_document": None,
        "final_answer": None,
        "audit_saved": False
    }

    result = ship_agent_app.invoke(state_input, config=config)

    session['extracted_params'] = dict(result.get('extracted_params') or {})
    session['is_param_complete'] = result.get('is_param_complete')
    session.modified = True

    completed_at = datetime.now()
    response_answer = result.get('final_answer') or "질의 처리가 완료되었습니다."
    raw_docs = result.get('retrieved_docs', [])
    formatted_references = _format_rag_references(raw_docs)

    # 실제 에이전트 판정 상태 추출
    approval_res, _ = _extract_approval_and_doc_status(result)
    task_type_res = result.get("task_type") or "knowledge_qa"

    exec_row = AgentExecution(
        user_id=session['user_id'],
        work_id=None,
        vision_id=None,
        conversation_id=session_id,
        request_type='작업지시' if task_type_res == 'doc_generation' else '일반 질의',
        agent_name='ShipSupervisorAgent',
        user_query=query,
        execution_status='SUCCESS',
        approval_result=approval_res,
        current_node='FINISH',
        state_version='2.0',
        input_data=json.dumps(state_input, ensure_ascii=False, default=str),
        state_data=json.dumps({
            "task_type": task_type_res,
            "extracted_params": result.get("extracted_params") or {},
            "missing_params": result.get("missing_params", []),
            "weather_data": result.get("weather_data")
        }, ensure_ascii=False, default=str),
        output_data=json.dumps(result.get("generated_document") or {}, ensure_ascii=False, default=str),
        response_text=response_answer,
        started_at=started_at,
        completed_at=completed_at
    )
    db.session.add(exec_row)
    db.session.commit()

    return jsonify({
        'status': 'success',
        'execution_id': exec_row.execution_id,
        'approval_result': approval_res,
        'answer': response_answer,
        'references': formatted_references,
        'retrieved_docs': formatted_references
    })


# ================= [기능 2: 이미지 + 모재정보 -> 작업지시 통합 파이프라인 API] =================
@api_bp.route('/api/work/analyze', methods=['POST'])
@login_required_api
def work_analyze():
    user_id = session['user_id']

    # 사용자가 실제 폼에 입력한 값만 정직하게 수신 (임의의 가짜 기본값 하드코딩 배제)
    work_name = request.form.get('work_name', '').strip()
    request_text = request.form.get('request_text', '').strip()
    block_no = request.form.get('block_no', '').strip()
    work_area = request.form.get('work_area', '').strip()
    process_type = request.form.get('process_type', '').strip()
    base_material = request.form.get('base_material', '').strip()
    raw_thickness = request.form.get('material_thickness', '').strip()
    joint_type = request.form.get('joint_type', '').strip()
    groove_type = request.form.get('groove_type', '').strip()
    welding_position = request.form.get('welding_position', '').strip()
    additional_input = request.form.get('additional_input', '').strip()
    simulate_obstacle = request.form.get('simulate_obstacle') == 'true'

    material_thickness = None
    if raw_thickness:
        try:
            material_thickness = float(raw_thickness)
        except ValueError:
            material_thickness = None

    img = request.files.get('image')
    img_path = 'simulated_frame.jpg'
    if img and img.filename:
        upload_folder = current_app.config.get('UPLOAD_FOLDER', './uploads')
        os.makedirs(upload_folder, exist_ok=True)
        img_path = os.path.join(upload_folder, img.filename)
        img.save(img_path)

    # 1. WORK_ORDERS 초기 저장 (상태: PROCESSING)
    order = WorkOrder(
        user_id=user_id,
        work_name=work_name or "현장 용접 작업",
        request_text=request_text,
        block_no=block_no,
        work_area=work_area,
        process_type=process_type,
        base_material=base_material,
        material_thickness=material_thickness,
        joint_type=joint_type,
        groove_type=groove_type,
        welding_position=welding_position,
        additional_input=additional_input,
        work_status='PROCESSING'
    )
    db.session.add(order)
    db.session.commit()

    # 2. 비전 분석 실행 및 기록
    v_out = analyze_welding_frame(img_path, include_defect=simulate_obstacle)
    v_analysis = VisionAnalysis(
        work_id=order.work_id,
        image_path=img_path,
        model_name=v_out.get('model_name', 'YOLOv11s-ShipWeld-Seg'),
        model_version=v_out.get('model_version', 'v2.4-2026'),
        analysis_status=v_out.get('analysis_status', 'COMPLETED'),
        image_width=v_out.get('image_width', 640),
        image_height=v_out.get('image_height', 360),
        result_json=json.dumps(v_out, ensure_ascii=False),
        started_at=v_out.get('started_at', datetime.now()),
        completed_at=v_out.get('completed_at', datetime.now())
    )
    db.session.add(v_analysis)
    db.session.commit()

    detection_rows = []
    for det in v_out.get('detections', []):
        detection_rows.append({
            "VISION_ID": v_analysis.vision_id,
            "CLASS_NAME": det.get('class_name'),
            "CONFIDENCE": det.get('confidence'),
            "COORDINATE_TYPE": det.get('coordinate_type', 'PIXEL'),
            "X_MIN": det.get('x_min'),
            "Y_MIN": det.get('y_min'),
            "X_MAX": det.get('x_max'),
            "Y_MAX": det.get('y_max'),
            "GEOMETRY_DATA": json.dumps(det.get('geometry_data', {}), ensure_ascii=False),
            "CREATED_AT": datetime.utcnow()
        })
    if detection_rows:
        db.session.bulk_insert_mappings(VisionDetection, detection_rows)
        db.session.commit()

    # 3. 에이전트 파이프라인 호출 (입력된 폼 필드만 동적으로 구성)
    session_id = session.get('conversation_id') or f"WORK_{order.work_id}"
    config = {"configurable": {"thread_id": session_id}}

    form_params = {}
    if base_material:
        form_params["material"] = base_material
    if material_thickness is not None:
        form_params["thickness"] = f"{material_thickness:g}T"
    if process_type:
        form_params["process"] = process_type
    if joint_type:
        form_params["joint_type"] = joint_type
    if welding_position:
        form_params["position"] = welding_position
    if work_area:
        form_params["location"] = work_area

    # 사용자가 작성한 요청문을 우선하되, 비어있으면 입력된 폼 제원을 자연스럽게 전달
    user_query_str = request_text if request_text else json.dumps(form_params, ensure_ascii=False)

    state_input = {
        "user_query": user_query_str,
        "messages": [HumanMessage(content=user_query_str)],
        "task_type": "doc_generation",
        "extracted_params": form_params,
        "vision_data": v_out,
        "search_status": "idle",
        "final_answer": None,
        "audit_saved": False
    }

    started_at = datetime.now()
    result = ship_agent_app.invoke(state_input, config=config)
    completed_at = datetime.now()

    generated_doc = result.get("generated_document") or {}
    instruction_text = result.get("final_answer") or "작업지시서 처리가 완료되었습니다."

    # 실제 안전 평가 및 비전 장애물 여부에 따른 승인 상태 산출
    approval_res, doc_status = _extract_approval_and_doc_status(result)

    # 4. AGENT_EXECUTIONS 저장 (실제 판정 결과 기록)
    exec_row = AgentExecution(
        user_id=user_id,
        work_id=order.work_id,
        vision_id=v_analysis.vision_id,
        conversation_id=session_id,
        request_type='작업지시',
        agent_name='ShipSupervisorAgent',
        user_query=user_query_str,
        execution_status='SUCCESS',
        approval_result=approval_res,
        current_node='FINISH',
        state_version='2.0',
        input_data=json.dumps(state_input, ensure_ascii=False, default=str),
        state_data=json.dumps({
            "params": result.get("extracted_params", form_params),
            "missing_params": result.get("missing_params", []),
            "weather_data": result.get("weather_data")
        }, ensure_ascii=False, default=str),
        output_data=json.dumps(generated_doc, ensure_ascii=False, default=str),
        response_text=instruction_text,
        started_at=started_at,
        completed_at=completed_at
    )
    db.session.add(exec_row)
    db.session.commit()

    # 5. WORK_INSTRUCTIONS 및 WORK_ORDERS 상태 업데이트
    inst_row = WorkInstruction(
        work_id=order.work_id,
        execution_id=exec_row.execution_id,
        version_no=1,
        instruction_text=instruction_text,
        instruction_data=json.dumps(generated_doc, ensure_ascii=False, default=str),
        source_references=json.dumps(result.get("retrieved_docs", []), ensure_ascii=False, default=str),
        approval_status=approval_res,
        instruction_status=doc_status
    )
    db.session.add(inst_row)

    order.work_status = approval_res
    db.session.commit()

    formatted_refs = _format_rag_references(result.get("retrieved_docs", []))

    return jsonify({
        'status': 'success',
        'work_id': order.work_id,
        'vision_id': v_analysis.vision_id,
        'execution_id': exec_row.execution_id,
        'instruction_id': inst_row.instruction_id,
        'approval_result': approval_res,
        'instruction_status': doc_status,
        'vision_result': v_out,
        'agent_result': {
            'final_answer': instruction_text,
            'generated_document': generated_doc
        },
        'rag_references': formatted_refs
    })


# ================= [상태 조회 API] =================
@api_bp.route('/api/system/status', methods=['GET'])
def system_status():
    counts = {
        'USERS': User.query.count(),
        'WORK_ORDERS': WorkOrder.query.count(),
        'VISION_ANALYSES': VisionAnalysis.query.count(),
        'VISION_DETECTIONS': VisionDetection.query.count(),
        'AGENT_EXECUTIONS': AgentExecution.query.count(),
        'WORK_INSTRUCTIONS': WorkInstruction.query.count(),
        'DOCUMENT_METADATA': DocumentMetadata.query.count()
    }
    orders = WorkOrder.query.order_by(WorkOrder.work_id.desc()).limit(8).all()
    history = []
    for o in orders:
        u = User.query.filter_by(user_id=o.user_id).first()
        thickness_str = f"{o.material_thickness:g}T" if o.material_thickness is not None else "미지정"
        history.append({
            'work_id': o.work_id,
            'username': u.username if u else 'unknown',
            'work_name': o.work_name,
            'block_no': o.block_no,
            'material': f"{o.base_material or '미지정'} ({thickness_str})",
            'joint_position': f"{o.joint_type or '-'} / {o.welding_position or '-'}",
            'work_status': o.work_status,
            'created_at': o.created_at.strftime('%H:%M:%S') if o.created_at else ''
        })
    return jsonify({'counts': counts, 'history': history})