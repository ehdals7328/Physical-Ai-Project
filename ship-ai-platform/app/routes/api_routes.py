import os, json, uuid
from functools import wraps
from flask import Blueprint, request, jsonify, render_template, session, redirect, url_for, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from app.models.schema import (
    db, User, WorkOrder, VisionAnalysis, VisionDetection,
    AgentExecution, WorkInstruction, DocumentMetadata
)
from modules.knowledge.retriever import KnowledgeStore
from modules.vision.detector import analyze_welding_frame
from modules.agent.supervisor import WeldingSupervisorAgent

api_bp = Blueprint('api', __name__)

# [보안 데코레이터 1] 비로그인 시 웹 페이지 접근 차단 -> 로그인 페이지로 이동
def login_required_page(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('api.login_page', next=request.path))
        return f(*args, **kwargs)
    return decorated

# [보안 데코레이터 2] 비로그인 시 API 작동 금지 (401 반환)
def login_required_api(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'status': 'unauthorized', 'message': '로그인이 필요한 서비스입니다. 세션이 만료되었거나 로그인되지 않았습니다.'}), 401
        return f(*args, **kwargs)
    return decorated

# ================= [멀티 페이지 화면 라우트] =================
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

# ================= [인증 API (비밀번호 해싱 & 세션)] =================
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
        is_active='Y'
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({'status': 'success', 'message': f'계정 [{username}] 생성 완료! 이제 로그인하세요.'})

@api_bp.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'status': 'error', 'message': '아이디 또는 비밀번호가 일치하지 않습니다.'}), 401

    if user.is_active != 'Y':
        return jsonify({'status': 'error', 'message': '비활성화된 계정입니다.'}), 403

    session.permanent = True
    session['user_id'] = user.user_id
    session['username'] = user.username
    session['display_name'] = user.display_name
    session['role'] = user.role
    session['conversation_id'] = f"CONV_{uuid.uuid4().hex[:10].upper()}"

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

# ================= [기능 1: 순수 LLM(RAG) 질의 API] =================
@api_bp.route('/api/rag/ask', methods=['POST'])
@login_required_api
def rag_ask():
    data = request.get_json() or {}
    query = data.get('query', '').strip()
    category = data.get('category', '').strip()
    if not query:
        return jsonify({'status': 'error', 'message': '질문을 입력하세요.'}), 400

    store = KnowledgeStore()
    rag_chunks = store.search(query=query, category=category if category else None, top_k=3)

    # 만약 ChromaDB가 비어있다면 Oracle DOCUMENT_METADATA에서 폴백 조회
    if not rag_chunks:
        docs = DocumentMetadata.query.limit(3).all()
        for d in docs:
            rag_chunks.append({
                "document_id": d.document_id,
                "document_name": d.document_name,
                "category": d.category,
                "source": d.source,
                "revision": d.revision,
                "chunk_text": f"[{d.document_name}] 조선소 표준 기술 기준 문서 (카테고리: {d.category})",
                "similarity_score": 0.75
            })

    agent = WeldingSupervisorAgent()
    res = agent.run_rag_qa(query, rag_chunks)

    exec_row = AgentExecution(
        user_id=session['user_id'],
        work_id=None,
        vision_id=None,
        conversation_id=session.get('conversation_id', 'CONV_DEFAULT'),
        request_type='일반 질의',
        agent_name=res['agent_name'],
        user_query=query,
        execution_status=res['execution_status'],
        approval_result=res['approval_result'],
        current_node=res['current_node'],
        state_version=res['state_version'],
        input_data=json.dumps(res['input_data'], ensure_ascii=False),
        state_data=json.dumps(res['state_data'], ensure_ascii=False),
        output_data=json.dumps(res['output_data'], ensure_ascii=False),
        response_text=res['response_text'],
        started_at=res['started_at'],
        completed_at=res['completed_at']
    )
    db.session.add(exec_row)
    db.session.commit()

    return jsonify({
        'status': 'success',
        'execution_id': exec_row.execution_id,
        'answer': res['response_text'],
        'references': rag_chunks
    })

# ================= [기능 2: 이미지 + 모재정보 -> 작업지시 통합 파이프라인 API] =================
@api_bp.route('/api/work/analyze', methods=['POST'])
@login_required_api
def work_analyze():
    user_id = session['user_id']
    work_name = request.form.get('work_name', '선미 블록 외판 맞대기 용접')
    request_text = request.form.get('request_text', 'AH36 16mm FCAW 용접 조건 및 안전 기준 확인 요청')
    block_no = request.form.get('block_no', 'B-402A')
    work_area = request.form.get('work_area', '2도크 선미부')
    process_type = request.form.get('process_type', 'FCAW')
    base_material = request.form.get('base_material', 'AH36')
    material_thickness = float(request.form.get('material_thickness', 16.0))
    joint_type = request.form.get('joint_type', 'BUTT (맞대기)')
    groove_type = request.form.get('groove_type', 'V-GROOVE')
    welding_position = request.form.get('welding_position', '2F (수평필렛)')
    additional_input = request.form.get('additional_input', '특이사항 없음')
    simulate_obstacle = request.form.get('simulate_obstacle') == 'true'

    img = request.files.get('image')
    img_path = 'simulated_frame.jpg'
    if img and img.filename:
        img_path = os.path.join(current_app.config['UPLOAD_FOLDER'], img.filename)
        img.save(img_path)

    # 1. WORK_ORDERS 저장 (16개 컬럼)
    order = WorkOrder(
        user_id=user_id,
        work_name=work_name,
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

    # 2. VISION_ANALYSES 저장 (13개 컬럼)
    v_out = analyze_welding_frame(img_path, include_defect=simulate_obstacle)
    v_analysis = VisionAnalysis(
        work_id=order.work_id,
        image_path=img_path,
        model_name=v_out['model_name'],
        model_version=v_out['model_version'],
        analysis_status=v_out['analysis_status'],
        image_width=v_out['image_width'],
        image_height=v_out['image_height'],
        result_json=json.dumps({"detections_count": len(v_out['detections'])}, ensure_ascii=False),
        started_at=v_out['started_at'],
        completed_at=v_out['completed_at']
    )
    db.session.add(v_analysis)
    db.session.commit()

    # 3. VISION_DETECTIONS 저장 (11개 컬럼 - 1:N 반복 저장)
    for det in v_out['detections']:
        db.session.add(VisionDetection(
            vision_id=v_analysis.vision_id,
            class_name=det['class_name'],
            confidence=det['confidence'],
            coordinate_type=det['coordinate_type'],
            x_min=det['x_min'],
            y_min=det['y_min'],
            x_max=det['x_max'],
            y_max=det['y_max'],
            geometry_data=json.dumps(det['geometry_data'], ensure_ascii=False)
        ))
    db.session.commit()

    # 4. RAG 검색
    store = KnowledgeStore()
    rag_query = f"{base_material} {material_thickness}mm {process_type} {request_text}"
    rag_chunks = store.search(query=rag_query, top_k=3)

    # 5. AGENT_EXECUTIONS 저장 (22개 컬럼)
    work_meta = {
        "work_id": order.work_id,
        "work_name": work_name,
        "block_no": block_no,
        "work_area": work_area,
        "process_type": process_type,
        "base_material": base_material,
        "material_thickness": material_thickness,
        "joint_type": joint_type,
        "groove_type": groove_type,
        "welding_position": welding_position
    }
    agent = WeldingSupervisorAgent()
    a_out = agent.run_work_supervisor(work_meta, v_out, rag_chunks)

    exec_row = AgentExecution(
        user_id=user_id,
        work_id=order.work_id,
        vision_id=v_analysis.vision_id,
        conversation_id=session.get('conversation_id', 'CONV_WORK'),
        request_type='작업지시',
        agent_name=a_out['agent_name'],
        user_query=request_text,
        execution_status=a_out['execution_status'],
        approval_result=a_out['approval_result'],
        current_node=a_out['current_node'],
        state_version=a_out['state_version'],
        input_data=json.dumps(a_out['input_data'], ensure_ascii=False),
        state_data=json.dumps(a_out['state_data'], ensure_ascii=False),
        output_data=json.dumps(a_out['output_data'], ensure_ascii=False),
        response_text=a_out['response_text'],
        started_at=a_out['started_at'],
        completed_at=a_out['completed_at']
    )
    db.session.add(exec_row)
    db.session.commit()

    # 6. WORK_INSTRUCTIONS 저장 (11개 컬럼)
    inst_row = WorkInstruction(
        work_id=order.work_id,
        execution_id=exec_row.execution_id,
        version_no=1,
        instruction_text=a_out['instruction_text'],
        instruction_data=json.dumps(a_out['instruction_data'], ensure_ascii=False),
        source_references=json.dumps(a_out['source_references'], ensure_ascii=False),
        approval_status=a_out['approval_result'],
        instruction_status='FINAL'
    )
    db.session.add(inst_row)

    order.work_status = a_out['approval_result']
    db.session.commit()

    return jsonify({
        'status': 'success',
        'work_id': order.work_id,
        'vision_id': v_analysis.vision_id,
        'execution_id': exec_row.execution_id,
        'instruction_id': inst_row.instruction_id,
        'vision_result': v_out,
        'agent_result': a_out,
        'rag_references': rag_chunks
    })

# ================= [DB 7개 테이블 실시간 카운트 & 최근 작업 이력 API] =================
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
        history.append({
            'work_id': o.work_id,
            'username': u.username if u else 'unknown',
            'work_name': o.work_name,
            'block_no': o.block_no,
            'material': f"{o.base_material} ({o.material_thickness}T)",
            'joint_position': f"{o.joint_type} / {o.welding_position}",
            'work_status': o.work_status,
            'created_at': o.created_at.strftime('%H:%M:%S')
        })
    return jsonify({'counts': counts, 'history': history})
