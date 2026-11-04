import os, json, uuid
from datetime import datetime
from functools import wraps
from flask import Blueprint, request, jsonify, render_template, session, redirect, url_for, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_core.messages import HumanMessage

from app.models.schema import (
    db, User, WorkOrder, VisionAnalysis, VisionDetection,
    AgentExecution, WorkInstruction, DocumentMetadata
)
from modules.knowledge.retriever import KnowledgeStore
from modules.vision.detector import analyze_welding_frame

# 메인 멀티에이전트 그래프 앱 임포트
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

# ================= [기능 1: RAG 질의 API] =================
@api_bp.route('/api/rag/ask', methods=['POST'])
@login_required_api
def rag_ask():
    data = request.get_json() or {}
    query = data.get('query', '').strip()
    category = data.get('category', '').strip()
    if not query:
        return jsonify({'status': 'error', 'message': '질문을 입력하세요.'}), 400

    started_at = datetime.now()
    session_id = session.get('conversation_id', f"SESSION_{uuid.uuid4().hex[:8]}")
    config = {"configurable": {"thread_id": session_id}}

    # 1. ShipAgentState 초기 입력값 설정
    state_input = {
        "user_query": query,
        "messages": [HumanMessage(content=query)],
        "task_type": None,
        "target_categories": [category] if category and category != "전체 카테고리 검색" else [],
        "need_weather": False,
        "extracted_params": {},
        "is_param_complete": None,
        "missing_params": [],
        "search_status": "idle",
        "search_queries": [],
        "retrieved_docs": [],
        "weather_data": None,
        "generated_document": None,
        "final_answer": None,
        "audit_saved": False
    }
    
    result = ship_agent_app.invoke(state_input, config=config)
    completed_at = datetime.now()

    response_answer = result.get('final_answer') or "질의 처리가 완료되었습니다."
    raw_docs = result.get('retrieved_docs', [])

# 2. 프론트엔드 undefined 및 DOC_ID 누락 방지 완전 매핑
    formatted_references = []
    for idx, doc in enumerate(raw_docs, start=1):
        meta = doc.get("metadata", {}) or {}
        
        # 문서 식별자
        chunk_id = (
            meta.get("doc_id") 
            or meta.get("id") 
            or doc.get("doc_id") 
            or doc.get("id") 
            or f"CHROMA_{idx:03d}"
        )
        
        # 문서명
        doc_name = (
            meta.get("document_name") 
            or meta.get("title") 
            or doc.get("document_name") 
            or "사내 기술/안전 규격서"
        )
        
        # 페이지 번호 (페이지 번호 또는 p.1 표기)
        page_val = meta.get("page") or meta.get("page_start") or meta.get("page_number") or meta.get("page_no") or 1
        page_str = f"p.{page_val}"
        
        # ⭕ [핵심 추가] 프론트엔드 rag_chat.html의 ${r.source}와 1:1 매핑
        source_val = meta.get("source") or doc.get("source") or page_str
        
        # 리비전/개정정보
        rev_val = meta.get("revision") or meta.get("rev") or "UNKNOWN"
        
        # 유사도 점수
        sim = meta.get("similarity") or meta.get("similarity_score") or doc.get("similarity_score")
        sim_str = f"{float(sim):.2f}" if sim is not None else "0.85"
        
        # 본문 내용
        txt = doc.get("content") or doc.get("chunk_text") or meta.get("chunk_text") or ""

        # 프론트엔드가 doc.xxx 또는 doc.metadata.xxx 어디로 접근해도 매핑되도록 보강
        enriched_meta = dict(meta)
        enriched_meta.update({
            "doc_id": chunk_id,
            "id": chunk_id,
            "document_id": chunk_id,
            "document_name": doc_name,
            "title": doc_name,
            "source": source_val,         # 👈 추가
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
            # 최상위 키 대응
            "id": chunk_id,
            "doc_id": chunk_id,
            "document_id": chunk_id,
            "category": doc.get("category", "규격"),
            "document_name": doc_name,
            "title": doc_name,
            "source": source_val,         # 👈 프론트엔드가 찾는 핵심 키 (${r.source})
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
            # metadata 객체 대응
            "metadata": enriched_meta
        })
        
    # DB 감사 기록
    exec_row = AgentExecution(
        user_id=session['user_id'],
        work_id=None,
        vision_id=None,
        conversation_id=session_id,
        request_type='일반 질의',
        agent_name='ShipSupervisorAgent',
        user_query=query,
        execution_status='SUCCESS',
        approval_result='PASS',
        current_node='FINISH',
        state_version='2.0',
        input_data=json.dumps(state_input, ensure_ascii=False, default=str),
        state_data=json.dumps({"task_type": result.get("task_type")}, ensure_ascii=False),
        output_data=json.dumps(result.get("generated_document") or {}, ensure_ascii=False),
        response_text=response_answer,
        started_at=started_at,
        completed_at=completed_at
    )
    db.session.add(exec_row)
    db.session.commit()

    return jsonify({
        'status': 'success',
        'execution_id': exec_row.execution_id,
        'answer': response_answer,
        'references': formatted_references,
        'retrieved_docs': formatted_references
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
        upload_folder = current_app.config.get('UPLOAD_FOLDER', './uploads')
        os.makedirs(upload_folder, exist_ok=True)
        img_path = os.path.join(upload_folder, img.filename)
        img.save(img_path)

    # 1. WORK_ORDERS 저장
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

    # 2. 비전 분석
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

    # 3. 에이전트 파이프라인 호출
    session_id = session.get('conversation_id', f"WORK_{order.work_id}")
    config = {"configurable": {"thread_id": session_id}}
    query_composite = f"{work_area}에서 {base_material} {material_thickness}T {process_type} {joint_type} 작업지시서 작성해줘. {request_text}"

    state_input = {
        "user_query": query_composite,
        "messages": [HumanMessage(content=query_composite)],
        "task_type": "doc_generation",
        "target_categories": ["manufacturing_tech", "safety_health"],
        "need_weather": True,
        "extracted_params": {
            "material": base_material,
            "thickness": f"{material_thickness}T",
            "process": process_type,
            "joint_type": joint_type,
            "location": work_area
        },
        "is_param_complete": "yes",
        "missing_params": [],
        "search_status": "idle",
        "search_queries": [],
        "retrieved_docs": [],
        "weather_data": None,
        "vision_data": v_out,  # [개선] 비전 검출 결과(용접선, 장애물)를 에이전트 상태로 명시적 전달
        "generated_document": None,
        "final_answer": None,
        "audit_saved": False
    }

    started_at = datetime.now()
    result = ship_agent_app.invoke(state_input, config=config)
    completed_at = datetime.now()

    generated_doc = result.get("generated_document") or {}
    instruction_text = result.get("final_answer") or "작업지시서가 정상 발행되었습니다."

    # 4. AGENT_EXECUTIONS 저장
    exec_row = AgentExecution(
        user_id=user_id,
        work_id=order.work_id,
        vision_id=v_analysis.vision_id,
        conversation_id=session_id,
        request_type='작업지시',
        agent_name='ShipSupervisorAgent',
        user_query=query_composite,
        execution_status='SUCCESS',
        approval_result='PASS',
        current_node='FINISH',
        state_version='2.0',
        input_data=json.dumps(state_input, ensure_ascii=False, default=str),
        state_data=json.dumps({"params": state_input["extracted_params"]}, ensure_ascii=False),
        output_data=json.dumps(generated_doc, ensure_ascii=False),
        response_text=instruction_text,
        started_at=started_at,
        completed_at=completed_at
    )
    db.session.add(exec_row)
    db.session.commit()

    # 5. WORK_INSTRUCTIONS 저장
    inst_row = WorkInstruction(
        work_id=order.work_id,
        execution_id=exec_row.execution_id,
        version_no=1,
        instruction_text=instruction_text,
        instruction_data=json.dumps(generated_doc, ensure_ascii=False),
        source_references=json.dumps(result.get("retrieved_docs", []), ensure_ascii=False),
        approval_status='PASS',
        instruction_status='FINAL'
    )
    db.session.add(inst_row)

    order.work_status = 'PASS'
    db.session.commit()

    return jsonify({
        'status': 'success',
        'work_id': order.work_id,
        'vision_id': v_analysis.vision_id,
        'execution_id': exec_row.execution_id,
        'instruction_id': inst_row.instruction_id,
        'vision_result': v_out,
        'agent_result': {
            'final_answer': instruction_text,
            'generated_document': generated_doc
        },
        'rag_references': result.get("retrieved_docs", [])
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
        history.append({
            'work_id': o.work_id,
            'username': u.username if u else 'unknown',
            'work_name': o.work_name,
            'block_no': o.block_no,
            'material': f"{o.base_material} ({o.material_thickness}T)",
            'joint_position': f"{o.joint_type} / {o.welding_position}",
            'work_status': o.work_status,
            'created_at': o.created_at.strftime('%H:%M:%S') if o.created_at else ''
        })
    return jsonify({'counts': counts, 'history': history})