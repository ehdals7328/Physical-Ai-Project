"""
app/routes/api_routes.py
다중 DB(Oracle 16개 테이블) + 다중 RAG + 관리자 문서 관리 통합 Blueprint.
"""
import os
import json
import uuid
import hashlib
from datetime import datetime
from functools import wraps
from flask import Blueprint, request, jsonify, render_template, session, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from app.models.schema import (
    db, User, WorkOrder, VisionAnalysis, VisionDetection,
    AgentExecution, WorkInstruction, DocumentMetadata,
    VisionStreamLog, HumanApprovalLog,
    ErpCompany, ErpBranch, ErpWorker, ErpWeldingRobot,
    ErpBranchInventory, ErpEquipmentMaintenance, ErpMaterialTransaction
)
from modules.knowledge.retriever import get_knowledge_store
from modules.vision.detector import analyze_welding_frame
from modules.agent.supervisor import ship_agent_app

api_bp = Blueprint('api', __name__)

def login_required(f):
    @wraps(f)
    def dec(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'status': 'unauthorized', 'message': '로그인이 필요합니다.'}), 401
        return f(*args, **kwargs)
    return dec

def admin_required(f):
    @wraps(f)
    def dec(*args, **kwargs):
        if session.get('role') != 'ADMIN':
            return jsonify({'status': 'forbidden', 'message': '관리자(ADMIN) 권한이 필요합니다.'}), 403
        return f(*args, **kwargs)
    return dec

@api_bp.route('/')
def index(): return render_template('index.html')

@api_bp.route('/login')
def login_page(): return render_template('auth.html')

@api_bp.route('/api/auth/login', methods=['POST'])
def auth_login():
    data = request.get_json() or {}
    user = User.query.filter_by(username=data.get('username', '').strip()).first()
    if not user or not check_password_hash(user.password_hash, data.get('password', '').strip()):
        return jsonify({'status': 'error', 'message': '계정 정보가 일치하지 않습니다.'}), 401
    session.permanent = True
    session['user_id'] = user.user_id
    session['username'] = user.username
    session['display_name'] = user.display_name
    session['role'] = user.role
    return jsonify({'status': 'success', 'user': {'username': user.username, 'display_name': user.display_name, 'role': user.role}})

@api_bp.route('/api/auth/logout', methods=['POST'])
def auth_logout():
    session.clear()
    return jsonify({'status': 'success'})

@api_bp.route('/api/auth/session-check')
def auth_check():
    return jsonify({
        'logged_in': 'user_id' in session,
        'username': session.get('username'),
        'display_name': session.get('display_name'),
        'role': session.get('role')
    })

@api_bp.route('/api/system/status')
def system_status():
    store = get_knowledge_store()
    return jsonify({
        'ship_user_counts': {
            'USERS': User.query.count(), 'WORK_ORDERS': WorkOrder.query.count(),
            'VISION_ANALYSES': VisionAnalysis.query.count(), 'VISION_DETECTIONS': VisionDetection.query.count(),
            'AGENT_EXECUTIONS': AgentExecution.query.count(), 'WORK_INSTRUCTIONS': WorkInstruction.query.count(),
            'DOCUMENT_METADATA': DocumentMetadata.query.count(), 'VISION_STREAM_LOGS': VisionStreamLog.query.count(),
            'HUMAN_APPROVAL_LOGS': HumanApprovalLog.query.count(),
        },
        'ship_erp_counts': {
            'ERP_COMPANIES': ErpCompany.query.count(), 'ERP_BRANCHES': ErpBranch.query.count(),
            'ERP_WORKERS': ErpWorker.query.count(), 'ERP_WELDING_ROBOTS': ErpWeldingRobot.query.count(),
            'ERP_BRANCH_INVENTORY': ErpBranchInventory.query.count(), 'ERP_EQUIPMENT_MAINT': ErpEquipmentMaintenance.query.count(),
            'ERP_MATERIAL_TX': ErpMaterialTransaction.query.count(),
        },
        'rag_counts': {
            'PUBLIC_TECH_RAG': store.count(current_index_only=False),
            'INTERNAL_RULE_RAG': store.count_internal(),
        }
    })

@api_bp.route('/api/admin/internal-docs', methods=['GET'])
@login_required
def list_internal_docs():
    docs = DocumentMetadata.query.filter_by(access_level='INTERNAL', document_status='ACTIVE').order_by(DocumentMetadata.document_id.desc()).all()
    return jsonify({'status': 'success', 'documents': [{'document_id': d.document_id, 'document_name': d.document_name, 'source': d.source, 'created_at': d.created_at.strftime('%Y-%m-%d %H:%M') if d.created_at else ''} for d in docs]})

@api_bp.route('/api/admin/internal-docs/upload', methods=['POST'])
@login_required
@admin_required
def upload_internal_doc():
    if 'pdf_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'PDF 파일이 필요합니다.'}), 400
    file = request.files['pdf_file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'status': 'error', 'message': 'PDF 파일만 허용됩니다.'}), 400

    filename = secure_filename(file.filename) or f"internal_{uuid.uuid4().hex[:6]}.pdf"
    folder = current_app.config['INTERNAL_PDF_FOLDER']
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, filename)
    file.save(save_path)

    with open(save_path, 'rb') as f:
        f_hash = hashlib.sha256(f.read()).hexdigest()

    store = get_knowledge_store()
    chunks = []
    try:
        import fitz
        with fitz.open(save_path) as doc:
            for p_idx, page in enumerate(doc):
                text = page.get_text('text').strip()
                if len(text) >= 30:
                    chunks.append({'text': text, 'page_start': p_idx + 1, 'page_end': p_idx + 1, 'chunk_type': 'TEXT'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'PDF 텍스트 추출 실패: {e}'}), 500

    if not chunks:
        return jsonify({'status': 'error', 'message': 'PDF 내부에 유효한 텍스트가 없습니다.'}), 400

    doc_meta = DocumentMetadata(
        document_name=filename, category='사내보안내규', source=session.get('display_name', '관리자'),
        access_level='INTERNAL', file_path=save_path, file_hash=f_hash, ingest_status='COMPLETED',
        document_status='ACTIVE', uploaded_by=session['user_id']
    )
    db.session.add(doc_meta)
    db.session.commit()

    store.add_internal_document_chunks(document_id=doc_meta.document_id, document_name=filename, category='사내보안내규', source=doc_meta.source, chunks=chunks)
    return jsonify({'status': 'success', 'message': f'사내 규정 [{filename}] 등록 및 임베딩 완료 ({len(chunks)}개 청크)'})

@api_bp.route('/api/admin/internal-docs/delete/<int:doc_id>', methods=['DELETE', 'POST'])
@login_required
@admin_required
def delete_internal_doc(doc_id):
    meta = db.session.get(DocumentMetadata, doc_id)
    if not meta: return jsonify({'status': 'error', 'message': '문서를 찾을 수 없습니다.'}), 404
    meta.document_status = 'DELETED'
    db.session.commit()
    store = get_knowledge_store()
    store.delete_internal_document_chunks(doc_id)
    return jsonify({'status': 'success', 'message': f'사내 규정 #{doc_id} 영구 삭제 완료.'})

@api_bp.route('/api/rag/ask', methods=['POST'])
@login_required
def rag_ask():
    data = request.get_json() or {}
    query = data.get('query', '').strip()
    rag_mode = data.get('rag_mode', 'ALL')
    if not query: return jsonify({'status': 'error', 'message': '질의문을 입력하세요.'}), 400

    store = get_knowledge_store()
    results = store.search(query=query, top_k=4, rag_mode=rag_mode)
    answer_text = f"[다중 RAG 검색 완료] 총 {len(results)}건의 검증된 규격 청크를 인출하였습니다." if results else "💡 일치하는 조선소 규격서 또는 사내 안전 내규를 찾지 못했습니다."
    return jsonify({'status': 'success', 'answer': answer_text, 'references': results})

@api_bp.route('/api/erp/resources')
def get_erp_resources():
    branch_name = request.args.get('branch', '울산 1야드')
    branch = ErpBranch.query.filter(ErpBranch.branch_name.like(f"%{branch_name[:2]}%")).first() or ErpBranch.query.first()
    target_id = branch.branch_id if branch else 1
    target_name = branch.branch_name if branch else branch_name

    robots = ErpWeldingRobot.query.filter_by(branch_id=target_id).all()
    workers = ErpWorker.query.filter_by(branch_id=target_id).all()
    inventory = ErpBranchInventory.query.filter_by(branch_name=target_name).all()
    branches = [b.branch_name for b in ErpBranch.query.order_by(ErpBranch.branch_id).all()]

    return jsonify({
        'branch_name': target_name,
        'all_branches': branches,
        'available_robots': [{'robot_id': r.robot_id, 'serial': r.robot_serial, 'model': r.model_name, 'status': r.status} for r in robots],
        'available_workers': [{'worker_id': w.worker_id, 'name': w.worker_name, 'skill': w.skill_grade, 'status': w.status} for w in workers],
        'inventory': [{'inventory_id': i.inventory_id, 'name': i.material_name, 'qty': i.current_qty, 'unit': i.unit, 'is_low': i.current_qty < i.min_safety_qty} for i in inventory]
    })

@api_bp.route('/api/work/analyze', methods=['POST'])
@login_required
def work_analyze():
    order = WorkOrder(
        user_id=session['user_id'],
        work_name=f"{request.form.get('work_area', '울산 1야드')} {request.form.get('block_no', 'B-402')} 용접",
        request_text=request.form.get('request_text', ''),
        block_no=request.form.get('block_no', 'B-402'),
        work_area=request.form.get('work_area', '울산 1야드'),
        process_type=request.form.get('process_type', 'FCAW'),
        base_material=request.form.get('base_material', 'AH36'),
        material_thickness=float(request.form.get('material_thickness', 16.0)),
        joint_type=request.form.get('joint_type', 'BUTT'),
        work_status='PROCESSING'
    )
    db.session.add(order)
    db.session.commit()

    sim_obs = request.form.get('simulate_obstacle') == 'true'
    v_out = analyze_welding_frame('', include_defect=sim_obs)

    agent_state = {
        'user_query': order.work_name,
        'extracted_params': {'material': order.base_material, 'thickness': f"{order.material_thickness}T", 'process': order.process_type},
        'vision_data': v_out
    }
    agent_res = ship_agent_app.invoke(agent_state)

    order.work_status = 'HOLD' if sim_obs else 'PASS'
    db.session.commit()

    return jsonify({
        'status': 'success',
        'work_id': order.work_id,
        'approval_result': order.work_status,
        'instruction_text': agent_res.get('final_answer')
    })
