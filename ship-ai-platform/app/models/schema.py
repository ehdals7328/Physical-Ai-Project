"""
app/models/schema.py
조선소 Physical AI + 통합 ERP 다중 데이터베이스 스키마 정의 (총 16개 테이블).
- ship_user (9개): USERS, WORK_ORDERS, VISION_ANALYSES, VISION_DETECTIONS,
                   AGENT_EXECUTIONS, WORK_INSTRUCTIONS, DOCUMENT_METADATA,
                   VISION_STREAM_LOGS, HUMAN_APPROVAL_LOGS
- ship_erp (7개):  ERP_COMPANIES, ERP_BRANCHES, ERP_WORKERS, ERP_WELDING_ROBOTS,
                   ERP_BRANCH_INVENTORY, ERP_EQUIPMENT_MAINTENANCE, ERP_MATERIAL_TRANSACTIONS
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Identity

db = SQLAlchemy()

# -----------------------------------------------------------------------------
# [DB 1: ship_user] 공정 / Vision / Agent 이력 관리 도메인 (9개 테이블)
# -----------------------------------------------------------------------------
class User(db.Model):
    """사용자 계정 및 역할 관리 테이블 (USERS)."""
    __tablename__ = 'USERS'
    user_id = db.Column('USER_ID', db.Integer, Identity(start=1), primary_key=True)
    username = db.Column('USERNAME', db.String(50), unique=True, nullable=False)
    password_hash = db.Column('PASSWORD_HASH', db.String(255), nullable=False)
    email = db.Column('EMAIL', db.String(100))
    display_name = db.Column('DISPLAY_NAME', db.String(100))
    role = db.Column('ROLE', db.String(20), default='USER')
    is_active = db.Column('IS_ACTIVE', db.String(10), default='Y')
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)
    updated_at = db.Column('UPDATED_AT', db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class WorkOrder(db.Model):
    """현장 작업 요청 및 모재 사양 관리 테이블 (WORK_ORDERS)."""
    __tablename__ = 'WORK_ORDERS'
    work_id = db.Column('WORK_ID', db.Integer, Identity(start=1), primary_key=True)
    user_id = db.Column('USER_ID', db.Integer, db.ForeignKey('USERS.USER_ID'), nullable=False)
    work_name = db.Column('WORK_NAME', db.String(200))
    request_text = db.Column('REQUEST_TEXT', db.Text)
    block_no = db.Column('BLOCK_NO', db.String(50))
    work_area = db.Column('WORK_AREA', db.String(100))
    process_type = db.Column('PROCESS_TYPE', db.String(50))
    base_material = db.Column('BASE_MATERIAL', db.String(100))
    material_thickness = db.Column('MATERIAL_THICKNESS', db.Float)
    joint_type = db.Column('JOINT_TYPE', db.String(50))
    groove_type = db.Column('GROOVE_TYPE', db.String(50))
    welding_position = db.Column('WELDING_POSITION', db.String(20), default='1G')
    additional_input = db.Column('ADDITIONAL_INPUT', db.Text)
    work_status = db.Column('WORK_STATUS', db.String(30), default='PENDING')
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)
    updated_at = db.Column('UPDATED_AT', db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class VisionAnalysis(db.Model):
    """비전 검사 마스터 테이블 (VISION_ANALYSES)."""
    __tablename__ = 'VISION_ANALYSES'
    vision_id = db.Column('VISION_ID', db.Integer, Identity(start=1), primary_key=True)
    work_id = db.Column('WORK_ID', db.Integer, db.ForeignKey('WORK_ORDERS.WORK_ID'), nullable=False)
    image_path = db.Column('IMAGE_PATH', db.String(1000))
    model_name = db.Column('MODEL_NAME', db.String(100), default='YOLOv11s')
    model_version = db.Column('MODEL_VERSION', db.String(50), default='v2.4')
    analysis_status = db.Column('ANALYSIS_STATUS', db.String(20), default='COMPLETED')
    image_width = db.Column('IMAGE_WIDTH', db.Integer)
    image_height = db.Column('IMAGE_HEIGHT', db.Integer)
    result_json = db.Column('RESULT_JSON', db.Text)
    error_message = db.Column('ERROR_MESSAGE', db.Text)
    started_at = db.Column('STARTED_AT', db.DateTime)
    completed_at = db.Column('COMPLETED_AT', db.DateTime)
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)

class VisionDetection(db.Model):
    """비전 검출 객체 상세 테이블 (VISION_DETECTIONS)."""
    __tablename__ = 'VISION_DETECTIONS'
    detection_id = db.Column('DETECTION_ID', db.Integer, Identity(start=1), primary_key=True)
    vision_id = db.Column('VISION_ID', db.Integer, db.ForeignKey('VISION_ANALYSES.VISION_ID'), nullable=False)
    class_name = db.Column('CLASS_NAME', db.String(100))
    confidence = db.Column('CONFIDENCE', db.Float)
    coordinate_type = db.Column('COORDINATE_TYPE', db.String(20), default='PIXEL')
    x_min = db.Column('X_MIN', db.Float)
    y_min = db.Column('Y_MIN', db.Float)
    x_max = db.Column('X_MAX', db.Float)
    y_max = db.Column('Y_MAX', db.Float)
    geometry_data = db.Column('GEOMETRY_DATA', db.Text)
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)

class AgentExecution(db.Model):
    """에이전트 실행 및 상태 머신 로그 테이블 (AGENT_EXECUTIONS)."""
    __tablename__ = 'AGENT_EXECUTIONS'
    execution_id = db.Column('EXECUTION_ID', db.Integer, Identity(start=1), primary_key=True)
    user_id = db.Column('USER_ID', db.Integer, db.ForeignKey('USERS.USER_ID'), nullable=False)
    work_id = db.Column('WORK_ID', db.Integer, db.ForeignKey('WORK_ORDERS.WORK_ID'), nullable=True)
    vision_id = db.Column('VISION_ID', db.Integer, db.ForeignKey('VISION_ANALYSES.VISION_ID'), nullable=True)
    conversation_id = db.Column('CONVERSATION_ID', db.String(64))
    request_type = db.Column('REQUEST_TYPE', db.String(30))
    agent_name = db.Column('AGENT_NAME', db.String(100), default='ShipSupervisorAgent')
    user_query = db.Column('USER_QUERY', db.Text)
    execution_status = db.Column('EXECUTION_STATUS', db.String(20), default='RUNNING')
    approval_result = db.Column('APPROVAL_RESULT', db.String(20), default='PASS')
    current_node = db.Column('CURRENT_NODE', db.String(100))
    state_version = db.Column('STATE_VERSION', db.Integer, default=1)
    input_data = db.Column('INPUT_DATA', db.Text)
    state_data = db.Column('STATE_DATA', db.Text)
    output_data = db.Column('OUTPUT_DATA', db.Text)
    response_text = db.Column('RESPONSE_TEXT', db.Text)
    error_message = db.Column('ERROR_MESSAGE', db.Text)
    started_at = db.Column('STARTED_AT', db.DateTime)
    completed_at = db.Column('COMPLETED_AT', db.DateTime)
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)
    updated_at = db.Column('UPDATED_AT', db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class WorkInstruction(db.Model):
    """최종 표준 작업지시서 본문 테이블 (WORK_INSTRUCTIONS)."""
    __tablename__ = 'WORK_INSTRUCTIONS'
    instruction_id = db.Column('INSTRUCTION_ID', db.Integer, Identity(start=1), primary_key=True)
    work_id = db.Column('WORK_ID', db.Integer, db.ForeignKey('WORK_ORDERS.WORK_ID'), nullable=False)
    execution_id = db.Column('EXECUTION_ID', db.Integer, db.ForeignKey('AGENT_EXECUTIONS.EXECUTION_ID'), nullable=False)
    version_no = db.Column('VERSION_NO', db.Integer, default=1)
    instruction_text = db.Column('INSTRUCTION_TEXT', db.Text)
    instruction_data = db.Column('INSTRUCTION_DATA', db.Text)
    source_references = db.Column('SOURCE_REFERENCES', db.Text)
    approval_status = db.Column('APPROVAL_STATUS', db.String(20), default='PASS')
    instruction_status = db.Column('INSTRUCTION_STATUS', db.String(20), default='FINAL')
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)
    updated_at = db.Column('UPDATED_AT', db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DocumentMetadata(db.Model):
    """규격/사내문서 PDF 원본 카탈로그 테이블 (DOCUMENT_METADATA)."""
    __tablename__ = 'DOCUMENT_METADATA'
    document_id = db.Column('DOCUMENT_ID', db.Integer, Identity(start=1), primary_key=True)
    document_name = db.Column('DOCUMENT_NAME', db.String(300), nullable=False)
    category = db.Column('CATEGORY', db.String(50), nullable=False)
    source = db.Column('SOURCE', db.String(300))
    revision = db.Column('REVISION', db.String(100))
    access_level = db.Column('ACCESS_LEVEL', db.String(20), default='PUBLIC')
    file_path = db.Column('FILE_PATH', db.String(1000))
    file_hash = db.Column('FILE_HASH', db.String(64))
    ingest_status = db.Column('INGEST_STATUS', db.String(20), default='COMPLETED')
    document_status = db.Column('DOCUMENT_STATUS', db.String(20), default='ACTIVE')
    uploaded_by = db.Column('UPLOADED_BY', db.Integer, db.ForeignKey('USERS.USER_ID'), nullable=True)
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)
    updated_at = db.Column('UPDATED_AT', db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class VisionStreamLog(db.Model):
    """시계열 실시간 용접 토치 궤적 로그 테이블 (VISION_STREAM_LOGS)."""
    __tablename__ = 'VISION_STREAM_LOGS'
    stream_id = db.Column('STREAM_ID', db.Integer, Identity(start=1), primary_key=True)
    vision_id = db.Column('VISION_ID', db.Integer, db.ForeignKey('VISION_ANALYSES.VISION_ID'), nullable=False)
    frame_seq = db.Column('FRAME_SEQ', db.Integer, nullable=False)
    event_type = db.Column('EVENT_TYPE', db.String(50), default='TRACKING')
    torch_x = db.Column('TORCH_X', db.Float)
    torch_y = db.Column('TORCH_Y', db.Float)
    event_payload = db.Column('EVENT_PAYLOAD', db.Text)
    captured_at = db.Column('CAPTURED_AT', db.DateTime, default=datetime.utcnow)

class HumanApprovalLog(db.Model):
    """HITL 관리자 안전 승인 및 개입 로그 테이블 (HUMAN_APPROVAL_LOGS)."""
    __tablename__ = 'HUMAN_APPROVAL_LOGS'
    approval_id = db.Column('APPROVAL_ID', db.Integer, Identity(start=1), primary_key=True)
    work_id = db.Column('WORK_ID', db.Integer, db.ForeignKey('WORK_ORDERS.WORK_ID'), nullable=False)
    instruction_id = db.Column('INSTRUCTION_ID', db.Integer, db.ForeignKey('WORK_INSTRUCTIONS.INSTRUCTION_ID'), nullable=False)
    approver_id = db.Column('APPROVER_ID', db.Integer, nullable=False)
    approval_decision = db.Column('APPROVAL_DECISION', db.String(20), nullable=False)
    override_params = db.Column('OVERRIDE_PARAMS', db.Text)
    reject_reason = db.Column('REJECT_REASON', db.String(1000))
    action_timestamp = db.Column('ACTION_TIMESTAMP', db.DateTime, default=datetime.utcnow)

# -----------------------------------------------------------------------------
# [DB 2: ship_erp] 대기업 인사 / 협력사 / 로봇 / 자재 ERP 도메인 (7개 테이블)
# -----------------------------------------------------------------------------
class ErpCompany(db.Model):
    """원청 및 사외 협력업체 마스터 테이블 (ERP_COMPANIES)."""
    __bind_key__ = 'erp_db'
    __tablename__ = 'ERP_COMPANIES'
    company_id = db.Column('COMPANY_ID', db.Integer, Identity(start=1), primary_key=True)
    company_name = db.Column('COMPANY_NAME', db.String(100), nullable=False)
    business_number = db.Column('BUSINESS_NUMBER', db.String(50))
    company_type = db.Column('COMPANY_TYPE', db.String(20), default='PARTNER')
    address = db.Column('ADDRESS', db.String(300))
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)

class ErpBranch(db.Model):
    """사업장 및 야드/도크 구분 테이블 (ERP_BRANCHES)."""
    __bind_key__ = 'erp_db'
    __tablename__ = 'ERP_BRANCHES'
    branch_id = db.Column('BRANCH_ID', db.Integer, Identity(start=1), primary_key=True)
    branch_name = db.Column('BRANCH_NAME', db.String(100), nullable=False)
    company_id = db.Column('COMPANY_ID', db.Integer, nullable=False)
    location_desc = db.Column('LOCATION_DESC', db.String(300))
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)

class ErpWorker(db.Model):
    """현장 용접 기술자 및 숙련도 등급 테이블 (ERP_WORKERS)."""
    __bind_key__ = 'erp_db'
    __tablename__ = 'ERP_WORKERS'
    worker_id = db.Column('WORKER_ID', db.Integer, Identity(start=1), primary_key=True)
    company_id = db.Column('COMPANY_ID', db.Integer, nullable=False)
    branch_id = db.Column('BRANCH_ID', db.Integer, nullable=False)
    worker_name = db.Column('WORKER_NAME', db.String(50), nullable=False)
    worker_code = db.Column('WORKER_CODE', db.String(50), unique=True)
    skill_grade = db.Column('SKILL_GRADE', db.String(100))
    status = db.Column('STATUS', db.String(20), default='AVAILABLE')
    phone = db.Column('PHONE', db.String(50))
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)

class ErpWeldingRobot(db.Model):
    """자동화 용접 로봇 설비 마스터 테이블 (ERP_WELDING_ROBOTS)."""
    __bind_key__ = 'erp_db'
    __tablename__ = 'ERP_WELDING_ROBOTS'
    robot_id = db.Column('ROBOT_ID', db.Integer, Identity(start=1), primary_key=True)
    robot_serial = db.Column('ROBOT_SERIAL', db.String(50), unique=True, nullable=False)
    branch_id = db.Column('BRANCH_ID', db.Integer, nullable=False)
    model_name = db.Column('MODEL_NAME', db.String(100))
    payload_kg = db.Column('PAYLOAD_KG', db.Float)
    axis_count = db.Column('AXIS_COUNT', db.Integer, default=6)
    status = db.Column('STATUS', db.String(20), default='IDLE')
    last_inspection_date = db.Column('LAST_INSPECTION_DATE', db.DateTime)
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)

class ErpBranchInventory(db.Model):
    """지부별 실시간 자재 및 소모품 재고 테이블 (ERP_BRANCH_INVENTORY)."""
    __bind_key__ = 'erp_db'
    __tablename__ = 'ERP_BRANCH_INVENTORY'
    inventory_id = db.Column('INVENTORY_ID', db.Integer, Identity(start=1), primary_key=True)
    branch_id = db.Column('BRANCH_ID', db.Integer, nullable=True)
    branch_name = db.Column('BRANCH_NAME', db.String(100), nullable=False)
    material_code = db.Column('MATERIAL_CODE', db.String(50), nullable=False)
    material_category = db.Column('MATERIAL_CATEGORY', db.String(50))
    material_name = db.Column('MATERIAL_NAME', db.String(200), nullable=False)
    specification = db.Column('SPECIFICATION', db.String(200))
    current_qty = db.Column('CURRENT_QTY', db.Float, default=0.0)
    unit = db.Column('UNIT', db.String(30))
    min_safety_qty = db.Column('MIN_SAFETY_QTY', db.Float, default=10.0)
    updated_at = db.Column('UPDATED_AT', db.DateTime, default=datetime.utcnow)

class ErpEquipmentMaintenance(db.Model):
    """로봇 설비 정비 및 점검 이력 테이블 (ERP_EQUIPMENT_MAINTENANCE)."""
    __bind_key__ = 'erp_db'
    __tablename__ = 'ERP_EQUIPMENT_MAINTENANCE'
    maint_id = db.Column('MAINT_ID', db.Integer, Identity(start=1), primary_key=True)
    robot_id = db.Column('ROBOT_ID', db.Integer, nullable=False)
    robot_serial = db.Column('ROBOT_SERIAL', db.String(50), nullable=False)
    maint_type = db.Column('MAINT_TYPE', db.String(50), default='PREVENTIVE')
    description = db.Column('DESCRIPTION', db.String(500))
    technician_name = db.Column('TECHNICIAN_NAME', db.String(100))
    start_date = db.Column('START_DATE', db.DateTime, default=datetime.utcnow)
    end_date = db.Column('END_DATE', db.DateTime)
    status = db.Column('STATUS', db.String(20), default='COMPLETED')

class ErpMaterialTransaction(db.Model):
    """자재 입출고 수불 대장 테이블 (ERP_MATERIAL_TRANSACTIONS)."""
    __bind_key__ = 'erp_db'
    __tablename__ = 'ERP_MATERIAL_TRANSACTIONS'
    tx_id = db.Column('TX_ID', db.Integer, Identity(start=1), primary_key=True)
    inventory_id = db.Column('INVENTORY_ID', db.Integer, nullable=False)
    material_code = db.Column('MATERIAL_CODE', db.String(50), nullable=False)
    tx_type = db.Column('TX_TYPE', db.String(30), nullable=False)
    change_qty = db.Column('CHANGE_QTY', db.Float, nullable=False)
    balance_qty = db.Column('BALANCE_QTY', db.Float, nullable=False)
    work_id = db.Column('WORK_ID', db.Integer, nullable=True)
    handler_name = db.Column('HANDLER_NAME', db.String(100))
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)
