"""
[파일 역할] 2개의 분리된 Oracle 스키마(ship_user, ship_erp)에 매핑되는 SQLAlchemy ORM 모델 정의서입니다.
  - Schema 1 (기본 바인딩 `ship_user`): 작업자 계정, 용접 작업요청, Vision 분석/검출 좌표, Agent 실행로그, 최종 작업지시서, RAG 문서 메타데이터
  - Schema 2 (`__bind_key__ = 'erp_db'`, `ship_erp`): 원청/협력사 정보, 지부(야드), 용접 기술자, 용접 로봇 자산, 지부별 실시간 자재 재고
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Identity, Index

db = SQLAlchemy()


# =============================================================================
# [PART 1] 메인 작업·공정·Vision·Agent 관리 스키마 (ship_user) - 7개 테이블
# =============================================================================

class User(db.Model):
    """시스템 로그인 사용자 계정 및 직책/소속 관리 테이블 (USERS)"""
    __tablename__ = 'USERS'

    user_id = db.Column('USER_ID', db.Integer, Identity(start=1), primary_key=True)
    username = db.Column('USERNAME', db.String(50), unique=True, nullable=False)
    password_hash = db.Column('PASSWORD_HASH', db.String(255), nullable=False)
    email = db.Column('EMAIL', db.String(100))
    display_name = db.Column('DISPLAY_NAME', db.String(100))
    # 직책 권한: 'EXEC'(사장/임원), 'MANAGER'(야드관리자), 'WORKER'(현장작업자), 'PARTNER'(협력사관리자)
    role = db.Column('ROLE', db.String(20), default='WORKER')
    # ERP 연동용 소속 정보 (기본값 제공으로 기존 코드 완벽 호환)
    company_name = db.Column('COMPANY_NAME', db.String(100), default='HD한국조선해양')
    company_type = db.Column('COMPANY_TYPE', db.String(20), default='MAIN')  # 'MAIN'(원청) | 'PARTNER'(협력사)
    branch_name = db.Column('BRANCH_NAME', db.String(100), default='울산 1야드')
    department = db.Column('DEPARTMENT', db.String(50), default='선체생산부')
    is_active = db.Column('IS_ACTIVE', db.Boolean, default=True)
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)
    updated_at = db.Column('UPDATED_AT', db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_context_dict(self) -> dict:
        """로그인한 사용자의 소속/권한 정보를 AI-Agent 및 프론트엔드에 주입하기 위한 변환 함수"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "company_name": self.company_name or "HD한국조선해양",
            "company_type": self.company_type or "MAIN",
            "branch_name": self.branch_name or "울산 1야드",
            "department": self.department or "선체생산부",
        }


class WorkOrder(db.Model):
    """선체 블록 용접 작업 요청 및 제원 관리 테이블 (WORK_ORDERS)"""
    __tablename__ = 'WORK_ORDERS'
    __table_args__ = (
        Index('IDX_WORK_ORDERS_STATUS_DATE', 'WORK_STATUS', 'CREATED_AT'),
    )

    work_id = db.Column('WORK_ID', db.Integer, Identity(start=1), primary_key=True)
    user_id = db.Column('USER_ID', db.Integer, db.ForeignKey('USERS.USER_ID'), nullable=False)
    work_name = db.Column('WORK_NAME', db.String(200))
    request_text = db.Column('REQUEST_TEXT', db.Text)
    block_no = db.Column('BLOCK_NO', db.String(50))
    work_area = db.Column('WORK_AREA', db.String(100))  # 지부/야드명 ('울산 1야드' 등)
    process_type = db.Column('PROCESS_TYPE', db.String(50))
    base_material = db.Column('BASE_MATERIAL', db.String(100))
    material_thickness = db.Column('MATERIAL_THICKNESS', db.Float)
    joint_type = db.Column('JOINT_TYPE', db.String(50))
    groove_type = db.Column('GROOVE_TYPE', db.String(50))
    welding_position = db.Column('WELDING_POSITION', db.String(20))
    additional_input = db.Column('ADDITIONAL_INPUT', db.Text)
    work_status = db.Column('WORK_STATUS', db.String(30), default='PENDING')  # PENDING, ANALYZING, APPROVED, COMPLETED
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)
    updated_at = db.Column('UPDATED_AT', db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class VisionAnalysis(db.Model):
    """YOLOv11 비전 프레임 분석 메타데이터 테이블 (VISION_ANALYSES)"""
    __tablename__ = 'VISION_ANALYSES'

    vision_id = db.Column('VISION_ID', db.Integer, Identity(start=1), primary_key=True)
    work_id = db.Column('WORK_ID', db.Integer, db.ForeignKey('WORK_ORDERS.WORK_ID'), nullable=False)
    image_path = db.Column('IMAGE_PATH', db.String(1000))
    model_name = db.Column('MODEL_NAME', db.String(100))
    model_version = db.Column('MODEL_VERSION', db.String(50))
    analysis_status = db.Column('ANALYSIS_STATUS', db.String(20))
    image_width = db.Column('IMAGE_WIDTH', db.Integer)
    image_height = db.Column('IMAGE_HEIGHT', db.Integer)
    result_json = db.Column('RESULT_JSON', db.Text)
    error_message = db.Column('ERROR_MESSAGE', db.Text)
    started_at = db.Column('STARTED_AT', db.DateTime)
    completed_at = db.Column('COMPLETED_AT', db.DateTime)
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)


class VisionDetection(db.Model):
    """비전 모델이 검출한 개별 바운딩 박스(모재, 용접선 궤적, 간섭물) 저장 테이블 (VISION_DETECTIONS)"""
    __tablename__ = 'VISION_DETECTIONS'

    detection_id = db.Column('DETECTION_ID', db.Integer, Identity(start=1), primary_key=True)
    vision_id = db.Column('VISION_ID', db.Integer, db.ForeignKey('VISION_ANALYSES.VISION_ID'), nullable=False)
    class_name = db.Column('CLASS_NAME', db.String(100))
    confidence = db.Column('CONFIDENCE', db.Float)
    coordinate_type = db.Column('COORDINATE_TYPE', db.String(20), default='PIXEL')
    x_min = db.Column('X_MIN', db.Float)
    y_min = db.Column('Y_MIN', db.Float)
    x_max = db.Column('X_MAX', db.Float)
    y_max = db.Column('Y_max', db.Float)
    geometry_data = db.Column('GEOMETRY_DATA', db.Text)  # 용접선 궤적 좌표 배열 및 루트간격(root_gap_mm) JSON
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)


class AgentExecution(db.Model):
    """LangGraph 멀티에이전트 실행 상태 및 입출력 이력 로깅 테이블 (AGENT_EXECUTIONS)"""
    __tablename__ = 'AGENT_EXECUTIONS'

    execution_id = db.Column('EXECUTION_ID', db.Integer, Identity(start=1), primary_key=True)
    user_id = db.Column('USER_ID', db.Integer, db.ForeignKey('USERS.USER_ID'), nullable=False)
    work_id = db.Column('WORK_ID', db.Integer, db.ForeignKey('WORK_ORDERS.WORK_ID'), nullable=True)
    vision_id = db.Column('VISION_ID', db.Integer, db.ForeignKey('VISION_ANALYSES.VISION_ID'), nullable=True)
    conversation_id = db.Column('CONVERSATION_ID', db.String(64))
    request_type = db.Column('REQUEST_TYPE', db.String(30))
    agent_name = db.Column('AGENT_NAME', db.String(100))
    user_query = db.Column('USER_QUERY', db.Text)
    execution_status = db.Column('EXECUTION_STATUS', db.String(20))  # RUNNING, COMPLETED, FAILED
    approval_result = db.Column('APPROVAL_RESULT', db.String(20))    # WAITING, APPROVED, REJECTED
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
    """최종 생성 및 승인된 표준 용접 작업지시서 테이블 (WORK_INSTRUCTIONS)"""
    __tablename__ = 'WORK_INSTRUCTIONS'

    instruction_id = db.Column('INSTRUCTION_ID', db.Integer, Identity(start=1), primary_key=True)
    work_id = db.Column('WORK_ID', db.Integer, db.ForeignKey('WORK_ORDERS.WORK_ID'), nullable=False)
    execution_id = db.Column('EXECUTION_ID', db.Integer, db.ForeignKey('AGENT_EXECUTIONS.EXECUTION_ID'), nullable=False)
    version_no = db.Column('VERSION_NO', db.Integer, default=1)
    instruction_text = db.Column('INSTRUCTION_TEXT', db.Text)
    instruction_data = db.Column('INSTRUCTION_DATA', db.Text)      # 배정된 로봇/작업자/자재 JSON 포함
    source_references = db.Column('SOURCE_REFERENCES', db.Text)
    approval_status = db.Column('APPROVAL_STATUS', db.String(20), default='WAITING')  # WAITING, APPROVED
    instruction_status = db.Column('INSTRUCTION_STATUS', db.String(20), default='DRAFT')
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)
    updated_at = db.Column('UPDATED_AT', db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DocumentMetadata(db.Model):
    """RAG 지식베이스 원본 PDF 문서 메타데이터 및 버전/상태 관리 테이블 (DOCUMENT_METADATA)"""
    __tablename__ = 'DOCUMENT_METADATA'
    __table_args__ = (
        Index('IDX_DOC_META_STATUS_CAT', 'DOCUMENT_STATUS', 'CATEGORY'),
    )

    document_id = db.Column('DOCUMENT_ID', db.Integer, Identity(start=1), primary_key=True)
    document_name = db.Column('DOCUMENT_NAME', db.String(300), nullable=False)
    category = db.Column('CATEGORY', db.String(30), nullable=False)
    source = db.Column('SOURCE', db.String(300))
    revision = db.Column('REVISION', db.String(100))
    # PUBLIC(전체/협력사 열람 가능), INTERNAL(원청 내부용), CONFIDENTIAL(임원/관리자 전용)
    access_level = db.Column('ACCESS_LEVEL', db.String(20), default='PUBLIC')
    file_path = db.Column('FILE_PATH', db.String(1000))
    file_hash = db.Column('FILE_HASH', db.String(64))
    ingest_status = db.Column('INGEST_STATUS', db.String(20), default='COMPLETED')
    document_status = db.Column('DOCUMENT_STATUS', db.String(20), default='ACTIVE')
    uploaded_by = db.Column('UPLOADED_BY', db.Integer, db.ForeignKey('USERS.USER_ID'), nullable=True)
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)
    updated_at = db.Column('UPDATED_AT', db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# =============================================================================
# [PART 2] 조선소 ERP 자원·인력·로봇·협력사 관리 스키마 (ship_erp) - 5개 테이블
# (`__bind_key__ = 'erp_db'`를 통해 두 번째 오라클 DB 계정으로 자동 라우팅됨)
# =============================================================================

class ERPCompany(db.Model):
    """원청 대기업 및 외부 사내외 협력업체 마스터 테이블 (ERP_COMPANIES)"""
    __bind_key__ = 'erp_db'
    __tablename__ = 'ERP_COMPANIES'

    company_id = db.Column('COMPANY_ID', db.Integer, Identity(start=1), primary_key=True)
    company_code = db.Column('COMPANY_CODE', db.String(30), unique=True, nullable=False)
    company_name = db.Column('COMPANY_NAME', db.String(100), nullable=False)
    company_type = db.Column('COMPANY_TYPE', db.String(20), default='PARTNER')  # 'MAIN'(원청), 'PARTNER'(협력사)
    specialty = db.Column('SPECIALTY', db.String(100))                          # 전문 공정 (선체용접, 배관, 블록조립)
    contract_status = db.Column('CONTRACT_STATUS', db.String(20), default='ACTIVE')
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)


class ERPBranch(db.Model):
    """조선소 지부(야드/도크) 위치 및 관할 정보 테이블 (ERP_BRANCHES)"""
    __bind_key__ = 'erp_db'
    __tablename__ = 'ERP_BRANCHES'

    branch_id = db.Column('BRANCH_ID', db.Integer, Identity(start=1), primary_key=True)
    branch_code = db.Column('BRANCH_CODE', db.String(30), unique=True, nullable=False)
    branch_name = db.Column('BRANCH_NAME', db.String(100), nullable=False)      # '울산 1야드', '울산 2야드', '거제 1도크'
    location_city = db.Column('LOCATION_CITY', db.String(50), nullable=False)   # 기상청 MCP 조회용 ('울산', '거제')
    yard_manager = db.Column('YARD_MANAGER', db.String(50))
    is_active = db.Column('IS_ACTIVE', db.String(1), default='Y')


class ERPWorker(db.Model):
    """원청 및 협력업체 소속 용접 기술자 자격·배치 현황 테이블 (ERP_WORKERS)"""
    __bind_key__ = 'erp_db'
    __tablename__ = 'ERP_WORKERS'
    __table_args__ = (
        Index('IDX_ERP_WORKER_BRANCH', 'BRANCH_NAME', 'WORK_STATUS'),
    )

    worker_id = db.Column('WORKER_ID', db.Integer, Identity(start=1), primary_key=True)
    emp_no = db.Column('EMP_NO', db.String(30), unique=True, nullable=False)
    worker_name = db.Column('WORKER_NAME', db.String(50), nullable=False)
    company_id = db.Column('COMPANY_ID', db.Integer, db.ForeignKey('ERP_COMPANIES.COMPANY_ID'), nullable=False)
    branch_name = db.Column('BRANCH_NAME', db.String(100), nullable=False)
    department = db.Column('DEPARTMENT', db.String(50))
    role_title = db.Column('ROLE_TITLE', db.String(50))                         # '명장', '기장', '선임용접사', '기능사'
    welding_license = db.Column('WELDING_LICENSE', db.String(100))              # 'FCAW 6G, TIG 1급, KR선급 인증'
    safety_training_yn = db.Column('SAFETY_TRAINING_YN', db.String(1), default='Y')
    work_status = db.Column('WORK_STATUS', db.String(20), default='AVAILABLE')  # 'AVAILABLE'(대기), 'WORKING'(투입중), 'OFF'
    current_block = db.Column('CURRENT_BLOCK', db.String(50))
    updated_at = db.Column('UPDATED_AT', db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ERPWeldingRobot(db.Model):
    """자동화 용접 로봇 기종·배치 지부·가동 상태 관리 테이블 (ERP_WELDING_ROBOTS)"""
    __bind_key__ = 'erp_db'
    __tablename__ = 'ERP_WELDING_ROBOTS'
    __table_args__ = (
        Index('IDX_ERP_ROBOT_BRANCH', 'BRANCH_NAME', 'ROBOT_STATUS'),
    )

    robot_id = db.Column('ROBOT_ID', db.Integer, Identity(start=1), primary_key=True)
    robot_serial = db.Column('ROBOT_SERIAL', db.String(50), unique=True, nullable=False)
    robot_name = db.Column('ROBOT_NAME', db.String(100), nullable=False)
    model_spec = db.Column('MODEL_SPEC', db.String(100))
    branch_name = db.Column('BRANCH_NAME', db.String(100), nullable=False)
    current_block = db.Column('CURRENT_BLOCK', db.String(50))
    robot_status = db.Column('ROBOT_STATUS', db.String(20), default='IDLE')     # 'IDLE'(대기), 'RUNNING'(작업중), 'MAINTENANCE'(정비중)
    assigned_worker_id = db.Column('ASSIGNED_WORKER_ID', db.Integer, nullable=True)
    battery_or_health = db.Column('BATTERY_OR_HEALTH', db.Float, default=100.0)
    updated_at = db.Column('UPDATED_AT', db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ERPBranchInventory(db.Model):
    """지부별 실시간 용접 자재(와이어, 모재 강재, 보호가스) 재고 테이블 (ERP_BRANCH_INVENTORY)"""
    __bind_key__ = 'erp_db'
    __tablename__ = 'ERP_BRANCH_INVENTORY'
    __table_args__ = (
        Index('IDX_ERP_INV_BRANCH', 'BRANCH_NAME', 'MATERIAL_CATEGORY'),
    )

    inventory_id = db.Column('INVENTORY_ID', db.Integer, Identity(start=1), primary_key=True)
    branch_name = db.Column('BRANCH_NAME', db.String(100), nullable=False)
    material_code = db.Column('MATERIAL_CODE', db.String(50), nullable=False)
    material_category = db.Column('MATERIAL_CATEGORY', db.String(50), nullable=False)  # 'WIRE', 'STEEL', 'GAS'
    material_name = db.Column('MATERIAL_NAME', db.String(150), nullable=False)
    specification = db.Column('SPECIFICATION', db.String(150))
    current_qty = db.Column('CURRENT_QTY', db.Float, default=0.0)
    unit = db.Column('UNIT', db.String(20), nullable=False)                            # 'ROLL', 'SHEET', 'CYLINDER'
    min_safety_qty = db.Column('MIN_SAFETY_QTY', db.Float, default=10.0)
    updated_at = db.Column('UPDATED_AT', db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class VisionStreamLog(db.Model):
    __tablename__ = 'VISION_STREAM_LOGS'
    stream_id = db.Column('STREAM_ID', db.Integer, Identity(start=1), primary_key=True)
    vision_id = db.Column('VISION_ID', db.Integer, nullable=False)
    frame_seq = db.Column('FRAME_SEQ', db.Integer, nullable=False)
    event_type = db.Column('EVENT_TYPE', db.String(50), default='TRACKING')
    torch_x = db.Column('TORCH_X', db.Float)
    torch_y = db.Column('TORCH_Y', db.Float)
    event_payload = db.Column('EVENT_PAYLOAD', db.Text)
    captured_at = db.Column('CAPTURED_AT', db.DateTime, default=datetime.utcnow)


class HumanApprovalLog(db.Model):
    __tablename__ = 'HUMAN_APPROVAL_LOGS'
    approval_id = db.Column('APPROVAL_ID', db.Integer, Identity(start=1), primary_key=True)
    work_id = db.Column('WORK_ID', db.Integer, nullable=False)
    instruction_id = db.Column('INSTRUCTION_ID', db.Integer, nullable=False)
    approver_id = db.Column('APPROVER_ID', db.Integer, nullable=False)
    approval_decision = db.Column('APPROVAL_DECISION', db.String(20), nullable=False)
    override_params = db.Column('OVERRIDE_PARAMS', db.Text)
    reject_reason = db.Column('REJECT_REASON', db.String(1000))
    action_timestamp = db.Column('ACTION_TIMESTAMP', db.DateTime, default=datetime.utcnow)