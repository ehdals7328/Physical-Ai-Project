"""조선/해양 자율 용접 플랫폼 RDB 스키마 모델 정의 (Oracle Database 18c XE 호환).

[시스템 개요 및 엔티티 관계(ERD) 흐름]
1. USERS: 시스템 작업자, 관리자 계정 및 역할 관리.
2. WORK_ORDERS: 블록 번호, 모재, 판두께, 홈 형상 등 선체 용접 작업 지시서(User 1:N).
3. VISION_ANALYSES: YOLO 비전 검사 메타데이터 및 분석 파이프라인 상태(WorkOrder 1:N).
4. VISION_DETECTIONS: 용접선(Seam), 부재 경계 박스 좌표 및 신뢰도(VisionAnalysis 1:N).
5. AGENT_EXECUTIONS: LangGraph 멀티 에이전트 상태, 노드 전이 로그(User/WorkOrder/Vision N:1).
6. WORK_INSTRUCTIONS: RAG 검증 및 에이전트 승인을 거쳐 생성된 최종 용접 작업 지시서.
7. DOCUMENT_METADATA: RAG 지식 베이스(선급 규정, WPS, 안전 표준) 원본 문서 메타데이터.

[데이터 파이프라인 계층 구조]
   USERS
     │
     ├──▶ WORK_ORDERS ───▶ VISION_ANALYSES ───▶ VISION_DETECTIONS
     │          │                 │
     │          └────────┬────────┘
     │                   ▼
     ├──────────▶ AGENT_EXECUTIONS ──▶ WORK_INSTRUCTIONS
     │                   ▲
     ▼                   │ (RAG 참조)
DOCUMENT_METADATA ───────┘
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Identity

db = SQLAlchemy()

# 1. USERS (9개 열)
class User(db.Model):
    __tablename__ = 'USERS'
    user_id = db.Column('USER_ID', db.Integer, Identity(start=1), primary_key=True)
    username = db.Column('USERNAME', db.String(50), unique=True, nullable=False)
    password_hash = db.Column('PASSWORD_HASH', db.String(255), nullable=False)
    email = db.Column('EMAIL', db.String(100))
    display_name = db.Column('DISPLAY_NAME', db.String(100))
    role = db.Column('ROLE', db.String(20), default='USER')
    is_active = db.Column('IS_ACTIVE', db.String(1), default='Y')
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)
    updated_at = db.Column('UPDATED_AT', db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# 2. WORK_ORDERS (16개 열)
class WorkOrder(db.Model):
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
    welding_position = db.Column('WELDING_POSITION', db.String(20))
    additional_input = db.Column('ADDITIONAL_INPUT', db.Text)
    work_status = db.Column('WORK_STATUS', db.String(30), default='PENDING')
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)
    updated_at = db.Column('UPDATED_AT', db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# 3. VISION_ANALYSES (13개 열)
class VisionAnalysis(db.Model):
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

# 4. VISION_DETECTIONS (11개 열)
class VisionDetection(db.Model):
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

# 5. AGENT_EXECUTIONS (22개 열)
class AgentExecution(db.Model):
    __tablename__ = 'AGENT_EXECUTIONS'
    execution_id = db.Column('EXECUTION_ID', db.Integer, Identity(start=1), primary_key=True)
    user_id = db.Column('USER_ID', db.Integer, db.ForeignKey('USERS.USER_ID'), nullable=False)
    work_id = db.Column('WORK_ID', db.Integer, db.ForeignKey('WORK_ORDERS.WORK_ID'), nullable=True)
    vision_id = db.Column('VISION_ID', db.Integer, db.ForeignKey('VISION_ANALYSES.VISION_ID'), nullable=True)
    conversation_id = db.Column('CONVERSATION_ID', db.String(64))
    request_type = db.Column('REQUEST_TYPE', db.String(30))
    agent_name = db.Column('AGENT_NAME', db.String(100))
    user_query = db.Column('USER_QUERY', db.Text)
    execution_status = db.Column('EXECUTION_STATUS', db.String(20))
    approval_result = db.Column('APPROVAL_RESULT', db.String(20))
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

# 6. WORK_INSTRUCTIONS (11개 열)
class WorkInstruction(db.Model):
    __tablename__ = 'WORK_INSTRUCTIONS'
    instruction_id = db.Column('INSTRUCTION_ID', db.Integer, Identity(start=1), primary_key=True)
    work_id = db.Column('WORK_ID', db.Integer, db.ForeignKey('WORK_ORDERS.WORK_ID'), nullable=False)
    execution_id = db.Column('EXECUTION_ID', db.Integer, db.ForeignKey('AGENT_EXECUTIONS.EXECUTION_ID'), nullable=False)
    version_no = db.Column('VERSION_NO', db.Integer, default=1)
    instruction_text = db.Column('INSTRUCTION_TEXT', db.Text)
    instruction_data = db.Column('INSTRUCTION_DATA', db.Text)
    source_references = db.Column('SOURCE_REFERENCES', db.Text)
    approval_status = db.Column('APPROVAL_STATUS', db.String(20))
    instruction_status = db.Column('INSTRUCTION_STATUS', db.String(20), default='FINAL')
    created_at = db.Column('CREATED_AT', db.DateTime, default=datetime.utcnow)
    updated_at = db.Column('UPDATED_AT', db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# 7. DOCUMENT_METADATA (13개 열)
class DocumentMetadata(db.Model):
    __tablename__ = 'DOCUMENT_METADATA'
    document_id = db.Column('DOCUMENT_ID', db.Integer, Identity(start=1), primary_key=True)
    document_name = db.Column('DOCUMENT_NAME', db.String(300), nullable=False)
    category = db.Column('CATEGORY', db.String(30), nullable=False)
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
