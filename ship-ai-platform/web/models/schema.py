# web/models/schema.py
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Identity

db = SQLAlchemy()


# =====================================================================
# [바인딩 1: 기본 DB] Oracle SHIP_USER (10개 핵심 테이블)
# =====================================================================

# 1. USERS (사용자 및 권한 관리)
class User(db.Model):
    __tablename__ = "USERS"

    user_id = db.Column("USER_ID", db.Integer, Identity(start=1), primary_key=True)
    username = db.Column("USERNAME", db.String(50), unique=True, nullable=False)
    full_name = db.Column("FULL_NAME", db.String(100), nullable=False)
    display_name = db.Column("DISPLAY_NAME", db.String(100))
    password_hash = db.Column("PASSWORD_HASH", db.String(255), nullable=False)
    email = db.Column("EMAIL", db.String(150))
    user_type = db.Column("USER_TYPE", db.String(30), default="INTERNAL", nullable=False)
    role = db.Column("ROLE", db.String(30), default="WORKER", nullable=False)

    # 권한 플래그
    can_access_erp = db.Column("CAN_ACCESS_ERP", db.String(1), default="N", nullable=False)
    can_access_internal_rag = db.Column("CAN_ACCESS_INTERNAL_RAG", db.String(1), default="N", nullable=False)
    can_control_robot = db.Column("CAN_CONTROL_ROBOT", db.String(1), default="N", nullable=False)
    default_workspace = db.Column("DEFAULT_WORKSPACE", db.String(50))
    department_name = db.Column("DEPARTMENT_NAME", db.String(100))

    # ERP 논리 연동 코드
    erp_company_code = db.Column("ERP_COMPANY_CODE", db.String(50))
    erp_branch_code = db.Column("ERP_BRANCH_CODE", db.String(50))
    erp_employee_code = db.Column("ERP_EMPLOYEE_CODE", db.String(30))

    # 계정 상태 및 이력
    is_active = db.Column("IS_ACTIVE", db.String(1), default="Y", nullable=False)
    failed_login_attempts = db.Column("FAILED_LOGIN_ATTEMPTS", db.Integer, default=0, nullable=False)
    locked_until = db.Column("LOCKED_UNTIL", db.DateTime)
    password_changed_at = db.Column("PASSWORD_CHANGED_AT", db.DateTime)
    last_login_at = db.Column("LAST_LOGIN_AT", db.DateTime)
    created_by = db.Column("CREATED_BY", db.Integer, db.ForeignKey("USERS.USER_ID"))
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow, nullable=False)
    updated_by = db.Column("UPDATED_BY", db.Integer, db.ForeignKey("USERS.USER_ID"))
    updated_at = db.Column("UPDATED_AT", db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


# 2. WORK_ORDERS (용접 작업 오더)
class WorkOrder(db.Model):
    __tablename__ = "WORK_ORDERS"

    work_id = db.Column("WORK_ID", db.Integer, Identity(start=1), primary_key=True)
    work_no = db.Column("WORK_NO", db.String(50), unique=True, nullable=False)
    work_title = db.Column("WORK_TITLE", db.String(200), nullable=False)
    work_description = db.Column("WORK_DESCRIPTION", db.Text)

    ship_code = db.Column("SHIP_CODE", db.String(50), nullable=False)
    block_code = db.Column("BLOCK_CODE", db.String(50), nullable=False)
    work_region = db.Column("WORK_REGION", db.String(50), nullable=False)
    work_zone = db.Column("WORK_ZONE", db.String(50), nullable=False)

    process_code = db.Column("PROCESS_CODE", db.String(30), nullable=False)
    material_grade = db.Column("MATERIAL_GRADE", db.String(30), nullable=False)
    thickness_mm = db.Column("THICKNESS_MM", db.Numeric(10, 2), nullable=False)
    weld_position = db.Column("WELD_POSITION", db.String(30), nullable=False)
    wps_code = db.Column("WPS_CODE", db.String(50))

    status_code = db.Column("STATUS_CODE", db.String(20), default="READY", nullable=False)
    requested_by = db.Column("REQUESTED_BY", db.Integer, db.ForeignKey("USERS.USER_ID"), nullable=False)
    requested_at = db.Column("REQUESTED_AT", db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column("UPDATED_AT", db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


# 3. WORK_INSTRUCTIONS (AI 생성 용접 작업 지침서)
class WorkInstruction(db.Model):
    __tablename__ = "WORK_INSTRUCTIONS"

    instruction_id = db.Column("INSTRUCTION_ID", db.Integer, Identity(start=1), primary_key=True)
    work_id = db.Column("WORK_ID", db.Integer, db.ForeignKey("WORK_ORDERS.WORK_ID"), nullable=False)
    revision_no = db.Column("REVISION_NO", db.String(20), default="Rev.1", nullable=False)
    instruction_text = db.Column("INSTRUCTION_TEXT", db.Text, nullable=False)
    welding_params_json = db.Column("WELDING_PARAMS_JSON", db.Text, nullable=False)
    source_refs_json = db.Column("SOURCE_REFS_JSON", db.Text)
    external_env_json = db.Column("EXTERNAL_ENV_JSON", db.Text)
    approval_status = db.Column("APPROVAL_STATUS", db.String(20), default="DRAFT", nullable=False)
    approved_by = db.Column("APPROVED_BY", db.Integer, db.ForeignKey("USERS.USER_ID"))
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow, nullable=False)


# 4. AGENT_EXECUTIONS (LangGraph Agent 실행 및 도구 호출 로그)
class AgentExecution(db.Model):
    __tablename__ = "AGENT_EXECUTIONS"

    execution_id = db.Column("EXECUTION_ID", db.Integer, Identity(start=1), primary_key=True)
    session_id = db.Column("SESSION_ID", db.String(100), nullable=False)
    user_id = db.Column("USER_ID", db.Integer, db.ForeignKey("USERS.USER_ID"), nullable=False)
    work_id = db.Column("WORK_ID", db.Integer, db.ForeignKey("WORK_ORDERS.WORK_ID"))
    intent_type = db.Column("INTENT_TYPE", db.String(50), nullable=False)
    current_state = db.Column("CURRENT_STATE", db.String(50))
    selected_tools_json = db.Column("SELECTED_TOOLS_JSON", db.Text)
    execution_time_ms = db.Column("EXECUTION_TIME_MS", db.Numeric(10, 2))
    status_code = db.Column("STATUS_CODE", db.String(20), default="SUCCESS", nullable=False)
    error_message = db.Column("ERROR_MESSAGE", db.Text)
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow, nullable=False)


# 5. AGENT_CHAT_HISTORY (챗봇 대화 및 RAG 출처 이력)
class AgentChatHistory(db.Model):
    __tablename__ = "AGENT_CHAT_HISTORY"

    message_id = db.Column("MESSAGE_ID", db.Integer, Identity(start=1), primary_key=True)
    session_id = db.Column("SESSION_ID", db.String(100), nullable=False)
    user_id = db.Column("USER_ID", db.Integer, db.ForeignKey("USERS.USER_ID"), nullable=False)
    role_code = db.Column("ROLE_CODE", db.String(20), nullable=False)
    content = db.Column("CONTENT", db.Text, nullable=False)
    retrieval_chunks_json = db.Column("RETRIEVAL_CHUNKS_JSON", db.Text)
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow, nullable=False)


# 6. DOCUMENT_METADATA (RAG 사내외 기술문서 메타데이터)
class DocumentMetadata(db.Model):
    __tablename__ = "DOCUMENT_METADATA"

    document_id = db.Column("DOCUMENT_ID", db.Integer, Identity(start=1), primary_key=True)
    document_code = db.Column("DOCUMENT_CODE", db.String(50), unique=True, nullable=False)
    document_title = db.Column("DOCUMENT_TITLE", db.String(200), nullable=False)
    collection_type = db.Column("COLLECTION_TYPE", db.String(20), nullable=False)
    category_code = db.Column("CATEGORY_CODE", db.String(50), nullable=False)
    original_file_name = db.Column("ORIGINAL_FILE_NAME", db.String(255), nullable=False)
    storage_uri = db.Column("STORAGE_URI", db.String(500), nullable=False)
    file_hash_sha256 = db.Column("FILE_HASH_SHA256", db.String(64))
    vector_collection = db.Column("VECTOR_COLLECTION", db.String(100), nullable=False)
    ingest_status = db.Column("INGEST_STATUS", db.String(20), default="COMPLETED", nullable=False)
    uploaded_by = db.Column("UPLOADED_BY", db.Integer, db.ForeignKey("USERS.USER_ID"), nullable=False)
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow, nullable=False)


# 7. VISION_ANALYSES (현장 비전 캡처 및 분석 세션)
class VisionAnalysis(db.Model):
    __tablename__ = "VISION_ANALYSES"

    analysis_id = db.Column("ANALYSIS_ID", db.Integer, Identity(start=1), primary_key=True)
    work_id = db.Column("WORK_ID", db.Integer, db.ForeignKey("WORK_ORDERS.WORK_ID"))
    camera_code = db.Column("CAMERA_CODE", db.String(50), nullable=False)
    source_stream_uri = db.Column("SOURCE_STREAM_URI", db.String(500))
    original_image_uri = db.Column("ORIGINAL_IMAGE_URI", db.String(500), nullable=False)
    visualized_image_uri = db.Column("VISUALIZED_IMAGE_URI", db.String(500))
    image_width = db.Column("IMAGE_WIDTH", db.Integer, nullable=False)
    image_height = db.Column("IMAGE_HEIGHT", db.Integer, nullable=False)
    inference_time_ms = db.Column("INFERENCE_TIME_MS", db.Numeric(10, 2))
    welding_state = db.Column("WELDING_STATE", db.String(20), default="BEFORE_WELD", nullable=False)
    status_code = db.Column("STATUS_CODE", db.String(20), default="COMPLETED", nullable=False)
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow, nullable=False)


# 8. VISION_DETECTIONS (YOLOv11 객체 탐지 및 용접선 궤적)
class VisionDetection(db.Model):
    __tablename__ = "VISION_DETECTIONS"

    detection_id = db.Column("DETECTION_ID", db.Integer, Identity(start=1), primary_key=True)
    analysis_id = db.Column("ANALYSIS_ID", db.Integer, db.ForeignKey("VISION_ANALYSES.ANALYSIS_ID"), nullable=False)
    class_id = db.Column("CLASS_ID", db.Integer, nullable=False)
    class_name = db.Column("CLASS_NAME", db.String(50), nullable=False)
    confidence = db.Column("CONFIDENCE", db.Numeric(5, 4), nullable=False)
    tracker_id = db.Column("TRACKER_ID", db.Integer)
    bbox_x = db.Column("BBOX_X", db.Numeric(10, 2), nullable=False)
    bbox_y = db.Column("BBOX_Y", db.Numeric(10, 2), nullable=False)
    bbox_width = db.Column("BBOX_WIDTH", db.Numeric(10, 2), nullable=False)
    bbox_height = db.Column("BBOX_HEIGHT", db.Numeric(10, 2), nullable=False)
    centerline_waypoints_json = db.Column("CENTERLINE_WAYPOINTS_JSON", db.Text)


# 9. VISION_INSPECTIONS (목표 용접선 확정 및 품질 판정)
class VisionInspection(db.Model):
    __tablename__ = "VISION_INSPECTIONS"

    inspection_id = db.Column("INSPECTION_ID", db.Integer, Identity(start=1), primary_key=True)
    analysis_id = db.Column("ANALYSIS_ID", db.Integer, db.ForeignKey("VISION_ANALYSES.ANALYSIS_ID"), nullable=False)
    detection_id = db.Column("DETECTION_ID", db.Integer, db.ForeignKey("VISION_DETECTIONS.DETECTION_ID"),
                             nullable=False)
    work_id = db.Column("WORK_ID", db.Integer, db.ForeignKey("WORK_ORDERS.WORK_ID"), nullable=False)
    start_point_json = db.Column("START_POINT_JSON", db.String(200), nullable=False)
    end_point_json = db.Column("END_POINT_JSON", db.String(200), nullable=False)
    estimated_length_mm = db.Column("ESTIMATED_LENGTH_MM", db.Numeric(10, 2))
    safety_clearance_yn = db.Column("SAFETY_CLEARANCE_YN", db.String(1), default="Y", nullable=False)
    inspection_status = db.Column("INSPECTION_STATUS", db.String(20), default="BEFORE_WELD", nullable=False)
    defect_detail_json = db.Column("DEFECT_DETAIL_JSON", db.Text)
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow, nullable=False)


# 10. ROBOT_EXECUTIONS (로봇 제어 상태 및 실시간 텔레메트리)
class RobotExecution(db.Model):
    __tablename__ = "ROBOT_EXECUTIONS"

    robot_exec_id = db.Column("ROBOT_EXEC_ID", db.Integer, Identity(start=1), primary_key=True)
    work_id = db.Column("WORK_ID", db.Integer, db.ForeignKey("WORK_ORDERS.WORK_ID"))
    equipment_code = db.Column("EQUIPMENT_CODE", db.String(50), unique=True, nullable=False)
    robot_name = db.Column("ROBOT_NAME", db.String(100), nullable=False)
    robot_status = db.Column("ROBOT_STATUS", db.String(20), default="IDLE", nullable=False)
    current_location_desc = db.Column("CURRENT_LOCATION_DESC", db.String(200))
    battery_percent = db.Column("BATTERY_PERCENT", db.Integer, default=100, nullable=False)
    emergency_stop_yn = db.Column("EMERGENCY_STOP_YN", db.String(1), default="N", nullable=False)
    ros2_node_status = db.Column("ROS2_NODE_STATUS", db.String(20), default="CONNECTED", nullable=False)
    current_pose_json = db.Column("CURRENT_POSE_JSON", db.Text)
    active_instruction_id = db.Column("ACTIVE_INSTRUCTION_ID", db.Integer,
                                      db.ForeignKey("WORK_INSTRUCTIONS.INSTRUCTION_ID"))
    last_heartbeat_at = db.Column("LAST_HEARTBEAT_AT", db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column("UPDATED_AT", db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


# 11. COMMUNITY_BOARD (사내 문의·FAQ 및 익명 소통 게시판)
class CommunityBoard(db.Model):
    __tablename__ = "COMMUNITY_BOARD"

    board_id = db.Column("BOARD_ID", db.Integer, Identity(start=1), primary_key=True)
    parent_id = db.Column("PARENT_ID", db.Integer, db.ForeignKey("COMMUNITY_BOARD.BOARD_ID"))
    category = db.Column("CATEGORY", db.String(50), default="일반문의", nullable=False)
    title = db.Column("TITLE", db.String(200))
    content = db.Column("CONTENT", db.Text, nullable=False)
    writer_id = db.Column("WRITER_ID", db.Integer, db.ForeignKey("USERS.USER_ID"), nullable=False)
    writer_display = db.Column("WRITER_DISPLAY", db.String(100), nullable=False)
    is_anonymous = db.Column("IS_ANONYMOUS", db.String(1), default="N", nullable=False)
    view_count = db.Column("VIEW_COUNT", db.Integer, default=0, nullable=False)
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column("UPDATED_AT", db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


# =====================================================================
# [바인딩 2: 사내 ERP DB] Oracle SHIP_ERP (__bind_key__ = 'ship_erp')
# =====================================================================

class ErpBlockMaster(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_BLOCK_MASTER"

    block_id = db.Column("BLOCK_ID", db.String(50), primary_key=True)
    ship_code = db.Column("SHIP_CODE", db.String(50), nullable=False)
    block_name = db.Column("BLOCK_NAME", db.String(100), nullable=False)
    steel_type = db.Column("STEEL_TYPE", db.String(50), nullable=False)
    thickness_mm = db.Column("THICKNESS_MM", db.String(20), nullable=False)
    joint_type = db.Column("JOINT_TYPE", db.String(50), nullable=False)
    process = db.Column("PROCESS", db.String(50), nullable=False)
    target_dock = db.Column("TARGET_DOCK", db.String(50), nullable=False)
    branch_code = db.Column("BRANCH_CODE", db.String(50), nullable=False)
    weight_ton = db.Column("WEIGHT_TON", db.Numeric(10, 2))


class ErpBomInventory(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_BOM_INVENTORY"

    item_code = db.Column("ITEM_CODE", db.String(50), primary_key=True)
    item_name = db.Column("ITEM_NAME", db.String(100), nullable=False)
    material_category = db.Column("MATERIAL_CATEGORY", db.String(30), nullable=False)
    lot_no = db.Column("LOT_NO", db.String(50), unique=True, nullable=False)
    supplier_code = db.Column("SUPPLIER_CODE", db.String(50), nullable=False)
    branch_code = db.Column("BRANCH_CODE", db.String(50), nullable=False)
    total_qty = db.Column("TOTAL_QTY", db.Numeric(10, 2), nullable=False)
    reserved_qty = db.Column("RESERVED_QTY", db.Numeric(10, 2), default=0, nullable=False)
    unit = db.Column("UNIT", db.String(20), nullable=False)


class MesWorkOrderDispatch(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "MES_WORK_ORDER_DISPATCH"

    wo_number = db.Column("WO_NUMBER", db.String(50), primary_key=True)
    block_id = db.Column("BLOCK_ID", db.String(50), nullable=False)
    execution_type = db.Column("EXECUTION_TYPE", db.String(30), nullable=False)
    partner_id = db.Column("PARTNER_ID", db.String(50))
    allocated_lot_no = db.Column("ALLOCATED_LOT_NO", db.String(50), nullable=False)
    worker_id = db.Column("WORKER_ID", db.String(30), nullable=False)
    worker_name = db.Column("WORKER_NAME", db.String(50), nullable=False)
    assigned_vehicle_code = db.Column("ASSIGNED_VEHICLE_CODE", db.String(50))
    status = db.Column("STATUS", db.String(30), default="COMMITTED", nullable=False)
    created_at = db.Column("CREATED_AT", db.String(30), nullable=False)


class ErpWorkerCert(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_WORKER_CERT"

    worker_id = db.Column("WORKER_ID", db.String(30), primary_key=True)
    worker_name = db.Column("WORKER_NAME", db.String(50), nullable=False)
    employment_type = db.Column("EMPLOYMENT_TYPE", db.String(30), nullable=False)
    company_code = db.Column("COMPANY_CODE", db.String(50), nullable=False)
    branch_code = db.Column("BRANCH_CODE", db.String(50), nullable=False)
    department_name = db.Column("DEPARTMENT_NAME", db.String(100))
    role = db.Column("ROLE", db.String(30), nullable=False)
    phone = db.Column("PHONE", db.String(30))
    cert_code = db.Column("CERT_CODE", db.String(50), nullable=False)
    weld_process = db.Column("WELD_PROCESS", db.String(30), nullable=False)
    cert_expiry_date = db.Column("CERT_EXPIRY_DATE", db.String(20), nullable=False)
    status = db.Column("STATUS", db.String(20), default="VALID", nullable=False)


class ErpVehiclesEquipment(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_VEHICLES_EQUIPMENT"

    asset_code = db.Column("ASSET_CODE", db.String(50), primary_key=True)
    asset_name = db.Column("ASSET_NAME", db.String(100), nullable=False)
    asset_category = db.Column("ASSET_CATEGORY", db.String(30), nullable=False)
    company_code = db.Column("COMPANY_CODE", db.String(50), nullable=False)
    branch_code = db.Column("BRANCH_CODE", db.String(50), nullable=False)
    current_zone = db.Column("CURRENT_ZONE", db.String(50))
    max_load_ton = db.Column("MAX_LOAD_TON", db.Numeric(10, 2))
    driver_worker_id = db.Column("DRIVER_WORKER_ID", db.String(30))
    operation_status = db.Column("OPERATION_STATUS", db.String(20), default="STANDBY", nullable=False)


class ErpCompany(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_COMPANIES"

    company_code = db.Column("COMPANY_CODE", db.String(50), primary_key=True)
    company_name = db.Column("COMPANY_NAME", db.String(100), nullable=False)
    company_type = db.Column("COMPANY_TYPE", db.String(30), nullable=False)
    specialty_field = db.Column("SPECIALTY_FIELD", db.String(50))
    business_no = db.Column("BUSINESS_NO", db.String(30))
    representative_name = db.Column("REPRESENTATIVE_NAME", db.String(50))
    contact_phone = db.Column("CONTACT_PHONE", db.String(30))
    contract_status = db.Column("CONTRACT_STATUS", db.String(20), default="ACTIVE", nullable=False)
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow, nullable=False)


class ErpBranch(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_BRANCHES"

    branch_code = db.Column("BRANCH_CODE", db.String(50), primary_key=True)
    company_code = db.Column("COMPANY_CODE", db.String(50), nullable=False)
    region_name = db.Column("REGION_NAME", db.String(50), nullable=False)
    branch_name = db.Column("BRANCH_NAME", db.String(100), nullable=False)
    branch_type = db.Column("BRANCH_TYPE", db.String(30), nullable=False)
    work_zones_json = db.Column("WORK_ZONES_JSON", db.Text)
    address = db.Column("ADDRESS", db.String(255))
    status_code = db.Column("STATUS_CODE", db.String(20), default="ACTIVE", nullable=False)
    latitude = db.Column("LATITUDE", db.Numeric(10, 6))
    longitude = db.Column("LONGITUDE", db.Numeric(10, 6))
    camera_code = db.Column("CAMERA_CODE", db.String(50))
    default_video = db.Column("DEFAULT_VIDEO", db.String(100))