"""ship_user: 공정·채팅·RAG 메타데이터. 모든 FK는 이 DB 내부에서만 연결한다."""
from datetime import datetime
from sqlalchemy import Identity, UniqueConstraint, Index
from app.extensions import db

class AppUser(db.Model):
    __tablename__ = "APP_USERS"
    user_id = db.Column("USER_ID", db.BigInteger, Identity(), primary_key=True)
    login_id = db.Column("LOGIN_ID", db.String(100), nullable=False, unique=True)
    password_hash = db.Column("PASSWORD_HASH", db.String(255), nullable=False)
    user_name = db.Column("USER_NAME", db.String(100), nullable=False)
    email = db.Column("EMAIL", db.String(150))
    role_code = db.Column("ROLE_CODE", db.String(30), nullable=False, default="OPERATOR")
    erp_company_code = db.Column("ERP_COMPANY_CODE", db.String(50))
    erp_branch_code = db.Column("ERP_BRANCH_CODE", db.String(50))
    erp_employee_code = db.Column("ERP_EMPLOYEE_CODE", db.String(50))
    can_access_erp = db.Column("CAN_ACCESS_ERP", db.String(1), nullable=False, default="N")
    can_access_internal_rag = db.Column("CAN_ACCESS_INTERNAL_RAG", db.String(1), nullable=False, default="N")
    is_active = db.Column("IS_ACTIVE", db.String(1), nullable=False, default="Y")
    password_changed_at = db.Column("PASSWORD_CHANGED_AT", db.DateTime, default=datetime.utcnow)
    last_login_at = db.Column("LAST_LOGIN_AT", db.DateTime)
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column("UPDATED_AT", db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def is_sys_admin(self):
        return self.role_code == "SYS_ADMIN"

class Ship(db.Model):
    __tablename__ = "SHIPS"
    ship_id = db.Column("SHIP_ID", db.BigInteger, Identity(), primary_key=True)
    ship_code = db.Column("SHIP_CODE", db.String(50), nullable=False, unique=True)
    ship_name = db.Column("SHIP_NAME", db.String(200), nullable=False)
    customer_name = db.Column("CUSTOMER_NAME", db.String(200))
    ship_type_code = db.Column("SHIP_TYPE_CODE", db.String(50))
    class_code = db.Column("CLASS_CODE", db.String(50))
    status_code = db.Column("STATUS_CODE", db.String(30), default="ACTIVE")
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow)

class ShipBlock(db.Model):
    __tablename__ = "SHIP_BLOCKS"
    __table_args__ = (UniqueConstraint("SHIP_ID", "BLOCK_CODE", name="UK_SHIP_BLOCK_CODE"),)
    block_id = db.Column("BLOCK_ID", db.BigInteger, Identity(), primary_key=True)
    ship_id = db.Column("SHIP_ID", db.BigInteger, db.ForeignKey("SHIPS.SHIP_ID"), nullable=False)
    block_code = db.Column("BLOCK_CODE", db.String(80), nullable=False)
    block_name = db.Column("BLOCK_NAME", db.String(200))
    block_type_code = db.Column("BLOCK_TYPE_CODE", db.String(50))
    erp_branch_code = db.Column("ERP_BRANCH_CODE", db.String(50))
    location_code = db.Column("LOCATION_CODE", db.String(80))
    drawing_no = db.Column("DRAWING_NO", db.String(100))
    drawing_revision = db.Column("DRAWING_REVISION", db.String(40))
    status_code = db.Column("STATUS_CODE", db.String(30), default="ACTIVE")
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow)

class WeldJoint(db.Model):
    __tablename__ = "WELD_JOINTS"
    __table_args__ = (UniqueConstraint("BLOCK_ID", "JOINT_CODE", "DRAWING_REVISION", name="UK_BLOCK_JOINT_REV"),)
    joint_id = db.Column("JOINT_ID", db.BigInteger, Identity(), primary_key=True)
    block_id = db.Column("BLOCK_ID", db.BigInteger, db.ForeignKey("SHIP_BLOCKS.BLOCK_ID"), nullable=False)
    joint_code = db.Column("JOINT_CODE", db.String(100), nullable=False)
    drawing_no = db.Column("DRAWING_NO", db.String(100))
    drawing_revision = db.Column("DRAWING_REVISION", db.String(40), default="A")
    member_a_code = db.Column("MEMBER_A_CODE", db.String(80))
    member_b_code = db.Column("MEMBER_B_CODE", db.String(80))
    joint_type_code = db.Column("JOINT_TYPE_CODE", db.String(40), default="BUTT")
    groove_type_code = db.Column("GROOVE_TYPE_CODE", db.String(40))
    weld_position_code = db.Column("WELD_POSITION_CODE", db.String(20))
    material_a_grade = db.Column("MATERIAL_A_GRADE", db.String(80))
    material_b_grade = db.Column("MATERIAL_B_GRADE", db.String(80))
    thickness_a_mm = db.Column("THICKNESS_A_MM", db.Numeric(8, 3))
    thickness_b_mm = db.Column("THICKNESS_B_MM", db.Numeric(8, 3))
    design_length_mm = db.Column("DESIGN_LENGTH_MM", db.Numeric(12, 3))
    status_code = db.Column("STATUS_CODE", db.String(30), default="PLANNED")
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow)

class WeldingProcedure(db.Model):
    __tablename__ = "WELDING_PROCEDURES"
    __table_args__ = (UniqueConstraint("WPS_CODE", "REVISION_NO", name="UK_WPS_REVISION"),)
    wps_id = db.Column("WPS_ID", db.BigInteger, Identity(), primary_key=True)
    wps_code = db.Column("WPS_CODE", db.String(100), nullable=False)
    revision_no = db.Column("REVISION_NO", db.String(40), nullable=False)
    process_code = db.Column("PROCESS_CODE", db.String(30))
    material_group = db.Column("MATERIAL_GROUP", db.String(80))
    thickness_min_mm = db.Column("THICKNESS_MIN_MM", db.Numeric(8, 3))
    thickness_max_mm = db.Column("THICKNESS_MAX_MM", db.Numeric(8, 3))
    position_range = db.Column("POSITION_RANGE", db.String(100))
    joint_type_code = db.Column("JOINT_TYPE_CODE", db.String(40))
    filler_metal_code = db.Column("FILLER_METAL_CODE", db.String(80))
    shielding_gas_code = db.Column("SHIELDING_GAS_CODE", db.String(80))
    parameters_json = db.Column("PARAMETERS_JSON", db.Text)
    approval_status = db.Column("APPROVAL_STATUS", db.String(30), default="REFERENCE")
    effective_from = db.Column("EFFECTIVE_FROM", db.Date)
    effective_to = db.Column("EFFECTIVE_TO", db.Date)
    document_uri = db.Column("DOCUMENT_URI", db.String(1000))
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow)

class WorkOrder(db.Model):
    __tablename__ = "WORK_ORDERS"
    work_id = db.Column("WORK_ID", db.BigInteger, Identity(), primary_key=True)
    work_no = db.Column("WORK_NO", db.String(80), nullable=False, unique=True)
    ship_id = db.Column("SHIP_ID", db.BigInteger, db.ForeignKey("SHIPS.SHIP_ID"))
    block_id = db.Column("BLOCK_ID", db.BigInteger, db.ForeignKey("SHIP_BLOCKS.BLOCK_ID"))
    work_type_code = db.Column("WORK_TYPE_CODE", db.String(30), default="PRODUCTION")
    work_title = db.Column("WORK_TITLE", db.String(200), nullable=False)
    work_description = db.Column("WORK_DESCRIPTION", db.Text)
    process_code = db.Column("PROCESS_CODE", db.String(30))
    base_material_grade = db.Column("BASE_MATERIAL_GRADE", db.String(80))
    plate_thickness_mm = db.Column("PLATE_THICKNESS_MM", db.Numeric(8, 3))
    weld_position_code = db.Column("WELD_POSITION_CODE", db.String(20))
    erp_branch_code = db.Column("ERP_BRANCH_CODE", db.String(50))
    work_location_code = db.Column("WORK_LOCATION_CODE", db.String(80))
    priority_code = db.Column("PRIORITY_CODE", db.String(20), default="NORMAL")
    requested_by = db.Column("REQUESTED_BY", db.BigInteger, db.ForeignKey("APP_USERS.USER_ID"), nullable=False)
    requested_at = db.Column("REQUESTED_AT", db.DateTime, default=datetime.utcnow)
    status_code = db.Column("STATUS_CODE", db.String(30), default="REQUESTED")
    parent_work_id = db.Column("PARENT_WORK_ID", db.BigInteger, db.ForeignKey("WORK_ORDERS.WORK_ID"))
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow)
    updated_at = db.Column("UPDATED_AT", db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class WorkOrderJoint(db.Model):
    __tablename__ = "WORK_ORDER_JOINTS"
    __table_args__ = (UniqueConstraint("WORK_ID", "JOINT_ID", name="UK_WORK_JOINT"),)
    work_order_joint_id = db.Column("WORK_ORDER_JOINT_ID", db.BigInteger, Identity(), primary_key=True)
    work_id = db.Column("WORK_ID", db.BigInteger, db.ForeignKey("WORK_ORDERS.WORK_ID"), nullable=False)
    joint_id = db.Column("JOINT_ID", db.BigInteger, db.ForeignKey("WELD_JOINTS.JOINT_ID"), nullable=False)
    wps_id = db.Column("WPS_ID", db.BigInteger, db.ForeignKey("WELDING_PROCEDURES.WPS_ID"))
    sequence_no = db.Column("SEQUENCE_NO", db.Integer, default=1)
    planned_length_mm = db.Column("PLANNED_LENGTH_MM", db.Numeric(12, 3))
    status_code = db.Column("STATUS_CODE", db.String(30), default="PLANNED")
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow)

class WorkInstruction(db.Model):
    __tablename__ = "WORK_INSTRUCTIONS"
    __table_args__ = (UniqueConstraint("WORK_ORDER_JOINT_ID", "REVISION_NO", name="UK_INSTRUCTION_REV"),)
    instruction_id = db.Column("INSTRUCTION_ID", db.BigInteger, Identity(), primary_key=True)
    work_order_joint_id = db.Column("WORK_ORDER_JOINT_ID", db.BigInteger, db.ForeignKey("WORK_ORDER_JOINTS.WORK_ORDER_JOINT_ID"), nullable=False)
    wps_id = db.Column("WPS_ID", db.BigInteger, db.ForeignKey("WELDING_PROCEDURES.WPS_ID"))
    revision_no = db.Column("REVISION_NO", db.Integer, nullable=False, default=1)
    instruction_text = db.Column("INSTRUCTION_TEXT", db.Text)
    parameter_set_json = db.Column("PARAMETER_SET_JSON", db.Text)
    source_references_json = db.Column("SOURCE_REFERENCES_JSON", db.Text)
    approval_status = db.Column("APPROVAL_STATUS", db.String(30), default="DRAFT")
    created_by = db.Column("CREATED_BY", db.BigInteger, db.ForeignKey("APP_USERS.USER_ID"))
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow)

class ChatConversation(db.Model):
    __tablename__ = "CHAT_CONVERSATIONS"
    conversation_id = db.Column("CONVERSATION_ID", db.BigInteger, Identity(), primary_key=True)
    user_id = db.Column("USER_ID", db.BigInteger, db.ForeignKey("APP_USERS.USER_ID"), nullable=False)
    title = db.Column("TITLE", db.String(200))
    started_at = db.Column("STARTED_AT", db.DateTime, default=datetime.utcnow)

class ChatMessage(db.Model):
    __tablename__ = "CHAT_MESSAGES"
    message_id = db.Column("MESSAGE_ID", db.BigInteger, Identity(), primary_key=True)
    conversation_id = db.Column("CONVERSATION_ID", db.BigInteger, db.ForeignKey("CHAT_CONVERSATIONS.CONVERSATION_ID"), nullable=False)
    role_code = db.Column("ROLE_CODE", db.String(20), nullable=False)
    content = db.Column("CONTENT", db.Text, nullable=False)
    route_code = db.Column("ROUTE_CODE", db.String(30))
    model_name = db.Column("MODEL_NAME", db.String(100))
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow)

class RagDocument(db.Model):
    __tablename__ = "RAG_DOCUMENTS"
    document_id = db.Column("DOCUMENT_ID", db.BigInteger, Identity(), primary_key=True)
    document_code = db.Column("DOCUMENT_CODE", db.String(100), nullable=False, unique=True)
    document_title = db.Column("DOCUMENT_TITLE", db.String(300), nullable=False)
    collection_type = db.Column("COLLECTION_TYPE", db.String(20), nullable=False)  # PUBLIC / INTERNAL
    category_code = db.Column("CATEGORY_CODE", db.String(50), default="GENERAL")
    original_file_name = db.Column("ORIGINAL_FILE_NAME", db.String(255))
    storage_uri = db.Column("STORAGE_URI", db.String(1000))
    file_hash_sha256 = db.Column("FILE_HASH_SHA256", db.String(64), nullable=False)
    revision_no = db.Column("REVISION_NO", db.String(80))
    source_organization = db.Column("SOURCE_ORGANIZATION", db.String(200))
    owner_company_code = db.Column("OWNER_COMPANY_CODE", db.String(50))
    access_scope_code = db.Column("ACCESS_SCOPE_CODE", db.String(30), default="PUBLIC")
    vector_collection = db.Column("VECTOR_COLLECTION", db.String(100), nullable=False)
    ingest_status = db.Column("INGEST_STATUS", db.String(30), default="PENDING")
    document_status = db.Column("DOCUMENT_STATUS", db.String(30), default="ACTIVE")
    uploaded_by = db.Column("UPLOADED_BY", db.BigInteger, db.ForeignKey("APP_USERS.USER_ID"))
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow)
    updated_at = db.Column("UPDATED_AT", db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class RagRetrievalLog(db.Model):
    __tablename__ = "RAG_RETRIEVAL_LOGS"
    retrieval_id = db.Column("RETRIEVAL_ID", db.BigInteger, Identity(), primary_key=True)
    message_id = db.Column("MESSAGE_ID", db.BigInteger, db.ForeignKey("CHAT_MESSAGES.MESSAGE_ID"), nullable=False)
    collection_type = db.Column("COLLECTION_TYPE", db.String(20), nullable=False)
    query_text = db.Column("QUERY_TEXT", db.Text)
    top_k = db.Column("TOP_K", db.Integer)
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow)

class RagRetrievalItem(db.Model):
    __tablename__ = "RAG_RETRIEVAL_ITEMS"
    retrieval_item_id = db.Column("RETRIEVAL_ITEM_ID", db.BigInteger, Identity(), primary_key=True)
    retrieval_id = db.Column("RETRIEVAL_ID", db.BigInteger, db.ForeignKey("RAG_RETRIEVAL_LOGS.RETRIEVAL_ID"), nullable=False)
    document_id = db.Column("DOCUMENT_ID", db.BigInteger, db.ForeignKey("RAG_DOCUMENTS.DOCUMENT_ID"), nullable=False)
    vector_chunk_id = db.Column("VECTOR_CHUNK_ID", db.String(200))
    result_rank = db.Column("RESULT_RANK", db.Integer)
    similarity_score = db.Column("SIMILARITY_SCORE", db.Numeric(14, 10))
    page_number = db.Column("PAGE_NUMBER", db.Integer)
    source_excerpt = db.Column("SOURCE_EXCERPT", db.Text)

class WorkErpResourceRef(db.Model):
    __tablename__ = "WORK_ERP_RESOURCE_REFS"
    resource_ref_id = db.Column("RESOURCE_REF_ID", db.BigInteger, Identity(), primary_key=True)
    work_id = db.Column("WORK_ID", db.BigInteger, db.ForeignKey("WORK_ORDERS.WORK_ID"), nullable=False)
    resource_type = db.Column("RESOURCE_TYPE", db.String(30), nullable=False)
    erp_resource_code = db.Column("ERP_RESOURCE_CODE", db.String(100), nullable=False)
    erp_allocation_code = db.Column("ERP_ALLOCATION_CODE", db.String(100))
    resource_name_snapshot = db.Column("RESOURCE_NAME_SNAPSHOT", db.String(200))
    resource_snapshot_json = db.Column("RESOURCE_SNAPSHOT_JSON", db.Text)
    linked_at = db.Column("LINKED_AT", db.DateTime, default=datetime.utcnow)

Index("IX_WORK_ORDERS_STATUS", WorkOrder.status_code)
Index("IX_WORK_ORDERS_REQUESTED_AT", WorkOrder.requested_at)
Index("IX_RAG_DOC_SCOPE", RagDocument.collection_type, RagDocument.owner_company_code, RagDocument.document_status)
