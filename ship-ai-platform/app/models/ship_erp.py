"""ship_erp: 회사·인력·자격·장비·자재·재고·배정. ship_user와 FK를 공유하지 않는다."""
from datetime import datetime
from sqlalchemy import Identity, UniqueConstraint, Index
from app.extensions import db

class ErpCompany(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_COMPANIES"
    company_id = db.Column("COMPANY_ID", db.BigInteger, Identity(), primary_key=True)
    company_code = db.Column("COMPANY_CODE", db.String(50), nullable=False, unique=True)
    company_name = db.Column("COMPANY_NAME", db.String(200), nullable=False)
    company_type_code = db.Column("COMPANY_TYPE_CODE", db.String(30), default="SUBCONTRACTOR")
    company_tier_code = db.Column("COMPANY_TIER_CODE", db.String(20), default="NA")
    business_no = db.Column("BUSINESS_NO", db.String(30))
    representative_name = db.Column("REPRESENTATIVE_NAME", db.String(100))
    phone = db.Column("PHONE", db.String(50))
    email = db.Column("EMAIL", db.String(150))
    address = db.Column("ADDRESS", db.String(500))
    status_code = db.Column("STATUS_CODE", db.String(30), default="ACTIVE")
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow)

class ErpBranch(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_BRANCHES"
    __table_args__ = (UniqueConstraint("COMPANY_ID", "BRANCH_CODE", name="UK_ERP_BRANCH_CODE"),)
    branch_id = db.Column("BRANCH_ID", db.BigInteger, Identity(), primary_key=True)
    company_id = db.Column("COMPANY_ID", db.BigInteger, db.ForeignKey("ERP_COMPANIES.COMPANY_ID"), nullable=False)
    branch_code = db.Column("BRANCH_CODE", db.String(50), nullable=False)
    branch_name = db.Column("BRANCH_NAME", db.String(200), nullable=False)
    branch_type_code = db.Column("BRANCH_TYPE_CODE", db.String(30), default="YARD")
    address = db.Column("ADDRESS", db.String(500))
    location_desc = db.Column("LOCATION_DESC", db.String(500))
    status_code = db.Column("STATUS_CODE", db.String(30), default="ACTIVE")
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow)

class ErpEmployee(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_EMPLOYEES"
    employee_id = db.Column("EMPLOYEE_ID", db.BigInteger, Identity(), primary_key=True)
    employee_code = db.Column("EMPLOYEE_CODE", db.String(50), nullable=False, unique=True)
    company_id = db.Column("COMPANY_ID", db.BigInteger, db.ForeignKey("ERP_COMPANIES.COMPANY_ID"), nullable=False)
    branch_id = db.Column("BRANCH_ID", db.BigInteger, db.ForeignKey("ERP_BRANCHES.BRANCH_ID"), nullable=False)
    employee_name = db.Column("EMPLOYEE_NAME", db.String(100), nullable=False)
    job_role_code = db.Column("JOB_ROLE_CODE", db.String(50))
    job_title = db.Column("JOB_TITLE", db.String(80))
    employment_status = db.Column("EMPLOYMENT_STATUS", db.String(30), default="ACTIVE")
    phone = db.Column("PHONE", db.String(50))
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow)

class ErpEmployeeCertification(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_EMPLOYEE_CERTIFICATIONS"
    certification_id = db.Column("CERTIFICATION_ID", db.BigInteger, Identity(), primary_key=True)
    employee_id = db.Column("EMPLOYEE_ID", db.BigInteger, db.ForeignKey("ERP_EMPLOYEES.EMPLOYEE_ID"), nullable=False)
    certification_code = db.Column("CERTIFICATION_CODE", db.String(100))
    qualification_type = db.Column("QUALIFICATION_TYPE", db.String(40))
    standard_code = db.Column("STANDARD_CODE", db.String(80))
    welding_process_code = db.Column("WELDING_PROCESS_CODE", db.String(30))
    material_group = db.Column("MATERIAL_GROUP", db.String(80))
    position_range = db.Column("POSITION_RANGE", db.String(100))
    thickness_min_mm = db.Column("THICKNESS_MIN_MM", db.Numeric(8, 3))
    thickness_max_mm = db.Column("THICKNESS_MAX_MM", db.Numeric(8, 3))
    valid_from = db.Column("VALID_FROM", db.Date)
    valid_to = db.Column("VALID_TO", db.Date)
    verification_status = db.Column("VERIFICATION_STATUS", db.String(30), default="PENDING")
    certificate_uri = db.Column("CERTIFICATE_URI", db.String(1000))

class ErpEquipmentType(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_EQUIPMENT_TYPES"
    equipment_type_id = db.Column("EQUIPMENT_TYPE_ID", db.BigInteger, Identity(), primary_key=True)
    type_code = db.Column("TYPE_CODE", db.String(50), nullable=False, unique=True)
    type_name = db.Column("TYPE_NAME", db.String(120), nullable=False)
    description = db.Column("DESCRIPTION", db.String(500))

class ErpEquipment(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_EQUIPMENT"
    equipment_id = db.Column("EQUIPMENT_ID", db.BigInteger, Identity(), primary_key=True)
    equipment_code = db.Column("EQUIPMENT_CODE", db.String(80), nullable=False, unique=True)
    company_id = db.Column("COMPANY_ID", db.BigInteger, db.ForeignKey("ERP_COMPANIES.COMPANY_ID"), nullable=False)
    branch_id = db.Column("BRANCH_ID", db.BigInteger, db.ForeignKey("ERP_BRANCHES.BRANCH_ID"), nullable=False)
    equipment_type_id = db.Column("EQUIPMENT_TYPE_ID", db.BigInteger, db.ForeignKey("ERP_EQUIPMENT_TYPES.EQUIPMENT_TYPE_ID"), nullable=False)
    equipment_name = db.Column("EQUIPMENT_NAME", db.String(200))
    manufacturer = db.Column("MANUFACTURER", db.String(100))
    model_name = db.Column("MODEL_NAME", db.String(100))
    serial_no = db.Column("SERIAL_NO", db.String(100))
    operation_status = db.Column("OPERATION_STATUS", db.String(30), default="AVAILABLE")
    location_code = db.Column("LOCATION_CODE", db.String(80))
    equipment_spec_json = db.Column("EQUIPMENT_SPEC_JSON", db.Text)
    last_inspection_at = db.Column("LAST_INSPECTION_AT", db.DateTime)

class ErpEquipmentMaintenance(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_EQUIPMENT_MAINTENANCE"
    maintenance_id = db.Column("MAINTENANCE_ID", db.BigInteger, Identity(), primary_key=True)
    equipment_id = db.Column("EQUIPMENT_ID", db.BigInteger, db.ForeignKey("ERP_EQUIPMENT.EQUIPMENT_ID"), nullable=False)
    maintenance_no = db.Column("MAINTENANCE_NO", db.String(80), unique=True)
    maintenance_type = db.Column("MAINTENANCE_TYPE", db.String(30))
    description = db.Column("DESCRIPTION", db.Text)
    requested_at = db.Column("REQUESTED_AT", db.DateTime)
    started_at = db.Column("STARTED_AT", db.DateTime)
    completed_at = db.Column("COMPLETED_AT", db.DateTime)
    result_status = db.Column("RESULT_STATUS", db.String(30))
    technician_code = db.Column("TECHNICIAN_CODE", db.String(50))
    cost = db.Column("COST", db.Numeric(14, 2))
    report_uri = db.Column("REPORT_URI", db.String(1000))

class ErpEquipmentAssignment(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_EQUIPMENT_ASSIGNMENTS"
    equipment_assignment_id = db.Column("EQUIPMENT_ASSIGNMENT_ID", db.BigInteger, Identity(), primary_key=True)
    allocation_code = db.Column("ALLOCATION_CODE", db.String(100), nullable=False, unique=True)
    equipment_id = db.Column("EQUIPMENT_ID", db.BigInteger, db.ForeignKey("ERP_EQUIPMENT.EQUIPMENT_ID"), nullable=False)
    source_system_code = db.Column("SOURCE_SYSTEM_CODE", db.String(50), nullable=False)
    ext_work_no = db.Column("EXT_WORK_NO", db.String(80), nullable=False)
    assigned_from = db.Column("ASSIGNED_FROM", db.DateTime)
    assigned_to = db.Column("ASSIGNED_TO", db.DateTime)
    status_code = db.Column("STATUS_CODE", db.String(30), default="REQUESTED")
    request_idempotency_key = db.Column("REQUEST_IDEMPOTENCY_KEY", db.String(100), unique=True)

class ErpMaterial(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_MATERIALS"
    material_id = db.Column("MATERIAL_ID", db.BigInteger, Identity(), primary_key=True)
    material_code = db.Column("MATERIAL_CODE", db.String(80), nullable=False, unique=True)
    material_name = db.Column("MATERIAL_NAME", db.String(200), nullable=False)
    material_category = db.Column("MATERIAL_CATEGORY", db.String(80))
    material_grade = db.Column("MATERIAL_GRADE", db.String(100))
    unit_code = db.Column("UNIT_CODE", db.String(20))
    specification = db.Column("SPECIFICATION", db.String(500))
    is_active = db.Column("IS_ACTIVE", db.String(1), default="Y")

class ErpMaterialLot(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_MATERIAL_LOTS"
    material_lot_id = db.Column("MATERIAL_LOT_ID", db.BigInteger, Identity(), primary_key=True)
    material_id = db.Column("MATERIAL_ID", db.BigInteger, db.ForeignKey("ERP_MATERIALS.MATERIAL_ID"), nullable=False)
    supplier_company_id = db.Column("SUPPLIER_COMPANY_ID", db.BigInteger, db.ForeignKey("ERP_COMPANIES.COMPANY_ID"))
    material_lot_code = db.Column("MATERIAL_LOT_CODE", db.String(100), nullable=False, unique=True)
    heat_no = db.Column("HEAT_NO", db.String(100))
    batch_no = db.Column("BATCH_NO", db.String(100))
    received_at = db.Column("RECEIVED_AT", db.DateTime)
    certificate_uri = db.Column("CERTIFICATE_URI", db.String(1000))
    quality_status = db.Column("QUALITY_STATUS", db.String(30), default="APPROVED")

class ErpWarehouse(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_WAREHOUSES"
    warehouse_id = db.Column("WAREHOUSE_ID", db.BigInteger, Identity(), primary_key=True)
    branch_id = db.Column("BRANCH_ID", db.BigInteger, db.ForeignKey("ERP_BRANCHES.BRANCH_ID"), nullable=False)
    warehouse_code = db.Column("WAREHOUSE_CODE", db.String(50), nullable=False)
    warehouse_name = db.Column("WAREHOUSE_NAME", db.String(150), nullable=False)
    location_desc = db.Column("LOCATION_DESC", db.String(300))
    status_code = db.Column("STATUS_CODE", db.String(30), default="ACTIVE")

class ErpInventoryBalance(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_INVENTORY_BALANCES"
    __table_args__ = (UniqueConstraint("WAREHOUSE_ID", "MATERIAL_LOT_ID", name="UK_WH_MATERIAL_LOT"),)
    inventory_id = db.Column("INVENTORY_ID", db.BigInteger, Identity(), primary_key=True)
    warehouse_id = db.Column("WAREHOUSE_ID", db.BigInteger, db.ForeignKey("ERP_WAREHOUSES.WAREHOUSE_ID"), nullable=False)
    material_lot_id = db.Column("MATERIAL_LOT_ID", db.BigInteger, db.ForeignKey("ERP_MATERIAL_LOTS.MATERIAL_LOT_ID"), nullable=False)
    quantity = db.Column("QUANTITY", db.Numeric(16, 3), nullable=False, default=0)
    reserved_quantity = db.Column("RESERVED_QUANTITY", db.Numeric(16, 3), nullable=False, default=0)
    updated_at = db.Column("UPDATED_AT", db.DateTime, default=datetime.utcnow)

class ErpMaterialReservation(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_MATERIAL_RESERVATIONS"
    reservation_id = db.Column("RESERVATION_ID", db.BigInteger, Identity(), primary_key=True)
    reservation_code = db.Column("RESERVATION_CODE", db.String(100), nullable=False, unique=True)
    inventory_id = db.Column("INVENTORY_ID", db.BigInteger, db.ForeignKey("ERP_INVENTORY_BALANCES.INVENTORY_ID"), nullable=False)
    source_system_code = db.Column("SOURCE_SYSTEM_CODE", db.String(50), nullable=False)
    ext_work_no = db.Column("EXT_WORK_NO", db.String(80), nullable=False)
    reserved_quantity = db.Column("RESERVED_QUANTITY", db.Numeric(16, 3), nullable=False)
    status_code = db.Column("STATUS_CODE", db.String(30), default="RESERVED")
    idempotency_key = db.Column("IDEMPOTENCY_KEY", db.String(100), unique=True)
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow)

class ErpInventoryTransaction(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_INVENTORY_TRANSACTIONS"
    transaction_id = db.Column("TRANSACTION_ID", db.BigInteger, Identity(), primary_key=True)
    inventory_id = db.Column("INVENTORY_ID", db.BigInteger, db.ForeignKey("ERP_INVENTORY_BALANCES.INVENTORY_ID"), nullable=False)
    transaction_no = db.Column("TRANSACTION_NO", db.String(100), nullable=False, unique=True)
    transaction_type = db.Column("TRANSACTION_TYPE", db.String(30), nullable=False)
    change_quantity = db.Column("CHANGE_QUANTITY", db.Numeric(16, 3), nullable=False)
    balance_quantity = db.Column("BALANCE_QUANTITY", db.Numeric(16, 3), nullable=False)
    source_system_code = db.Column("SOURCE_SYSTEM_CODE", db.String(50))
    ext_work_no = db.Column("EXT_WORK_NO", db.String(80))
    handler_employee_code = db.Column("HANDLER_EMPLOYEE_CODE", db.String(50))
    transaction_at = db.Column("TRANSACTION_AT", db.DateTime, default=datetime.utcnow)

class ErpPersonnelAssignment(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_PERSONNEL_ASSIGNMENTS"
    personnel_assignment_id = db.Column("PERSONNEL_ASSIGNMENT_ID", db.BigInteger, Identity(), primary_key=True)
    allocation_code = db.Column("ALLOCATION_CODE", db.String(100), nullable=False, unique=True)
    employee_id = db.Column("EMPLOYEE_ID", db.BigInteger, db.ForeignKey("ERP_EMPLOYEES.EMPLOYEE_ID"), nullable=False)
    source_system_code = db.Column("SOURCE_SYSTEM_CODE", db.String(50), nullable=False)
    ext_work_no = db.Column("EXT_WORK_NO", db.String(80), nullable=False)
    role_code = db.Column("ROLE_CODE", db.String(50))
    assigned_from = db.Column("ASSIGNED_FROM", db.DateTime)
    assigned_to = db.Column("ASSIGNED_TO", db.DateTime)
    status_code = db.Column("STATUS_CODE", db.String(30), default="RESERVED")

class ErpIntegrationRequest(db.Model):
    __bind_key__ = "ship_erp"
    __tablename__ = "ERP_INTEGRATION_REQUESTS"
    request_id = db.Column("REQUEST_ID", db.BigInteger, Identity(), primary_key=True)
    idempotency_key = db.Column("IDEMPOTENCY_KEY", db.String(100), nullable=False, unique=True)
    source_system_code = db.Column("SOURCE_SYSTEM_CODE", db.String(50), nullable=False)
    request_type = db.Column("REQUEST_TYPE", db.String(50), nullable=False)
    ext_work_no = db.Column("EXT_WORK_NO", db.String(80))
    request_payload = db.Column("REQUEST_PAYLOAD", db.Text)
    response_payload = db.Column("RESPONSE_PAYLOAD", db.Text)
    status_code = db.Column("STATUS_CODE", db.String(30), default="RECEIVED")
    created_at = db.Column("CREATED_AT", db.DateTime, default=datetime.utcnow)

Index("IX_ERP_EMP_BRANCH_STATUS", ErpEmployee.branch_id, ErpEmployee.employment_status)
Index("IX_ERP_EQUIPMENT_BRANCH", ErpEquipment.branch_id, ErpEquipment.operation_status)
Index("IX_ERP_RESERVATION_WORK", ErpMaterialReservation.ext_work_no, ErpMaterialReservation.status_code)
