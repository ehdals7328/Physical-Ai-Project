# web/models/partner_b2b.py
from datetime import datetime
from .schema import db  # ⭐ 상대 경로(.schema)로 안전하게 db 가져오기

# 1. 대한특수철강 B2B 공개 자재 카탈로그 View
class B2BDaehanMaterial(db.Model):
    __bind_key__ = 'partner_pg'
    __tablename__ = 'v_b2b_daehan_materials'

    supplier_sku = db.Column(db.String(50), primary_key=True)
    partner_id = db.Column(db.String(50))
    hq_item_code = db.Column(db.String(50))
    item_name = db.Column(db.String(100))
    material_category = db.Column(db.String(30))
    supplier_lot_no = db.Column(db.String(50))
    available_qty = db.Column(db.Numeric(10, 2))
    unit = db.Column(db.String(20))
    unit_price_krw = db.Column(db.Numeric(12, 2))
    lead_time_hours = db.Column(db.Integer)


# 2. 동해블록용접 B2B 공개 가용 인력 View
class B2BDonghaeWorker(db.Model):
    __bind_key__ = 'partner_pg'
    __tablename__ = 'v_b2b_donghae_workers'

    ext_worker_id = db.Column(db.String(30), primary_key=True)
    partner_id = db.Column(db.String(50))
    worker_name = db.Column(db.String(50))
    specialty_role = db.Column(db.String(30))
    cert_code = db.Column(db.String(50))
    weld_process = db.Column(db.String(30))
    cert_expiry_date = db.Column(db.Date)
    daily_rate_krw = db.Column(db.Numeric(10, 0))
    current_region = db.Column(db.String(30))
    dispatch_status = db.Column(db.String(20))


# 3. 한국중량물운송 B2B 공개 가용 장비/로봇 View
class B2BHankookEquipment(db.Model):
    __bind_key__ = 'partner_pg'
    __tablename__ = 'v_b2b_hankook_equipment'

    ext_asset_code = db.Column(db.String(50), primary_key=True)
    partner_id = db.Column(db.String(50))
    asset_name = db.Column(db.String(100))
    asset_category = db.Column(db.String(30))
    current_region = db.Column(db.String(30))
    max_load_ton = db.Column(db.Numeric(10, 2))
    operator_name = db.Column(db.String(50))
    hourly_rental_krw = db.Column(db.Numeric(10, 0))
    availability_status = db.Column(db.String(20))


# 4. B2B 통합 주문 접수 창구 테이블
class B2BSalesOrder(db.Model):
    __bind_key__ = 'partner_pg'
    __tablename__ = 'b2b_sales_orders'

    b2b_order_no = db.Column(db.String(50), primary_key=True)
    client_company = db.Column(db.String(50), default='COMP-HQ', nullable=False)
    hq_wo_number = db.Column(db.String(50), nullable=False)
    partner_id = db.Column(db.String(50), nullable=False)
    order_type = db.Column(db.String(30), nullable=False)
    ordered_ref_code = db.Column(db.String(50), nullable=False)
    order_qty = db.Column(db.Numeric(10, 2), default=1, nullable=False)
    allocated_lot_or_id = db.Column(db.String(50), nullable=False)
    delivery_yard_zone = db.Column(db.String(50), nullable=False)
    order_status = db.Column(db.String(20), default='CONFIRMED', nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# web/models/partner_b2b.py 맨 아래(또는 맨 위)에 추가
class B2BCompanyLocation(db.Model):
    """외부 협력사가 공개한 사업장 위치 및 계약기간 View (내부 임대료/기밀 메모 제외)"""
    __bind_key__ = 'partner_pg'
    __tablename__ = 'v_b2b_company_locations'

    partner_id = db.Column(db.String(50), primary_key=True)
    company_name = db.Column(db.String(100))
    specialty_desc = db.Column(db.String(100))
    representative = db.Column(db.String(50))
    contract_period = db.Column(db.String(50))
    address = db.Column(db.String(200))
    latitude = db.Column(db.Numeric(10, 6))
    longitude = db.Column(db.Numeric(10, 6))