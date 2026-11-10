"""
modules/agent/tools/erp_tools.py
Oracle SHIP_ERP 및 PostgreSQL PARTNER_B2B 듀얼 RDB 연동 도구
"""

import os
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from sqlalchemy import create_engine, text

from config import Config

# 1. Oracle SHIP_ERP 엔진
_ERP_URI = Config.SQLALCHEMY_BINDS.get("ship_erp", Config.SQLALCHEMY_DATABASE_URI)
_ENGINE_OPTS = getattr(Config, "SQLALCHEMY_ENGINE_OPTIONS", {"pool_pre_ping": True})
erp_engine = create_engine(_ERP_URI, **_ENGINE_OPTS)

# 2. PostgreSQL PARTNER_B2B 엔진
_PG_URI = Config.SQLALCHEMY_BINDS.get("partner_pg")
pg_engine = create_engine(_PG_URI, pool_pre_ping=True) if _PG_URI else None


def _convert_decimals(obj: Any) -> Any:
    """Oracle NUMBER 및 PostgreSQL NUMERIC 타입을 JSON 호환 타입으로 재귀 변환"""
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, dict):
        return {k: _convert_decimals(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_decimals(item) for item in obj]
    return obj


# ==============================================================================
# 1. 사내 ERP 전체 작업자 명단 실시간 DB 조회 도구 (Oracle: ERP_WORKER_CERT)
# ==============================================================================
@tool
def list_erp_workers(branch_code: Optional[str] = None) -> Dict[str, Any]:
    """Oracle SHIP_ERP의 ERP_WORKER_CERT 테이블에서 등록된 전체 작업자 명단을 조회합니다."""
    params = {}
    if branch_code and str(branch_code).strip() and str(branch_code).upper() != "NONE":
        sql = """
            SELECT worker_id, worker_name, employment_type, company_code, branch_code,
                   department_name, role, phone, cert_code, weld_process,
                   cert_expiry_date, status
            FROM ERP_WORKER_CERT
            WHERE UPPER(TRIM(branch_code)) = UPPER(TRIM(:branch_code))
            ORDER BY worker_id ASC
        """
        params["branch_code"] = str(branch_code).strip()
    else:
        sql = """
            SELECT worker_id, worker_name, employment_type, company_code, branch_code,
                   department_name, role, phone, cert_code, weld_process,
                   cert_expiry_date, status
            FROM ERP_WORKER_CERT
            ORDER BY worker_id ASC
        """

    today_str = datetime.now().strftime("%Y-%m-%d")
    workers_list = []

    try:
        with erp_engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
            for r in rows:
                exp_str = str(r["cert_expiry_date"] or "").strip()
                st_str = str(r["status"] or "").strip().upper()
                is_qualified = (st_str == "VALID") and (not exp_str or exp_str >= today_str)

                workers_list.append({
                    "worker_id": str(r["worker_id"] or "").strip(),
                    "worker_name": str(r["worker_name"] or "").strip(),
                    "employment_type": r["employment_type"],
                    "company_code": r["company_code"],
                    "branch_code": r["branch_code"],
                    "department_name": r["department_name"],
                    "role": r["role"],
                    "phone": r["phone"],
                    "cert_code": r["cert_code"],
                    "weld_process": r["weld_process"],
                    "cert_expiry_date": exp_str,
                    "raw_status": st_str,
                    "status": "투입 가능" if is_qualified else f"투입 불가 ({st_str})",
                    "is_valid": is_qualified,
                })

        return _convert_decimals({
            "status": "SUCCESS",
            "total_count": len(workers_list),
            "workers": workers_list,
        })
    except Exception as e:
        print(f"❌ [list_erp_workers 오류]: {e}")
        return {"status": "ERROR", "error": str(e), "total_count": 0, "workers": []}


# ==============================================================================
# 2. 원청(본사) SHIP_ERP 통합 진단 도구 (Oracle 4대 마스터 테이블 동적 연동)
# ==============================================================================
@tool
def check_hq_erp_resources(
    block_id: Optional[str] = None,
    material: Optional[str] = None,
    worker_name: Optional[str] = None,
    equipment: Optional[str] = None,
    branch_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    [SHIP_ERP 4대 마스터 통합 조회]
    1) ERP_BLOCK_MASTER: 블록 번호로 호선, 강종, 두께, 공정, 중량, 배정 야드 자동 인출
    2) ERP_BOM_INVENTORY: 강종/공정에 맞는 강판(STEEL_PLATE) 및 와이어(FILLER_WIRE) 실가용 재고 확인
    3) ERP_WORKER_CERT: 투입 작업자의 선급 용접 자격증 유효기간 검증
    4) ERP_VEHICLES_EQUIPMENT: 블록 중량을 감당할 수 있는 대기 중(STANDBY) 차량/장비 동적 매칭
    """
    result: Dict[str, Any] = {
        "status": "SUCCESS",
        "block_master": None,
        "inventory": {},
        "vehicles_equipment": [],
        "assigned_vehicle_code": None,
        "worker_qualification": None,
        "stock_available": True,
        "worker_qualified": True,
        "require_outsourcing": False,
        "outsourcing_reason": None,
        "assigned_lot_no": None,
        "selected_wire_code": None,
        "selected_plate_code": None,
    }

    block_weight = 0.0
    target_process = None
    today_str = datetime.now().strftime("%Y-%m-%d")

    try:
        with erp_engine.connect() as conn:
            # 1) ERP_BLOCK_MASTER 조회
            if block_id:
                sql_block = """
                    SELECT block_id, ship_code, block_name, steel_type, thickness_mm,
                           joint_type, process, target_dock, branch_code, weight_ton
                    FROM ERP_BLOCK_MASTER
                    WHERE UPPER(block_id) = :block_id
                """
                b_row = conn.execute(
                    text(sql_block), {"block_id": block_id.upper().strip()}
                ).mappings().first()

                if b_row:
                    block_weight = float(b_row["weight_ton"] or 0.0)
                    target_process = b_row["process"]
                    result["block_master"] = {
                        "block_id": b_row["block_id"],
                        "ship_code": b_row["ship_code"],
                        "block_name": b_row["block_name"],
                        "material": b_row["steel_type"],
                        "thickness": str(b_row["thickness_mm"]),
                        "joint_type": b_row["joint_type"],
                        "process": b_row["process"],
                        "location": b_row["target_dock"],
                        "branch_code": b_row["branch_code"],
                        "weight_ton": block_weight,
                    }
                    if not material:
                        material = b_row["steel_type"]
                    if not branch_code:
                        branch_code = b_row["branch_code"]

            # 2) ERP_BOM_INVENTORY 조회 (카테고리/속성 기반 동적 매칭)
            sql_inv = """
                SELECT item_code, item_name, material_category, lot_no,
                       supplier_code, branch_code, total_qty, reserved_qty, unit
                FROM ERP_BOM_INVENTORY
            """
            inv_rows = conn.execute(text(sql_inv)).mappings().all()

            matched_plate = None
            matched_wire = None

            for r in inv_rows:
                code = str(r["item_code"])
                cat = str(r["material_category"] or "").upper()
                total = float(r["total_qty"] or 0.0)
                reserved = float(r["reserved_qty"] or 0.0)
                avail = round(max(0.0, total - reserved), 2)

                item_info = {
                    "item_code": code,
                    "item_name": r["item_name"],
                    "material_category": cat,
                    "lot_no": r["lot_no"],
                    "supplier_code": r["supplier_code"],
                    "branch_code": r["branch_code"],
                    "total_qty": total,
                    "reserved_qty": reserved,
                    "available_qty": avail,
                    "unit": r["unit"],
                }
                result["inventory"][code] = item_info

                if cat == "STEEL_PLATE":
                    if material and material.upper() in code.upper():
                        matched_plate = item_info
                    elif matched_plate is None:
                        matched_plate = item_info

                elif cat == "FILLER_WIRE":
                    if target_process and target_process.upper() in code.upper():
                        matched_wire = item_info
                    elif material and "EH36" in material.upper() and "SAW" in code.upper():
                        matched_wire = item_info
                    elif matched_wire is None:
                        matched_wire = item_info

            if matched_plate:
                result["assigned_lot_no"] = matched_plate["lot_no"]
                result["selected_plate_code"] = matched_plate["item_code"]

            if matched_wire:
                result["selected_wire_code"] = matched_wire["item_code"]
                if not result["assigned_lot_no"]:
                    result["assigned_lot_no"] = matched_wire["lot_no"]

                if matched_wire["available_qty"] <= 0:
                    result["stock_available"] = False
                    result["require_outsourcing"] = True
                    result["outsourcing_reason"] = (
                        f"MATERIAL_SHORTAGE: 용접 와이어({matched_wire['item_code']}) 가용 재고 소진 "
                        f"(공급사 {matched_wire.get('supplier_code')} 긴급 발주 필요)"
                    )

            # 3) ERP_VEHICLES_EQUIPMENT 대기 장비/차량 동적 조회
            sql_veh = """
                SELECT asset_code, asset_name, asset_category, company_code,
                       branch_code, current_zone, max_load_ton, driver_worker_id, operation_status
                FROM ERP_VEHICLES_EQUIPMENT
            """
            veh_rows = conn.execute(text(sql_veh)).mappings().all()

            hq_standby_tp = None
            ext_standby_tp = None
            fallback_standby_veh = None

            for v in veh_rows:
                v_dict = dict(v)
                result["vehicles_equipment"].append(v_dict)

                op_stat = str(v["operation_status"] or "").upper()
                a_cat = str(v["asset_category"] or "").upper()
                max_ton = float(v["max_load_ton"] or 0.0)

                if op_stat == "STANDBY":
                    if fallback_standby_veh is None:
                        fallback_standby_veh = v["asset_code"]

                    if a_cat == "VEHICLE_TRANSPORTER" and max_ton >= block_weight:
                        if str(v["company_code"] or "").upper() == "COMP-HQ":
                            hq_standby_tp = v["asset_code"]
                        else:
                            ext_standby_tp = v["asset_code"]

            result["assigned_vehicle_code"] = hq_standby_tp or fallback_standby_veh

            if equipment and any(k in equipment.upper() for k in ["트랜스포터", "운송", "배차", "TP", "TRANSPORTER"]):
                if not hq_standby_tp:
                    result["stock_available"] = False
                    result["require_outsourcing"] = True
                    result["outsourcing_reason"] = "TRANSPORT_SHORTAGE: 본사 가용 트랜스포터(STANDBY) 배차 불가"
                    if ext_standby_tp:
                        result["assigned_vehicle_code"] = ext_standby_tp

            # 4) ERP_WORKER_CERT 작업자 자격 검증
            if worker_name and str(worker_name).strip() not in ("None", "미배정", "UNASSIGNED"):
                target_name = str(worker_name).strip()
                sql_w = """
                    SELECT worker_id, worker_name, role, cert_code,
                           cert_expiry_date, status, weld_process, company_code, branch_code
                    FROM ERP_WORKER_CERT
                    WHERE worker_name = :w_name
                       OR UPPER(TRIM(worker_id)) = :w_id
                       OR worker_name LIKE :w_like
                """
                w_row = conn.execute(
                    text(sql_w),
                    {
                        "w_name": target_name,
                        "w_id": target_name.upper(),
                        "w_like": f"%{target_name}%",
                    },
                ).mappings().first()

                if w_row:
                    exp_str = str(w_row["cert_expiry_date"] or "").strip()
                    st_str = str(w_row["status"] or "").strip().upper()
                    is_valid = (st_str == "VALID") and (not exp_str or exp_str >= today_str)

                    result["worker_qualification"] = {
                        "worker_id": str(w_row["worker_id"] or "").strip(),
                        "worker_name": str(w_row["worker_name"] or "").strip(),
                        "role": w_row["role"],
                        "cert_code": w_row["cert_code"],
                        "cert_expiry_date": exp_str,
                        "status": st_str,
                        "weld_process": w_row["weld_process"],
                        "company_code": w_row["company_code"],
                        "branch_code": w_row["branch_code"],
                        "is_valid": is_valid,
                    }
                    result["worker_qualified"] = is_valid
                else:
                    result["worker_qualification"] = {
                        "worker_name": target_name,
                        "is_valid": False,
                        "reason": f"ERP_WORKER_CERT 원장에 등록되지 않은 작업자 ('{target_name}')",
                    }
                    result["worker_qualified"] = False
            else:
                result["worker_qualification"] = {
                    "worker_name": "미지정",
                    "is_valid": False,
                    "reason": "투입 작업자가 지정되지 않았습니다.",
                }
                result["worker_qualified"] = False

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"SHIP_ERP 조회 실패: {str(e)}"

    return _convert_decimals(result)


# ==============================================================================
# 3. 외부 협력사 PostgreSQL B2B 공개용 View 실시간 조회 도구 (query_partner_erp_b2b)
# ==============================================================================
@tool
def query_partner_erp_b2b(
    shortage_reason: str = "MATERIAL_SHORTAGE",
    material: str = "",
    required_wire_kg: float = 20.0,
    target_region: str = "울산",
) -> Dict[str, Any]:
    """
    [PostgreSQL PARTNER_B2B 공개용 View 실시간 조회]
    본사 자재 결품, 장비 부족 또는 외주 인력 필요 시 외부 협력사 View를 조회합니다:
    - v_b2b_daehan_materials: 대한특수철강 가용 자재 및 납품 리드타임
    - v_b2b_hankook_equipment: 한국중량물운송 STANDBY 대기 중장비/차량
    - v_b2b_donghae_workers: 동해블록용접 투입 가능(AVAILABLE) 외주 용접 인력
    """
    if not pg_engine:
        return {"status": "ERROR", "error": "PARTNER_PG_DATABASE_URL이 설정되지 않았습니다."}

    is_transport = any(k in shortage_reason.upper() for k in ["TRANSPORT", "운송", "트랜스포터", "HEAVY", "FL", "TP"])
    is_worker = any(k in shortage_reason.upper() for k in ["WORKER", "인력", "용접사", "MANPOWER"])

    try:
        with pg_engine.connect() as conn:
            # 1) 중량물 운송/배차 부족 시 -> v_b2b_hankook_equipment 조회
            if is_transport:
                sql = """
                    SELECT partner_id, ext_asset_code, asset_name, asset_category,
                           current_region, max_load_ton, operator_name, hourly_rental_krw, availability_status
                    FROM v_b2b_hankook_equipment
                    WHERE availability_status = 'STANDBY'
                    ORDER BY max_load_ton DESC
                    LIMIT 1
                """
                row = conn.execute(text(sql)).mappings().first()
                if row:
                    return _convert_decimals({
                        "b2b_inquiry_timestamp": datetime.now().isoformat(),
                        "source": "PostgreSQL PARTNER_B2B (v_b2b_hankook_equipment)",
                        "selected_partner": {
                            "partner_id": row["partner_id"],
                            "partner_name": "한국중량물운송(주)",
                            "company_type": "SUBCONTRACTOR",
                            "specialty_field": "LOGISTICS",
                            "assigned_vehicle_code": row["ext_asset_code"],
                            "asset_name": row["asset_name"],
                            "certified_driver": row["operator_name"],
                            "max_load_ton": row["max_load_ton"],
                            "hourly_rental_krw": row["hourly_rental_krw"],
                            "lead_time_hours": 1.5,
                            "can_accept_order": True,
                        },
                    })

            # 2) 외주 인력 지원 필요 시 -> v_b2b_donghae_workers 조회
            elif is_worker:
                sql = """
                    SELECT partner_id, ext_worker_id, worker_name, specialty_role,
                           cert_code, weld_process, TO_CHAR(cert_expiry_date, 'YYYY-MM-DD') AS cert_expiry_date,
                           daily_rate_krw, current_region, dispatch_status
                    FROM v_b2b_donghae_workers
                    WHERE dispatch_status = 'AVAILABLE'
                    ORDER BY daily_rate_krw ASC
                    LIMIT 1
                """
                row = conn.execute(text(sql)).mappings().first()
                if row:
                    return _convert_decimals({
                        "b2b_inquiry_timestamp": datetime.now().isoformat(),
                        "source": "PostgreSQL PARTNER_B2B (v_b2b_donghae_workers)",
                        "selected_partner": {
                            "partner_id": row["partner_id"],
                            "partner_name": "동해블록용접(주)",
                            "company_type": "SUBCONTRACTOR",
                            "specialty_field": "BLOCK_WELDING",
                            "assigned_worker_id": row["ext_worker_id"],
                            "worker_name": row["worker_name"],
                            "cert_code": row["cert_code"],
                            "weld_process": row["weld_process"],
                            "daily_rate_krw": row["daily_rate_krw"],
                            "can_accept_order": True,
                        },
                    })

            # 3) 기본: 자재 결품 시 -> v_b2b_daehan_materials 조회
            else:
                sql = """
                    SELECT partner_id, supplier_sku, hq_item_code, item_name,
                           material_category, supplier_lot_no, available_qty, unit,
                           unit_price_krw, lead_time_hours
                    FROM v_b2b_daehan_materials
                    WHERE (:mat = '' OR material_category ILIKE :mat_like OR item_name ILIKE :mat_like)
                      AND available_qty > 0
                    ORDER BY available_qty DESC
                    LIMIT 1
                """
                row = conn.execute(text(sql), {"mat": material, "mat_like": f"%{material}%"}).mappings().first()
                if row:
                    return _convert_decimals({
                        "b2b_inquiry_timestamp": datetime.now().isoformat(),
                        "source": "PostgreSQL PARTNER_B2B (v_b2b_daehan_materials)",
                        "selected_partner": {
                            "partner_id": row["partner_id"],
                            "partner_name": "대한특수철강",
                            "company_type": "SUPPLIER",
                            "specialty_field": "STEEL_SUPPLY",
                            "supplier_sku": row["supplier_sku"],
                            "item_name": row["item_name"],
                            "partner_lot_no": row["supplier_lot_no"],
                            "available_qty": row["available_qty"],
                            "unit": row["unit"],
                            "unit_price_krw": row["unit_price_krw"],
                            "lead_time_hours": row["lead_time_hours"],
                            "can_accept_order": True,
                        },
                    })

    except Exception as e:
        return {"status": "ERROR", "error": f"PostgreSQL B2B View 조회 실패: {str(e)}"}

    return {"status": "EMPTY", "selected_partner": {}}


# ==============================================================================
# 4. [Logical Bridge] 트랜잭션 마감 (Oracle MES 원장 + PostgreSQL b2b_sales_orders)
# ==============================================================================
@tool
def commit_erp_work_order(
    block_id: str,
    material: str,
    allocated_lot_no: str,
    worker_name: str,
    worker_id: str,
    work_no: Optional[str] = None,
    assigned_vehicle_code: Optional[str] = None,
    wire_item_code: Optional[str] = None,
    is_outsourced: bool = False,
    partner_id: Optional[str] = None,
    reserve_wire_kg: float = 15.0,
) -> Dict[str, Any]:
    """
    최종 ERP 트랜잭션 커밋
    - 본사 Oracle: MES_WORK_ORDER_DISPATCH 레코드 생성 및 ERP_BOM_INVENTORY 재고 선점
    - 협력사 PostgreSQL: 외주 발주 시 b2b_sales_orders 테이블에 자동 수주 등록
    """
    if not worker_id or worker_id in ("UNASSIGNED", "미배정", "None"):
        return {
            "erp_transaction_status": "ERROR",
            "work_order_number": work_no or "WO-FAILED",
            "error": "외래키 무결성 위배 방지: 유효한 작업자 사번(WORKER_ID)이 없어 원장에 커밋할 수 없습니다.",
        }

    wo_number = work_no or f"WO-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    exec_type = "OUTSOURCED_B2B" if is_outsourced else "IN_HOUSE"
    target_partner = partner_id if (is_outsourced and partner_id) else "COMP-HQ"
    created_at_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    remaining_avail = 0.0

    try:
        # [Step 1] 본사 Oracle SHIP_ERP 트랜잭션 마감
        with erp_engine.begin() as conn:
            target_wire = wire_item_code
            if not target_wire:
                w_row = conn.execute(
                    text("SELECT item_code FROM ERP_BOM_INVENTORY WHERE material_category = 'FILLER_WIRE' FETCH FIRST 1 ROWS ONLY")
                ).mappings().first()
                if w_row:
                    target_wire = w_row["item_code"]

            if not is_outsourced and target_wire:
                conn.execute(
                    text("UPDATE ERP_BOM_INVENTORY SET reserved_qty = NVL(reserved_qty, 0) + :qty WHERE item_code = :code"),
                    {"qty": reserve_wire_kg, "code": target_wire},
                )
                inv_row = conn.execute(
                    text("SELECT total_qty, reserved_qty FROM ERP_BOM_INVENTORY WHERE item_code = :code"),
                    {"code": target_wire},
                ).mappings().first()
                if inv_row:
                    remaining_avail = round(max(0.0, float(inv_row["total_qty"] or 0) - float(inv_row["reserved_qty"] or 0)), 2)

            check_sql = "SELECT COUNT(*) FROM MES_WORK_ORDER_DISPATCH WHERE wo_number = :wo_number"
            cnt = conn.execute(text(check_sql), {"wo_number": wo_number}).scalar()

            if cnt and cnt > 0:
                update_sql = """
                    UPDATE MES_WORK_ORDER_DISPATCH SET
                        block_id = :block_id,
                        execution_type = :exec_type,
                        partner_id = :partner_id,
                        allocated_lot_no = :lot_no,
                        worker_id = :worker_id,
                        worker_name = :worker_name,
                        assigned_vehicle_code = :vehicle,
                        status = 'COMMITTED'
                    WHERE wo_number = :wo_number
                """
                conn.execute(text(update_sql), {
                    "wo_number": wo_number,
                    "block_id": str(block_id or "B-104"),
                    "exec_type": exec_type,
                    "partner_id": target_partner,
                    "lot_no": str(allocated_lot_no or "LOT-DEFAULT"),
                    "worker_id": str(worker_id),
                    "worker_name": str(worker_name),
                    "vehicle": assigned_vehicle_code,
                })
            else:
                insert_sql = """
                    INSERT INTO MES_WORK_ORDER_DISPATCH (
                        wo_number, block_id, execution_type, partner_id,
                        allocated_lot_no, worker_id, worker_name,
                        assigned_vehicle_code, status, created_at
                    )
                    VALUES (
                        :wo_number, :block_id, :exec_type, :partner_id,
                        :lot_no, :worker_id, :worker_name,
                        :vehicle, 'COMMITTED', :created_at
                    )
                """
                conn.execute(text(insert_sql), {
                    "wo_number": wo_number,
                    "block_id": str(block_id or "B-104"),
                    "exec_type": exec_type,
                    "partner_id": target_partner,
                    "lot_no": str(allocated_lot_no or "LOT-DEFAULT"),
                    "worker_id": str(worker_id),
                    "worker_name": str(worker_name),
                    "vehicle": assigned_vehicle_code,
                    "created_at": created_at_str,
                })

        # [Step 2] 외주 발주인 경우 협력사 PostgreSQL b2b_sales_orders 테이블에 자동 수주 등록
        b2b_order_no = f"B2B-PO-{wo_number}"
        if is_outsourced and pg_engine:
            try:
                order_type = "HEAVY_TRANSPORT" if "HANKOOK" in str(target_partner) else "WELDING_MATERIAL"
                insert_b2b_sql = """
                    INSERT INTO b2b_sales_orders (
                        b2b_order_no, client_company, hq_wo_number, partner_id,
                        order_type, ordered_ref_code, order_qty, allocated_lot_or_id,
                        delivery_yard_zone, order_status, created_at
                    ) VALUES (
                        :b2b_no, 'COMP-HQ', :wo_no, :p_id,
                        :o_type, :ref_code, :qty, :lot_or_id,
                        :yard_zone, 'CONFIRMED', CURRENT_TIMESTAMP
                    )
                    ON CONFLICT (b2b_order_no) DO UPDATE SET order_status = 'CONFIRMED'
                """
                with pg_engine.begin() as pg_conn:
                    pg_conn.execute(text(insert_b2b_sql), {
                        "b2b_no": b2b_order_no,
                        "wo_no": wo_number,
                        "p_id": target_partner,
                        "o_type": order_type,
                        "ref_code": wire_item_code or "WIRE-CSF71T",
                        "qty": reserve_wire_kg,
                        "lot_or_id": allocated_lot_no or "EXT-LOT-DEFAULT",
                        "yard_zone": "ZONE-A-BAY1",
                    })
            except Exception as e:
                print(f"⚠️ [PostgreSQL b2b_sales_orders 등록 실패]: {e}")

        return _convert_decimals({
            "erp_transaction_status": "COMMITTED",
            "work_order_number": wo_number,
            "execution_type": exec_type,
            "partner_id": target_partner,
            "allocated_lot_no": allocated_lot_no,
            "worker_id": worker_id,
            "worker_name": worker_name,
            "assigned_vehicle_code": assigned_vehicle_code,
            "reserved_wire_kg": reserve_wire_kg if not is_outsourced else 0.0,
            "remaining_hq_wire_kg": remaining_avail,
            "b2b_po_ack": f"PO-ACK-{wo_number}-{target_partner}" if is_outsourced else None,
            "b2b_sales_order_no": b2b_order_no if is_outsourced else None,
            "committed_at": created_at_str,
        })
    except Exception as e:
        print(f"❌ [commit_erp_work_order 커밋 에러 상세]: {e}")
        return {
            "erp_transaction_status": "ERROR",
            "work_order_number": wo_number,
            "error": f"MES_WORK_ORDER_DISPATCH 커밋 실패: {str(e)}",
        }