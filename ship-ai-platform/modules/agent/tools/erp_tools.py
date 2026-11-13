"""
modules/agent/tools/erp_tools.py
본사 Oracle SHIP_ERP 및 외부 협력사 PostgreSQL B2B 연동 도구
- config.py 및 .env의 SHIP_ERP_DATABASE_URL / PARTNER_PG_DATABASE_URL 직접 로드
- 용접 공정(WPS) 상호 일치 검증 및 대상 야드 우선 배차 로직 탑재
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain_core.tools import tool
from sqlalchemy import create_engine, text

# 프로젝트 루트의 .env 로드 (config.py와 동일한 환경변수 사용)
PLATFORM_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PLATFORM_ROOT / ".env")

# ==============================================================================
# 1. SQLAlchemy DB 엔진 설정 (.env의 SHIP_ERP_DATABASE_URL / PARTNER_PG_DATABASE_URL 사용)
# ==============================================================================
ORACLE_URI = os.getenv("SHIP_ERP_DATABASE_URL")
PG_URI = os.getenv("PARTNER_PG_DATABASE_URL")

try:
    erp_engine = create_engine(ORACLE_URI, pool_pre_ping=True) if ORACLE_URI else None
except Exception as e:
    print(f"⚠️ [Oracle SHIP_ERP 엔진 초기화 경고]: {e}")
    erp_engine = None

try:
    pg_engine = create_engine(PG_URI, pool_pre_ping=True) if PG_URI else None
except Exception as e:
    print(f"⚠️ [PostgreSQL B2B 엔진 초기화 경고]: {e}")
    pg_engine = None


# ==============================================================================
# 2. 동적 유틸리티 및 마스터 메타데이터 조회 함수
# ==============================================================================
def calculate_required_wire_kg(
    estimated_length_mm: Optional[float] = None,
    weight_ton: Optional[float] = None,
) -> float:
    """비전 계측 용접선 길이(mm) 또는 블록 중량(ton)을 기반으로 필요 와이어 소모량(kg) 산출"""
    if estimated_length_mm and float(estimated_length_mm) > 0:
        return round((float(estimated_length_mm) / 1000.0) * 6.25, 2)
    if weight_ton and float(weight_ton) > 0:
        return round(float(weight_ton) * 0.08, 2)
    return 15.0


def get_branch_context_from_db(branch_code: Optional[str]) -> Dict[str, Any]:
    """ERP_BRANCHES 테이블에서 야드 코드에 매핑된 지역명 및 기본 작업구역 조회"""
    if not erp_engine or not branch_code:
        return {}
    try:
        with erp_engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT branch_code, company_code, region_name, branch_name
                    FROM ERP_BRANCHES
                    WHERE branch_code = :code
                """),
                {"code": branch_code},
            ).mappings().first()
            if row:
                res = dict(row)
                res["default_zone"] = "DOCK-1" if "GEOJE" in str(res.get("branch_code", "")).upper() else "ZONE-A-BAY1"
                return res
    except Exception as e:
        print(f"⚠️ [get_branch_context_from_db 오류]: {e}")
    return {}


def get_active_partner_from_db(specialty_field: str = "BLOCK_WELDING") -> Dict[str, Any]:
    """ERP_COMPANIES 테이블에서 전문 분야 또는 구분에 맞는 활성 업체 정보 동적 조회"""
    if not erp_engine:
        return {"company_code": "COMP-HQ", "partner_id": "VEND-DONGHAE-WELD", "partner_name": "동해블록용접(주)"}
    try:
        with erp_engine.connect() as conn:
            if specialty_field == "HQ":
                row = conn.execute(
                    text("""
                        SELECT company_code, company_name, company_type, specialty_field
                        FROM ERP_COMPANIES
                        WHERE company_type = 'HQ' AND contract_status = 'ACTIVE'
                        FETCH FIRST 1 ROWS ONLY
                    """)
                ).mappings().first()
            else:
                row = conn.execute(
                    text("""
                        SELECT company_code, company_name, company_type, specialty_field
                        FROM ERP_COMPANIES
                        WHERE specialty_field = :spec AND contract_status = 'ACTIVE'
                        FETCH FIRST 1 ROWS ONLY
                    """),
                    {"spec": specialty_field},
                ).mappings().first()

            if row:
                data = dict(row)
                data["partner_id"] = data["company_code"]
                data["partner_name"] = data["company_name"]
                return data
    except Exception as e:
        print(f"⚠️ [get_active_partner_from_db 오류]: {e}")

    return {
        "company_code": "VEND-DONGHAE-WELD",
        "partner_id": "VEND-DONGHAE-WELD",
        "partner_name": "동해블록용접(주)",
        "specialty_field": specialty_field,
    }


def list_registered_partners_from_db() -> List[Dict[str, Any]]:
    """ERP_COMPANIES 테이블에서 본사(HQ)를 제외한 활성 외부 협력사 목록을 동적으로 인출"""
    if not erp_engine:
        return []
    try:
        with erp_engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT company_code, company_name, company_type, specialty_field
                    FROM ERP_COMPANIES
                    WHERE company_type != 'HQ' AND contract_status = 'ACTIVE'
                    ORDER BY company_code ASC
                """)
            ).mappings().all()
            return [dict(r) for r in rows]
    except Exception:
        return []


# ==============================================================================
# 3. LangChain 에이전트 호출용 @tool 함수들
# ==============================================================================
@tool
def list_erp_workers(branch_code: Optional[str] = None) -> Dict[str, Any]:
    """Oracle ERP_WORKER_CERT 원장에서 사내외 전체 작업자 명단 및 자격 상태를 조회합니다."""
    if not erp_engine:
        return {"status": "ERROR", "workers": []}

    try:
        with erp_engine.connect() as conn:
            if branch_code:
                rows = conn.execute(
                    text("""
                        SELECT worker_id, worker_name, employment_type, company_code, branch_code,
                               department_name, role, cert_code, weld_process, cert_expiry_date, status
                        FROM ERP_WORKER_CERT
                        WHERE branch_code = :branch
                        ORDER BY worker_id ASC
                    """),
                    {"branch": branch_code},
                ).mappings().all()
            else:
                rows = conn.execute(
                    text("""
                        SELECT worker_id, worker_name, employment_type, company_code, branch_code,
                               department_name, role, cert_code, weld_process, cert_expiry_date, status
                        FROM ERP_WORKER_CERT
                        ORDER BY worker_id ASC
                    """)
                ).mappings().all()

            workers = []
            for r in rows:
                d = dict(r)
                d["is_valid"] = str(d.get("status", "")).upper() == "VALID"
                workers.append(d)

            return {"status": "SUCCESS", "workers": workers}
    except Exception as e:
        return {"status": "ERROR", "error_message": str(e), "workers": []}


@tool
def check_hq_erp_resources(
    block_id: Optional[str] = None,
    material: Optional[str] = None,
    worker_name: Optional[str] = None,
    equipment: Optional[str] = None,
    branch_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    HQ Oracle SHIP_ERP 원장을 조회하여 블록 제원, 자재 재고, 작업자 자격/공정 일치 여부,
    해당 야드의 대기 차량 및 외주 필요 여부를 정밀 진단합니다.
    """
    result: Dict[str, Any] = {
        "status": "SUCCESS",
        "hq_company_code": "COMP-HQ",
        "block_master": None,
        "branch_info": None,
        "worker_qualification": None,
        "worker_qualified": True,
        "stock_available": True,
        "require_outsourcing": False,
        "outsourcing_reason": None,
        "inventory": {},
        "selected_plate_code": None,
        "selected_wire_code": None,
        "selected_wire_supplier": None,
        "assigned_lot_no": None,
        "vehicles_equipment": [],
        "assigned_vehicle_code": None,
    }

    if not erp_engine:
        result["status"] = "ERROR"
        result["error_message"] = "SHIP_ERP_DATABASE_URL이 설정되지 않았습니다."
        return result

    try:
        with erp_engine.connect() as conn:
            # 1. 블록 마스터 제원 조회
            if block_id:
                b_row = conn.execute(
                    text("""
                        SELECT block_id, ship_code, block_name, steel_type, thickness_mm,
                               joint_type, process, target_dock, branch_code, weight_ton
                        FROM ERP_BLOCK_MASTER
                        WHERE block_id = :block_id
                    """),
                    {"block_id": block_id},
                ).mappings().first()

                if b_row:
                    result["block_master"] = {
                        "block_id": b_row["block_id"],
                        "ship_code": b_row["ship_code"],
                        "block_name": b_row["block_name"],
                        "material": b_row["steel_type"],
                        "thickness": b_row["thickness_mm"],
                        "joint_type": b_row["joint_type"],
                        "process": b_row["process"],
                        "location": b_row["target_dock"],
                        "branch_code": b_row["branch_code"],
                        "weight_ton": float(b_row["weight_ton"]) if b_row["weight_ton"] else 0.0,
                    }
                    if not branch_code:
                        branch_code = b_row["branch_code"]

            target_branch = branch_code or (
                result["block_master"]["branch_code"] if result["block_master"] else "BR-GEOJE-01"
            )

            # 2. 야드(Branch) 정보 조회
            br_row = conn.execute(
                text("""
                    SELECT branch_code, company_code, region_name, branch_name
                    FROM ERP_BRANCHES
                    WHERE branch_code = :branch_code
                """),
                {"branch_code": target_branch},
            ).mappings().first()

            if br_row:
                result["hq_company_code"] = br_row["company_code"]
                result["branch_info"] = {
                    "branch_code": br_row["branch_code"],
                    "company_code": br_row["company_code"],
                    "region_name": br_row["region_name"],
                    "branch_name": br_row["branch_name"],
                    "default_zone": "DOCK-1" if "GEOJE" in str(br_row["branch_code"]).upper() else "ZONE-A-BAY1",
                }
            # 👇 [추가] 2-1. 해당 야드 소속 현장 반장/관리자(검토자) 동적 조회
            foreman_row = conn.execute(
                text("""
                    SELECT worker_name, worker_id
                    FROM ERP_WORKER_CERT
                    WHERE branch_code = :b_code
                      AND UPPER(role) IN ('WELD_LEADER', 'FOREMAN', 'MANAGER', 'ENGINEER')
                      AND status = 'VALID'
                    ORDER BY CASE UPPER(role)
                        WHEN 'WELD_LEADER' THEN 1
                        WHEN 'FOREMAN' THEN 2
                        WHEN 'MANAGER' THEN 3
                        ELSE 4
                    END
                    FETCH FIRST 1 ROWS ONLY
                """),
                {"b_code": target_branch},
            ).mappings().first()

            result["foreman_name"] = (
                f"{foreman_row['worker_name']}({foreman_row['worker_id']})"
                if foreman_row
                else None
            )

            # 3. 작업자 선급 자격증 유효성 및 용접 공정(WPS) 일치 검증
            if worker_name:
                # 👇 [추가] "최선임 반장", "김철수 용접사", "윤로봇 님" 등에서 직책/호칭 제거 후 순수 성명 추출
                clean_worker_name = re.sub(
                    r"\s*(반장|작업자|용접사|용접공|오퍼레이터|기사|엔지니어|팀장|님)$",
                    "",
                    str(worker_name).strip(),
                ).strip()

                w_row = conn.execute(
                    text("""
                        SELECT worker_id, worker_name, role, company_code, branch_code,
                               cert_code, weld_process, cert_expiry_date, status
                        FROM ERP_WORKER_CERT
                        WHERE worker_name = :worker_name
                        ORDER BY cert_expiry_date DESC
                        FETCH FIRST 1 ROWS ONLY
                    """),
                    {"worker_name": clean_worker_name},  # 👈 여기 파라미터도 clean_worker_name으로 변경
                ).mappings().first()

                if w_row:
                    is_cert_valid = (str(w_row["status"]).upper() == "VALID")
                    req_process = result["block_master"].get("process") if result["block_master"] else None
                    worker_proc = w_row["weld_process"]

                    is_proc_matched = True
                    proc_mismatch_reason = None

                    if req_process and worker_proc and str(req_process).upper() != "NA" and str(worker_proc).upper() != "NA":
                        if str(req_process).upper() != str(worker_proc).upper():
                            is_proc_matched = False
                            proc_mismatch_reason = (
                                f"공정 불일치: 대상 블록({block_id})의 지정 공정은 '{req_process}'이나 "
                                f"작업자({w_row['worker_name']})의 인증 공정은 '{worker_proc}'입니다."
                            )

                    is_fully_qualified = is_cert_valid and is_proc_matched

                    result["worker_qualification"] = {
                        "worker_id": w_row["worker_id"],
                        "worker_name": w_row["worker_name"],
                        "role": w_row["role"],
                        "company_code": w_row["company_code"],
                        "branch_code": w_row["branch_code"],
                        "cert_code": w_row["cert_code"],
                        "weld_process": worker_proc,
                        "cert_expiry_date": str(w_row["cert_expiry_date"]),
                        "status": w_row["status"] if is_proc_matched else "PROCESS_MISMATCH",
                        "is_valid": is_fully_qualified,
                        "is_process_matched": is_proc_matched,
                        "mismatch_reason": proc_mismatch_reason,
                    }
                    result["worker_qualified"] = is_fully_qualified

                    if not is_fully_qualified:
                        result["require_outsourcing"] = True
                        result["outsourcing_reason"] = (
                            proc_mismatch_reason or f"작업자 자격 상태 부적격({w_row['status']})"
                        )
                else:
                    result["worker_qualified"] = False
                    result["require_outsourcing"] = True
                    result["outsourcing_reason"] = f"본사 원장 미등록 작업자('{worker_name}') - 외주 인력 조회 필요"
                    result["worker_qualification"] = {
                        "worker_name": worker_name,
                        "status": "NOT_FOUND",
                        "is_valid": False,
                        "is_process_matched": False,
                        "mismatch_reason": f"인사 원장에서 작업자 '{worker_name}'을 찾을 수 없습니다.",
                    }

            # 4. BOM 자재 재고 및 Lot 조회
            inv_rows = conn.execute(
                text("""
                    SELECT item_code, item_name, material_category, total_qty,
                           reserved_qty, unit, branch_code, lot_no, supplier_code
                    FROM ERP_BOM_INVENTORY
                    ORDER BY item_code ASC
                """)
            ).mappings().all()

            for r in inv_rows:
                total_q = float(r["total_qty"])
                res_q = float(r["reserved_qty"])
                avail_q = max(0.0, total_q - res_q)
                item_dict = {
                    "item_code": r["item_code"],
                    "item_name": r["item_name"],
                    "material_category": r["material_category"],
                    "total_qty": total_q,
                    "reserved_qty": res_q,
                    "available_qty": avail_q,
                    "unit": r["unit"],
                    "branch_code": r["branch_code"],
                    "lot_no": r["lot_no"],
                    "supplier_code": r["supplier_code"],
                }
                result["inventory"][r["item_code"]] = item_dict

                if r["material_category"] == "STEEL_PLATE" and avail_q > 0:
                    if not result["assigned_lot_no"] or (material and material.upper() in r["item_code"].upper()):
                        result["assigned_lot_no"] = r["lot_no"]
                        result["selected_plate_code"] = r["item_code"]

                if r["material_category"] == "FILLER_WIRE":
                    req_proc = str((result["block_master"] or {}).get("process") or "").upper()
                    item_text = f"{r['item_code']} {r['item_name']}".upper()
                    is_proc_wire = bool(req_proc and req_proc in item_text)

                    if not result["selected_wire_code"] or is_proc_wire:
                        result["selected_wire_code"] = r["item_code"]
                        result["selected_wire_supplier"] = r["supplier_code"]
                        if avail_q < 15.0:
                            result["stock_available"] = False
                            result["require_outsourcing"] = True
                            result["outsourcing_reason"] = f"사내 용접 와이어({r['item_code']}) 가용 재고 부족({avail_q}kg)"

            # 5. 야드별 대기 차량 및 중장비 조회 (본사 및 상주 협력사 대기 장비 포함)
            eq_rows = conn.execute(
                text("""
                    SELECT asset_code, asset_name, asset_category, company_code,
                           branch_code, current_zone, max_load_ton, driver_worker_id, operation_status
                    FROM ERP_VEHICLES_EQUIPMENT
                    ORDER BY asset_code ASC
                """)
            ).mappings().all()

            all_eq = [dict(r) for r in eq_rows]
            result["vehicles_equipment"] = all_eq

            # 대상 야드(target_branch) 내 STANDBY 장비 중 운반차량(VEHICLE_*) 우선, 없으면 크레인(CRANE) 배정
            branch_standby = [
                eq for eq in all_eq
                if str(eq.get("branch_code", "")).upper() == str(target_branch).upper()
                and str(eq.get("operation_status", "")).upper() == "STANDBY"
            ]
            selected_eq = None
            for eq in branch_standby:
                if str(eq.get("asset_category", "")).upper().startswith("VEHICLE_"):
                    selected_eq = eq
                    break
            if not selected_eq and branch_standby:
                selected_eq = branch_standby[0]

            result["assigned_vehicle_code"] = (
                selected_eq["asset_code"] if selected_eq else None
            )

    except Exception as e:
        result["status"] = "ERROR"
        result["error_message"] = str(e)

    return result


@tool
def query_partner_erp_b2b(
    outsourcing_domain: Optional[str] = None,
    material: Optional[str] = None,
    process: Optional[str] = None,
    partner_id: Optional[str] = None,
    worker_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    외부 협력사 PostgreSQL B2B ERP View(v_b2b_daehan_materials, v_b2b_donghae_workers, v_b2b_hankook_transporters)를
    조회하여 실시간 외주 자재 단가/재고, 가용 용접사, 중량물 트랜스포터 배차 가능 현황을 반환합니다.
    """
    domain = (outsourcing_domain or "WELDING_MATERIAL").upper()
    timestamp_iso = datetime.now().isoformat()

    if not pg_engine:
        return {
            "status": "ERROR",
            "error_message": "PARTNER_PG_DATABASE_URL이 설정되지 않았습니다.",
            "b2b_inquiry_timestamp": timestamp_iso,
        }

    try:
        with pg_engine.connect() as conn:
            # 1. 용접 자재 / 철판 협력사 (대한특수철강)
            if domain == "WELDING_MATERIAL":
                rows = conn.execute(
                    text("SELECT * FROM v_b2b_daehan_materials")
                ).mappings().all()
                items = [dict(r) for r in rows]
                selected = None
                if items:
                    for it in items:
                        if "WIRE" in str(it.get("supplier_sku", "")).upper() or "와이어" in str(it.get("item_name", "")):
                            selected = dict(it)
                            break
                    if not selected:
                        selected = dict(items[0])

                    selected.setdefault("partner_id", "VEND-DAEHAN-MAT")
                    selected.setdefault("partner_name", "대한특수철강")
                    selected.setdefault("business_type", "WELDING_MATERIAL")

                return {
                    "status": "SUCCESS",
                    "b2b_inquiry_timestamp": timestamp_iso,
                    "source": "PostgreSQL PARTNER_B2B (v_b2b_daehan_materials)",
                    "available_items": items,
                    "selected_partner": selected,
                }

            # 2. 외주 용접 인력 협력사 (동해블록용접)
            elif domain == "OUTSOURCED_WORKER":
                rows = conn.execute(
                    text("SELECT * FROM v_b2b_donghae_workers")
                ).mappings().all()
                workers = [dict(r) for r in rows]
                selected = None

                clean_w_name = None
                if worker_name:
                    clean_w_name = re.sub(
                        r"\s*(반장|작업자|용접사|용접공|오퍼레이터|기사|엔지니어|팀장|님)$",
                        "",
                        str(worker_name).strip(),
                    ).strip()

                if workers:
                    if clean_w_name:
                        for w in workers:
                            if str(w.get("worker_name", "")).strip() == clean_w_name:
                                selected = dict(w)
                                break
                    else:
                        req_proc = str(process or "").upper().strip()
                        for w in workers:
                            if w.get("can_accept_order", True) and (
                                not req_proc or str(w.get("weld_process", "")).upper() == req_proc
                            ):
                                selected = dict(w)
                                break
                        if not selected:
                            for w in workers:
                                if w.get("can_accept_order", True):
                                    selected = dict(w)
                                    break

                if selected:
                    w_id = (
                        selected.get("assigned_worker_id")
                        or selected.get("ext_worker_id")
                        or selected.get("worker_id")
                    )
                    selected["assigned_worker_id"] = w_id
                    selected.setdefault("partner_id", "VEND-DONGHAE-WELD")
                    selected.setdefault("partner_name", "동해블록용접(주)")
                    selected.setdefault("business_type", "OUTSOURCED_WORKER")

                return {
                    "status": "SUCCESS",
                    "b2b_inquiry_timestamp": timestamp_iso,
                    "source": "PostgreSQL PARTNER_B2B (v_b2b_donghae_workers)",
                    "available_workers": workers,
                    "selected_partner": selected,
                }

            # 3. 중량물 운송 / 트랜스포터 협력사 (한국중량물운송)
            else:
                transporters = []
                source_name = "PostgreSQL PARTNER_B2B (v_b2b_hankook_transporters)"
                try:
                    rows = conn.execute(
                        text("SELECT * FROM v_b2b_hankook_transporters")
                    ).mappings().all()
                    transporters = [dict(r) for r in rows]
                except Exception:
                    # PostgreSQL에 v_b2b_hankook_transporters 뷰가 없는 경우 Oracle 협력사 장비 원장에서 동적 폴백 조회
                    if erp_engine:
                        with erp_engine.connect() as ora_conn:
                            ora_rows = ora_conn.execute(
                                text("""
                                    SELECT e.asset_code AS assigned_vehicle_code,
                                           e.asset_name,
                                           e.asset_category,
                                           e.company_code AS partner_id,
                                           c.company_name AS partner_name,
                                           e.branch_code,
                                           e.current_zone,
                                           e.operation_status,
                                           e.max_load_ton,
                                           e.driver_worker_id
                                    FROM ERP_VEHICLES_EQUIPMENT e
                                    LEFT JOIN ERP_COMPANIES c ON e.company_code = c.company_code
                                    WHERE e.company_code != 'COMP-HQ'
                                    ORDER BY e.asset_code ASC
                                """)
                            ).mappings().all()
                            transporters = [dict(r) for r in ora_rows]
                            source_name = "ERP_VEHICLES_EQUIPMENT (Partner Asset View)"

                selected = None
                if transporters:
                    selected = dict(transporters[0])
                    selected.setdefault("partner_id", "VEND-HANKOOK-TP")
                    selected.setdefault("partner_name", "한국중량물운송(주)")
                    selected.setdefault("business_type", "HEAVY_TRANSPORT")

                return {
                    "status": "SUCCESS",
                    "b2b_inquiry_timestamp": timestamp_iso,
                    "source": source_name,
                    "available_transporters": transporters,
                    "selected_partner": selected,
                }

    except Exception as e:
        return {
            "status": "ERROR",
            "error_message": str(e),
            "b2b_inquiry_timestamp": timestamp_iso,
        }


@tool
def commit_erp_work_order(
    block_id: str,
    material: str,
    allocated_lot_no: str,
    worker_name: str,
    worker_id: Optional[str] = None,
    assigned_vehicle_code: Optional[str] = None,
    wire_item_code: Optional[str] = None,
    is_outsourced: bool = False,
    partner_id: str = "COMP-HQ",
    reserve_wire_kg: float = 15.0,
    delivery_yard_zone: str = "ZONE-A-BAY1",
) -> Dict[str, Any]:
    """관리자 승인(HITL) 완료 후 Oracle MES 원장 커밋 및 외주 시 PostgreSQL B2B 수주 원장을 동시 갱신합니다."""
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    committed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wo_no = f"WO-{now_str}"
    exec_type = "OUTSOURCED_B2B" if is_outsourced else "IN_HOUSE"

    remaining_wire = 0.0
    reserved_applied = 0.0
    resolved_worker_id = worker_id or "EMP-1002"

    # 1. Oracle SHIP_ERP 원장 커밋 (재고 선점 + MES_WORK_ORDER_DISPATCH 등록)
    with erp_engine.begin() as conn:
        if not worker_id and worker_name:
            w_row = conn.execute(
                text("SELECT worker_id FROM ERP_WORKER_CERT WHERE worker_name = :w_name FETCH FIRST 1 ROWS ONLY"),
                {"w_name": worker_name},
            ).mappings().first()
            if w_row:
                resolved_worker_id = w_row["worker_id"]

        if wire_item_code:
            wire_row = conn.execute(
                text("""
                    SELECT item_code, total_qty, reserved_qty
                    FROM ERP_BOM_INVENTORY
                    WHERE item_code = :i_code
                    FETCH FIRST 1 ROWS ONLY
                """),
                {"i_code": wire_item_code},
            ).mappings().first()

            if wire_row:
                tot_q = float(wire_row["total_qty"] or 0)
                res_q = float(wire_row["reserved_qty"] or 0)
                avail_q = max(0.0, tot_q - res_q)
                reserved_applied = min(float(reserve_wire_kg), avail_q)
                remaining_wire = avail_q - reserved_applied

                if reserved_applied > 0:
                    conn.execute(
                        text("""
                            UPDATE ERP_BOM_INVENTORY
                            SET reserved_qty = reserved_qty + :add_qty
                            WHERE item_code = :i_code
                        """),
                        {"add_qty": reserved_applied, "i_code": wire_item_code},
                    )

        conn.execute(
            text("""
                INSERT INTO MES_WORK_ORDER_DISPATCH (
                    wo_number, block_id, execution_type, partner_id,
                    allocated_lot_no, worker_id, worker_name,
                    assigned_vehicle_code, status, created_at
                ) VALUES (
                    :wo_no, :b_id, :exec_type, :p_id,
                    :lot_no, :w_id, :w_name,
                    :veh_code, 'COMMITTED', CURRENT_TIMESTAMP
                )
            """),
            {
                "wo_no": wo_no,
                "b_id": block_id,
                "exec_type": exec_type,
                "p_id": partner_id,
                "lot_no": allocated_lot_no,
                "w_id": resolved_worker_id,
                "w_name": worker_name,
                "veh_code": assigned_vehicle_code,
            },
        )

    # 2. 외주 연계(OUTSOURCED_B2B)인 경우 PostgreSQL PARTNER_B2B 수주 원장 동시 기록
    po_no = None
    b2b_ack = None
    if is_outsourced:
        po_no = f"PO-{now_str}"
        b2b_ack = f"ACK-{wo_no}-{partner_id}"
        try:
            with pg_engine.begin() as p_conn:
                p_conn.execute(
                    text("""
                        INSERT INTO b2b_sales_orders (
                            b2b_order_no, client_company, partner_id, ref_wo_number,
                            block_id, material_grade, delivery_zone, order_status, created_at
                        ) VALUES (
                            :po_no, '한국스마트조선해양', :p_id, :wo_no,
                            :b_id, :mat, :zone, 'CONFIRMED', CURRENT_TIMESTAMP
                        )
                    """),
                    {
                        "po_no": po_no,
                        "p_id": partner_id,
                        "wo_no": wo_no,
                        "b_id": block_id,
                        "mat": material,
                        "zone": delivery_yard_zone,
                    },
                )
        except Exception as e:
            print(f"⚠️ [PostgreSQL B2B 수주 원장 기록 참고]: {e}")

    return {
        "erp_transaction_status": "COMMITTED",
        "work_order_number": wo_no,
        "execution_type": exec_type,
        "partner_id": partner_id,
        "po_number": po_no,
        "b2b_order_number": po_no,
        "b2b_sales_order_no": po_no,
        "b2b_po_ack": b2b_ack,
        "allocated_lot_no": allocated_lot_no,
        "worker_id": resolved_worker_id,
        "worker_name": worker_name,
        "assigned_vehicle_code": assigned_vehicle_code,
        "reserved_wire_kg": reserved_applied,
        "remaining_hq_wire_kg": remaining_wire,
        "committed_at": committed_at,
    }