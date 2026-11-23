"""
modules/agent/tools/erp_tools.py
본사 Oracle SHIP_ERP 및 외부 협력사 FastMCP B2B 게이트웨이 연동 도구
- 본사 Oracle(SHIP_ERP_DATABASE_URL)만 직접 접속하며, 협력사 PostgreSQL은 partner_mcp_client를 통해 MCP/A2A 통신으로 분리
- 용접 공정(WPS) 상호 일치 검증, 중량물 정격하중 안전율(90% 이하) 진단, 대상 야드 우선 배차 및 외주 실패 보상 롤백 탑재
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain_core.tools import tool
from sqlalchemy import create_engine, text

from modules.agent.tools.partner_mcp_client import (
    dispatch_partner_b2b_order_via_mcp,
    query_partner_b2b_via_mcp,
)

# 프로젝트 루트의 .env 로드 (config.py와 동일한 환경변수 사용)
PLATFORM_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PLATFORM_ROOT / ".env")

# ==============================================================================
# 1. SQLAlchemy DB 엔진 설정 (본사 Oracle SHIP_ERP 전용)
# ==============================================================================
ORACLE_URI = os.getenv("SHIP_ERP_DATABASE_URL")

try:
    erp_engine = create_engine(ORACLE_URI, pool_pre_ping=True) if ORACLE_URI else None
except Exception as e:
    print(f"⚠️ [Oracle SHIP_ERP 엔진 초기화 경고]: {e}")
    erp_engine = None


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


def evaluate_heavy_lifting_capacity(
    block_weight_ton: Optional[float],
    vehicles_equipment: List[Dict[str, Any]],
    target_branch: Optional[str] = None,
    requested_equipment: Optional[str] = None,
) -> Dict[str, Any]:
    """
    [산업안전보건규칙 제38조 실무 기준]
    블록 중량 + 줄걸이(Rigging) 자중 5%를 합산한 총 인양하중과 장비 정격하중(max_load_ton)을 대조해
    안전 하중률(85% 이하 정상 / 85~90% 주의 / 90% 초과 과하중 차단)을 판정합니다.
    """
    weight = float(block_weight_ton or 0.0)
    rigging_ton = round(weight * 0.05, 2)
    total_lifting_ton = round(weight + rigging_ton, 2)

    lifting_assets = [
        eq for eq in (vehicles_equipment or [])
        if any(
            k in str(eq.get("asset_category", "")).upper()
            for k in ("CRANE", "TRANSPORTER", "FORKLIFT", "VEHICLE")
        )
    ]

    selected_eq = None
    req_eq_upper = str(requested_equipment or "").upper().strip()

    # 1순위: 사용자가 명시한 장비명/코드 매칭
    if req_eq_upper:
        for eq in lifting_assets:
            text_key = (
                f"{eq.get('asset_code', '')} "
                f"{eq.get('asset_name', '')} "
                f"{eq.get('asset_category', '')}"
            ).upper()
            if (
                req_eq_upper in text_key
                or ("크레인" in req_eq_upper and "CRANE" in text_key)
                or ("트랜스포터" in req_eq_upper and "TRANSPORTER" in text_key)
                or ("지게차" in req_eq_upper and "FORKLIFT" in text_key)
            ):
                selected_eq = eq
                break

    # 2순위: 대상 야드 내 대기(STANDBY) 장비 중 총 인양하중(안전율 90% 이내)을 감당할 수 있는 최적 장비 자동 선정
    if not selected_eq:
        candidates = [
            eq for eq in lifting_assets
            if str(eq.get("operation_status", "")).upper() == "STANDBY"
            and (
                not target_branch
                or str(eq.get("branch_code", "")).upper() == str(target_branch).upper()
            )
        ]
        safe_candidates = [
            eq for eq in candidates
            if float(eq.get("max_load_ton") or 0) * 0.90 >= total_lifting_ton
        ]
        if safe_candidates:
            selected_eq = sorted(
                safe_candidates, key=lambda x: float(x.get("max_load_ton") or 9999)
            )[0]
        elif candidates:
            selected_eq = sorted(
                candidates, key=lambda x: float(x.get("max_load_ton") or 0), reverse=True
            )[0]

    # 3순위: 타 야드 포함 안전 인양 가능한 대안 장비(외주 200톤 트랜스포터 등) 탐색
    recommended_alternatives = [
        eq for eq in lifting_assets
        if str(eq.get("operation_status", "")).upper() == "STANDBY"
        and float(eq.get("max_load_ton") or 0) * 0.90 >= total_lifting_ton
    ]

    if not selected_eq or weight <= 0:
        return {
            "status": "NO_EQUIPMENT" if not selected_eq else "WEIGHT_MISSING",
            "is_safe": False,
            "need_outsourcing": bool(weight > 0),
            "block_weight_ton": weight,
            "rigging_weight_ton": rigging_ton,
            "total_lifting_ton": total_lifting_ton,
            "selected_equipment": selected_eq,
            "max_load_ton": float((selected_eq or {}).get("max_load_ton") or 0.0),
            "load_ratio_pct": 0.0,
            "recommended_alternatives": recommended_alternatives,
            "reason": "대상 야드에 투입 가능한 중량물 양중/운반 장비가 없거나 블록 중량이 미지정되었습니다.",
        }

    max_load = float(selected_eq.get("max_load_ton") or 0.0)
    load_ratio = round((total_lifting_ton / max_load) * 100.0, 1) if max_load > 0 else 999.0
    is_standby = str(selected_eq.get("operation_status", "")).upper() == "STANDBY"
    is_external = str(selected_eq.get("company_code", "COMP-HQ")).upper() != "COMP-HQ"

    if not is_standby:
        status_code, is_safe = "EQUIPMENT_BUSY", False
        reason = (
            f"지정 장비 {selected_eq.get('asset_name')}({selected_eq.get('asset_code')})가 "
            f"현재 운행/정비 중입니다."
        )
    elif load_ratio > 90.0:
        status_code, is_safe = "OVERLOAD", False
        reason = (
            f"정격하중 안전율 초과(위험): 총 인양하중 {total_lifting_ton}톤"
            f"(블록 {weight}톤 + 줄걸이 {rigging_ton}톤) / 장비 정격 {max_load}톤 "
            f"➔ 하중률 {load_ratio}% (법정 허용 안전기준 90% 초과)"
        )
    elif load_ratio > 85.0:
        status_code, is_safe = "CONDITIONAL_PASS", True
        reason = (
            f"정격하중 주의 구간(하중률 {load_ratio}%): "
            f"저속 인양 및 시험 인양(지상 30cm 정지 확인) 필수"
        )
    else:
        status_code, is_safe = "PASS", True
        reason = (
            f"정격하중 안전 기준 충족(하중률 {load_ratio}%, "
            f"여유 하중 {round(max_load - total_lifting_ton, 1)}톤)"
        )

    return {
        "status": status_code,
        "is_safe": is_safe,
        "need_outsourcing": is_external or (status_code == "OVERLOAD"),
        "block_weight_ton": weight,
        "rigging_weight_ton": rigging_ton,
        "total_lifting_ton": total_lifting_ton,
        "selected_equipment": selected_eq,
        "max_load_ton": max_load,
        "load_ratio_pct": load_ratio,
        "recommended_alternatives": recommended_alternatives,
        "reason": reason,
    }


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
                res["default_zone"] = (
                    "DOCK-1"
                    if "GEOJE" in str(res.get("branch_code", "")).upper()
                    else "ZONE-A-BAY1"
                )
                return res
    except Exception as e:
        print(f"⚠️ [get_branch_context_from_db 오류]: {e}")
    return {}


def get_active_partner_from_db(specialty_field: str = "BLOCK_WELDING") -> Dict[str, Any]:
    """ERP_COMPANIES 테이블에서 전문 분야 또는 구분에 맞는 활성 업체 정보 동적 조회"""
    if not erp_engine:
        return {
            "company_code": "COMP-HQ",
            "partner_id": "VEND-DONGHAE-WELD",
            "partner_name": "동해블록용접(주)",
        }
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
    """Oracle ERP_WORKER_CERT 원장에서 사내외 전체 작업자 명단, 자격 상태 및 실시간 투입(is_busy) 여부를 조회합니다."""
    if not erp_engine:
        return {"status": "ERROR", "workers": []}

    try:
        with erp_engine.connect() as conn:
            busy_rows = conn.execute(
                text("""
                    SELECT worker_id, worker_name, block_id, execution_type
                    FROM MES_WORK_ORDER_DISPATCH
                    WHERE UPPER(status) = 'COMMITTED'
                """)
            ).mappings().all()
            busy_ids = {
                str(r["worker_id"]).strip()
                for r in busy_rows
                if r.get("worker_id") and str(r.get("execution_type", "")).upper() == "IN_HOUSE"
            }
            busy_names = {str(r["worker_name"]).strip() for r in busy_rows if r.get("worker_name")}

            eq_busy_rows = conn.execute(
                text("""
                    SELECT driver_worker_id
                    FROM ERP_VEHICLES_EQUIPMENT
                    WHERE UPPER(operation_status) = 'IN_USE'
                      AND driver_worker_id IS NOT NULL
                """)
            ).mappings().all()
            for er in eq_busy_rows:
                busy_ids.add(str(er["driver_worker_id"]).strip())

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
                w_id = str(d.get("worker_id") or "").strip()
                w_nm = str(d.get("worker_name") or "").strip()
                d["is_busy"] = (w_id in busy_ids) or (w_nm in busy_names)
                d["is_valid"] = (str(d.get("status", "")).upper() == "VALID") and (not d["is_busy"])
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
    process: Optional[str] = None,
) -> Dict[str, Any]:
    """
    HQ Oracle SHIP_ERP 원장을 조회하여 전사 야드, 블록 제원, 자재 재고,
    작업자 자격/공정 일치 및 실시간 중복 투입(MES_WORK_ORDER_DISPATCH COMMITTED) 여부,
    해당 야드의 대기 차량 및 외주 필요 여부를 정밀 진단합니다.
    """
    result: Dict[str, Any] = {
        "status": "SUCCESS",
        "hq_company_code": "COMP-HQ",
        "all_branches": [],
        "block_master": None,
        "branch_info": None,
        "foreman_name": None,
        "worker_qualification": None,
        "worker_qualified": True,
        "available_workers": [],
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
        "available_blocks": [],
    }

    if not erp_engine:
        result["status"] = "ERROR"
        return result

    try:
        with erp_engine.connect() as conn:
            # 0. 전사 야드(ERP_BRANCHES) 전체 마스터 조회 및 동적 권역(region_name) 매핑
            all_br_rows = conn.execute(
                text("SELECT * FROM ERP_BRANCHES ORDER BY branch_code ASC")
            ).mappings().all()
            result["all_branches"] = [dict(r) for r in all_br_rows]
            branch_region_map = {
                str(r["branch_code"]).strip().upper(): str(r["region_name"]).strip()
                for r in all_br_rows
                if r.get("branch_code") and r.get("region_name")
            }

            # 1. 블록 마스터 제원 조회
            if block_id:
                norm_block_id = str(block_id).strip().upper()
                b_row = conn.execute(
                    text("""
                        SELECT block_id, ship_code, block_name, steel_type, thickness_mm,
                               joint_type, process, target_dock, branch_code, weight_ton
                        FROM ERP_BLOCK_MASTER
                        WHERE UPPER(TRIM(block_id)) = :block_id
                    """),
                    {"block_id": norm_block_id},
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
                        "work_zone": b_row["target_dock"],
                        "location": b_row["target_dock"],
                        "branch_code": b_row["branch_code"],
                        "weight_ton": float(b_row["weight_ton"]) if b_row["weight_ton"] else 0.0,
                    }

            # 블록 마스터에 지정된 실제 작업 야드가 있으면 로그인 사용자 기본 야드보다 최우선 적용
            if result["block_master"] and result["block_master"].get("branch_code"):
                target_branch = result["block_master"]["branch_code"]
            else:
                target_branch = branch_code or "BR-GEOJE-01"

            # 1-1. 활성 작업 배정(MES_WORK_ORDER_DISPATCH) 및 전사 등록 블록 마스터 전체 조회
            active_block_rows = conn.execute(
                text("""
                    SELECT DISTINCT UPPER(TRIM(block_id)) AS block_id
                    FROM MES_WORK_ORDER_DISPATCH
                    WHERE UPPER(status) = 'COMMITTED'
                """)
            ).mappings().all()
            active_block_set = {r["block_id"] for r in active_block_rows if r.get("block_id")}

            all_block_rows = conn.execute(
                text("""
                    SELECT block_id, ship_code, block_name, steel_type AS material,
                           thickness_mm AS thickness, joint_type, process,
                           target_dock, branch_code, weight_ton
                    FROM ERP_BLOCK_MASTER
                    ORDER BY block_id ASC
                """)
            ).mappings().all()
            avail_blocks = []
            for r in all_block_rows:
                bd = dict(r)
                bd["is_active_work"] = str(bd.get("block_id") or "").strip().upper() in active_block_set
                avail_blocks.append(bd)
            result["available_blocks"] = avail_blocks

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
                    "default_zone": (
                        "DOCK-1"
                        if "GEOJE" in str(br_row["branch_code"]).upper()
                        else "ZONE-A-BAY1"
                    ),
                }

            # 2-1. 해당 야드 소속 현장 반장/관리자(검토자) 동적 조회
            foreman_row = conn.execute(
                text("""
                    SELECT worker_name, worker_id
                    FROM ERP_WORKER_CERT
                    WHERE branch_code = :b_code
                      AND UPPER(role) IN ('WELD_LEADER', 'FOREMAN', 'MANAGER', 'ENGINEER')
                      AND UPPER(status) = 'VALID'
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

            # 3. 현재 활성 작업 배정 원장(MES_WORK_ORDER_DISPATCH) 및 가동 중 장비 운전자 조회 (중복 투입 방지용)
            active_dispatch_rows = conn.execute(
                text("""
                    SELECT worker_id, worker_name, block_id, wo_number, execution_type
                    FROM MES_WORK_ORDER_DISPATCH
                    WHERE UPPER(status) = 'COMMITTED'
                """)
            ).mappings().all()

            busy_by_id: Dict[str, Dict[str, Any]] = {}
            busy_by_name: Dict[str, Dict[str, Any]] = {}
            for d_row in active_dispatch_rows:
                info = {
                    "block_id": d_row["block_id"],
                    "wo_number": d_row["wo_number"],
                }
                # 외주(OUTSOURCED_B2B)의 FK 대리 매핑(본사 반장 사번)은 본사 인력 잠금에서 제외
                if d_row.get("worker_id") and str(d_row.get("execution_type", "")).upper() == "IN_HOUSE":
                    busy_by_id[str(d_row["worker_id"]).strip()] = info
                if d_row.get("worker_name"):
                    busy_by_name[str(d_row["worker_name"]).strip()] = info

            eq_busy_rows = conn.execute(
                text("""
                    SELECT asset_code, driver_worker_id
                    FROM ERP_VEHICLES_EQUIPMENT
                    WHERE UPPER(operation_status) = 'IN_USE'
                      AND driver_worker_id IS NOT NULL
                """)
            ).mappings().all()
            for er in eq_busy_rows:
                drv_id = str(er["driver_worker_id"]).strip()
                if drv_id and drv_id not in busy_by_id:
                    busy_by_id[drv_id] = {
                        "block_id": er["asset_code"],
                        "wo_number": "EQUIPMENT_IN_USE",
                    }

            # 3-1. 전체 작업자 자격 및 실시간 투입 현황 요약 조회
            all_workers = conn.execute(
                text("""
                    SELECT worker_id, worker_name, employment_type, company_code,
                           branch_code, role, cert_code, weld_process,
                           cert_expiry_date, status
                    FROM ERP_WORKER_CERT
                    ORDER BY worker_id ASC
                """)
            ).mappings().all()

            enriched_workers = []
            for r in all_workers:
                w_dict = dict(r)
                w_id_str = str(w_dict.get("worker_id") or "").strip()
                w_nm_str = str(w_dict.get("worker_name") or "").strip()
                w_br_str = str(w_dict.get("branch_code") or "").strip().upper()
                active_job = busy_by_id.get(w_id_str) or busy_by_name.get(w_nm_str)

                w_dict["region_name"] = branch_region_map.get(w_br_str, "")
                w_dict["is_busy"] = bool(active_job)
                w_dict["assigned_block_id"] = active_job["block_id"] if active_job else None
                w_dict["assigned_wo_number"] = active_job["wo_number"] if active_job else None
                enriched_workers.append(w_dict)

            result["available_workers"] = enriched_workers

            # 3-2. 특정 작업자 선급 자격증 유효성, 용접 공정(WPS) 일치, 야드 권역 일치 및 중복 투입 정밀 검증
            if worker_name:
                clean_worker_name = re.sub(
                    r"\s*(반장|작업자|용접사|용접공|오퍼레이터|기사|엔지니어|팀장|님|씨)$",
                    "",
                    str(worker_name).strip(),
                ).strip()
                if not clean_worker_name:
                    clean_worker_name = str(worker_name).strip()

                w_row = conn.execute(
                    text("""
                        SELECT worker_id, worker_name, role, company_code, branch_code,
                               cert_code, weld_process, cert_expiry_date, status
                        FROM ERP_WORKER_CERT
                        WHERE worker_name = :worker_name
                           OR UPPER(TRIM(worker_id)) = UPPER(TRIM(:worker_name))
                        ORDER BY cert_expiry_date DESC
                        FETCH FIRST 1 ROWS ONLY
                    """),
                    {"worker_name": clean_worker_name},
                ).mappings().first()

                if w_row:
                    is_cert_valid = str(w_row["status"]).upper() == "VALID"
                    req_process = (
                        result["block_master"].get("process")
                        if result["block_master"]
                        else process
                    )
                    worker_proc = w_row["weld_process"]

                    active_job = (
                        busy_by_id.get(str(w_row["worker_id"]).strip())
                        or busy_by_name.get(str(w_row["worker_name"]).strip())
                    )
                    is_already_busy = bool(active_job)
                    busy_reason = None
                    if is_already_busy:
                        busy_reason = (
                            f"작업자 중복 투입 불가: '{w_row['worker_name']}({w_row['worker_id']})' 인원은 "
                            f"현재 블록 '{active_job['block_id']}'(오더: {active_job['wo_number']})에 이미 투입 중입니다."
                        )

                    is_proc_matched = True
                    proc_mismatch_reason = None
                    if (
                        req_process
                        and worker_proc
                        and str(req_process).upper() != "NA"
                        and str(worker_proc).upper() != "NA"
                    ):
                        if str(req_process).upper() != str(worker_proc).upper():
                            is_proc_matched = False
                            proc_mismatch_reason = (
                                f"공정 불일치: 대상 블록({block_id})의 지정 공정은 '{req_process}'이나 "
                                f"작업자({w_row['worker_name']})의 인증 공정은 '{worker_proc}'입니다."
                            )

                    worker_branch_upper = str(w_row["branch_code"] or "").strip().upper()
                    target_branch_upper = str(target_branch or "").strip().upper()
                    worker_region = branch_region_map.get(worker_branch_upper, "")
                    target_region = branch_region_map.get(target_branch_upper, "")

                    is_yard_matched = True
                    yard_mismatch_reason = None
                    if (
                        result["block_master"]
                        and worker_region
                        and target_region
                        and worker_region != target_region
                    ):
                        is_yard_matched = False
                        yard_mismatch_reason = (
                            f"야드 지역 불일치: 대상 블록({block_id})은 '{target_region}' 권역({target_branch}) 작업이나, "
                            f"작업자({w_row['worker_name']})는 '{worker_region}' 권역({w_row['branch_code']}) 상주 인원입니다."
                        )

                    is_fully_qualified = (
                        is_cert_valid
                        and is_proc_matched
                        and is_yard_matched
                        and (not is_already_busy)
                    )

                    if is_already_busy:
                        final_w_status = "ALREADY_ASSIGNED"
                    elif not is_proc_matched:
                        final_w_status = "PROCESS_MISMATCH"
                    elif not is_yard_matched:
                        final_w_status = "YARD_MISMATCH"
                    else:
                        final_w_status = w_row["status"]

                    result["worker_qualification"] = {
                        "worker_id": w_row["worker_id"],
                        "worker_name": w_row["worker_name"],
                        "role": w_row["role"],
                        "company_code": w_row["company_code"],
                        "branch_code": w_row["branch_code"],
                        "worker_region": worker_region,
                        "target_branch_code": target_branch,
                        "target_region": target_region,
                        "cert_code": w_row["cert_code"],
                        "weld_process": worker_proc,
                        "cert_expiry_date": str(w_row["cert_expiry_date"]),
                        "status": final_w_status,
                        "is_valid": is_fully_qualified,
                        "is_busy": is_already_busy,
                        "assigned_block_id": active_job["block_id"] if active_job else None,
                        "is_process_matched": is_proc_matched,
                        "is_yard_matched": is_yard_matched,
                        "mismatch_reason": busy_reason or proc_mismatch_reason or yard_mismatch_reason,
                    }
                    result["worker_qualified"] = is_fully_qualified

                    if not is_fully_qualified:
                        result["require_outsourcing"] = True
                        result["outsourcing_reason"] = (
                            busy_reason
                            or proc_mismatch_reason
                            or yard_mismatch_reason
                            or f"작업자 자격 상태 부적격({w_row['status']})"
                        )
                else:
                    ext_active = busy_by_name.get(clean_worker_name)
                    if ext_active:
                        result["worker_qualified"] = False
                        result["worker_qualification"] = {
                            "worker_name": clean_worker_name,
                            "status": "ALREADY_ASSIGNED",
                            "is_valid": False,
                            "is_busy": True,
                            "assigned_block_id": ext_active["block_id"],
                            "is_process_matched": True,
                            "mismatch_reason": (
                                f"외주 작업자 중복 투입 불가: '{clean_worker_name}' 인원은 "
                                f"현재 블록 '{ext_active['block_id']}'(오더: {ext_active['wo_number']})에 이미 투입 중입니다."
                            ),
                        }
                    else:
                        result["worker_qualified"] = False
                        result["worker_qualification"] = {
                            "worker_name": clean_worker_name,
                            "status": "NOT_FOUND",
                            "is_valid": False,
                            "is_process_matched": False,
                            "mismatch_reason": f"인사 원장에서 작업자 '{clean_worker_name}'을 찾을 수 없습니다.",
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

            eff_material = (
                result["block_master"].get("material")
                if result["block_master"]
                else material
            )
            eff_process = (
                result["block_master"].get("process")
                if result["block_master"]
                else process
            )
            plate_match_score = -1
            wire_match_score = -1
            selected_wire_avail = None

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

                item_text_upper = f"{r['item_code']} {r['item_name']}".upper()
                is_same_branch = (
                    str(r.get("branch_code") or "").upper() == str(target_branch).upper()
                )

                if r["material_category"] == "STEEL_PLATE" and avail_q > 0:
                    score = 0
                    if eff_material and str(eff_material).upper() in item_text_upper:
                        score += 2
                    if is_same_branch:
                        score += 1
                    if score > plate_match_score:
                        plate_match_score = score
                        result["assigned_lot_no"] = r["lot_no"]
                        result["selected_plate_code"] = r["item_code"]

                if r["material_category"] == "FILLER_WIRE":
                    score = 0
                    if eff_process and str(eff_process).upper() in item_text_upper:
                        score += 2
                    if is_same_branch:
                        score += 1
                    if score > wire_match_score:
                        wire_match_score = score
                        result["selected_wire_code"] = r["item_code"]
                        result["selected_wire_supplier"] = r["supplier_code"]
                        selected_wire_avail = avail_q

            if selected_wire_avail is not None and selected_wire_avail < 15.0:
                result["stock_available"] = False
                result["require_outsourcing"] = True
                result["outsourcing_reason"] = (
                    f"사내 용접 와이어({result['selected_wire_code']}) 가용 재고 부족({selected_wire_avail}kg)"
                )

            # 5. 장비 및 운반 차량 배정 - 작업 대상 야드(target_branch) 우선 매칭
            veh_rows = conn.execute(
                text("""
                    SELECT asset_code, asset_name, asset_category, company_code,
                           branch_code, current_zone, operation_status, max_load_ton, driver_worker_id
                    FROM ERP_VEHICLES_EQUIPMENT
                    ORDER BY CASE WHEN branch_code = :target_branch THEN 0 ELSE 1 END, asset_code ASC
                """),
                {"target_branch": target_branch},
            ).mappings().all()

            assigned_veh = None
            for v in veh_rows:
                v_dict = {
                    "asset_code": v["asset_code"],
                    "asset_name": v["asset_name"],
                    "asset_category": v["asset_category"],
                    "company_code": v["company_code"],
                    "branch_code": v["branch_code"],
                    "current_zone": v["current_zone"],
                    "operation_status": v["operation_status"],
                    "max_load_ton": float(v["max_load_ton"]) if v["max_load_ton"] else 0.0,
                    "driver_worker_id": v["driver_worker_id"],
                }
                result["vehicles_equipment"].append(v_dict)

                if (
                    not assigned_veh
                    and v_dict["branch_code"] == target_branch
                    and v_dict["operation_status"] == "STANDBY"
                    and "VEHICLE" in str(v_dict["asset_category"]).upper()
                ):
                    assigned_veh = v_dict["asset_code"]

            if not assigned_veh:
                for v in result["vehicles_equipment"]:
                    if v["branch_code"] == target_branch and v["operation_status"] == "STANDBY":
                        assigned_veh = v["asset_code"]
                        break

            if not assigned_veh:
                for v in result["vehicles_equipment"]:
                    if v["operation_status"] == "STANDBY":
                        assigned_veh = v["asset_code"]
                        break

            result["assigned_vehicle_code"] = assigned_veh

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
    외부 협력사 FastMCP B2B 게이트웨이 서버(partner_mcp_server.py)를 호출하여
    실시간 외주 자재 단가/재고, 가용 용접사, 중량물 트랜스포터 배차 가능 현황을 반환합니다.
    """
    domain = (outsourcing_domain or "WELDING_MATERIAL").upper()
    mcp_res = query_partner_b2b_via_mcp(
        outsourcing_domain=domain,
        material=material,
        process=process,
        partner_id=partner_id,
        worker_name=worker_name,
    )

    # 중량물 운송(HEAVY_TRANSPORT) 시 협력사 MCP에 등록된 트랜스포터 뷰가 비어 있으면 본사 협력사 자산 원장 폴백 조회
    if (
        domain not in ("WELDING_MATERIAL", "OUTSOURCED_WORKER")
        and mcp_res.get("status") == "SUCCESS"
        and not mcp_res.get("available_transporters")
        and erp_engine
    ):
        try:
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
                if transporters:
                    selected = dict(transporters[0])
                    selected.setdefault("partner_id", "VEND-HANKOOK-TP")
                    selected.setdefault("partner_name", "한국중량물운송(주)")
                    selected.setdefault("business_type", "HEAVY_TRANSPORT")
                    mcp_res["available_transporters"] = transporters
                    mcp_res["selected_partner"] = selected
                    mcp_res["source"] = "ERP_VEHICLES_EQUIPMENT (Partner Asset Fallback)"
        except Exception as e:
            print(f"⚠️ [HEAVY_TRANSPORT Oracle 폴백 조회 참고]: {e}")

    return mcp_res


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
    """
    관리자 승인(HITL) 완료 후 Oracle MES 원장 커밋 및 외주 시 협력사 FastMCP 게이트웨이를 통해 수주 원장을 동시 갱신합니다.
    - [패치 1] 미등록 블록 번호 입력 시 임의 블록(B-104 등) 대체 금지 — 즉시 커밋 중단
    - [패치 3] 사내 직영 시 요청 와이어 수량 전체 확보 검증 — 재고 부족 시 부분 선점 없이 커밋 중단
    - 외주 수주 접수 실패 시 본사 Oracle 원장 오더를 자동 보상 롤백(ROLLED_BACK)
    """
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    committed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wo_no = f"WO-{now_str}"
    exec_type = "OUTSOURCED_B2B" if is_outsourced else "IN_HOUSE"

    if not erp_engine:
        return {
            "erp_transaction_status": "ERROR",
            "purchase_status": "FAILED",
            "error_message": "SHIP_ERP_DATABASE_URL이 설정되지 않았습니다.",
            "work_order_number": wo_no,
            "execution_type": exec_type,
        }

    remaining_wire = 0.0
    reserved_applied = 0.0
    resolved_worker_id = worker_id or "EMP-1002"

    try:
        # 1. Oracle SHIP_ERP 원장 커밋 (블록/재고 엄격 검증 + 재고 선점 + MES_WORK_ORDER_DISPATCH 등록)
        with erp_engine.begin() as conn:
            if not worker_id and worker_name:
                w_row = conn.execute(
                    text(
                        "SELECT worker_id FROM ERP_WORKER_CERT "
                        "WHERE worker_name = :w_name FETCH FIRST 1 ROWS ONLY"
                    ),
                    {"w_name": worker_name},
                ).mappings().first()
                if w_row:
                    resolved_worker_id = w_row["worker_id"]

            fk_worker_id = conn.execute(
                text("SELECT worker_id FROM ERP_WORKER_CERT WHERE worker_id = :wid"),
                {"wid": resolved_worker_id},
            ).scalar()

            # [패치 1] FK_DISPATCH_BLOCK 무결성 보장: 유효한 등록 블록인지 엄격 검증 (임의 블록 대체 금지)
            fk_block_id = conn.execute(
                text("""
                    SELECT block_id
                    FROM ERP_BLOCK_MASTER
                    WHERE UPPER(TRIM(block_id)) = UPPER(TRIM(:bid))
                """),
                {"bid": block_id},
            ).scalar()

            if not fk_block_id:
                raise ValueError(
                    f"유효하지 않은 블록 번호('{block_id}')입니다. "
                    "ERP_BLOCK_MASTER에 등록된 블록만 작업 오더를 커밋할 수 있습니다."
                )

            # [Pre-commit 재검증 1] 동일 블록 중복 오더 차단
            dup_block = conn.execute(
                text("""
                    SELECT wo_number
                    FROM MES_WORK_ORDER_DISPATCH
                    WHERE UPPER(TRIM(block_id)) = UPPER(TRIM(:bid))
                      AND UPPER(status) = 'COMMITTED'
                    FETCH FIRST 1 ROWS ONLY
                """),
                {"bid": fk_block_id},
            ).scalar()
            if dup_block:
                raise ValueError(
                    f"블록 '{fk_block_id}'은 이미 확정된 작업오더({dup_block})가 진행 중이어서 중복 커밋할 수 없습니다."
                )

            # [Pre-commit 재검증 2] 사내 직영 작업자 중복 투입 차단
            if not is_outsourced and fk_worker_id:
                dup_worker = conn.execute(
                    text("""
                        SELECT wo_number
                        FROM MES_WORK_ORDER_DISPATCH
                        WHERE worker_id = :wid
                          AND UPPER(execution_type) = 'IN_HOUSE'
                          AND UPPER(status) = 'COMMITTED'
                        FETCH FIRST 1 ROWS ONLY
                    """),
                    {"wid": fk_worker_id},
                ).scalar()
                if dup_worker:
                    raise ValueError(
                        f"작업자 '{worker_name}({fk_worker_id})'는 이미 작업오더({dup_worker})에 투입 중입니다."
                    )

            # [Pre-commit 재검증 3] 사내 배차 차량/장비 STANDBY 상태 확인
            if assigned_vehicle_code:
                veh_row = conn.execute(
                    text("""
                        SELECT asset_code, operation_status
                        FROM ERP_VEHICLES_EQUIPMENT
                        WHERE UPPER(TRIM(asset_code)) = UPPER(TRIM(:veh_code))
                    """),
                    {"veh_code": assigned_vehicle_code},
                ).mappings().first()
                if veh_row and str(veh_row["operation_status"]).upper() != "STANDBY":
                    raise ValueError(
                        f"장비 '{assigned_vehicle_code}'는 현재 '{veh_row['operation_status']}' 상태여서 중복 배차할 수 없습니다."
                    )

            # 외부 협력사(PostgreSQL) 전용 작업자인 경우 본사 현장 담당자(EMP-*)를 FK 대리 매핑
            if not fk_worker_id:
                fk_worker_id = conn.execute(
                    text("""
                        SELECT worker_id FROM ERP_WORKER_CERT
                        WHERE status = 'VALID'
                        ORDER BY CASE WHEN role = 'WELD_LEADER' THEN 0 ELSE 1 END, worker_id ASC
                        FETCH FIRST 1 ROWS ONLY
                    """)
                ).scalar() or "EMP-1002"

            # [패치 3] 사내 직영 시 요청 수량 전체 확보 검증 (부분 선점 후 COMMITTED 방지)
            if wire_item_code and not is_outsourced:
                wire_row = conn.execute(
                    text("""
                        SELECT item_code, total_qty, reserved_qty
                        FROM ERP_BOM_INVENTORY
                        WHERE item_code = :i_code
                        FETCH FIRST 1 ROWS ONLY
                    """),
                    {"i_code": wire_item_code},
                ).mappings().first()

                if not wire_row:
                    raise ValueError(
                        f"자재 품번('{wire_item_code}')이 ERP_BOM_INVENTORY에 존재하지 않습니다."
                    )

                tot_q = float(wire_row["total_qty"] or 0)
                res_q = float(wire_row["reserved_qty"] or 0)
                avail_q = max(0.0, tot_q - res_q)
                req_q = float(reserve_wire_kg)

                if avail_q < req_q:
                    raise ValueError(
                        f"사내 자재({wire_item_code}) 가용 재고 부족: "
                        f"요청 {req_q}kg > 가용 {avail_q}kg (부분 선점 불가 — 커밋 중단)"
                    )

                reserved_applied = req_q
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
                        :veh_code, 'COMMITTED', TO_CHAR(CURRENT_TIMESTAMP, 'YYYY-MM-DD HH24:MI:SS')
                    )
                """),
                {
                    "wo_no": wo_no,
                    "b_id": fk_block_id,
                    "exec_type": exec_type,
                    "p_id": partner_id,
                    "lot_no": allocated_lot_no,
                    "w_id": fk_worker_id,
                    "w_name": worker_name,
                    "veh_code": assigned_vehicle_code,
                },
            )
            if assigned_vehicle_code:
                conn.execute(
                    text("""
                        UPDATE ERP_VEHICLES_EQUIPMENT
                        SET operation_status = 'IN_USE'
                        WHERE UPPER(TRIM(asset_code)) = UPPER(TRIM(:veh_code))
                          AND UPPER(operation_status) = 'STANDBY'
                    """),
                    {"veh_code": assigned_vehicle_code},
                )
    except Exception as e:
        print(f"⚠️ [Oracle MES 원장 커밋 오류]: {e}")
        return {
            "erp_transaction_status": "ERROR",
            "purchase_status": "FAILED",
            "error_message": str(e),
            "work_order_number": wo_no,
            "execution_type": exec_type,
        }

    # 2. 외주 연계(OUTSOURCED_B2B)인 경우 협력사 FastMCP 게이트웨이를 통해 수주 원장 동시 기록
    po_no = None
    b2b_ack = None
    purchase_status = "CONFIRMED"
    if is_outsourced:
        po_no = f"PO-{now_str}"
        b2b_ack = f"ACK-{wo_no}-{partner_id}"
        mcp_order_res = dispatch_partner_b2b_order_via_mcp(
            po_no=po_no,
            wo_no=wo_no,
            partner_id=partner_id,
            ordered_ref_code=wire_item_code or resolved_worker_id,
            order_qty=reserve_wire_kg,
            allocated_lot_or_id=allocated_lot_no or resolved_worker_id,
            delivery_yard_zone=delivery_yard_zone,
        )
        if mcp_order_res.get("status") not in ("CONFIRMED", "SUCCESS"):
            print(f"🚨 [외주 발주 실패 ➔ 본사 원장 보상 롤백 실행]: {mcp_order_res}")
            try:
                with erp_engine.begin() as rb_conn:
                    if assigned_vehicle_code:
                        rb_conn.execute(
                            text("""
                                UPDATE ERP_VEHICLES_EQUIPMENT
                                SET operation_status = 'STANDBY'
                                WHERE UPPER(TRIM(asset_code)) = UPPER(TRIM(:veh_code))
                            """),
                            {"veh_code": assigned_vehicle_code},
                        )

                    if wire_item_code and reserved_applied > 0:
                        rb_conn.execute(
                            text("""
                                UPDATE ERP_BOM_INVENTORY
                                SET reserved_qty = GREATEST(0, reserved_qty - :sub_qty)
                                WHERE item_code = :i_code
                            """),
                            {"sub_qty": reserved_applied, "i_code": wire_item_code},
                        )
                    rb_conn.execute(
                        text("""
                            UPDATE MES_WORK_ORDER_DISPATCH
                            SET status = 'ROLLED_BACK'
                            WHERE wo_number = :wo_no
                        """),
                        {"wo_no": wo_no},
                    )
            except Exception as rb_err:
                print(f"🚨 [보상 롤백 실패 — 수동 확인 필요]: {rb_err}")

            return {
                "erp_transaction_status": "ROLLED_BACK",
                "purchase_status": "FAILED",
                "work_order_number": wo_no,
                "execution_type": exec_type,
                "partner_id": partner_id,
                "po_number": po_no,
                "error_message": (
                    "협력사 수주 접수 실패로 본사 ERP 선점 및 오더를 자동 보상 롤백했습니다: "
                    f"{mcp_order_res.get('error_message')}"
                ),
                "committed_at": committed_at,
            }

        if mcp_order_res.get("b2b_ack"):
            b2b_ack = mcp_order_res["b2b_ack"]
        purchase_status = "CONFIRMED"

    return {
        "erp_transaction_status": "COMMITTED",
        "purchase_status": purchase_status,
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

# ==============================================================================
# 4. ReAct 에이전트 자율 탐색(Auto-Discovery)용 원자적 도구 3종
# ==============================================================================
@tool
def find_pending_blocks(
    branch_code: Optional[str] = None,
    process: Optional[str] = None,
    exclude_blocks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    ERP_BLOCK_MASTER와 MES_WORK_ORDER_DISPATCH를 대조해 현재 야드에서
    작업 편성 가능한 대기(미투입) 블록 후보 목록을 우선순위순으로 조회합니다.
    """
    if not erp_engine:
        return {"status": "ERROR", "blocks": [], "error_message": "Oracle 연결 없음"}

    valid_procs = {"FCAW", "SAW", "GMAW", "SMAW", "GTAW", "TIG", "MIG"}
    if process and str(process).strip().upper() not in valid_procs:
        process = None

    excluded = {str(b).strip().upper() for b in (exclude_blocks or []) if b}
    try:
        with erp_engine.connect() as conn:
            br_rows = conn.execute(
                text("SELECT branch_code FROM ERP_BRANCHES")
            ).mappings().all()
            valid_branch_codes = {
                str(r["branch_code"]).strip().upper()
                for r in br_rows
                if r.get("branch_code")
            }
            if branch_code and str(branch_code).strip().upper() not in valid_branch_codes:
                branch_code = None

            active_rows = conn.execute(
                text("""
                    SELECT DISTINCT UPPER(TRIM(block_id)) AS block_id
                    FROM MES_WORK_ORDER_DISPATCH
                    WHERE UPPER(status) = 'COMMITTED'
                """)
            ).mappings().all()
            active_blocks = {r["block_id"] for r in active_rows if r.get("block_id")}

            rows = conn.execute(
                text("""
                    SELECT b.block_id, b.ship_code, b.block_name, b.steel_type AS material,
                           b.thickness_mm AS thickness, b.joint_type, b.process,
                           b.target_dock AS work_zone, b.branch_code, b.weight_ton,
                           br.region_name
                    FROM ERP_BLOCK_MASTER b
                    LEFT JOIN ERP_BRANCHES br ON b.branch_code = br.branch_code
                    ORDER BY b.block_id ASC
                """)
            ).mappings().all()

            candidates = []
            for r in rows:
                d = dict(r)
                bid = str(d.get("block_id") or "").strip().upper()
                if bid in excluded:
                    continue
                if branch_code and str(d.get("branch_code") or "").upper() != str(branch_code).upper():
                    continue
                if process and str(d.get("process") or "").upper() != str(process).upper():
                    continue
                d["weight_ton"] = float(d["weight_ton"]) if d.get("weight_ton") is not None else 0.0
                d["is_active_work"] = bid in active_blocks
                candidates.append(d)

            # 아직 작업 배정되지 않은(is_active_work=False) 블록을 우선 정렬
            candidates.sort(key=lambda x: (x["is_active_work"], x["block_id"]))
            return {
                "status": "SUCCESS",
                "total_candidates": len(candidates),
                "blocks": candidates,
            }
    except Exception as e:
        return {"status": "ERROR", "blocks": [], "error_message": str(e)}


@tool
def find_eligible_workers(
    branch_code: Optional[str] = None,
    process: Optional[str] = None,
    exclude_workers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    대상 야드 권역과 용접 공정에 부합하고, 선급 자격증이 유효(VALID)하며
    현재 다른 블록에 중복 투입(COMMITTED)되지 않은 대기(IDLE) 작업자 후보를 우선순위순으로 조회합니다.
    """
    if not erp_engine:
        return {"status": "ERROR", "workers": [], "error_message": "Oracle 연결 없음"}

    valid_procs = {"FCAW", "SAW", "GMAW", "SMAW", "GTAW", "TIG", "MIG"}
    if process and str(process).strip().upper() not in valid_procs:
        process = None

    excluded = {str(w).strip() for w in (exclude_workers or []) if w}
    try:
        with erp_engine.connect() as conn:
            br_rows = conn.execute(
                text("SELECT branch_code, region_name FROM ERP_BRANCHES")
            ).mappings().all()
            br_region_map = {
                str(r["branch_code"]).strip().upper(): str(r["region_name"]).strip()
                for r in br_rows
                if r.get("branch_code")
            }
            if branch_code and str(branch_code).strip().upper() not in br_region_map:
                branch_code = None

            target_region = (
                br_region_map.get(str(branch_code).strip().upper(), "")
                if branch_code
                else ""
            )

            busy_rows = conn.execute(
                text("""
                    SELECT worker_id, worker_name, block_id, execution_type
                    FROM MES_WORK_ORDER_DISPATCH
                    WHERE UPPER(status) = 'COMMITTED'
                """)
            ).mappings().all()
            busy_ids = {
                str(r["worker_id"]).strip()
                for r in busy_rows
                if r.get("worker_id") and str(r.get("execution_type", "")).upper() == "IN_HOUSE"
            }
            busy_names = {str(r["worker_name"]).strip() for r in busy_rows if r.get("worker_name")}

            eq_busy_rows = conn.execute(
                text("""
                    SELECT driver_worker_id
                    FROM ERP_VEHICLES_EQUIPMENT
                    WHERE UPPER(operation_status) = 'IN_USE'
                      AND driver_worker_id IS NOT NULL
                """)
            ).mappings().all()
            for er in eq_busy_rows:
                busy_ids.add(str(er["driver_worker_id"]).strip())

            w_rows = conn.execute(
                text("""
                    SELECT worker_id, worker_name, employment_type, company_code,
                           branch_code, role, cert_code, weld_process,
                           cert_expiry_date, status
                    FROM ERP_WORKER_CERT
                    WHERE UPPER(status) = 'VALID'
                    ORDER BY worker_id ASC
                """)
            ).mappings().all()

            eligible = []
            for r in w_rows:
                d = dict(r)
                w_id = str(d.get("worker_id") or "").strip()
                w_nm = str(d.get("worker_name") or "").strip()
                if w_nm in excluded or w_id in excluded:
                    continue
                if w_id in busy_ids or w_nm in busy_names:
                    continue

                w_proc = str(d.get("weld_process") or "").upper()
                if process and w_proc != str(process).upper():
                    continue

                w_br = str(d.get("branch_code") or "").strip().upper()
                w_region = br_region_map.get(w_br, "")
                if target_region and w_region and w_region != target_region:
                    continue

                d["region_name"] = w_region
                d["is_hq"] = str(d.get("company_code") or "").upper() == "COMP-HQ"
                eligible.append(d)

            # 본사 직영(COMP-HQ) 우선, 그 다음 동일 야드 코드 우선 정렬
            eligible.sort(
                key=lambda x: (
                    not x["is_hq"],
                    str(x.get("branch_code") or "").upper() != str(branch_code or "").upper(),
                    x["worker_id"],
                )
            )
            return {
                "status": "SUCCESS",
                "target_branch": branch_code,
                "target_region": target_region,
                "required_process": process,
                "eligible_count": len(eligible),
                "workers": eligible,
            }
    except Exception as e:
        return {"status": "ERROR", "workers": [], "error_message": str(e)}


@tool
def check_yard_materials_and_equipment(
    branch_code: Optional[str] = None,
    process: Optional[str] = None,
    weight_ton: Optional[float] = None,
    requested_equipment: Optional[str] = None,
) -> Dict[str, Any]:
    """
    특정 야드의 가용 용접 와이어·강판 재고 및 대기(STANDBY) 양중/운반 장비를 점검하고,
    블록 중량(weight_ton)이 주어진 경우 정격하중 안전율(90% 이하)까지 함께 계산합니다.
    """
    erp_res = check_hq_erp_resources.invoke({
        "branch_code": branch_code,
        "process": process,
        "equipment": requested_equipment,
    })
    lifting_eval = None
    if weight_ton is not None and float(weight_ton) > 0:
        lifting_eval = evaluate_heavy_lifting_capacity(
            block_weight_ton=float(weight_ton),
            vehicles_equipment=erp_res.get("vehicles_equipment") or [],
            target_branch=branch_code,
            requested_equipment=requested_equipment,
        )

    standby_eq = [
        eq for eq in (erp_res.get("vehicles_equipment") or [])
        if str(eq.get("operation_status", "")).upper() == "STANDBY"
    ]
    return {
        "status": erp_res.get("status"),
        "branch_code": branch_code,
        "stock_available": erp_res.get("stock_available"),
        "selected_wire_code": erp_res.get("selected_wire_code"),
        "selected_plate_code": erp_res.get("selected_plate_code"),
        "assigned_lot_no": erp_res.get("assigned_lot_no"),
        "outsourcing_reason": erp_res.get("outsourcing_reason"),
        "standby_equipment": standby_eq,
        "lifting_capacity_eval": lifting_eval,
    }