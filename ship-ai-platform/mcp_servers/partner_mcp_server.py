"""
partner_mcp_server.py
독립 FastMCP + A2A 외부 협력사 B2B 게이트웨이 서버 (port=8001, streamable-http)
- MCP: 외부 협력사 PostgreSQL(PARTNER_PG_DATABASE_URL) 원장 조회 및 수주(PO) 발주 등록
- A2A: Agent Card 공개(/.well-known/agent-card.json) 및 JSON-RPC 2.0(/a2a) 자율 협의·보상 취소
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine, text
from starlette.requests import Request
from starlette.responses import JSONResponse

# 불필요한 스트림 종료 로그 억제 (weather_server.py와 동일)
logging.getLogger("mcp.server.streamable_http").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PLATFORM_ROOT / ".env")

PG_URI = os.getenv("PARTNER_PG_DATABASE_URL")
PARTNER_A2A_BASE_URL = os.getenv("PARTNER_A2A_BASE_URL", "http://127.0.0.1:8001")
PARTNER_A2A_API_KEY = os.getenv("PARTNER_A2A_API_KEY", "")  # 설정 시 Bearer 인증 강제

try:
    pg_engine = create_engine(PG_URI, pool_pre_ping=True) if PG_URI else None
except Exception as e:
    print(f"🚨 [Partner MCP PostgreSQL 엔진 초기화 실패]: {e}")
    pg_engine = None

mcp = FastMCP("partner-b2b-server", host="0.0.0.0", port=8001)

# A2A Task 상태 메모리 저장소 (감사 추적 및 tasks/get 조회용)
A2A_TASK_STORE: Dict[str, Dict[str, Any]] = {}


@mcp.tool()
def query_partner_b2b_catalog(
    outsourcing_domain: str = "WELDING_MATERIAL",
    material: str = "",
    process: str = "",
    partner_id: str = "",
    worker_name: str = "",
) -> str:
    """
    외부 협력사 PostgreSQL B2B ERP View(v_b2b_daehan_materials, v_b2b_donghae_workers, v_b2b_hankook_transporters)를
    조회하여 실시간 외주 자재 단가/재고, 가용 용접사, 중량물 트랜스포터 배차 가능 현황을 JSON 문자열로 반환합니다.
    """
    domain = (outsourcing_domain or "WELDING_MATERIAL").upper().strip()
    timestamp_iso = datetime.now().isoformat()
    print(
        f"\n📡 [Partner FastMCP 조회 수신] domain='{domain}', "
        f"material='{material}', process='{process}', worker='{worker_name}'"
    )

    if not pg_engine:
        return json.dumps(
            {
                "status": "ERROR",
                "error_message": "협력사 서버에 PARTNER_PG_DATABASE_URL이 설정되지 않았습니다.",
                "b2b_inquiry_timestamp": timestamp_iso,
            },
            ensure_ascii=False,
            default=str,
        )

    try:
        with pg_engine.connect() as conn:
            # 전체 등록 협력사 공개 뷰(v_b2b_company_locations) 공통 조회
            all_companies = []
            try:
                comp_rows = conn.execute(
                    text("SELECT * FROM v_b2b_company_locations")
                ).mappings().all()
                all_companies = [dict(r) for r in comp_rows]
            except Exception as comp_err:
                print(f"⚠️ [v_b2b_company_locations 조회 참고]: {comp_err}")

            # 0. 등록 협력사 전체 현황 조회
            if domain == "ALL_PARTNERS":
                res = {
                    "status": "SUCCESS",
                    "b2b_inquiry_timestamp": timestamp_iso,
                    "source": "FastMCP Partner Gateway -> PostgreSQL (v_b2b_company_locations)",
                    "registered_companies": all_companies,
                    "selected_partner": None,
                }
                return json.dumps(res, ensure_ascii=False, default=str)

            # 1. 용접 자재 / 철판 협력사 (대한특수철강)
            if domain == "WELDING_MATERIAL":
                rows = conn.execute(
                    text("SELECT * FROM v_b2b_daehan_materials")
                ).mappings().all()
                items = [dict(r) for r in rows]
                selected = None
                if items:
                    for it in items:
                        if (
                            "WIRE" in str(it.get("supplier_sku", "")).upper()
                            or "와이어" in str(it.get("item_name", ""))
                        ):
                            selected = dict(it)
                            break
                    if not selected:
                        selected = dict(items[0])

                    selected.setdefault("partner_id", "VEND-DAEHAN-MAT")
                    selected.setdefault("partner_name", "대한특수철강")
                    selected.setdefault("business_type", "WELDING_MATERIAL")

                res = {
                    "status": "SUCCESS",
                    "b2b_inquiry_timestamp": timestamp_iso,
                    "source": "FastMCP Partner Gateway -> PostgreSQL (v_b2b_daehan_materials)",
                    "available_items": items,
                    "selected_partner": selected,
                }
                return json.dumps(res, ensure_ascii=False, default=str)

            # 2. 외주 용접 인력 협력사 (동해블록용접 / 남해해양플랜트용접 등)
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
                    if not selected and process:
                        req_proc = str(process).upper().strip()
                        for w in workers:
                            if (
                                w.get("can_accept_order", True)
                                and str(w.get("weld_process", "")).upper() == req_proc
                            ):
                                selected = dict(w)
                                break
                    if not selected:
                        for w in workers:
                            if w.get("can_accept_order", True):
                                selected = dict(w)
                                break
                    if not selected:
                        selected = dict(workers[0])

                    p_id = selected.get("partner_id") or "VEND-DONGHAE-WELD"
                    partner_name_map = {
                        "VEND-DONGHAE-WELD": "동해블록용접(주)",
                        "VEND-NAMHAE-ENG": "남해해양플랜트용접",
                        "VEND-KOREA-CAD": "한국선체설계외주(주)",
                    }
                    selected["partner_id"] = p_id
                    if not selected.get("partner_name"):
                        selected["partner_name"] = partner_name_map.get(p_id, p_id)
                    selected.setdefault("business_type", "OUTSOURCED_WORKER")

                res = {
                    "status": "SUCCESS",
                    "b2b_inquiry_timestamp": timestamp_iso,
                    "source": "FastMCP Partner Gateway -> PostgreSQL (v_b2b_donghae_workers)",
                    "available_workers": workers,
                    "selected_partner": selected,
                }
                return json.dumps(res, ensure_ascii=False, default=str)

            # 3. 중량물 운송 / 트랜스포터 협력사 (한국중량물운송)
            else:
                transporters = []
                try:
                    rows = conn.execute(
                        text("SELECT * FROM v_b2b_hankook_transporters")
                    ).mappings().all()
                    transporters = [dict(r) for r in rows]
                except Exception as view_err:
                    print(f"⚠️ [v_b2b_hankook_transporters 조회 참고]: {view_err}")

                selected = None
                if transporters:
                    selected = dict(transporters[0])
                    selected.setdefault("partner_id", "VEND-HANKOOK-TP")
                    selected.setdefault("partner_name", "한국중량물운송(주)")
                    selected.setdefault("business_type", "HEAVY_TRANSPORT")

                res = {
                    "status": "SUCCESS",
                    "b2b_inquiry_timestamp": timestamp_iso,
                    "source": "FastMCP Partner Gateway -> PostgreSQL (v_b2b_hankook_transporters)",
                    "available_transporters": transporters,
                    "selected_partner": selected,
                }
                return json.dumps(res, ensure_ascii=False, default=str)

    except Exception as e:
        return json.dumps(
            {
                "status": "ERROR",
                "error_message": str(e),
                "b2b_inquiry_timestamp": timestamp_iso,
            },
            ensure_ascii=False,
            default=str,
        )


@mcp.tool()
def create_partner_b2b_order(
    po_no: str,
    wo_no: str,
    partner_id: str,
    ordered_ref_code: str,
    order_qty: float = 15.0,
    allocated_lot_or_id: str = "",
    delivery_yard_zone: str = "ZONE-A-BAY1",
) -> str:
    """
    본사 관리자 승인(HITL) 완료 시 협력사 PostgreSQL 수주 원장(b2b_sales_orders)에 B2B 발주(PO)를 접수합니다.
    """
    print(f"\n📨 [Partner FastMCP 수주 접수] PO={po_no}, WO={wo_no}, Partner={partner_id}")
    b2b_ack = f"ACK-{wo_no}-{partner_id}"

    if not pg_engine:
        return json.dumps(
            {
                "status": "ERROR",
                "error_message": "PARTNER_PG_DATABASE_URL 미설정",
            },
            ensure_ascii=False,
        )

    try:
        with pg_engine.begin() as p_conn:
            p_conn.execute(
                text("""
                    INSERT INTO b2b_sales_orders (
                        b2b_order_no, client_company, hq_wo_number, partner_id,
                        order_type, ordered_ref_code, order_qty,
                        allocated_lot_or_id, delivery_yard_zone, order_status, created_at
                    ) VALUES (
                        :po_no, 'COMP-HQ', :wo_no, :p_id,
                        'OUTSOURCED_B2B', :ref_code, :qty,
                        :lot_or_id, :zone, 'CONFIRMED', CURRENT_TIMESTAMP
                    )
                """),
                {
                    "po_no": po_no,
                    "wo_no": wo_no,
                    "p_id": partner_id,
                    "ref_code": ordered_ref_code,
                    "qty": float(order_qty),
                    "lot_or_id": allocated_lot_or_id,
                    "zone": delivery_yard_zone,
                },
            )
        return json.dumps(
            {
                "status": "CONFIRMED",
                "po_no": po_no,
                "b2b_ack": b2b_ack,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {
                "status": "ERROR",
                "po_no": po_no,
                "b2b_ack": b2b_ack,
                "error_message": str(e),
            },
            ensure_ascii=False,
        )


# ==============================================================================
# A2A (Agent-to-Agent) Agent Card 및 JSON-RPC 2.0 자율 협의 계층
# ==============================================================================
def _build_agent_card() -> Dict[str, Any]:
    """A2A 표준 Agent Card 명세 (/.well-known/agent-card.json 및 하위호환 /.well-known/agent.json 공용)"""
    return {
        "name": "K-Shipyard B2B Partner Supply & Logistics Agent",
        "description": (
            "조선소 협력사(대한특수철강·동해블록용접·한국중량물운송) 통합 B2B 영업·배차 에이전트. "
            "내부 원가 및 타사 스케줄 기밀을 보호(Privacy Boundary)하면서 용접 자재 분할납품·긴급 ETA, "
            "중량물 트랜스포터 하중·이동 조율, 선급 인증 외주 용접사 매칭을 자율 협의합니다."
        ),
        "url": f"{PARTNER_A2A_BASE_URL.rstrip('/')}/a2a",
        "version": "1.0.0",
        "protocolVersion": "0.3.0",
        "provider": {
            "organization": "Partner B2B Consortium (Daehan / Donghae / Hankook)",
            "url": PARTNER_A2A_BASE_URL,
        },
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "authentication": {
            "schemes": ["Bearer"] if PARTNER_A2A_API_KEY else ["None"],
        },
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json"],
        "skills": [
            {
                "id": "negotiate_material_supply",
                "name": "용접 자재 긴급 납기(ETA) 및 분할 납품 자율 협의",
                "description": (
                    "요청 수량과 희망 납기를 내부 재고·리드타임과 대조해 즉시 출고, "
                    "분할 납품(Split Delivery), 동급 선급 대체 LOT안을 역제안합니다."
                ),
                "tags": ["WELDING_MATERIAL", "RFQ", "SPLIT_DELIVERY"],
            },
            {
                "id": "negotiate_heavy_transport",
                "name": "중량물 블록 운송·양중 장비 톤수 및 야드 이동 조율",
                "description": (
                    "블록 중량(ton)과 목표 야드를 기준으로 정격하중 안전율(90% 이하)을 "
                    "만족하는 트랜스포터·크레인 배차와 야드 간 이동 ETA를 산출합니다."
                ),
                "tags": ["HEAVY_TRANSPORT", "LIFTING_PLAN", "ETA"],
            },
            {
                "id": "match_certified_welder",
                "name": "선급 인증 외주 용접사 제약 기반 매칭 및 대체 역제안",
                "description": (
                    "요청 공정(FCAW/SAW/GMAW)·야드 권역·유효 선급 자격을 충족하는 "
                    "가용 용접사를 선별하고 투입 가능 일정을 회신합니다."
                ),
                "tags": ["OUTSOURCED_WORKER", "CERT_MATCH"],
            },
        ],
    }


def _evaluate_partner_a2a_rfq(rfq: Dict[str, Any]) -> Dict[str, Any]:
    """협력사 자체 DB를 근거로 내부 제약조건을 판단해 A2A 협상안(Offer / Counter-Offer)을 생성"""
    domain = str(rfq.get("outsourcing_domain") or "WELDING_MATERIAL").upper().strip()
    urgency = str(rfq.get("urgency") or "NORMAL").upper().strip()
    req_qty = float(rfq.get("requested_qty") or 0)
    req_weight_ton = float(rfq.get("weight_ton") or 0)
    target_branch = str(rfq.get("target_branch") or "BR-ULSAN-01").upper().strip()
    target_region = "거제" if "GEOJE" in target_branch else "울산"
    process = str(rfq.get("process") or "").upper().strip()
    worker_name = str(rfq.get("worker_name") or "").strip()
    req_item = str(rfq.get("requested_item_name") or "").strip()

    now_iso = datetime.now().isoformat()
    privacy_notice = (
        "협력사 내부 매입 원가, 타 조선소 수주 단가 및 타사 배차 상세 일정은 "
        "Privacy Boundary 정책에 따라 비공개 처리되었으며, 협의 가능한 납기·가용 조건만 회신되었습니다."
    )

    if not pg_engine:
        return {
            "a2a_task_status": "failed",
            "business_status": "REJECTED",
            "negotiation_decision": "REJECTED",
            "summary": "협력사 내부 ERP DB 연결 불가로 협의 실패",
            "privacy_boundary_notice": privacy_notice,
            "timestamp": now_iso,
        }

    try:
        with pg_engine.connect() as conn:
            # ------------------------------------------------------------------
            # 1. 용접 자재(WELDING_MATERIAL) 분할 납품 / 긴급 ETA 자율 협의
            # ------------------------------------------------------------------
            if domain == "WELDING_MATERIAL":
                rows = conn.execute(
                    text("SELECT * FROM v_b2b_daehan_materials")
                ).mappings().all()
                items = [dict(r) for r in rows]
                if not items:
                    return {
                        "a2a_task_status": "completed",
                        "business_status": "REJECTED",
                        "negotiation_decision": "REJECTED",
                        "summary": "협력사 자재 카탈로그에 가용 품목이 없습니다.",
                        "privacy_boundary_notice": privacy_notice,
                        "timestamp": now_iso,
                    }

                matched = None
                for it in items:
                    txt = (
                        f"{it.get('hq_item_code', '')} "
                        f"{it.get('supplier_sku', '')} "
                        f"{it.get('item_name', '')}"
                    ).upper()
                    if req_item and req_item.upper() in txt:
                        matched = dict(it)
                        break
                    if process and process in txt:
                        matched = dict(it)
                        break
                if not matched:
                    for it in items:
                        if (
                            "WIRE" in str(it.get("supplier_sku", "")).upper()
                            or "와이어" in str(it.get("item_name", ""))
                        ):
                            matched = dict(it)
                            break
                matched = matched or dict(items[0])

                avail_qty = float(
                    matched.get("available_qty")
                    or matched.get("stock_qty")
                    or matched.get("qty")
                    or 0
                )
                base_price = float(matched.get("unit_price_krw") or 4500)
                base_lead_h = float(matched.get("lead_time_hours") or 4)
                eff_qty = req_qty if req_qty > 0 else 100.0
                partner_name = matched.get("partner_name") or "대한특수철강"
                partner_id = matched.get("partner_id") or "VEND-DAEHAN-MAT"

                options = []
                if avail_qty >= eff_qty:
                    if urgency == "URGENT" and base_lead_h > 1.5:
                        agreed_eta_h = max(1.0, round(base_lead_h * 0.5, 1))
                        agreed_price = int(round(base_price * 1.05))
                        decision = "COUNTER_OFFER"
                        summary = (
                            f"{partner_name}: 요청 수량 {eff_qty:g}kg 전량 보유 확인(가용 {avail_qty:g}kg). "
                            f"긴급 직송 전환 시 납기를 {base_lead_h:g}시간 ➔ {agreed_eta_h:g}시간으로 단축 가능 "
                            f"(긴급 배차 5% 반영 {agreed_price:,}원/kg)."
                        )
                    elif eff_qty >= 500:
                        agreed_eta_h = base_lead_h
                        agreed_price = int(round(base_price * 0.97))
                        decision = "ACCEPTED"
                        summary = (
                            f"{partner_name}: {eff_qty:g}kg 대량 발주 확인(가용 {avail_qty:g}kg). "
                            f"단가 3% 협의 할인({agreed_price:,}원/kg) 적용, {agreed_eta_h:g}시간 내 일괄 납품 가능."
                        )
                    else:
                        agreed_eta_h = base_lead_h
                        agreed_price = int(round(base_price))
                        decision = "ACCEPTED"
                        summary = (
                            f"{partner_name}: 요청 수량 {eff_qty:g}kg 전량 즉시 출고 가능 "
                            f"({agreed_eta_h:g}시간 내 도착, {agreed_price:,}원/kg)."
                        )
                    options.append({
                        "option_code": "OPT-FULL-DELIVERY",
                        "label": "전량 일괄 납품",
                        "first_batch_qty": eff_qty,
                        "second_batch_qty": 0,
                        "eta_hours": agreed_eta_h,
                        "unit_price_krw": agreed_price,
                        "lot_no": matched.get("partner_lot_no") or matched.get("lot_no"),
                    })
                else:
                    shortage = round(eff_qty - avail_qty, 2)
                    alt_item = next(
                        (
                            dict(it)
                            for it in items
                            if it.get("supplier_sku") != matched.get("supplier_sku")
                            and (
                                "WIRE" in str(it.get("supplier_sku", "")).upper()
                                or "와이어" in str(it.get("item_name", ""))
                            )
                            and float(it.get("available_qty") or it.get("stock_qty") or 0) >= shortage
                        ),
                        None,
                    )
                    decision = "COUNTER_OFFER"
                    agreed_eta_h = base_lead_h
                    agreed_price = int(round(base_price))
                    options.append({
                        "option_code": "OPT-SPLIT-DELIVERY",
                        "label": f"분할 납품 (1차 {avail_qty:g}kg 즉시 + 2차 {shortage:g}kg 익일 오전 08:00)",
                        "first_batch_qty": avail_qty,
                        "second_batch_qty": shortage,
                        "eta_hours": base_lead_h,
                        "second_batch_eta_hours": 16.0,
                        "unit_price_krw": agreed_price,
                        "lot_no": matched.get("partner_lot_no") or matched.get("lot_no"),
                    })
                    if alt_item:
                        options.append({
                            "option_code": "OPT-MIXED-EQUIV-LOT",
                            "label": (
                                f"동급 선급 대체 합산 납품 (주품목 {avail_qty:g}kg + "
                                f"대체품 {alt_item.get('item_name')} {shortage:g}kg)"
                            ),
                            "first_batch_qty": avail_qty,
                            "alt_batch_qty": shortage,
                            "alt_item_code": alt_item.get("hq_item_code") or alt_item.get("supplier_sku"),
                            "alt_lot_no": alt_item.get("partner_lot_no") or alt_item.get("lot_no"),
                            "eta_hours": max(base_lead_h, float(alt_item.get("lead_time_hours") or 2)),
                            "unit_price_krw": int(round(float(alt_item.get("unit_price_krw") or base_price))),
                        })
                    summary = (
                        f"{partner_name}: 요청 {eff_qty:g}kg 대비 즉시 가용 재고 {avail_qty:g}kg(부족 {shortage:g}kg). "
                        f"1차 {avail_qty:g}kg {base_lead_h:g}시간 내 우선 출고 + 잔여 {shortage:g}kg 분할/대체 납품안 역제안."
                    )

                return {
                    "a2a_task_status": "input-required" if decision == "COUNTER_OFFER" else "completed",
                    "business_status": "NEGOTIATED",
                    "negotiation_decision": decision,
                    "partner_id": partner_id,
                    "partner_name": partner_name,
                    "domain": domain,
                    "requested_qty": eff_qty,
                    "available_qty": avail_qty,
                    "agreed_eta_hours": agreed_eta_h,
                    "agreed_unit_price_krw": agreed_price,
                    "proposal_options": options,
                    "summary": summary,
                    "privacy_boundary_notice": privacy_notice,
                    "negotiation_trace": [
                        f"[HQ ➔ Partner RFQ] 품목={matched.get('item_name')}, 요청수량={eff_qty:g}kg, 긴급도={urgency}",
                        f"[Partner 제약 분석] 가용재고={avail_qty:g}kg, 기본리드타임={base_lead_h:g}h",
                        f"[Partner ➔ HQ 응답] 판정={decision} | {summary}",
                    ],
                    "timestamp": now_iso,
                }

            # ------------------------------------------------------------------
            # 2. 중량물 운송/양중(HEAVY_TRANSPORT) 하중·야드 이동시간 자율 협의
            # ------------------------------------------------------------------
            if domain == "HEAVY_TRANSPORT":
                transporters = []
                try:
                    rows = conn.execute(
                        text("SELECT * FROM v_b2b_hankook_transporters")
                    ).mappings().all()
                    transporters = [dict(r) for r in rows]
                except Exception:
                    transporters = []

                required_cap = round((req_weight_ton * 1.05) / 0.9, 1) if req_weight_ton > 0 else 50.0
                chosen = None
                for tp in transporters:
                    cap = float(tp.get("max_load_ton") or 200.0)
                    if cap >= required_cap:
                        chosen = dict(tp)
                        break
                if not chosen and transporters:
                    chosen = dict(transporters[0])

                partner_id = (chosen or {}).get("partner_id") or "VEND-HANKOOK-TP"
                partner_name = (chosen or {}).get("partner_name") or "한국중량물운송(주)"
                veh_code = (
                    (chosen or {}).get("assigned_vehicle_code")
                    or (chosen or {}).get("asset_code")
                    or "VEH-TP-GEOJE-01"
                )
                max_load = float((chosen or {}).get("max_load_ton") or 200.0)
                veh_branch = str((chosen or {}).get("branch_code") or "BR-GEOJE-01").upper()
                base_rate = int(
                    float(
                        (chosen or {}).get("trip_rate_krw")
                        or (chosen or {}).get("unit_price_krw")
                        or 1800000
                    )
                )

                same_region = (
                    (target_region == "거제" and "GEOJE" in veh_branch)
                    or (target_region == "울산" and "ULSAN" in veh_branch)
                )
                if same_region:
                    eta_h = 1.0 if urgency == "URGENT" else 2.0
                    decision = "ACCEPTED"
                    agreed_rate = base_rate
                    move_note = f"동일 권역({target_region}) 대기 차량 즉시 투입"
                else:
                    eta_h = 3.0 if urgency == "URGENT" else 4.0
                    decision = "COUNTER_OFFER"
                    agreed_rate = int(round(base_rate * 1.15))
                    move_note = (
                        f"현재 차량 위치({veh_branch})에서 {target_branch} 회송 이동 "
                        f"{eta_h:g}시간 소요(회송비 15% 포함)"
                    )

                load_ratio = (
                    round(((req_weight_ton * 1.05) / max_load) * 100, 1)
                    if (req_weight_ton > 0 and max_load > 0)
                    else 0.0
                )
                summary = (
                    f"{partner_name}: {veh_code}(정격 {max_load:g}톤) 배정 가능"
                    + (f" (블록 {req_weight_ton:g}톤 기준 하중률 {load_ratio}% 안전)" if req_weight_ton > 0 else "")
                    + f". {move_note}, 제안 단가 {agreed_rate:,}원/회."
                )

                return {
                    "a2a_task_status": "input-required" if decision == "COUNTER_OFFER" else "completed",
                    "business_status": "NEGOTIATED",
                    "negotiation_decision": decision,
                    "partner_id": partner_id,
                    "partner_name": partner_name,
                    "domain": domain,
                    "assigned_vehicle_code": veh_code,
                    "max_load_ton": max_load,
                    "load_ratio_pct": load_ratio,
                    "agreed_eta_hours": eta_h,
                    "agreed_unit_price_krw": agreed_rate,
                    "proposal_options": [{
                        "option_code": "OPT-HEAVY-DISPATCH",
                        "label": f"{veh_code} ({max_load:g}톤) 배차 · ETA {eta_h:g}시간",
                        "vehicle_code": veh_code,
                        "max_load_ton": max_load,
                        "eta_hours": eta_h,
                        "unit_price_krw": agreed_rate,
                    }],
                    "summary": summary,
                    "privacy_boundary_notice": privacy_notice,
                    "negotiation_trace": [
                        f"[HQ ➔ Partner RFQ] 대상야드={target_branch}, 블록중량={req_weight_ton:g}톤, 긴급도={urgency}",
                        f"[Partner 제약 분석] 차량={veh_code}({max_load:g}톤, 위치={veh_branch}), 안전하중률={load_ratio}%",
                        f"[Partner ➔ HQ 응답] 판정={decision} | {summary}",
                    ],
                    "timestamp": now_iso,
                }

            # ------------------------------------------------------------------
            # 3. 외주 용접 인력(OUTSOURCED_WORKER) 자격·야드 매칭 및 대체 역제안
            # ------------------------------------------------------------------
            rows = conn.execute(
                text("SELECT * FROM v_b2b_donghae_workers")
            ).mappings().all()
            workers = [dict(r) for r in rows]
            exact_worker = None
            substitute_workers = []

            for w in workers:
                w_nm = str(w.get("worker_name") or "").strip()
                w_proc = str(w.get("weld_process") or "").upper()
                can_accept = bool(w.get("can_accept_order", True))
                if worker_name and w_nm == worker_name and can_accept:
                    if not process or w_proc == process:
                        exact_worker = dict(w)
                if can_accept and (not process or w_proc == process):
                    substitute_workers.append(dict(w))

            chosen_w = exact_worker or (substitute_workers[0] if substitute_workers else None)
            if not chosen_w:
                return {
                    "a2a_task_status": "completed",
                    "business_status": "REJECTED",
                    "negotiation_decision": "REJECTED",
                    "summary": f"요청 공정({process or '전체'})을 충족하는 가용 외주 용접사가 현재 없습니다.",
                    "privacy_boundary_notice": privacy_notice,
                    "timestamp": now_iso,
                }

            decision = "ACCEPTED" if (exact_worker or not worker_name) else "COUNTER_OFFER"
            partner_id = chosen_w.get("partner_id") or "VEND-DONGHAE-WELD"
            partner_name = chosen_w.get("partner_name") or "동해블록용접(주)"
            daily_rate = int(float(chosen_w.get("daily_rate_krw") or 350000))
            eta_h = 1.5 if urgency == "URGENT" else 3.0
            summary = (
                f"{partner_name}: 선급 자격({chosen_w.get('cert_code')}, {chosen_w.get('weld_process')}) 보유 "
                f"'{chosen_w.get('worker_name')}' 투입 가능 (ETA {eta_h:g}시간, 일당 {daily_rate:,}원)."
            )
            if decision == "COUNTER_OFFER":
                summary = (
                    f"{partner_name}: 지정 인원 '{worker_name}' 투입 불가(타 일정/공정 미일치) ➔ "
                    f"동일 공정 대체 용접사 '{chosen_w.get('worker_name')}'({chosen_w.get('cert_code')}) 역제안 "
                    f"(ETA {eta_h:g}시간, 일당 {daily_rate:,}원)."
                )

            return {
                "a2a_task_status": "input-required" if decision == "COUNTER_OFFER" else "completed",
                "business_status": "NEGOTIATED",
                "negotiation_decision": decision,
                "partner_id": partner_id,
                "partner_name": partner_name,
                "domain": domain,
                "recommended_worker": chosen_w,
                "agreed_eta_hours": eta_h,
                "agreed_unit_price_krw": daily_rate,
                "proposal_options": [{
                    "option_code": "OPT-WELDER-ASSIGN",
                    "label": f"{chosen_w.get('worker_name')} ({chosen_w.get('cert_code')}/{chosen_w.get('weld_process')})",
                    "worker_name": chosen_w.get("worker_name"),
                    "eta_hours": eta_h,
                    "unit_price_krw": daily_rate,
                }],
                "summary": summary,
                "privacy_boundary_notice": privacy_notice,
                "negotiation_trace": [
                    f"[HQ ➔ Partner RFQ] 요청공정={process or '무관'}, 지정작업자={worker_name or '미지정'}, 야드={target_branch}",
                    f"[Partner 제약 분석] 가용후보={len(substitute_workers)}명 중 선급자격·공정 최적 매칭",
                    f"[Partner ➔ HQ 응답] 판정={decision} | {summary}",
                ],
                "timestamp": now_iso,
            }
    except Exception as e:
        return {
            "a2a_task_status": "failed",
            "business_status": "FAILED",
            "negotiation_decision": "ERROR",
            "summary": f"협력사 A2A 협의 처리 오류: {str(e)}",
            "privacy_boundary_notice": privacy_notice,
            "timestamp": now_iso,
        }


@mcp.custom_route("/.well-known/agent-card.json", methods=["GET"])
async def a2a_agent_card_v03(request: Request) -> JSONResponse:
    return JSONResponse(_build_agent_card())


@mcp.custom_route("/.well-known/agent.json", methods=["GET"])
async def a2a_agent_card_legacy(request: Request) -> JSONResponse:
    return JSONResponse(_build_agent_card())


@mcp.custom_route("/a2a", methods=["POST"])
async def a2a_jsonrpc_endpoint(request: Request) -> JSONResponse:
    """A2A JSON-RPC 2.0 엔드포인트 (tasks/send, tasks/get, tasks/cancel)"""
    if PARTNER_A2A_API_KEY:
        auth_hdr = request.headers.get("Authorization", "")
        if auth_hdr != f"Bearer {PARTNER_A2A_API_KEY}":
            return JSONResponse(
                {"error": {"code": -32001, "message": "Unauthorized A2A Bearer Token"}},
                status_code=401,
            )

    body = await request.json()
    rpc_id = body.get("id", str(uuid.uuid4()))
    method = body.get("method", "tasks/send")
    params = body.get("params") or {}

    if method == "tasks/get":
        tid = params.get("id")
        task = A2A_TASK_STORE.get(tid)
        if not task:
            return JSONResponse(
                {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32004, "message": "Task not found"}},
                status_code=404,
            )
        return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": task})

    if method == "tasks/cancel":
        po_no = params.get("po_no")
        cancelled = False
        if po_no and pg_engine:
            try:
                with pg_engine.begin() as p_conn:
                    res = p_conn.execute(
                        text("UPDATE b2b_sales_orders SET order_status = 'CANCELLED' WHERE b2b_order_no = :po"),
                        {"po": po_no},
                    )
                    cancelled = (res.rowcount or 0) > 0
            except Exception as e:
                return JSONResponse(
                    {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32000, "message": str(e)}},
                    status_code=500,
                )
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {"po_no": po_no, "cancelled": cancelled, "status": {"state": "canceled"}},
        })

    # 기본: tasks/send (RFQ 자율 협의 실행)
    task_id = params.get("id") or f"a2a-task-{uuid.uuid4().hex[:8]}"
    message = params.get("message") or {}
    parts = message.get("parts") or []
    rfq_payload: Dict[str, Any] = {}
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "data" and isinstance(p.get("data"), dict):
            rfq_payload.update(p["data"])

    print(f"\n🤝 [Partner A2A Task 수신] task_id={task_id}, domain={rfq_payload.get('outsourcing_domain')}")
    negotiation_result = _evaluate_partner_a2a_rfq(rfq_payload)

    task_record = {
        "id": task_id,
        "status": {
            "state": negotiation_result.get("a2a_task_status", "completed"),
            "timestamp": negotiation_result.get("timestamp"),
        },
        "artifacts": [
            {
                "name": "partner_b2b_negotiation_proposal",
                "mimeType": "application/json",
                "parts": [{"type": "data", "data": negotiation_result}],
            }
        ],
    }
    A2A_TASK_STORE[task_id] = task_record
    print(f"🤝 [Partner A2A Task 응답] task_id={task_id}, decision={negotiation_result.get('negotiation_decision')}")
    return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": task_record})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")