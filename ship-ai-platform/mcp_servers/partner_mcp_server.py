"""
partner_mcp_server.py
독립 FastMCP 외부 협력사 B2B 게이트웨이 서버 (port=8001, streamable-http)
- 외부 협력사 PostgreSQL(PARTNER_PG_DATABASE_URL) 원장 조회 및 수주(PO) 발주 등록 전담
- 본사 Oracle DB와 완전히 분리된 사외 협력사 망(Network) 역할 수행
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine, text

# 불필요한 스트림 종료 로그 억제 (weather_server.py와 동일)
logging.getLogger("mcp.server.streamable_http").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PLATFORM_ROOT / ".env")

PG_URI = os.getenv("PARTNER_PG_DATABASE_URL")

try:
    pg_engine = create_engine(PG_URI, pool_pre_ping=True) if PG_URI else None
except Exception as e:
    print(f"🚨 [Partner MCP PostgreSQL 엔진 초기화 실패]: {e}")
    pg_engine = None

mcp = FastMCP("partner-b2b-server", host="0.0.0.0", port=8001)


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
    print(f"\n📡 [Partner FastMCP 조회 수신] domain='{domain}', material='{material}', process='{process}', worker='{worker_name}'")

    if not pg_engine:
        return json.dumps({
            "status": "ERROR",
            "error_message": "협력사 서버에 PARTNER_PG_DATABASE_URL이 설정되지 않았습니다.",
            "b2b_inquiry_timestamp": timestamp_iso,
        }, ensure_ascii=False, default=str)

    try:
        with pg_engine.connect() as conn:
            # 1. 용접 자재 / 철판 협력사 (대한특수철강)
            if domain == "WELDING_MATERIAL":
                rows = conn.execute(text("SELECT * FROM v_b2b_daehan_materials")).mappings().all()
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
                rows = conn.execute(text("SELECT * FROM v_b2b_donghae_workers")).mappings().all()
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
                            if w.get("can_accept_order", True) and str(w.get("weld_process", "")).upper() == req_proc:
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
                    rows = conn.execute(text("SELECT * FROM v_b2b_hankook_transporters")).mappings().all()
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
        return json.dumps({
            "status": "ERROR",
            "error_message": str(e),
            "b2b_inquiry_timestamp": timestamp_iso,
        }, ensure_ascii=False, default=str)


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
        return json.dumps({
            "status": "ERROR",
            "error_message": "PARTNER_PG_DATABASE_URL 미설정",
        }, ensure_ascii=False)

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
        return json.dumps({
            "status": "CONFIRMED",
            "po_no": po_no,
            "b2b_ack": b2b_ack,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "ERROR",
            "po_no": po_no,
            "b2b_ack": b2b_ack,
            "error_message": str(e),
        }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")