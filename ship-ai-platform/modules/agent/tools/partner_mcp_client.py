"""
modules/agent/tools/partner_mcp_client.py
FastMCP 외부 협력사 B2B 게이트웨이 서버(streamable-http) 통신 클라이언트
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

PARTNER_MCP_SERVER_URL = os.getenv("PARTNER_MCP_SERVER_URL", "http://127.0.0.1:8001/mcp")


async def _call_partner_b2b_catalog_async(
    outsourcing_domain: Optional[str] = None,
    material: Optional[str] = None,
    process: Optional[str] = None,
    partner_id: Optional[str] = None,
    worker_name: Optional[str] = None,
) -> Dict[str, Any]:
    domain = (outsourcing_domain or "WELDING_MATERIAL").upper()
    print(f"\n📤 [Client -> Partner FastMCP 조회 요청] URL={PARTNER_MCP_SERVER_URL}, domain='{domain}'")

    async with streamable_http_client(PARTNER_MCP_SERVER_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool(
                "query_partner_b2b_catalog",
                arguments={
                    "outsourcing_domain": domain,
                    "material": material or "",
                    "process": process or "",
                    "partner_id": partner_id or "",
                    "worker_name": worker_name or "",
                },
            )

            raw_text = result.content[0].text if (result.content and len(result.content) > 0) else ""
            parsed = json.loads(raw_text)
            selected = parsed.get("selected_partner") or {}
            print(f"📥 [Partner FastMCP -> Client 응답]: status={parsed.get('status')}, 선정협력사={selected.get('partner_name', 'N/A')}")
            return parsed


def query_partner_b2b_via_mcp(
    outsourcing_domain: Optional[str] = None,
    material: Optional[str] = None,
    process: Optional[str] = None,
    partner_id: Optional[str] = None,
    worker_name: Optional[str] = None,
) -> Dict[str, Any]:
    """동기 LangGraph 노드/도구에서 협력사 FastMCP 카탈로그를 호출하는 진입점"""
    try:
        return asyncio.run(
            _call_partner_b2b_catalog_async(
                outsourcing_domain=outsourcing_domain,
                material=material,
                process=process,
                partner_id=partner_id,
                worker_name=worker_name,
            )
        )
    except Exception as e:
        return {
            "status": "CONNECTION_ERROR",
            "error_message": f"Partner FastMCP 서버({PARTNER_MCP_SERVER_URL}) 통신 장애: {str(e)}",
            "b2b_inquiry_timestamp": datetime.now().isoformat(),
        }


async def _call_partner_b2b_order_async(
    po_no: str,
    wo_no: str,
    partner_id: str,
    ordered_ref_code: str,
    order_qty: float,
    allocated_lot_or_id: str,
    delivery_yard_zone: str,
) -> Dict[str, Any]:
    print(f"\n📤 [Client -> Partner FastMCP 발주 전송] PO={po_no}, Partner={partner_id}")

    async with streamable_http_client(PARTNER_MCP_SERVER_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool(
                "create_partner_b2b_order",
                arguments={
                    "po_no": po_no,
                    "wo_no": wo_no,
                    "partner_id": partner_id,
                    "ordered_ref_code": ordered_ref_code,
                    "order_qty": float(order_qty),
                    "allocated_lot_or_id": allocated_lot_or_id,
                    "delivery_yard_zone": delivery_yard_zone,
                },
            )

            raw_text = result.content[0].text if (result.content and len(result.content) > 0) else ""
            parsed = json.loads(raw_text)
            print(f"📥 [Partner FastMCP 수주 응답]: {parsed}")
            return parsed


def dispatch_partner_b2b_order_via_mcp(
    po_no: str,
    wo_no: str,
    partner_id: str,
    ordered_ref_code: str,
    order_qty: float = 15.0,
    allocated_lot_or_id: str = "",
    delivery_yard_zone: str = "ZONE-A-BAY1",
) -> Dict[str, Any]:
    """동기 커밋 도구에서 협력사 FastMCP로 수주(PO) 발주를 전송하는 진입점"""
    try:
        return asyncio.run(
            _call_partner_b2b_order_async(
                po_no=po_no,
                wo_no=wo_no,
                partner_id=partner_id,
                ordered_ref_code=ordered_ref_code,
                order_qty=order_qty,
                allocated_lot_or_id=allocated_lot_or_id,
                delivery_yard_zone=delivery_yard_zone,
            )
        )
    except Exception as e:
        return {
            "status": "CONNECTION_ERROR",
            "po_no": po_no,
            "b2b_ack": f"ACK-{wo_no}-{partner_id}",
            "error_message": f"Partner FastMCP 서버({PARTNER_MCP_SERVER_URL}) 발주 전송 실패: {str(e)}",
        }