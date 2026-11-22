"""
modules/agent/tools/partner_mcp_client.py
FastMCP 외부 협력사 B2B 게이트웨이 서버(streamable-http) 및 A2A JSON-RPC 통신 클라이언트
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

PARTNER_MCP_SERVER_URL = os.getenv("PARTNER_MCP_SERVER_URL", "http://127.0.0.1:8001/mcp")
PARTNER_A2A_BASE_URL = os.getenv(
    "PARTNER_A2A_BASE_URL",
    PARTNER_MCP_SERVER_URL.replace("/mcp", "").rstrip("/"),
)
PARTNER_A2A_API_KEY = os.getenv("PARTNER_A2A_API_KEY", "")


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
            print(
                f"📥 [Partner FastMCP -> Client 응답]: status={parsed.get('status')}, "
                f"선정협력사={selected.get('partner_name', 'N/A')}"
            )
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


# ==============================================================================
# A2A (Agent-to-Agent) Discovery · Negotiation · Compensating Cancel
# ==============================================================================
def _a2a_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if PARTNER_A2A_API_KEY:
        headers["Authorization"] = f"Bearer {PARTNER_A2A_API_KEY}"
    return headers


def discover_partner_agent_card(timeout_sec: float = 4.0) -> Dict[str, Any]:
    """협력사 A2A Agent Card 자동 탐색 (/.well-known/agent-card.json -> /.well-known/agent.json 폴백)"""
    for path in ("/.well-known/agent-card.json", "/.well-known/agent.json"):
        url = f"{PARTNER_A2A_BASE_URL}{path}"
        try:
            with httpx.Client(timeout=timeout_sec) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    card = resp.json()
                    print(f"🪪 [A2A Discovery 성공] {card.get('name')} ({url})")
                    return {"status": "SUCCESS", "card_url": url, "agent_card": card}
        except Exception:
            continue
    return {"status": "UNAVAILABLE", "agent_card": None}


def negotiate_partner_via_a2a(
    outsourcing_domain: str,
    target_branch: str = "BR-ULSAN-01",
    requested_qty: Optional[float] = None,
    weight_ton: Optional[float] = None,
    material: str = "",
    process: str = "",
    worker_name: str = "",
    requested_item_name: str = "",
    urgency: str = "NORMAL",
    timeout_sec: float = 8.0,
) -> Dict[str, Any]:
    """협력사 A2A 에이전트(/a2a)에 RFQ Task를 전송하고 제약 기반 역제안(Counter-Offer)을 수신합니다."""
    discovery = discover_partner_agent_card()
    card = discovery.get("agent_card") or {}
    a2a_endpoint = card.get("url") or f"{PARTNER_A2A_BASE_URL}/a2a"

    task_id = f"rfq-{uuid.uuid4().hex[:8]}"
    payload = {
        "jsonrpc": "2.0",
        "id": task_id,
        "method": "tasks/send",
        "params": {
            "id": task_id,
            "message": {
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": f"{outsourcing_domain} B2B 공급·납기·조건 협의 요청 (긴급도: {urgency})",
                    },
                    {
                        "type": "data",
                        "data": {
                            "outsourcing_domain": outsourcing_domain,
                            "target_branch": target_branch,
                            "requested_qty": requested_qty,
                            "weight_ton": weight_ton,
                            "material": material,
                            "process": process,
                            "worker_name": worker_name,
                            "requested_item_name": requested_item_name,
                            "urgency": urgency,
                        },
                    },
                ],
            },
        },
    }

    try:
        with httpx.Client(timeout=timeout_sec) as client:
            resp = client.post(a2a_endpoint, json=payload, headers=_a2a_headers())
            resp.raise_for_status()
            rpc_res = resp.json()

        result = rpc_res.get("result") or {}
        artifacts = result.get("artifacts") or []
        proposal = {}
        if artifacts:
            for part in (artifacts[0].get("parts") or []):
                if part.get("type") == "data":
                    proposal = part.get("data") or {}
                    break

        return {
            "status": "SUCCESS",
            "a2a_task_id": result.get("id", task_id),
            "a2a_task_state": (result.get("status") or {}).get("state", "completed"),
            "agent_card_name": card.get("name", "Partner B2B Agent"),
            "agent_card_url": discovery.get("card_url"),
            "proposal": proposal,
        }
    except Exception as e:
        print(f"⚠️ [A2A 협의 폴백]: {e}")
        return {
            "status": "UNAVAILABLE",
            "a2a_task_id": task_id,
            "a2a_task_state": "failed",
            "error_message": str(e),
            "proposal": {},
        }


def cancel_partner_b2b_order_via_a2a(po_no: str, timeout_sec: float = 5.0) -> Dict[str, Any]:
    """본사 후속 커밋 장애 시 협력사 측 수주 취소 보상(tasks/cancel) 요청"""
    payload = {
        "jsonrpc": "2.0",
        "id": f"cancel-{uuid.uuid4().hex[:6]}",
        "method": "tasks/cancel",
        "params": {"po_no": po_no},
    }
    try:
        with httpx.Client(timeout=timeout_sec) as client:
            resp = client.post(
                f"{PARTNER_A2A_BASE_URL}/a2a",
                json=payload,
                headers=_a2a_headers(),
            )
            resp.raise_for_status()
            return resp.json().get("result") or {"cancelled": False}
    except Exception as e:
        return {"cancelled": False, "error": str(e)}