"""
modules/agent/tools/gmail_mcp_client.py
해외 공개 Gmail MCP (@gongrzhe/server-gmail-autoauth-mcp) stdio 연동 클라이언트
- 관리자 HITL 최종 승인 시 협력사 담당자 이메일로 작업지시서 PDF 및 외주 발주 주문서(PO) PDF 자동 첨부 발송
"""

import asyncio
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(env_key: str, default_rel: str) -> str:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return str((PROJECT_ROOT / default_rel).resolve())
    p = Path(raw)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return str(p)


def _run_async(coro):
    """이미 이벤트 루프가 실행 중인 스레드에서도 안전하게 비동기 코루틴 실행"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box: Dict[str, Any] = {}
    err_box: Dict[str, BaseException] = {}

    def _runner():
        try:
            result_box["value"] = asyncio.run(coro)
        except BaseException as exc:
            err_box["error"] = exc

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=45)
    if "error" in err_box:
        raise err_box["error"]
    return result_box.get("value")



def _ensure_fresh_gmail_token(oauth_path: str, cred_path: str) -> None:
    """Node.js(node-fetch)의 Premature close 버그 우회를 위해 만료 5분 전 파이썬으로 토큰 사전 갱신"""
    import json, time, urllib.request, urllib.parse
    try:
        with open(cred_path, "r", encoding="utf-8") as f:
            creds = json.load(f)
        expiry_ms = int(creds.get("expiry_date") or 0)
        # 만료까지 5분(300초) 이상 남아있으면 갱신 생략
        if expiry_ms > (time.time() + 300) * 1000:
            return
        refresh_token = creds.get("refresh_token")
        if not refresh_token:
            return
        with open(oauth_path, "r", encoding="utf-8") as f:
            raw_keys = json.load(f)
        keys = raw_keys.get("installed") or raw_keys.get("web") or {}
        data = urllib.parse.urlencode({
            "client_id": keys["client_id"],
            "client_secret": keys["client_secret"],
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }).encode("utf-8")
        req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            new_tok = json.loads(resp.read().decode("utf-8"))
        if new_tok.get("access_token"):
            creds["access_token"] = new_tok["access_token"]
            creds["expiry_date"] = int((time.time() + int(new_tok.get("expires_in", 3600)) - 60) * 1000)
            with open(cred_path, "w", encoding="utf-8") as f:
                json.dump(creds, f, indent=2)
            print("🔄 [Gmail OAuth] Python 사전 토큰 갱신 완료")
    except Exception as e:
        print(f"⚠️ [Gmail OAuth 사전 갱신 참고]: {e}")


async def _call_gmail_mcp_send_email(
    to_list: List[str],
    subject: str,
    body: str,
    attachments: Optional[List[str]] = None,
) -> Dict[str, Any]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    npx_bin = (
        shutil.which("npx.cmd")
        or shutil.which("npx")
        or (
            r"C:\Program Files\nodejs\npx.cmd"
            if os.path.exists(r"C:\Program Files\nodejs\npx.cmd")
            else None
        )
    )
    if not npx_bin:
        return {
            "status": "SKIPPED",
            "reason": "Node.js(npx)가 설치되어 있지 않아 Gmail MCP 발송을 건너뛰었습니다.",
        }

    oauth_path = _resolve_path("GMAIL_OAUTH_PATH", "gcp-oauth.keys.json")
    cred_path = _resolve_path("GMAIL_CREDENTIALS_PATH", ".gmail-mcp-credentials.json")

    if not Path(oauth_path).exists():
        return {
            "status": "SKIPPED",
            "reason": f"OAuth 키 파일 미존재: {oauth_path}",
        }
    if not Path(cred_path).exists():
        return {
            "status": "SKIPPED",
            "reason": f"Gmail 인증 토큰 파일 미존재({cred_path}). 먼저 'npx -y @gongrzhe/server-gmail-autoauth-mcp auth'를 실행하세요.",
        }

    valid_attachments = [
        str(Path(p).resolve())
        for p in (attachments or [])
        if p and Path(p).exists()
    ]

    _ensure_fresh_gmail_token(oauth_path, cred_path)
    mcp_env = os.environ.copy()
    mcp_env["GMAIL_OAUTH_PATH"] = oauth_path
    mcp_env["GMAIL_CREDENTIALS_PATH"] = cred_path

    server_params = StdioServerParameters(
        command=npx_bin,
        args=["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
        env=mcp_env,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_resp = await session.list_tools()
            available_tools = {t.name: t for t in tools_resp.tools}
            if "send_email" not in available_tools:
                return {
                    "status": "ERROR",
                    "reason": f"Gmail MCP 서버에서 send_email 도구를 찾지 못했습니다. (가용 도구: {list(available_tools.keys())})",
                }

            args_payload: Dict[str, Any] = {
                "to": to_list,
                "subject": subject,
                "body": body,
            }
            if valid_attachments:
                args_payload["attachments"] = valid_attachments

            call_res = await session.call_tool("send_email", arguments=args_payload)
            texts = [
                getattr(c, "text", str(c))
                for c in (call_res.content or [])
            ]
            is_err = bool(getattr(call_res, "isError", False))
            return {
                "status": "ERROR" if is_err else "SUCCESS",
                "to": to_list,
                "attachments": [Path(a).name for a in valid_attachments],
                "response": "\n".join(texts).strip(),
            }


def send_work_order_dispatch_email(
    work_order_number: str,
    po_number: Optional[str],
    partner_name: str,
    block_id: str,
    branch_name: str,
    work_zone: str,
    process_code: str,
    material_grade: str,
    thickness_mm: Any,
    worker_name: str,
    approver_label: str,
    pdf_paths: List[str],
    recipient_email: Optional[str] = None,
) -> Dict[str, Any]:
    """
    관리자 최종 승인 완료 후 협력사 담당자 이메일로 공식 작업지시서 및 발주 주문서(PO) PDF를 첨부 발송합니다.
    """
    target_email = (recipient_email or os.getenv("PARTNER_NOTIFY_EMAIL", "")).strip()
    if not target_email:
        return {
            "status": "SKIPPED",
            "reason": ".env에 PARTNER_NOTIFY_EMAIL이 설정되어 있지 않아 메일 발송을 생략했습니다.",
        }

    po_line = f"- 외주 발주 번호 (PO): {po_number}\n" if po_number else ""
    subject = (
        f"[K-스마트조선소] 공식 작업지시서 및 외주 발주 확정 안내 "
        f"({work_order_number} / 블록 {block_id})"
    )
    body = (
        f"수신: {partner_name} 담당자 귀하\n"
        f"발신: K-스마트조선소 통합생산관리본부 (승인자: {approver_label})\n\n"
        f"귀사와 연계된 선박 블록 용접 작업이 관리자 최종 승인(APPROVED) 및 ERP/B2B 원장 커밋 완료되어 "
        f"공식 작업지시서 및 외주 발주 주문서(PO) PDF를 송부드립니다.\n\n"
        f"■ 작업 및 발주 확정 요약\n"
        f"- 작업지시 번호 (WO): {work_order_number}\n"
        f"{po_line}"
        f"- 수주 협력사: {partner_name}\n"
        f"- 대상 블록 / 규격: {block_id} ({material_grade} / {thickness_mm}mm / {process_code})\n"
        f"- 납품·투입 야드 / 구역: {branch_name} ({work_zone})\n"
        f"- 배정 담당 작업자: {worker_name}\n\n"
        f"첨부된 PDF 문서를 확인하시어 현장 안전 수칙 및 WPS 시공 조건에 따라 작업 착수해 주시기 바랍니다.\n"
        f"감사합니다."
    )

    try:
        res = _run_async(
            _call_gmail_mcp_send_email(
                to_list=[target_email],
                subject=subject,
                body=body,
                attachments=pdf_paths,
            )
        )
        print(f"📧 [Gmail MCP 발송 결과]: {res}")
        return res or {"status": "ERROR", "reason": "빈 응답 반환"}
    except Exception as exc:
        print(f"⚠️ [Gmail MCP 발송 예외]: {type(exc).__name__} - {exc}")
        return {
            "status": "ERROR",
            "reason": f"{type(exc).__name__}: {str(exc)}",
        }