"""
weather_mcp_client.py
FastMCP 기반 실시간 현장 기상 데이터 수집 클라이언트
- Team 1의 Function Calling이 정제한 도시명을 받아 FastMCP 서버(streamable-http) 도구 호출
"""
import os
import asyncio
import re
from typing import Dict, Any
from mcp import ClientSession

try:
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    from mcp.client.sse import sse_client as streamable_http_client

MCP_SERVER_HTTP_URL = os.getenv("WEATHER_MCP_URL", "http://127.0.0.1:8000/mcp")


async def _fetch_from_server(city_name: str) -> Dict[str, Any]:
    async with streamable_http_client(MCP_SERVER_HTTP_URL) as client_streams:
        if isinstance(client_streams, (tuple, list)):
            read_stream, write_stream = client_streams[0], client_streams[1]
        else:
            read_stream, write_stream = client_streams.read_stream, client_streams.write_stream

        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool(
                "get_current_weather",
                arguments={"location": city_name}
            )

            raw_text = result.content[0].text if result.content else ""

            if "찾을 수 없습니다" in raw_text or "실패" in raw_text:
                return {
                    "location": city_name,
                    "status": "NOT_FOUND",
                    "error": raw_text,
                    "condition": "위치 조회 실패",
                    "source": "FastMCP_StreamableHTTP"
                }

            temp_match = re.search(r"기온:\s*([-+]?\d*\.?\d+)℃", raw_text)
            wind_match = re.search(r"풍속:\s*([-+]?\d*\.?\d+)m/s", raw_text)
            hum_match = re.search(r"습도:\s*([-+]?\d*\.?\d+)%", raw_text)

            temperature = float(temp_match.group(1)) if temp_match else None
            wind_speed = float(wind_match.group(1)) if wind_match else None
            humidity = float(hum_match.group(1)) if hum_match else None

            return {
                "location": city_name,
                "status": "SUCCESS",
                "temperature": temperature,
                "wind_speed": wind_speed,
                "humidity": humidity,
                "condition": "정상 수신",
                "raw_text": raw_text,
                "source": "FastMCP_StreamableHTTP"
            }


def get_realtime_weather_from_mcp(location: str) -> Dict[str, Any]:
    """외부 MCP 서버와 연동하여 실시간 기상 데이터를 수집"""
    clean_loc = (location or "").strip()
    if not clean_loc:
        return {
            "status": "NEED_LOCATION",
            "message": "확인하려는 작업 지역명을 입력해 주세요."
        }

    try:
        return asyncio.run(_fetch_from_server(clean_loc))
    except Exception as e:
        return {
            "location": clean_loc,
            "status": "ERROR",
            "error": str(e),
            "condition": f"서버 통신 실패 ({type(e).__name__})",
            "source": "fallback_offline"
        }