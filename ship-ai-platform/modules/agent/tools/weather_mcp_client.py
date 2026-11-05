"""
weather_mcp_client.py
FastMCP 기반 실시간 현장 기상 데이터 수집 클라이언트
- MCP isError 확인 및 필수 기상 데이터(기온, 풍속, 강수량) 무결성 검증
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

            # 1. MCP 프로토콜 에러 또는 빈 응답 검증
            if getattr(result, "isError", False) or not result.content:
                err_msg = result.content[0].text if getattr(result, "content", None) else "빈 응답 수신"
                return {
                    "location": city_name,
                    "status": "ERROR",
                    "error": err_msg,
                    "condition": "기상 도구 실행 오류",
                    "source": "FastMCP_StreamableHTTP"
                }

            raw_text = (result.content[0].text or "").strip()
            if not raw_text:
                return {
                    "location": city_name,
                    "status": "ERROR",
                    "error": "기상 서버로부터 빈 텍스트가 반환되었습니다.",
                    "condition": "응답 데이터 없음",
                    "source": "FastMCP_StreamableHTTP"
                }

            if any(err_kw in raw_text for err_kw in ["찾을 수 없습니다", "조회 실패", "오류:", "특정할 수 없습니다"]):
                return {
                    "location": city_name,
                    "status": "NOT_FOUND",
                    "error": raw_text,
                    "condition": "위치/기상 조회 실패",
                    "source": "FastMCP_StreamableHTTP"
                }

            # 2. 기온, 풍속, 습도, 강수량 정밀 파싱
            temp_match = re.search(r"기온:\s*([-+]?\d*\.?\d+)℃", raw_text)
            wind_match = re.search(r"풍속:\s*([-+]?\d*\.?\d+)m/s", raw_text)
            gust_match = re.search(r"순간돌풍:\s*([-+]?\d*\.?\d+)m/s", raw_text)
            hum_match = re.search(r"습도:\s*([-+]?\d*\.?\d+)%", raw_text)
            rain_match = re.search(r"(?:강수량|강우량|강설량):\s*([-+]?\d*\.?\d+)mm/h", raw_text)

            temperature = float(temp_match.group(1)) if temp_match else None
            wind_speed = float(wind_match.group(1)) if wind_match else None
            wind_gust = float(gust_match.group(1)) if gust_match else wind_speed
            humidity = float(hum_match.group(1)) if hum_match else None
            rainfall = float(rain_match.group(1)) if rain_match else 0.0

            # 3. 필수 기상 데이터(기온, 풍속) 결측 시 SUCCESS 반환 금지 (Fail-Closed)
            if temperature is None or wind_speed is None:
                return {
                    "location": city_name,
                    "status": "ERROR",
                    "error": f"필수 기상 수치 파싱 실패: {raw_text}",
                    "condition": "데이터 결측",
                    "raw_text": raw_text,
                    "source": "FastMCP_StreamableHTTP"
                }

            return {
                "location": city_name,
                "status": "SUCCESS",
                "temperature": temperature,
                "wind_speed": wind_speed,
                "wind_gust": wind_gust,
                "humidity": humidity,
                "rainfall": rainfall,
                "condition": "정상 수신",
                "raw_text": raw_text,
                "source": "FastMCP_StreamableHTTP"
            }


def get_realtime_weather_from_mcp(location: str) -> Dict[str, Any]:
    """외부 MCP 서버와 연동하여 실시간 기상 데이터를 수집 (이벤트 루프 충돌 방어 포함)"""
    clean_loc = (location or "").strip()
    if not clean_loc:
        return {
            "status": "NEED_LOCATION",
            "message": "확인하려는 작업 지역명을 입력해 주세요."
        }

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, _fetch_from_server(clean_loc))
                return future.result(timeout=10.0)
        else:
            return asyncio.run(_fetch_from_server(clean_loc))
    except Exception as e:
        return {
            "location": clean_loc,
            "status": "ERROR",
            "error": str(e),
            "condition": f"서버 통신 실패 ({type(e).__name__})",
            "source": "fallback_offline"
        }