"""
weather_mcp_client.py
FastMCP weather_server 전용 비동기 클라이언트 (환경변수 분리 및 지명 매핑 최적화)
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

# .env에서 URL 로드 (기본값 로컬 8000번)
MCP_SERVER_HTTP_URL = os.getenv("WEATHER_MCP_URL", "http://127.0.0.1:8000/mcp")

# 빈번한 조선소 지명에 대한 초고속 딕셔너리 매핑 (LLM 1~2초 지연 제거)
KNOWN_SHIPYARD_LOCATIONS = {
    "울산": "Ulsan", "울산독": "Ulsan", "방어진": "Ulsan", "미포": "Ulsan",
    "거제": "Geoje", "거제독": "Geoje", "옥포": "Geoje", "통영": "Tongyeong",
    "부산": "Busan", "영도": "Busan", "신선대": "Busan",
    "포항": "Pohang", "목포": "Mokpo", "삼호": "Mokpo", "인천": "Incheon"
}

def resolve_location_fast(location_query: str) -> str:
    """지명 매핑 테이블을 우선 검사하여 속도를 극대화"""
    if not location_query:
        return "Ulsan"
    
    query = location_query.strip()
    for kor, eng in KNOWN_SHIPYARD_LOCATIONS.items():
        if kor in query:
            return eng
            
    # 영문 도시명이 직접 들어온 경우
    eng_match = re.search(r"[a-zA-Z]+", query)
    if eng_match:
        return eng_match.group(0).capitalize()

    return "Ulsan"

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

            temp_match = re.search(r"기온:\s*([-+]?\d*\.?\d+)℃", raw_text)
            wind_match = re.search(r"풍속:\s*([-+]?\d*\.?\d+)m/s", raw_text)
            hum_match = re.search(r"습도:\s*([-+]?\d*\.?\d+)%", raw_text)

            temperature = float(temp_match.group(1)) if temp_match else 15.0
            wind_speed = float(wind_match.group(1)) if wind_match else 3.5
            humidity = float(hum_match.group(1)) if hum_match else 50.0

            return {
                "location": city_name,
                "temperature": temperature,
                "wind_speed": wind_speed,
                "humidity": humidity,
                "condition": "정상 수신",
                "raw_text": raw_text,
                "source": "FastMCP_StreamableHTTP"
            }

def get_realtime_weather_from_mcp(location: str = "울산") -> Dict[str, Any]:
    """Team 2(search.py) 서브그래프에서 호출하는 Fail-safe 기상 수집 함수"""
    target_city = resolve_location_fast(location)
    try:
        return asyncio.run(_fetch_from_server(target_city))
    except Exception as e:
        return {
            "location": location,
            "target_city": target_city,
            "temperature": 18.0,
            "wind_speed": 2.5,
            "humidity": 60.0,
            "condition": f"오프라인 기본값 ({type(e).__name__})",
            "source": "fallback_offline"
        }