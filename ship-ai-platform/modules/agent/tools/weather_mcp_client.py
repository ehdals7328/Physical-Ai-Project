"""
modules/agent/tools/weather_mcp_client.py
FastMCP 기상 서버(streamable-http) 통신 및 표준 기상 수치 파서
"""

import asyncio
import json
import os
import re
from typing import Any, Dict
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

WEATHER_MCP_SERVER_URL = os.getenv("WEATHER_MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")


def _sanitize_location_name(raw_loc: str) -> str:
    """
    현장 수식어('야외 독', '조선소' 등)만 제거하고,
    도시명 변환 및 지오코딩은 FastMCP 서버(weather_server.py)의 동적 정제기에 완전히 위임
    (city_map 딕셔너리 하드코딩 배제)
    """
    if not raw_loc:
        return "Ulsan"
    
    cleaned = raw_loc.strip()
    for noise in ["야외 독", "야외독", "조선소", "야드", "공장", "현장", "도크", "독"]:
        cleaned = cleaned.replace(noise, "").strip()
    
    return cleaned if cleaned else raw_loc.strip()


def _extract_weather_metrics(raw_text: str) -> Dict[str, Any]:
    """
    FastMCP 응답 텍스트에서 JSON 객체 파싱 또는 다국어 정규식 추출
    """
    # 1. 원문이 JSON 문자열인 경우 우선 파싱
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return {
                "temperature": parsed.get("temperature") or parsed.get("temp"),
                "wind_speed": parsed.get("wind_speed") or parsed.get("wind"),
                "humidity": parsed.get("humidity", 50.0),
                "precipitation": parsed.get("precipitation") or parsed.get("rainfall", 0.0),
            }
    except Exception:
        pass

    # 2. 한/영 혼용 텍스트에서 정규식 수치 추출
    # 기온 매칭: '기온: 15.0', 'temp: 15.0', 'temperature: 15.0' 등
    temp_match = re.search(r"(?:기온|온도|temp(?:erature)?)\s*[:=]?\s*([-+]?\d*\.?\d+)", raw_text, re.IGNORECASE)
    # 풍속 매칭: '풍속: 2.5', 'wind: 2.5', 'wind_speed: 2.5' 등
    wind_match = re.search(r"(?:풍속|바람|wind(?:_speed)?)\s*[:=]?\s*([-+]?\d*\.?\d+)", raw_text, re.IGNORECASE)
    # 습도 매칭
    hum_match = re.search(r"(?:습도|humidity)\s*[:=]?\s*([-+]?\d*\.?\d+)", raw_text, re.IGNORECASE)
    # 강수량 매칭
    rain_match = re.search(r"(?:강수(?:량)?|rain(?:fall)?|precipitation)\s*[:=]?\s*([-+]?\d*\.?\d+)", raw_text, re.IGNORECASE)

    return {
        "temperature": float(temp_match.group(1)) if temp_match else None,
        "wind_speed": float(wind_match.group(1)) if wind_match else None,
        "humidity": float(hum_match.group(1)) if hum_match else 50.0,
        "precipitation": float(rain_match.group(1)) if rain_match else 0.0,
    }


async def _call_mcp_weather_async(location: str) -> Dict[str, Any]:
    target_loc = _sanitize_location_name(location)
    print(f"\n📤 [Client -> FastMCP 요청] 전달 인자: '{target_loc}' (원래 location: '{location}')")

    async with streamable_http_client(WEATHER_MCP_SERVER_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool(
                "get_current_weather",
                arguments={"location": target_loc},
            )

            raw_text = result.content[0].text if (result.content and len(result.content) > 0) else ""
            print(f"📥 [FastMCP -> Client 원문 응답]:\n   \"{raw_text}\"")

            metrics = _extract_weather_metrics(raw_text)
            print(f"📊 [파싱된 수치]: Temp={metrics['temperature']}, Wind={metrics['wind_speed']}")

            # 서버 응답 텍스트는 왔으나 기온/풍속 수치가 전혀 파싱되지 않는 경우
            if metrics["temperature"] is None or metrics["wind_speed"] is None:
                return {
                    "location": target_loc,
                    "status": "PARSE_ERROR",
                    "raw_text": raw_text,
                    "error": f"서버 원문에서 기상 수치를 파싱할 수 없습니다. 원문: {raw_text}",
                }

            return {
                "location": target_loc,
                "temperature": metrics["temperature"],
                "wind_speed": metrics["wind_speed"],
                "humidity": metrics["humidity"],
                "precipitation": metrics["precipitation"],
                "raw_text": raw_text,
                "status": "SUCCESS",
                "source": "FastMCP_Server_8000",
            }


def get_realtime_weather_from_mcp(location: str = "Ulsan") -> Dict[str, Any]:
    """동기 실행 진입점"""
    try:
        return asyncio.run(_call_mcp_weather_async(location))
    except Exception as e:
        return {
            "location": location,
            "status": "CONNECTION_ERROR",
            "error": f"FastMCP 서버({WEATHER_MCP_SERVER_URL}) 통신 장애: {str(e)}",
        }