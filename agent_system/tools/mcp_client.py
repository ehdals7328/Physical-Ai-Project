"""agent_system/tools/mcp_client.py

FastMCP streamable-http 모드 연결 클라이언트 (언패킹 버그 수정 버전)
"""

import asyncio
import re
from typing import Any, Dict
from mcp import ClientSession

# streamable-http 클라이언트
try:
  from mcp.client.streamable_http import streamable_http_client
except ImportError:
  from mcp.client.sse import sse_client as streamable_http_client

MCP_SERVER_HTTP_URL = "http://127.0.0.1:8000/mcp"


async def _fetch_from_server(location: str) -> Dict[str, Any]:
  # 💡 반환값을 단일 streams 객체로 받은 뒤 내부 스트림 추출
  async with streamable_http_client(MCP_SERVER_HTTP_URL) as client_streams:
    # 튜플 형태(2개 또는 3개)이거나 단일 객체일 경우 모두 대응
    if isinstance(client_streams, (tuple, list)):
      read_stream, write_stream = client_streams[0], client_streams[1]
    else:
      read_stream, write_stream = (
          client_streams.read_stream,
          client_streams.write_stream,
      )

    async with ClientSession(read_stream, write_stream) as session:
      await session.initialize()

      result = await session.call_tool(
          "get_current_weather", arguments={"location": location}
      )

      raw_text = result.content[0].text if result.content else ""

      temp_match = re.search(r"기온:\s*([-+]?\d*\.?\d+)℃", raw_text)
      wind_match = re.search(r"풍속:\s*([-+]?\d*\.?\d+)m/s", raw_text)
      hum_match = re.search(r"습도:\s*([-+]?\d*\.?\d+)%", raw_text)

      temperature = float(temp_match.group(1)) if temp_match else 15.0
      wind_speed = float(wind_match.group(1)) if wind_match else 3.5
      humidity = float(hum_match.group(1)) if hum_match else 50.0

      return {
          "location": location,
          "temperature": temperature,
          "wind_speed": wind_speed,
          "humidity": humidity,
          "condition": "정상 수신",
          "raw_text": raw_text,
          "source": "Local_FastMCP_StreamableHTTP",
      }


def get_realtime_weather_from_mcp(location: str = "Ulsan") -> Dict[str, Any]:
  loc_en = "Ulsan"
  if "거제" in location:
    loc_en = "Geoje"
  elif "부산" in location:
    loc_en = "Busan"

  try:
    return asyncio.run(_fetch_from_server(loc_en))
  except ExceptionGroup as eg:
    for sub_exc in eg.exceptions:
      print(f"[세부 sub-exception 원인] {type(sub_exc).__name__}: {sub_exc}")
    return _fallback(location, "taskgroup_sub_exception")
  except Exception as e:
    print(f"[일반 통신 실패] {type(e).__name__}: {e}")
    return _fallback(location, str(e))


def _fallback(location: str, reason: str) -> Dict[str, Any]:
  return {
      "location": location,
      "temperature": 3.0,
      "wind_speed": 11.2,
      "humidity": 68.0,
      "condition": f"오프라인 기본값 ({reason})",
      "source": "fallback_offline",
  }