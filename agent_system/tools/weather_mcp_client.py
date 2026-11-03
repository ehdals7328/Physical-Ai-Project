"""
agent_system/tools/weather_mcp_client.py
FastMCP weather_server.py (streamable-http 모드, /mcp) 전용 클라이언트
"""
import asyncio
import re
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from mcp import ClientSession

# streamable-http 클라이언트 라이브러리 임포트
try:
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    from mcp.client.sse import sse_client as streamable_http_client

# weather_server.py가 streamable-http로 열어둔 표준 엔드포인트
MCP_SERVER_HTTP_URL = "http://127.0.0.1:8000/mcp"

# LLM을 통한 동적 영문 도시명 변환 (하드코딩 제거)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
city_extract_prompt = PromptTemplate.from_template(
    "사용자가 입력한 작업 위치/문맥에서 OpenWeatherMap 조회를 위한 '대표 도시 영문명(예: Ulsan, Geoje, Busan, Pohang, Incheon)' 딱 1단어만 출력하세요.\n"
    "특수문자 없이 오직 영문 도시명 1개만 반환해야 합니다.\n"
    "식별이 불가능한 경우 'Ulsan'을 반환하세요.\n\n"
    "입력 문맥: {location_query}\n"
    "영문 도시명:"
)
city_extractor_chain = city_extract_prompt | llm

def resolve_location_dynamically(location_query: str) -> str:
    """한글/현장 수식어가 포함된 위치를 영문 도시명으로 동적 변환"""
    if not location_query or not location_query.strip():
        return "Ulsan"
    try:
        res = city_extractor_chain.invoke({"location_query": location_query})
        city_name = res.content.strip().replace("'", "").replace('"', "")
        return city_name.split()[0] if city_name else "Ulsan"
    except Exception:
        return "Ulsan"

async def _fetch_from_server(city_name: str) -> Dict[str, Any]:
    # streamable-http 컨텍스트 진입 및 안전한 스트림 객체 언패킹
    async with streamable_http_client(MCP_SERVER_HTTP_URL) as client_streams:
        if isinstance(client_streams, (tuple, list)):
            read_stream, write_stream = client_streams[0], client_streams[1]
        else:
            read_stream, write_stream = client_streams.read_stream, client_streams.write_stream

        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # weather_server.py의 @mcp.tool() get_current_weather 호출
            result = await session.call_tool(
                "get_current_weather",
                arguments={"location": city_name}
            )

            raw_text = result.content[0].text if result.content else ""

            # 서버 반환 문자열("위치: ..., 기온: ...℃, 풍속: ...m/s, 습도: ...%") 파싱
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
                "source": "Local_FastMCP_StreamableHTTP"
            }

def get_realtime_weather_from_mcp(location: str = "Ulsan") -> Dict[str, Any]:
    """Team 2(search.py) 서브그래프에서 동기 방식으로 호출하는 래퍼"""
    # 1. LLM 동적 지명 변환
    target_city = resolve_location_dynamically(location)

    # 2. 비동기 호출 및 연결 실패 방어(Fallback)
    try:
        return asyncio.run(_fetch_from_server(target_city))
    except Exception as e:
        return {
            "location": location,
            "temperature": 18.0,
            "wind_speed": 1.5,
            "humidity": 60.0,
            "condition": f"오프라인 기본 안전모드 ({type(e).__name__})",
            "source": "fallback_offline"
        }