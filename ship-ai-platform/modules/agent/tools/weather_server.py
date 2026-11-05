"""
독립 FastMCP 기상 서버 (weather_server.py)
- 하드코딩 지명 리스트 배제 및 OpenWeatherMap 실측 데이터 무결성 검증
"""
import os
import logging
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# 불필요한 스트림 종료 로그 억제
logging.getLogger("mcp.server.streamable_http").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)

load_dotenv()

weatherapi_key = os.getenv("WEATHER_API_KEY") or os.getenv("OPENWEATHER_API_KEY")
openai_client = OpenAI()

mcp = FastMCP("weather-server", host="0.0.0.0", port=8000)


def extract_clean_city(query: str) -> str:
    """사용자가 입력한 지명/문장에서 OpenWeatherMap 조회가 가능한 영문 대표 도시명 1단어 추출"""
    clean_query = (query or "").strip()
    if not clean_query:
        return ""

    # 이미 순수 영문 단일 단어(알파벳)로 넘어온 경우 그대로 사용
    if clean_query.isascii() and clean_query.isalpha():
        return clean_query

    prompt = (
        "다음 입력에서 기상 조회가 가능한 대표 도시의 영문명 딱 1단어만 출력하세요.\n"
        "특정 도시를 식별할 수 없으면 'NONE'이라고만 출력하세요.\n"
        "설명 없이 영문 도시명 1단어만 반환할 것."
    )
    try:
        res = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": clean_query}
            ],
            temperature=0,
            max_tokens=10
        )
        city = res.choices[0].message.content.strip().split()[0].replace(".", "").replace("'", "")
        return "" if city.upper() == "NONE" else city
    except Exception:
        return ""


def get_geo(city_query: str):
    """지오코딩 조회 (URL 파라미터 안전 바인딩)"""
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city_query, "limit": 1, "appid": weatherapi_key}
    try:
        res = requests.get(url, params=params, timeout=5.0)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]
    except Exception:
        pass
    return None


@mcp.tool()
def get_current_weather(location: str) -> str:
    """[MCP Tool] 실시간 기상 조회 (결측치 0.0 대체 금지 원칙 적용)"""
    if not location or not location.strip():
        return "오류: 조회할 도시/지역명이 입력되지 않았습니다."

    if not weatherapi_key:
        return "오류: 기상 API 인증키(WEATHER_API_KEY)가 설정되지 않았습니다."

    target_city = extract_clean_city(location)
    if not target_city:
        return f"조회 실패: '{location}'에서 검색 가능한 도시명을 특정할 수 없습니다."

    geo = get_geo(f"{target_city},KR") or get_geo(target_city)
    if not geo or "lat" not in geo or "lon" not in geo:
        return f"조회 실패: '{location}' ({target_city}) 위치의 좌표를 찾을 수 없습니다."

    weather_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": geo["lat"],
        "lon": geo["lon"],
        "appid": weatherapi_key,
        "units": "metric",
        "lang": "kr"
    }

    try:
        res = requests.get(weather_url, params=params, timeout=5.0)
        if res.status_code != 200:
            return f"조회 실패: {location} 기상 API 응답 오류 (HTTP {res.status_code})"

        data = res.json()
        main_data = data.get("main") or {}
        wind_data = data.get("wind") or {}

        # 필수 계측값(기온, 풍속) 결측 확인 (임의로 0.0으로 대체하지 않음)
        temp = main_data.get("temp")
        wind_speed = wind_data.get("speed")
        if temp is None or wind_speed is None:
            return f"조회 실패: {location}의 핵심 기상 센서 데이터(기온/풍속)가 결측되었습니다."

        feels_like = main_data.get("feels_like", temp)
        humidity = main_data.get("humidity")
        wind_gust = wind_data.get("gust", wind_speed)
        weather_desc = (data.get("weather") or [{}])[0].get("description", "정보 없음")

        rain_1h = (data.get("rain") or {}).get("1h", 0.0)
        snow_1h = (data.get("snow") or {}).get("1h", 0.0)
        total_precip = float(rain_1h or 0.0) + float(snow_1h or 0.0)

        hum_str = f"{humidity}%" if humidity is not None else "결측"
        return (
            f"[{location} 실시간 기상] 상태: {weather_desc}, 기온: {temp}℃(체감: {feels_like}℃), "
            f"풍속: {wind_speed}m/s(순간돌풍: {wind_gust}m/s), 습도: {hum_str}, 강수량: {total_precip}mm/h"
        )

    except Exception as e:
        return f"조회 실패: {location} 날씨 조회 중 통신 예외 발생 ({e})"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")