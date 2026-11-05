"""
agent_system/tools/weather_server.py
독립 FastMCP 기상 서버
"""
import os
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

load_dotenv()

weatherapi_key = os.getenv("WEATHER_API_KEY") or os.getenv("OPENWEATHER_API_KEY")
openai_client = OpenAI()

mcp = FastMCP("weather-server", host="0.0.0.0", port=8000)


def extract_clean_city(query: str) -> str:
    """사용자가 입력한 복합 지명에서 영문 대표 도시명 1단어만 추출"""
    clean_query = (query or "").strip()
    if not clean_query:
        return ""

    # 기본 영문 도시명 처리
    low = clean_query.lower()
    if low in ["seoul", "busan", "daegu", "ulsan", "geoje", "pohang", "incheon", "changwon"]:
        return clean_query

    prompt = (
        "다음 문장이나 지명에서 OpenWeatherMap API 조회가 가능한 대한민국 대표 도시의 영문명 딱 1단어만 출력하세요.\n"
        "특정 도시를 식별할 수 없거나 날씨와 무관하면 'NONE'이라고만 출력하세요.\n"
        "예: 대구 -> Daegu, 울산 야외 독 -> Ulsan, 거제 조선소 -> Geoje\n"
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


def get_geo(city: str):
    """지오코딩 조회 및 IndexError: 0 방지 방어 코드"""
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={weatherapi_key}"
    try:
        res = requests.get(url, timeout=5.0)
        data = res.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0]
    except Exception:
        pass
    return None


@mcp.tool()
def get_current_weather(location: str) -> str:
    """[MCP Tool] 실시간 기상 조회"""
    if not location or not location.strip():
        return "조회할 도시/지역명이 입력되지 않았습니다."

    target_city = extract_clean_city(location)
    if not target_city:
        return f"'{location}'에서 검색 가능한 영문 도시명을 특정할 수 없습니다."

    # 1. 대한민국 국가코드(,KR) 우선 부여 후 좌표 검색
    geo = get_geo(f"{target_city},KR")
    if not geo:
        geo = get_geo(target_city)

    if not geo:
        return f"{location} ({target_city}) 위치의 좌표를 찾을 수 없습니다."

    # 한국어 날씨 설명(lang=kr) 및 섭씨 단위(units=metric) 적용
    weather_url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"lat={geo['lat']}&lon={geo['lon']}&appid={weatherapi_key}&units=metric&lang=kr"
    )
    
    try:
        res = requests.get(weather_url, timeout=5.0)
        if res.status_code != 200:
            return f"{location}의 날씨 정보를 가져오지 못했습니다. (HTTP {res.status_code})"

        data = res.json()
        
        # 1) 기본 온습도 및 체감온도
        temp = data.get("main", {}).get("temp", 0.0)
        feels_like = data.get("main", {}).get("feels_like", temp)
        humidity = data.get("main", {}).get("humidity", 0)
        
        # 2) 풍속 및 순간 돌풍(gust)
        wind_speed = data.get("wind", {}).get("speed", 0.0)
        wind_gust = data.get("wind", {}).get("gust", wind_speed)  # 없으면 기본 풍속 대체
        
        # 3) 날씨 상태(비, 눈, 맑음 등)
        weather_desc = data.get("weather", [{}])[0].get("description", "정보 없음")
        
        # 4) 강수량 (비 또는 눈 체크)
        rain_1h = data.get("rain", {}).get("1h", 0.0)
        snow_1h = data.get("snow", {}).get("1h", 0.0)
        
        precip_text = ""
        if rain_1h > 0:
            precip_text = f", 강우량: {rain_1h}mm/h"
        elif snow_1h > 0:
            precip_text = f", 강설량: {snow_1h}mm/h"
        else:
            precip_text = ", 강수: 없음"

        # 포맷 문자열 조합
        return (
            f"[{location} 실시간 기상] 상태: {weather_desc}, 기온: {temp}℃(체감: {feels_like}℃), "
            f"풍속: {wind_speed}m/s(순간돌풍: {wind_gust}m/s), 습도: {humidity}%{precip_text}"
        )
        
    except Exception as e:
        return f"{location} 날씨 조회 실패: {e}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")