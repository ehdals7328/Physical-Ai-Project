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
        return "Ulsan"

    # 기본 영문 도시명 처리
    low = clean_query.lower()
    if low in ["seoul", "busan", "daegu", "ulsan", "geoje", "pohang", "incheon", "changwon"]:
        return clean_query

    prompt = (
        "다음 문장이나 지명에서 OpenWeatherMap API 조회가 가능한 대한민국 대표 도시의 영문명 딱 1단어만 출력하세요.\n"
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
        return res.choices[0].message.content.strip().split()[0].replace(".", "").replace("'", "")
    except Exception:
        return "Ulsan"


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
def get_current_weather(location: str = "Ulsan") -> str:
    """[MCP Tool] 실시간 기상 조회"""
    target_city = extract_clean_city(location)

    # 1. 대한민국 국가코드(,KR)를 우선 부여하여 한국 도시 우선 검색
    geo = get_geo(f"{target_city},KR")
    if not geo:
        geo = get_geo(target_city)

    if not geo:
        return f"{location} ({target_city}) 위치의 좌표를 찾을 수 없습니다."

    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={geo['lat']}&lon={geo['lon']}&appid={weatherapi_key}&units=metric"
    try:
        res = requests.get(weather_url, timeout=5.0)
        if res.status_code != 200:
            return f"{location}의 날씨 정보를 가져오지 못했습니다."

        data = res.json()
        temp = data["main"]["temp"]
        wind = data["wind"]["speed"]
        humidity = data["main"]["humidity"]

        return f"[{location} 실시간 기상] 기온: {temp}℃, 풍속: {wind}m/s, 습도: {humidity}%"
    except Exception as e:
        return f"{location} 날씨 조회 실패: {e}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")