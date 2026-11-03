# C:\Physical-Ai-Project\agent_system\tools\weather_server.py
import os
from pathlib import Path
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# 1. 상위 폴더(agent_system)의 .env 파일 명시적 로드
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

weatherapi_key = os.getenv("WEATHER_API_KEY") or os.getenv("OPENWEATHER_API_KEY")

# 2. host와 port 설정을 FastMCP 생성자에서 정의
mcp = FastMCP("weather-server", host="0.0.0.0", port=8000)


@mcp.tool()
def get_current_weather(location: str = "Ulsan") -> str:
  """조선소 작업 현장의 실시간 기상(기온, 풍속, 습도)을 조회합니다."""
  geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={weatherapi_key}"
  geo_res = requests.get(geo_url).json()
  if not geo_res:
    return f"{location} 위치를 찾을 수 없습니다."

  lat, lon = geo_res[0]["lat"], geo_res[0]["lon"]
  weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={weatherapi_key}&units=metric"
  w_data = requests.get(weather_url).json()

  temp = w_data["main"]["temp"]
  wind = w_data["wind"]["speed"]
  humidity = w_data["main"]["humidity"]

  return (
      f"위치: {location}, 기온: {temp}℃, 풍속: {wind}m/s, 습도:"
      f" {humidity}%"
  )


if __name__ == "__main__":
  # 3. run() 에는 transport="streamable-http"만 전달
  mcp.run(transport="streamable-http")