# 테스트를 위한 임시 도구
"""
agent_system/tools/mock_tools.py
Team 2(검색팀)와 통합 전 임시로 동작하는 Mock 데이터 도구 모음
"""
from typing import Dict, List


def mock_chroma_rag(query: str) -> List[str]:
  """Chroma Vector DB 규격 검색 Mock 도구"""
  query_lower = query.lower()

  if "ss400" in query_lower or "10t" in query_lower or "용접" in query_lower:
    return [
        (
            "[사내 WPS-SS400-01] SS400 10T 맞대기 용접 표준: FCAW 공정, 전류"
            " 210~230A (권장 220A), 전압 23~25V (권장 24V), 주행속도"
            " 4.5~5.5mm/s, 루트 간격 허용공차 2~4mm (권장 3.5mm)"
        ),
        (
            "[KOSHA 안전보건지침 M-01] 모재 온도가 5℃ 이하인 저온 환경에서는 수소"
            " 유입에 따른 균열 방지를 위해 작업 전 100℃ 이상 예열을 반드시"
            " 실시해야 함."
        ),
    ]
  elif "크레인" in query_lower or "인양" in query_lower or "중량" in query_lower:
    return [
        (
            "[중량물 취급 안전규칙 제38조] 블록 인양 시 슬링벨트 및 샤클 결속"
            " 각도는 60도 이내로 유지하며, 와이어로프 안전계수는 5 이상 확보할 것."
        ),
        (
            "[악천후 작업 제한 기준] 순간풍속이 10m/s를 초과하는 경우 타워크레인 및"
            " 지브크레인의 중량물 인양 작업을 즉각 중단해야 함."
        ),
    ]
  else:
    return [
        (
            "[일반 조선 가공 지침] 규정되지 않은 특수 사양의 경우 선급 협회(KR,"
            " DNV) 표준 규정 및 사내 용접관리자 승인 기준을 준용함."
        )
    ]


def mock_weather_mcp(location: str) -> Dict:
  """303호 기상 MCP 서버 Mock 도구"""
  # 저온 예열 및 강풍 조건을 테스트하기 위한 현장 기상 반환
  return {
      "location": location,
      "temperature": 3.0,  # 5도 이하 -> 예열 지침 발동
      "wind_speed": 11.2,  # 10m/s 초과 -> 크레인 작업 중지 기준
      "humidity": 68,
      "condition": "구름 많음 (저온/강풍 주의)",
  }