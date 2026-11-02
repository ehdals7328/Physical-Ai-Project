"""
Domain Tools Package
규칙 기반 현장 안전 및 공정 판정 로컬 @tool 모음
"""
from typing import Dict, Any, List
from langchain_core.tools import tool

@tool
def validate_environmental_rules(
    temperature: float, 
    wind_speed: float, 
    task_type: str = "weld_order"
) -> Dict[str, Any]:
    """
    현장 기상 환경(기온, 풍속)에 따른 필수 안전 규정 준수 여부를 검증합니다.
    
    Args:
        temperature: 실시간 외기 온도 (℃)
        wind_speed: 실시간 풍속 (m/s)
        task_type: 작업 유형 ('weld_order', 'heavy_plan', 'general_qa')
        
    Returns:
        Dict containing status, safety flag, and mandatory rules to follow.
    """
    alerts: List[str] = []
    status = "PASS"

    # 1. 저온 작업 환경 규정 (KOSHA 지침)
    if task_type == "weld_order" and temperature <= 5.0:
        alerts.append("저온 환경 규정 준수: 모재 100℃ 이상 예열 작업 필수 (KOSHA 지침)")

    # 2. 강풍 환경 규정 (크레인 인양 중지 및 방풍막 설치)
    if wind_speed >= 10.0:
        alerts.append("강풍 경보: 풍속 10m/s 초과로 크레인 인양 및 고소 작업 즉시 중지 (STOP)")
        status = "STOP"
    elif wind_speed >= 2.0 and task_type == "weld_order":
        alerts.append("방풍 대책 수립: 차폐가스 보호를 위해 작업 구역 이동식 방풍막 설치 필수")

    return {
        "status": status,
        "is_safe": status == "PASS",
        "mandatory_rules": alerts,
        "environment": {
            "temperature_c": temperature,
            "wind_speed_ms": wind_speed
        }
    }