"""
Domain Tools Package
RAG에서 인출된 사내 규격 기준치와 실시간 센서 계측값을 대조하여
결정론적으로 PASS / STOP / 조치사항을 판정하는 안전 엔진
"""
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool

@tool
def evaluate_environmental_safety(
    current_temp: float,
    current_wind: float,
    thresholds: Optional[Dict[str, float]] = None,
    task_context: str = "용접 작업"
) -> Dict[str, Any]:
    """
    RAG 규격 문서에서 인출한 동적 기준치(thresholds)와 실시간 기상 데이터를 비교 검증합니다.
    """
    limits = thresholds or {
        "preheat_temp_limit": 5.0,     # 예열 필요 외기온도 하한선
        "wind_screen_limit": 2.0,      # 방풍막 필요 풍속
        "work_stop_wind_limit": 10.0   # 작업 중지 풍속
    }

    alerts: List[str] = []
    status = "PASS"

    preheat_limit = limits.get("preheat_temp_limit", 5.0)
    wind_screen_limit = limits.get("wind_screen_limit", 2.0)
    stop_wind_limit = limits.get("work_stop_wind_limit", 10.0)

    # 1. 외기온도 대조 (RAG 문서 기준선 반영)
    if current_temp <= preheat_limit:
        alerts.append(f"사내 안전규격 준수: 외기온도가 규정 기준({preheat_limit}℃ 이하)에 도달하여 모재 예열(100℃ 이상) 조치가 필수적입니다.")

    # 2. 풍속 대조 (RAG 문서 기준선 반영)
    if current_wind >= stop_wind_limit:
        status = "STOP"
        alerts.append(f"강풍 위험 경보: 실시간 풍속이 규정 허용 한계({stop_wind_limit}m/s)를 초과하여 즉시 작업을 중지(STOP)해야 합니다.")
    elif current_wind >= wind_screen_limit:
        alerts.append(f"방풍 조치 필요: 실시간 풍속({current_wind}m/s)이 차폐가스 보호 기준({wind_screen_limit}m/s)을 초과하므로 이동식 방풍막을 설치하십시오.")

    return {
        "status": status,
        "is_safe": (status == "PASS"),
        "mandatory_rules": alerts,
        "applied_thresholds": {
            "source": "RAG 사내 규격 문서 인출 기준",
            "preheat_limit_c": preheat_limit,
            "wind_stop_limit_ms": stop_wind_limit
        },
        "measured": {
            "temperature_c": current_temp,
            "wind_speed_ms": current_wind
        }
    }


# ⭕ [기존 호출부 완벽 호환용 래퍼 및 별칭]
@tool
def validate_environmental_rules(
    temperature: float, 
    wind_speed: float, 
    task_type: str = "doc_generation"
) -> Dict[str, Any]:
    """구버전 파이프라인 호출부와의 호환성을 위한 래퍼 함수"""
    return evaluate_environmental_safety.invoke({
        "current_temp": temperature,
        "current_wind": wind_speed,
        "thresholds": None,
        "task_context": task_type
    })