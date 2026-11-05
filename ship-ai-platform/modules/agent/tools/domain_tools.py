"""
Domain Tools Package
물리적 환경 계측값과 규격 임계치(Thresholds) 대조 기반 안전 인터록 엔진
"""
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool


@tool
def evaluate_environmental_safety(
    current_temp: Optional[float],
    current_wind: Optional[float],
    rainfall: Optional[float] = 0.0,
    thresholds: Optional[Dict[str, Any]] = None,
    task_context: Optional[str] = "현장 작업",
    task_type: Optional[str] = None,
    is_outdoor: bool = True
) -> Dict[str, Any]:
    """
    현장 기상 실측값과 규격 임계치(thresholds)를 대조하여 안전성(PASS/CONDITIONAL_PASS/STOP)을 판정합니다.
    Fail-Closed 원칙: 기온이나 풍속 결측 시 즉시 작업 중단(STOP).
    """
    # 1. 입력 기상 데이터 무결성 검증
    if current_temp is None or current_wind is None:
        return {
            "status": "STOP",
            "safety_level": "FORBIDDEN",
            "is_safe": False,
            "mandatory_rules": ["기상 센서 데이터 결측으로 안전 평가 불가"],
            "prerequisites": [],
            "action_message": "기상 센서 확인 불가로 작업 즉시 보류",
            "environment": {"temp": current_temp, "wind": current_wind, "rain": rainfall}
        }

    rain_val = max(0.0, float(rainfall or 0.0))
    wind_val = max(0.0, float(current_wind))
    temp_val = float(current_temp)

    # 2. 동적 기준치 유연 추출 (단일 수치 및 중첩 Dict 구조 모두 수용)
    def _extract_val(v: Any, default_val: float) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict):
            for sub_k in ["max", "limit", "value", "target", "min"]:
                if sub_k in v and isinstance(v[sub_k], (int, float)):
                    return float(v[sub_k])
        return default_val

    raw_limits = thresholds or {}
    wind_stop_limit = _extract_val(
        raw_limits.get("work_stop_wind_limit") or raw_limits.get("wind_speed") or raw_limits.get("wind_limit"),
        10.0
    )
    wind_caution_limit = _extract_val(
        raw_limits.get("weld_wind_screen_limit") or raw_limits.get("screen_limit"),
        2.0
    )
    temp_preheat_limit = _extract_val(
        raw_limits.get("preheat_temp_limit") or raw_limits.get("temperature") or raw_limits.get("temp_limit"),
        5.0
    )
    preheat_target = _extract_val(raw_limits.get("preheat_target_temp"), 100.0)

    alerts: List[str] = []
    prerequisites: List[str] = []
    stop_reasons: List[str] = []

    # 3. 최우선 전면 중지(STOP) 검사
    if is_outdoor and wind_val >= wind_stop_limit:
        stop_reasons.append(f"강풍 위험: 실측 풍속 {wind_val}m/s가 안전 한계치({wind_stop_limit}m/s) 이상이므로 작업을 중단하십시오.")

    if is_outdoor and rain_val > 0.0:
        stop_reasons.append(f"강우 위험: 강우({rain_val}mm/h)로 인한 안전사고 위험으로 야외 작업을 중지하십시오.")

    if stop_reasons:
        return {
            "status": "STOP",
            "safety_level": "FORBIDDEN",
            "is_safe": False,
            "mandatory_rules": stop_reasons,
            "prerequisites": [],
            "action_message": "작업 전면 중단 (기상 정상화 시까지 착수 불가)",
            "environment": {"temp": temp_val, "wind": wind_val, "rain": rain_val}
        }

    # 4. 조건부 안전 조치(CONDITIONAL_PASS) 검사
    if is_outdoor and wind_val >= wind_caution_limit:
        alerts.append(f"풍속 주의: 실측 풍속 {wind_val}m/s (기준 {wind_caution_limit}m/s 이상) 환경이므로 현장 이동식 방풍막 설치 필수")
        prerequisites.append("차폐용 이동식 방풍막 설치 완료 확인")

    if temp_val <= temp_preheat_limit:
        alerts.append(f"저온 환경: 실측 기온 {temp_val}℃ (기준 {temp_preheat_limit}℃ 이하)이므로 모재 {preheat_target:.0f}℃ 이상 예열 필수")
        prerequisites.append(f"모재 {preheat_target:.0f}℃ 이상 사전 예열 및 표면 온도 측정 완료 확인")

    # 5. 최종 판정 확정
    if prerequisites:
        status = "CONDITIONAL_PASS"
        safety_level = "REQUIRES_ACTION"
        action_msg = "조건부 승인 (선행 안전조치 확인 필요)"
    else:
        status = "PASS"
        safety_level = "CLEAR"
        action_msg = "환경 조건 양호 (표준 작업 착수 가능)"

    process_label = task_context or task_type or "현장 공정"

    return {
        "status": status,
        "safety_level": safety_level,
        "is_safe": (safety_level == "CLEAR"),
        "process": process_label,
        "mandatory_rules": alerts,
        "prerequisites": prerequisites,
        "action_message": action_msg,
        "environment": {"temp": temp_val, "wind": wind_val, "rain": rain_val}
    }


@tool
def validate_environmental_rules(
    temperature: float,
    wind_speed: float,
    task_type: str = "doc_generation"
) -> Dict[str, Any]:
    """기존 호출부 호환용 래퍼 함수"""
    return evaluate_environmental_safety.invoke({
        "current_temp": temperature,
        "current_wind": wind_speed,
        "rainfall": 0.0,
        "thresholds": None,
        "task_context": task_type,
        "task_type": task_type,
        "is_outdoor": True
    })