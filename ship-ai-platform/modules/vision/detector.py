"""
modules/vision/detector.py
조선소 용접선 및 결함 탐지 비전 모듈 (팀원 연계용 표준 인터페이스 스텁).
"""
from datetime import datetime
from typing import Dict, Any

def analyze_welding_frame(image_path: str = "", include_defect: bool = False) -> Dict[str, Any]:
    """
    현장 용접 이미지 또는 비디오 프레임을 수신하여 BBox 및 이음선 궤적을 반환합니다.
    (추후 팀원의 실제 YOLOv11s 가중치 추론 코드가 이 인터페이스 규격에 맞추어 결합됩니다.)
    """
    started = datetime.utcnow()
    detections = [
        {
            "class_name": "BASE_METAL_PLATE",
            "label_ko": "모재(상판)",
            "confidence": 0.9742,
            "coordinate_type": "PIXEL",
            "x_min": 60.0, "y_min": 40.0, "x_max": 580.0, "y_max": 180.0,
            "color": "#0d6efd",
            "geometry_data": {"type": "polygon", "zone": "upper_plate"}
        },
        {
            "class_name": "STIFFENER_PLATE",
            "label_ko": "보강판(하판)",
            "confidence": 0.9518,
            "coordinate_type": "PIXEL",
            "x_min": 60.0, "y_min": 182.0, "x_max": 580.0, "y_max": 320.0,
            "color": "#198754",
            "geometry_data": {"type": "polygon", "zone": "lower_plate"}
        },
        {
            "class_name": "WELD_SEAM_LINE",
            "label_ko": "용접 이음새(궤적)",
            "confidence": 0.9385,
            "coordinate_type": "PIXEL",
            "x_min": 65.0, "y_min": 176.0, "x_max": 575.0, "y_max": 186.0,
            "color": "#ff2e4c",
            "geometry_data": {
                "type": "seam_centerline",
                "seam_trajectory": [[65, 181], [200, 181], [350, 180], [480, 181], [575, 181]],
                "root_gap_mm": 2.1
            }
        }
    ]

    if include_defect:
        detections.append({
            "class_name": "CLAMP_OBSTACLE",
            "label_ko": "클램프 간섭물(위험)",
            "confidence": 0.9120,
            "coordinate_type": "PIXEL",
            "x_min": 420.0, "y_min": 145.0, "x_max": 505.0, "y_max": 215.0,
            "color": "#ffc107",
            "geometry_data": {"type": "obstacle", "distance_to_seam_mm": 12.0}
        })

    completed = datetime.utcnow()
    return {
        "model_name": "YOLOv11s-ShipWeld-Seg",
        "model_version": "v2.4-2026",
        "analysis_status": "SUCCESS",
        "image_width": 640,
        "image_height": 360,
        "started_at": started,
        "completed_at": completed,
        "detections": detections,
        "seam_trajectory": [[65, 181], [200, 181], [350, 180], [480, 181], [575, 181]]
    }
