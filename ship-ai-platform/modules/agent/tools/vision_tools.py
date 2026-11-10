"""
modules/agent/tools/vision_tools.py
Oracle SHIP_USER DB의 VISION_ANALYSES, VISION_DETECTIONS, VISION_INSPECTIONS, ROBOT_EXECUTIONS 실시간 조회 도구
"""

import json
import math
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from sqlalchemy import create_engine, text

from config import Config

_USER_URI = Config.SQLALCHEMY_DATABASE_URI
_ENGINE_OPTS = getattr(Config, "SQLALCHEMY_ENGINE_OPTIONS", {"pool_pre_ping": True})
user_engine = create_engine(_USER_URI, **_ENGINE_OPTS)


def _calc_pixel_length_mm(waypoints: List[List[float]], scale_mm_per_px: float = 1.25) -> float:
    if not waypoints or len(waypoints) < 2:
        return 0.0
    total_px = 0.0
    for i in range(len(waypoints) - 1):
        dx = float(waypoints[i + 1][0]) - float(waypoints[i][0])
        dy = float(waypoints[i + 1][1]) - float(waypoints[i][1])
        total_px += math.sqrt(dx * dx + dy * dy)
    return round(total_px * scale_mm_per_px, 1)


@tool
def get_vision_inspection_status(
    work_id: Optional[str] = None,
    camera_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    [Oracle SHIP_USER 비전 & 로봇 상태 통합 조회]
    VISION_ANALYSES, VISION_DETECTIONS, VISION_INSPECTIONS 및 ROBOT_EXECUTIONS 테이블을 조회하여
    용접 상태(WELDING_STATE), 시작/종료 좌표, 예상 용접 길이, 간섭 객체 여부(SAFETY_CLEARANCE_YN),
    로봇 비상정지(EMERGENCY_STOP_YN) 상태를 반환합니다.
    """
    try:
        with user_engine.connect() as conn:
            # 1) 최신 VISION_ANALYSES 세션 조회
            sql_ana = """
                SELECT analysis_id, work_id, camera_code, original_image_uri,
                       visualized_image_uri, image_width, image_height,
                       status_code, welding_state
                FROM VISION_ANALYSES
                WHERE (:work_id IS NULL OR TO_CHAR(work_id) = :work_id)
                  AND (:camera_code IS NULL OR camera_code = :camera_code)
                ORDER BY created_at DESC
                FETCH FIRST 1 ROWS ONLY
            """
            ana_row = conn.execute(
                text(sql_ana),
                {"work_id": str(work_id) if work_id else None, "camera_code": camera_code},
            ).mappings().first()

            # 조건에 맞는 행이 없으면 가장 최근 분석 세션 1건 조회
            if not ana_row:
                ana_row = conn.execute(
                    text(
                        "SELECT analysis_id, work_id, camera_code, original_image_uri, "
                        "visualized_image_uri, image_width, image_height, status_code, welding_state "
                        "FROM VISION_ANALYSES ORDER BY created_at DESC FETCH FIRST 1 ROWS ONLY"
                    )
                ).mappings().first()

            if not ana_row:
                return {
                    "status": "NO_DATA",
                    "inspection_status": "BEFORE_WELD",
                    "welding_state": "BEFORE_WELD",
                    "safety_clearance_yn": "Y",
                    "detections": [],
                }

            analysis_id = ana_row["analysis_id"]
            actual_work_id = ana_row["work_id"] or work_id

            # 2) 해당 세션의 VISION_DETECTIONS 조회
            sql_det = """
                SELECT detection_id, class_id, class_name, confidence, tracker_id,
                       bbox_x, bbox_y, bbox_width, bbox_height, centerline_waypoints_json
                FROM VISION_DETECTIONS
                WHERE analysis_id = :analysis_id
            """
            det_rows = conn.execute(text(sql_det), {"analysis_id": analysis_id}).mappings().all()

            detections = []
            waypoints = []
            interfering_objects = []

            for d in det_rows:
                c_name = str(d["class_name"] or "").lower()
                wp_raw = d["centerline_waypoints_json"]
                wp_parsed = json.loads(wp_raw) if isinstance(wp_raw, str) and wp_raw.strip() else (wp_raw or [])
                if c_name == "weld_joint" and wp_parsed:
                    waypoints = wp_parsed

                if c_name in ("worker", "person", "obstacle"):
                    interfering_objects.append(c_name)

                detections.append({
                    "detection_id": d["detection_id"],
                    "class_id": d["class_id"],
                    "class_name": d["class_name"],
                    "confidence": float(d["confidence"] or 0.0),
                    "tracker_id": d["tracker_id"],
                    "bbox": {
                        "x": d["bbox_x"],
                        "y": d["bbox_y"],
                        "width": d["bbox_width"],
                        "height": d["bbox_height"],
                    },
                    "centerline_waypoints": wp_parsed if c_name == "weld_joint" else None,
                })

            # 3) VISION_INSPECTIONS 조회
            sql_ins = """
                SELECT inspection_id, start_point_json, end_point_json,
                       estimated_length_mm, safety_clearance_yn,
                       inspection_status, defect_detail_json
                FROM VISION_INSPECTIONS
                WHERE analysis_id = :analysis_id
                ORDER BY created_at DESC
                FETCH FIRST 1 ROWS ONLY
            """
            ins_row = conn.execute(text(sql_ins), {"analysis_id": analysis_id}).mappings().first()

            start_pt = waypoints[0] if waypoints else None
            end_pt = waypoints[-1] if waypoints else None
            est_len = _calc_pixel_length_mm(waypoints) if waypoints else None
            safety_yn = "N" if interfering_objects else "Y"
            ins_status = "BEFORE_WELD"
            defect_detail = None

            if ins_row:
                if ins_row["start_point_json"]:
                    sp = ins_row["start_point_json"]
                    start_pt = json.loads(sp) if isinstance(sp, str) else sp
                if ins_row["end_point_json"]:
                    ep = ins_row["end_point_json"]
                    end_pt = json.loads(ep) if isinstance(ep, str) else ep
                if ins_row["estimated_length_mm"] is not None:
                    est_len = float(ins_row["estimated_length_mm"])
                if ins_row["safety_clearance_yn"]:
                    safety_yn = str(ins_row["safety_clearance_yn"]).upper()
                if ins_row["inspection_status"]:
                    ins_status = str(ins_row["inspection_status"]).upper()
                if ins_row["defect_detail_json"]:
                    dd = ins_row["defect_detail_json"]
                    defect_detail = json.loads(dd) if isinstance(dd, str) else dd

            # 4) ROBOT_EXECUTIONS 상태 확인 (비상정지 등)
            robot_status = None
            try:
                r_row = conn.execute(
                    text(
                        "SELECT equipment_code, robot_name, robot_status, emergency_stop_yn, ros2_node_status "
                        "FROM ROBOT_EXECUTIONS "
                        "WHERE (:work_id IS NULL OR TO_CHAR(work_id) = :work_id) "
                        "FETCH FIRST 1 ROWS ONLY"
                    ),
                    {"work_id": str(actual_work_id) if actual_work_id else None},
                ).mappings().first()
                if r_row:
                    robot_status = dict(r_row)
                    if str(r_row["emergency_stop_yn"] or "N").upper() == "Y":
                        safety_yn = "N"
                        interfering_objects.append("ROBOT_EMERGENCY_STOP")
            except Exception:
                pass

            return {
                "status": "SUCCESS",
                "analysis_id": analysis_id,
                "work_id": actual_work_id,
                "camera_code": ana_row["camera_code"],
                "welding_state": ana_row["welding_state"] or "BEFORE_WELD",
                "inspection_status": ins_status,
                "safety_clearance_yn": safety_yn,
                "interfering_objects": interfering_objects,
                "start_point_json": start_pt,
                "end_point_json": end_pt,
                "estimated_length_mm": est_len,
                "centerline_waypoints": waypoints,
                "defect_detail_json": defect_detail,
                "detections": detections,
                "robot_execution": robot_status,
            }

    except Exception as e:
        return {
            "status": "ERROR",
            "error": f"VISION DB 조회 오류: {str(e)}",
            "inspection_status": "BEFORE_WELD",
            "welding_state": "BEFORE_WELD",
            "safety_clearance_yn": "Y",
            "detections": [],
        }