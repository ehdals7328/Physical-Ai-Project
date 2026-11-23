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


def _lookup_work_order_meta(conn, work_id: Any) -> Dict[str, Any]:
    """WORK_ORDERS 테이블에서 연결된 블록/작업번호 정보를 안전하게 보조 조회합니다."""
    if work_id is None:
        return {}
    try:
        row = conn.execute(
            text("SELECT * FROM WORK_ORDERS WHERE work_id = :wid FETCH FIRST 1 ROWS ONLY"),
            {"wid": work_id},
        ).mappings().first()
        if not row:
            return {}
        return {
            "work_no": row.get("work_no"),
            "block_id": row.get("block_id") or row.get("block_code"),
            "work_status": row.get("status") or row.get("work_status"),
        }
    except Exception:
        return {}


@tool
def get_vision_inspection_status(
    work_id: Optional[str] = None,
    camera_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    [Oracle SHIP_USER 비전 & 로봇 상태 단건 조회 - 작업지시서 및 안전 인터록용]
    - NO_DATA: 계측 이력이 없는 상태 (safety_clearance_yn=None 반환, 안전 확인 완료와 명확히 분리)
    - ERROR: DB/조회 장애 발생 시 Fail-Closed 원칙에 따라 safety_clearance_yn='N' 반환
    """
    try:
        with user_engine.connect() as conn:
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

            if not ana_row:
                ana_row = conn.execute(
                    text(
                        "SELECT analysis_id, work_id, camera_code, original_image_uri, "
                        "visualized_image_uri, image_width, image_height, status_code, welding_state "
                        "FROM VISION_ANALYSES ORDER BY created_at DESC FETCH FIRST 1 ROWS ONLY"
                    )
                ).mappings().first()

            # [패치 2-1] 분석 이력 미존재(NO_DATA) 시 안전 확인 완료('Y')와 분리
            if not ana_row:
                return {
                    "status": "NO_DATA",
                    "inspection_status": "NO_DATA",
                    "welding_state": "BEFORE_WELD",
                    "safety_clearance_yn": None,
                    "interfering_objects": [],
                    "detections": [],
                }

            analysis_id = ana_row["analysis_id"]
            actual_work_id = ana_row["work_id"] or work_id

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
                wp_parsed = (
                    json.loads(wp_raw)
                    if isinstance(wp_raw, str) and wp_raw.strip()
                    else (wp_raw or [])
                )
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

            sql_ins = """
                SELECT *
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
                if ins_row.get("start_point_json"):
                    sp = ins_row["start_point_json"]
                    start_pt = json.loads(sp) if isinstance(sp, str) else sp
                if ins_row.get("end_point_json"):
                    ep = ins_row["end_point_json"]
                    end_pt = json.loads(ep) if isinstance(ep, str) else ep
                if ins_row.get("estimated_length_mm") is not None:
                    est_len = float(ins_row["estimated_length_mm"])
                if ins_row.get("safety_clearance_yn"):
                    safety_yn = str(ins_row["safety_clearance_yn"]).upper()
                if ins_row.get("inspection_status"):
                    ins_status = str(ins_row["inspection_status"]).upper()
                if ins_row.get("defect_detail_json"):
                    dd = ins_row["defect_detail_json"]
                    defect_detail = json.loads(dd) if isinstance(dd, str) else dd

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
        # [패치 2-2] 조회 실패(ERROR) 시 Fail-Closed 원칙 적용 ('Y' 절대 반환 금지)
        return {
            "status": "ERROR",
            "error": f"VISION DB 조회 오류: {str(e)}",
            "inspection_status": "ERROR",
            "welding_state": "UNKNOWN",
            "safety_clearance_yn": "N",
            "interfering_objects": ["VISION_DB_ERROR"],
            "detections": [],
        }

@tool
def get_vision_overview_for_qa(
    block_id: Optional[str] = None,
    camera_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    [자연어 비전 관제 질의 전용 조회 도구]
    "용접해야 할 부분 있어?", "지금 비전 탐지 상황 알려줘", "결함이나 간섭 위험 있어?" 등의 질의에 대해
    Oracle SHIP_USER의 VISION_ANALYSES, VISION_INSPECTIONS, VISION_DETECTIONS, ROBOT_EXECUTIONS를
    종합 집계하여 원시 픽셀 좌표 없이 깔끔한 상태 브리핑 데이터를 반환합니다.
    """
    try:
        with user_engine.connect() as conn:
            sql_base = """
                SELECT
                    va.analysis_id,
                    va.work_id,
                    va.camera_code,
                    va.welding_state,
                    va.status_code AS analysis_status,
                    va.created_at AS captured_at,
                    vi.inspection_id,
                    vi.safety_clearance_yn,
                    vi.inspection_status,
                    vi.defect_detail_json
                FROM VISION_ANALYSES va
                LEFT JOIN VISION_INSPECTIONS vi
                    ON va.analysis_id = vi.analysis_id
                {where_clause}
                ORDER BY va.created_at DESC NULLS LAST, va.analysis_id DESC
                FETCH FIRST 20 ROWS ONLY
            """

            norm_cam = camera_code.strip() if camera_code else None
            rows = conn.execute(
                text(
                    sql_base.format(
                        where_clause="WHERE UPPER(va.camera_code) = UPPER(:camera_code)"
                        if norm_cam
                        else ""
                    )
                ),
                {"camera_code": norm_cam} if norm_cam else {},
            ).mappings().all()

            # camera_code가 야드/구역명 등으로 넘어와 0건일 때 전체 최신 기록으로 자동 폴백
            camera_fallback_used = False
            if not rows and norm_cam:
                camera_fallback_used = True
                rows = conn.execute(
                    text(sql_base.format(where_clause=""))
                ).mappings().all()

            def _build_record(r) -> Dict[str, Any]:
                aid = r["analysis_id"]
                wid = r["work_id"]
                wo_meta = _lookup_work_order_meta(conn, wid)
                resolved_block = wo_meta.get("block_id")

                # 탐지된 객체 요약 (좌표 제외, 클래스별 개수만 집계)
                det_rows = conn.execute(
                    text(
                        "SELECT class_name, COUNT(*) AS cnt "
                        "FROM VISION_DETECTIONS WHERE analysis_id = :aid GROUP BY class_name"
                    ),
                    {"aid": aid},
                ).mappings().all()
                detected_summary = {str(d["class_name"]): int(d["cnt"]) for d in det_rows}
                has_weld_joint = detected_summary.get("weld_joint", 0) > 0

                w_state = str(r["welding_state"] or "BEFORE_WELD").upper()
                i_status = str(r["inspection_status"] or "BEFORE_WELD").upper()
                clearance = str(r["safety_clearance_yn"] or "Y").upper()
                est_len_mm = (
                    float(r["estimated_length_mm"])
                    if r.get("estimated_length_mm") is not None
                    else None
                )

                defect_raw = r["defect_detail_json"]
                defect_info = (
                    json.loads(defect_raw)
                    if isinstance(defect_raw, str) and defect_raw.strip()
                    else defect_raw
                )

                return {
                    "analysis_id": aid,
                    "work_id": wid,
                    "work_no": wo_meta.get("work_no"),
                    "block_id": resolved_block or (f"WORK-{wid}" if wid else "미매핑 블록"),
                    "camera_code": r["camera_code"],
                    "welding_state": w_state,
                    "inspection_status": i_status,
                    "weld_joint_detected": has_weld_joint,
                    "estimated_length_mm": est_len_mm,
                    "detected_objects": detected_summary,
                    "safety_clearance_yn": clearance,
                    "defect_detail": defect_info,
                    "captured_at": str(r["captured_at"]) if r["captured_at"] else None,
                }

            all_items = [_build_record(r) for r in rows]

            # block_id 필터 적용 — 단, 일치 건이 0건이면 전체 최신 비전 로그로 자동 폴백
            block_fallback_used = False
            if block_id:
                filtered = [
                    it for it in all_items
                    if str(it.get("block_id", "")).upper() == str(block_id).upper()
                ]
                if filtered:
                    records = filtered
                else:
                    records = all_items
                    block_fallback_used = True
            else:
                records = all_items

            pending_weld_targets: List[Dict[str, Any]] = []
            in_welding_targets: List[Dict[str, Any]] = []
            completed_targets: List[Dict[str, Any]] = []
            hazard_or_defect_targets: List[Dict[str, Any]] = []

            for item in records:
                w_state = item["welding_state"]
                i_status = item["inspection_status"]
                clearance = item["safety_clearance_yn"]

                if w_state == "BEFORE_WELD" or i_status == "BEFORE_WELD":
                    pending_weld_targets.append(item)
                elif w_state == "IN_WELDING":
                    in_welding_targets.append(item)
                else:
                    completed_targets.append(item)

                if clearance == "N" or i_status == "DEFECT":
                    hazard_or_defect_targets.append(item)

            # 로봇 실행 상태 요약
            robots: List[Dict[str, Any]] = []
            try:
                r_rows = conn.execute(
                    text(
                        "SELECT equipment_code, robot_name, robot_status, current_location_desc, "
                        "battery_percent, emergency_stop_yn, ros2_node_status "
                        "FROM ROBOT_EXECUTIONS ORDER BY robot_exec_id ASC"
                    )
                ).mappings().all()
                for rb in r_rows:
                    robots.append(dict(rb))
            except Exception:
                pass

            return {
                "status": "SUCCESS" if records else "NO_DATA",
                "total_records": len(records),
                "camera_fallback_used": camera_fallback_used,
                "block_fallback_used": block_fallback_used,
                "summary_counts": {
                    "pending_weld_count": len(pending_weld_targets),
                    "in_welding_count": len(in_welding_targets),
                    "completed_count": len(completed_targets),
                    "hazard_or_defect_count": len(hazard_or_defect_targets),
                },
                "pending_weld_targets": pending_weld_targets,
                "in_welding_targets": in_welding_targets,
                "completed_targets": completed_targets,
                "hazard_or_defect_targets": hazard_or_defect_targets,
                "robots": robots,
                "all_recent_records": records[:10],
            }

    except Exception as e:
        return {
            "status": "ERROR",
            "error": f"비전 관제 현황 조회 오류: {str(e)}",
            "total_records": 0,
        }