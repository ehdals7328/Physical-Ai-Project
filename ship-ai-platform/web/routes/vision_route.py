# web/routes/vision_route.py
import base64
import json
import os
import threading
import time
import requests
import urllib3
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from flask import Blueprint, render_template, Response, jsonify, request, session, redirect, url_for, flash
from werkzeug.utils import secure_filename

from web.models import (
    db,
    User,
    WorkOrder,
    ErpBlockMaster,
    VisionAnalysis,
    VisionDetection,
    VisionInspection,
    RobotExecution,
    ErpCompany,
    ErpBranch,
    B2BCompanyLocation,
    B2BDaehanMaterial,
    B2BDonghaeWorker,
    B2BHankookEquipment,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

vision_bp = Blueprint("vision", __name__, url_prefix="/vision")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

RUNPOD_ENDPOINT_ID = (os.getenv("RUNPOD_ENDPOINT_ID") or "").strip()
RUNPOD_API_KEY = (os.getenv("RUNPOD_API_KEY") or "").strip()
RUNPOD_SERVERLESS_URL = (
    f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}" if RUNPOD_ENDPOINT_ID else ""
)
RUNPOD_BASE = (
    os.getenv("RUNPOD_BASE")
    or os.getenv("VISION_API_URL")
    or ""
).strip().rstrip("/")
RUNPOD_STREAM_URL = f"{RUNPOD_BASE}/video_feed" if RUNPOD_BASE else ""
RUNPOD_JSON_URL = f"{RUNPOD_BASE}/detections" if RUNPOD_BASE else ""
RUNPOD_PAUSE_URL = f"{RUNPOD_BASE}/pause" if RUNPOD_BASE else ""
RUNPOD_CHANGE_VIDEO_URL = f"{RUNPOD_BASE}/change_video/" if RUNPOD_BASE else ""

# DB 기본값(video1.mp4 등)과 RunPod 파일명/단계 매핑
VIDEO_FILENAME_ALIASES = {
    "video1.mp4": "Test_1.mp4",
    "video2.mp4": "ocr_test.mp4",
    "video3.mp4": "seg_Test.mp4",
    "transport": "ocr_test.mp4",
    "assembly": "Test_1.mp4",
    "welding": "seg_Test.mp4",
}

STAGE_MAP = {
    "transport": "transport",
    "test1": "transport",
    "test1.png": "transport",
    "ocr_test.mp4": "transport",
    "video2.mp4": "transport",
    "assembly": "assembly",
    "test2": "assembly",
    "test2.png": "assembly",
    "test2.mp4": "assembly",
    "test_1.mp4": "assembly",
    "video1.mp4": "assembly",
    "welding": "welding",
    "test3": "welding",
    "test3.png": "welding",
    "test3.mp4": "welding",
    "seg_test.mp4": "welding",
    "video3.mp4": "welding",
}

# ⭐ RunPod Serverless 최신 결과 메모리 캐시 (웹 실시간 표시용 · 블로킹 방지)
_RUNPOD_STATE_LOCK = threading.Lock()
_RUNPOD_FETCH_LOCK = threading.Lock()
_RUNPOD_STATE: dict = {
    "stage": "assembly",
    "bay_id": "",
    "image_bytes": b"",
    "image_data_url": "",
    "detection_data": {"image": {"width": 960, "height": 540}, "source": "WAIT", "stage": "assembly", "predictions": []},
    "is_fetching": False,
    "last_error": "",
    "updated_at": 0.0,
}

# ⭐ DB 중복 적재 방지 캐시 (시나리오 그대로 유지 — DB 과적재 차단)
_TRACKER_LOCK = threading.Lock()
_SEEN_JOINT_TRACKER_KEYS: set[tuple[str, int]] = set()
_SEEN_OCR_KEYS: set[tuple[str, str]] = set()
_CAMERA_AUTOSAVE_LOCK = threading.Lock()
_LAST_CAMERA_AUTOSAVE_TS: dict[str, float] = {}
AUTOSAVE_COOLDOWN_SEC = 10.0
_LAST_MANUAL_SAVE_TS: dict[str, float] = {}
MANUAL_SAVE_COOLDOWN_SEC = 3.0

MAX_DISK_FILES_PER_FOLDER = 40
RECORDS_PER_PAGE = 5
VISION_CAPTURE_DIR = Path(__file__).resolve().parents[1] / "static" / "captures"

CLASS_LABEL_KO = {
    "weld_joint": "용접선",
    "weld_seam": "용접선",
    "seam": "용접선",
    "weld_bead": "용접비드",
    "ship_block": "선박블록",
    "block": "선박블록",
    "transporter": "트랜스포터",
    "vehicle_transporter": "트랜스포터",
    "forklift": "지게차",
    "truck": "운반차량",
    "vehicle": "운반차량",
    "crane_hook": "크레인후크",
    "crane": "크레인",
    "crane_lifting": "크레인 인양 중",
    "hook": "크레인후크",
    "worker": "작업자",
    "person": "작업자",
    "flame": "용접불꽃",
    "welding_arc": "용접아크",
    "spark": "용접스파크",
    "torch": "용접토치",
    "robot": "용접로봇",
    "helmet": "안전모",
    "obstacle": "장애물",
}


def _get_default_camera_code() -> str:
    try:
        br = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).first()
        if br and br.camera_code:
            return br.camera_code
    except Exception:
        pass
    return ""


def _get_default_work_id() -> int | None:
    try:
        wo = WorkOrder.query.order_by(WorkOrder.work_id.desc()).first()
        if wo:
            return wo.work_id
    except Exception:
        pass
    return None


def _use_serverless() -> bool:
    return bool(RUNPOD_SERVERLESS_URL and RUNPOD_API_KEY)


def _runpod_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if RUNPOD_API_KEY:
        headers["Authorization"] = f"Bearer {RUNPOD_API_KEY}"
    return headers


def _resolve_stage(name: str | None) -> str:
    s = (name or "").strip().lower()
    if s in STAGE_MAP:
        return STAGE_MAP[s]
    if "ocr" in s or "transport" in s or "video2" in s:
        return "transport"
    if "seg" in s or "robot" in s or "weld" in s or "video3" in s:
        return "welding"
    return "assembly"


def _bbox_xyxy_to_box(bbox: list) -> dict:
    if not bbox or len(bbox) < 4:
        return {}
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return {
        "x": round((x1 + x2) / 2.0, 1),
        "y": round((y1 + y2) / 2.0, 1),
        "width": round(x2 - x1, 1),
        "height": round(y2 - y1, 1),
    }


def _normalize_incoming_json(raw: dict) -> dict:
    """
    RunPod에서 들어온 JSON(Serverless payload 또는 Pod predictions JSON)에서
    모든 탐지 객체(block, transporter, worker, crane, weld_joint, torch, spark 등)를
    누락 없이 추출하여 웹 탐지 객체 목록으로 정리합니다.
    """
    if not isinstance(raw, dict):
        return {"image": {"width": 960, "height": 540}, "source": "EMPTY", "predictions": []}

    stage = str(raw.get("stage") or "").lower()
    width, height = 1920, 1080
    predictions: list[dict] = []
    seen_keys: set[tuple] = set()

    def _add_pred(
        cls_raw: str,
        conf_val: float = 0.0,
        bbox=None,
        ocr_text: str = "",
        extra: dict | None = None,
    ):
        cls_clean = str(cls_raw or "object").strip()
        if not cls_clean:
            cls_clean = "object"
        c_low = cls_clean.lower()
        conf_f = round(float(conf_val or 0.0), 4)
        box_dict = {}
        bbox_sig = ()
        if isinstance(bbox, dict) and bbox:
            box_dict = bbox
            bbox_sig = (round(float(bbox.get("x", 0)), 0), round(float(bbox.get("y", 0)), 0))
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            box_dict = _bbox_xyxy_to_box(list(bbox))
            bbox_sig = tuple(round(float(v), 0) for v in bbox[:4])

        # 같은 객체가 blocks와 detections 양쪽에 중복 포함된 경우 1건으로 병합
        norm_cls = "block" if c_low in ("block", "ship_block") else c_low
        dedup_key = (norm_cls, round(conf_f, 2), bbox_sig)
        if dedup_key in seen_keys:
            if ocr_text:
                for prev in predictions:
                    p_low = str(prev.get("class", "")).lower()
                    p_norm = "block" if p_low in ("block", "ship_block") else p_low
                    if p_norm == norm_cls and not prev.get("ocr_text"):
                        prev["ocr_text"] = ocr_text
            return
        seen_keys.add(dedup_key)

        item = {
            "class": cls_clean,
            "class_ko": CLASS_LABEL_KO.get(c_low, cls_clean),
            "confidence": conf_f,
            "box": box_dict,
        }
        if ocr_text:
            item["ocr_text"] = str(ocr_text).strip()
        if extra:
            for k, v in extra.items():
                if k not in item and v is not None:
                    item[k] = v
        predictions.append(item)

    def _scan_detection_list(items, default_cls: str = "object", fallback_ocr: str = ""):
        if not isinstance(items, list):
            return
        for obj in items:
            if not isinstance(obj, dict):
                continue
            cls_raw = str(
                obj.get("class")
                or obj.get("class_name")
                or obj.get("label")
                or obj.get("name")
                or default_cls
            )
            conf_raw = (
                obj.get("confidence")
                if obj.get("confidence") is not None
                else (obj.get("conf") if obj.get("conf") is not None else obj.get("score", 0.0))
            )
            ocr_val = str(obj.get("ocr_id") or obj.get("ocr_text") or obj.get("block_id") or "").strip()
            if not ocr_val and cls_raw.lower() in ("block", "ship_block"):
                ocr_val = fallback_ocr
            bbox_val = obj.get("bbox") or obj.get("box")
            extra = {}
            if obj.get("tracker_id") is not None:
                extra["tracker_id"] = obj.get("tracker_id")
            if obj.get("class_id") is not None:
                extra["class_id"] = obj.get("class_id")
            if obj.get("centerline_waypoints"):
                extra["centerline_waypoints"] = obj.get("centerline_waypoints")
            _add_pred(cls_raw, conf_raw, bbox=bbox_val, ocr_text=ocr_val, extra=extra)

    def _scan_stage_dict(sd: dict, default_ocr: str = ""):
        nonlocal width, height
        if not isinstance(sd, dict):
            return
        res = sd.get("image_resolution")
        if isinstance(res, (list, tuple)) and len(res) >= 2:
            width, height = int(res[0]), int(res[1])

        # 알려진 객체 배열 전부 추출 (blocks, detections, transporters, workers, cranes 등)
        key_defaults = (
            ("blocks", "ship_block"),
            ("detections", "object"),
            ("all_detections", "object"),
            ("objects", "object"),
            ("predictions", "object"),
            ("transporters", "transporter"),
            ("vehicles", "transporter"),
            ("workers", "worker"),
            ("persons", "worker"),
            ("cranes", "crane"),
        )
        known_keys = {k for k, _ in key_defaults}
        for k, def_cls in key_defaults:
            if k in sd:
                _scan_detection_list(sd.get(k), default_cls=def_cls, fallback_ocr=default_ocr)

        # 혹시 handler.py가 다른 키 이름의 리스트로 객체를 내려줘도 자동 수집
        for k, val in sd.items():
            if k in known_keys or k in ("image_resolution", "weld_seam_centerline", "torch_bbox"):
                continue
            if isinstance(val, list) and val and isinstance(val[0], dict):
                if any(fld in val[0] for fld in ("class", "class_name", "label", "conf", "confidence", "bbox")):
                    _scan_detection_list(val, default_cls=k.rstrip("s") or "object", fallback_ocr=default_ocr)

    # 케이스 1: 최상위 predictions/detections 리스트가 직접 내려오는 경우
    if isinstance(raw.get("image"), dict):
        width = int(raw["image"].get("width") or width)
        height = int(raw["image"].get("height") or height)
    if isinstance(raw.get("predictions"), list):
        _scan_detection_list(raw.get("predictions"))
    if isinstance(raw.get("detections"), list):
        _scan_detection_list(raw.get("detections"))

    # 케이스 2: Serverless stage payload (transport_data / assembly_data / welding_data)
    if "transport_data" in raw and isinstance(raw.get("transport_data"), dict):
        td = raw["transport_data"]
        top_ocr = str(td.get("block_id") or td.get("ocr_id") or td.get("ocr_text") or "").strip()
        _scan_stage_dict(td, default_ocr=top_ocr)

        # 만약 handler가 transporter/worker를 리스트 대신 boolean/count 요약 키로만 내려준 경우에도 표시
        has_tp = any(p["class"].lower() in ("transporter", "vehicle_transporter", "truck", "vehicle") for p in predictions)
        if not has_tp and (td.get("transporter_detected") or int(td.get("transporter_count") or 0) > 0):
            _add_pred("transporter", float(td.get("transporter_conf") or 1.0))

        has_wk = any(p["class"].lower() in ("worker", "person") for p in predictions)
        wk_cnt = int(td.get("worker_count") or 0)
        if not has_wk and (td.get("worker_detected") or wk_cnt > 0):
            for _ in range(max(1, wk_cnt)):
                _add_pred("worker", float(td.get("worker_conf") or 1.0))

        if not predictions and top_ocr:
            _add_pred("ship_block", float(td.get("ocr_confidence") or 0.0), ocr_text=top_ocr)

    if "assembly_data" in raw and isinstance(raw.get("assembly_data"), dict):
        ad = raw["assembly_data"]
        _scan_stage_dict(ad)
        if ad.get("crane_lifting"):
            _add_pred("crane_lifting", 1.0)

    if "welding_data" in raw and isinstance(raw.get("welding_data"), dict):
        wd = raw["welding_data"]
        _scan_stage_dict(wd)
        if wd.get("seam_detected"):
            pts = wd.get("weld_seam_centerline") or []
            _add_pred(
                "weld_joint",
                1.0,
                extra={
                    "class_ko": f"용접선 (중심점 {wd.get('point_count', len(pts))}개)",
                    "centerline_waypoints": pts,
                },
            )
            # class_ko override
            for p in predictions:
                if p["class"] == "weld_joint":
                    p["class_ko"] = f"용접선 (중심점 {wd.get('point_count', len(pts))}개)"
        if wd.get("torch_detected"):
            _add_pred("torch", 1.0, bbox=wd.get("torch_bbox") or [])
        if wd.get("spark_detected"):
            _add_pred("spark", 1.0)

    # 디버그용 수신 키 로그 (터미널에서 RunPod가 실제 어떤 JSON 키를 보내는지 즉시 확인 가능)
    try:
        sub_keys = {
            k: list(raw[k].keys())
            for k in ("transport_data", "assembly_data", "welding_data")
            if isinstance(raw.get(k), dict)
        }
        print(f"[Vision JSON] stage={stage or 'live'} preds={len(predictions)} top_keys={list(raw.keys())} sub_keys={sub_keys}")
    except Exception:
        pass

    return {
        "image": {"width": width, "height": height},
        "source": raw.get("source") or "LIVE",
        "stage": stage,
        "bay_id": raw.get("bay_id") or "",
        "timestamp": raw.get("timestamp"),
        "predictions": predictions,
    }


def _call_runpod_serverless(stage: str = "assembly", bay_id: str = "", image_base64: str | None = None) -> dict:
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    body: dict = {"input": {"stage": stage, "bay_id": bay_id}}
    if image_base64:
        body["input"]["image_base64"] = image_base64

    res = requests.post(f"{RUNPOD_SERVERLESS_URL}/runsync", headers=headers, json=body, timeout=90)
    res.raise_for_status()
    data = res.json()

    job_id = data.get("id")
    for _ in range(30):
        if data.get("status") not in ("IN_QUEUE", "IN_PROGRESS"):
            break
        time.sleep(2)
        st = requests.get(f"{RUNPOD_SERVERLESS_URL}/status/{job_id}", headers=headers, timeout=30)
        st.raise_for_status()
        data = st.json()

    if data.get("status") != "COMPLETED":
        raise RuntimeError(f"RunPod 추론 실패: {data.get('status')}")

    output = data.get("output") or {}
    if "error" in output:
        raise RuntimeError(str(output["error"]))

    b64_img = output.get("annotated_image_base64") or ""
    img_bytes = base64.b64decode(b64_img) if b64_img else b""
    image_data_url = f"data:image/jpeg;base64,{b64_img}" if b64_img else ""
    b64_vid = output.get("annotated_video_base64") or ""
    video_data_url = output.get("annotated_video_url") or (f"data:video/mp4;base64,{b64_vid}" if b64_vid else "")
    detection_data = _normalize_incoming_json(output.get("payload") or output)

    return {
        "stage": stage,
        "bay_id": bay_id,
        "image_bytes": img_bytes,
        "image_data_url": image_data_url,
        "video_data_url": video_data_url,
        "detection_data": detection_data,
        "is_fetching": False,
        "last_error": "",
        "updated_at": time.time(),
    }


def _fetch_serverless_stage(stage: str, bay_id: str = "", force: bool = False) -> dict:
    target_stage = _resolve_stage(stage)
    with _RUNPOD_STATE_LOCK:
        if (
            not force
            and _RUNPOD_STATE["image_bytes"]
            and _RUNPOD_STATE["stage"] == target_stage
        ):
            return dict(_RUNPOD_STATE)

    with _RUNPOD_FETCH_LOCK:
        with _RUNPOD_STATE_LOCK:
            if (
                not force
                and _RUNPOD_STATE["image_bytes"]
                and _RUNPOD_STATE["stage"] == target_stage
            ):
                return dict(_RUNPOD_STATE)
            _RUNPOD_STATE["is_fetching"] = True
            _RUNPOD_STATE["stage"] = target_stage

        try:
            result = _call_runpod_serverless(stage=target_stage, bay_id=bay_id)
            with _RUNPOD_STATE_LOCK:
                _RUNPOD_STATE.update(result)
                return dict(_RUNPOD_STATE)
        except Exception as e:
            with _RUNPOD_STATE_LOCK:
                _RUNPOD_STATE["is_fetching"] = False
                _RUNPOD_STATE["last_error"] = str(e)
            raise


def _prune_old_disk_files(folder: Path, max_keep: int = MAX_DISK_FILES_PER_FOLDER):
    if not folder.exists():
        return
    try:
        files = sorted(
            (p for p in folder.iterdir() if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old_file in files[max_keep:]:
            old_file.unlink(missing_ok=True)
    except Exception:
        pass


def _extract_valid_ocr_text(predictions: list[dict]) -> str | None:
    for p in predictions:
        raw = str(p.get("block_id") or p.get("ocr_text") or p.get("ocr_id") or "").strip()
        if raw and raw.lower() not in ("scanning...", "none", "null", "-"):
            return raw[:100]
    return None


def _format_detected_objects_ko(dets: list) -> str:
    if not dets:
        return "객체 없음"
    counts: dict[str, int] = {}
    for d in dets:
        raw = (d.class_name or "unknown").lower()
        label = CLASS_LABEL_KO.get(raw, d.class_name or "객체")
        counts[label] = counts.get(label, 0) + 1
    return ", ".join(f"{k} {v}" for k, v in counts.items())


def _query_inspections_paginated(
    page: int = 1,
    per_page: int = RECORDS_PER_PAGE,
    camera_filter: str = "ALL",
    date_filter: str = "",
    keyword: str = "",
    branch_filter: str = "ALL",
) -> dict:
    try:
        page = max(1, int(page))
    except Exception:
        page = 1

    base_q = (
        db.session.query(VisionInspection, VisionAnalysis, WorkOrder)
        .join(VisionAnalysis, VisionInspection.analysis_id == VisionAnalysis.analysis_id)
        .outerjoin(WorkOrder, VisionInspection.work_id == WorkOrder.work_id)
    )
    if camera_filter and camera_filter != "ALL":
        base_q = base_q.filter(VisionAnalysis.camera_code == camera_filter)
    if branch_filter and branch_filter != "ALL":
        base_q = base_q.filter(WorkOrder.work_region == branch_filter)
    if date_filter:
        try:
            d0 = datetime.strptime(date_filter.strip(), "%Y-%m-%d")
            base_q = base_q.filter(
                VisionInspection.created_at >= d0,
                VisionInspection.created_at < d0 + timedelta(days=1),
            )
        except Exception:
            pass
    if keyword:
        kw = f"%{keyword.strip()}%"
        base_q = base_q.filter(
            db.or_(
                VisionAnalysis.camera_code.ilike(kw),
                VisionInspection.ocr_text.ilike(kw),
                WorkOrder.block_code.ilike(kw),
                WorkOrder.work_no.ilike(kw),
            )
        )

    total_count = base_q.count()
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    page = min(page, total_pages)
    rows = (
        base_q.order_by(VisionInspection.inspection_id.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    analysis_ids = [ana.analysis_id for _, ana, _ in rows]
    det_map: dict[int, list] = {}
    if analysis_ids:
        for det in VisionDetection.query.filter(VisionDetection.analysis_id.in_(analysis_ids)).all():
            det_map.setdefault(det.analysis_id, []).append(det)

    items = []
    for insp, ana, wo in rows:
        dets = det_map.get(ana.analysis_id, [])
        items.append({
            "inspection_id": insp.inspection_id,
            "analysis_id": ana.analysis_id,
            "camera_code": ana.camera_code,
            "work_no": wo.work_no if wo else "-",
            "block_code": wo.block_code if wo else "-",
            "ocr_text": insp.ocr_text or "-",
            "detected_summary": _format_detected_objects_ko(dets),
            "safety_clearance_yn": insp.safety_clearance_yn,
            "inspection_status": insp.inspection_status,
            "created_at": insp.created_at.strftime("%Y-%m-%d %H:%M") if insp.created_at else "-",
        })

    return {
        "items": items,
        "page": page,
        "total_pages": total_pages,
        "total_count": total_count,
    }


# ---------------------------------------------------------------------
# 1. 현장 관제 메인 페이지 (/vision/)
# ---------------------------------------------------------------------
@vision_bp.route("/")
def index():
    if "user_id" not in session:
        flash("해당 서비스는 로그인 후 이용하실 수 있습니다.", "warning")
        return redirect(url_for("auth.login"))

    user = User.query.get_or_404(session["user_id"])
    robots = RobotExecution.query.order_by(RobotExecution.robot_exec_id.asc()).all()
    branches = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).all()
    initial_records = _query_inspections_paginated(page=1, per_page=RECORDS_PER_PAGE)

    active_orders_raw = (
        WorkOrder.query
        .filter(WorkOrder.status_code.in_(["IN_PROGRESS", "READY"]))
        .order_by(WorkOrder.work_id.desc())
        .all()
    )
    all_orders_summary = [
        {
            "work_id": wo.work_id,
            "work_no": wo.work_no,
            "work_title": wo.work_title,
            "ship_code": wo.ship_code,
            "block_code": wo.block_code,
            "work_region": wo.work_region,
            "work_zone": wo.work_zone,
            "process_code": wo.process_code,
            "material_grade": wo.material_grade,
            "thickness_mm": float(wo.thickness_mm) if wo.thickness_mm is not None else 0.0,
            "weld_position": wo.weld_position,
            "status_code": wo.status_code,
        }
        for wo in active_orders_raw
    ]

    all_blocks = ErpBlockMaster.query.order_by(ErpBlockMaster.ship_code.asc(), ErpBlockMaster.block_id.asc()).all()
    robot_detail_video = os.getenv("ROBOT_DETAIL_VIDEO", "seg_Test.mp4")

    branch_cards = []
    for br in branches:
        raw_vid = br.default_video or ""
        mapped_vid = VIDEO_FILENAME_ALIASES.get(raw_vid.lower(), raw_vid)
        br_orders = [o for o in all_orders_summary if o["work_region"] == br.branch_code]
        br_blocks = [b for b in all_blocks if b.branch_code == br.branch_code]
        is_ocr_vid = "ocr" in mapped_vid.lower()
        st = "TRANSPORT" if is_ocr_vid else "WORKING"

        if br_orders:
            primary_ship = br_orders[0]["ship_code"]
            primary_block = br_orders[0]["block_code"]
            primary_work_id = br_orders[0]["work_id"]
        elif br_blocks:
            primary_ship = br_blocks[0].ship_code
            primary_block = br_blocks[0].block_id
            primary_work_id = active_orders_raw[0].work_id if active_orders_raw else (_get_default_work_id() or 0)
        else:
            primary_ship = "-"
            primary_block = "-"
            primary_work_id = active_orders_raw[0].work_id if active_orders_raw else (_get_default_work_id() or 0)

        branch_cards.append({
            "branch_code": br.branch_code,
            "branch_name": br.branch_name,
            "region_name": br.region_name,
            "address": br.address or "",
            "camera_code": br.camera_code or f"CAM-{br.branch_code}",
            "default_video": mapped_vid,
            "work_stage": st,
            "ocr_video": mapped_vid if is_ocr_vid else VIDEO_FILENAME_ALIASES.get("video2.mp4", "ocr_test.mp4"),
            "cctv_video": mapped_vid if not is_ocr_vid else VIDEO_FILENAME_ALIASES.get("video1.mp4", "Test_1.mp4"),
            "robot_video": robot_detail_video,
            "primary_ship": primary_ship,
            "primary_block": primary_block,
            "primary_work_id": primary_work_id,
            "active_order_count": len(br_orders),
            "orders": br_orders,
        })

    selected_branch = request.args.get("branch", "").strip()
    if not selected_branch and branch_cards:
        selected_branch = branch_cards[0]["branch_code"]
    current_card = next((c for c in branch_cards if c["branch_code"] == selected_branch), branch_cards[0] if branch_cards else None)
    selected_stage = current_card["work_stage"] if current_card else "WORKING"

    coords = [(float(b.latitude), float(b.longitude)) for b in branches if b.latitude is not None and b.longitude is not None]
    map_center = [
        round(sum(c[0] for c in coords) / len(coords), 4),
        round(sum(c[1] for c in coords) / len(coords), 4),
    ] if coords else [35.18, 129.07]

    return render_template(
        "service/vision.html",
        map_center=map_center,
        me=user,
        initial_records=initial_records,
        robots=robots,
        branches=branches,
        branch_cards=branch_cards,
        current_card=current_card,
        active_orders=all_orders_summary,
        selected_branch=selected_branch,
        selected_stage=selected_stage,
        init_stage="transport" if selected_stage == "TRANSPORT" else "assembly",
        use_serverless=_use_serverless(),
    )


# ---------------------------------------------------------------------
# 2. 야드 및 협력사 위치 지도 API
# ---------------------------------------------------------------------
@vision_bp.route("/api/map-markers", methods=["GET"])
def get_map_markers():
    if "user_id" not in session:
        return jsonify({"status": "ERROR"}), 401

    user = User.query.get_or_404(session["user_id"])
    can_see_partners = (user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER"))

    active_branches = (
        ErpBranch.query
        .filter(ErpBranch.status_code == "ACTIVE", ErpBranch.latitude.isnot(None))
        .order_by(ErpBranch.branch_code.asc())
        .all()
    )

    hq_branches = []
    for br in active_branches:
        raw_vid = br.default_video or ""
        mapped_vid = VIDEO_FILENAME_ALIASES.get(raw_vid.lower(), raw_vid)
        is_ocr_vid = "ocr" in mapped_vid.lower()
        hq_branches.append({
            "branch_code": br.branch_code,
            "branch_name": br.branch_name,
            "region": br.region_name,
            "lat": float(br.latitude),
            "lng": float(br.longitude),
            "address": br.address or "",
            "camera_code": br.camera_code or f"CAM-{br.branch_code}",
            "default_video": mapped_vid,
            "work_stage": "TRANSPORT" if is_ocr_vid else "WORKING",
        })

    partner_markers = []
    if can_see_partners:
        active_hq_contracts = (
            ErpCompany.query
            .filter(ErpCompany.company_type != "HQ", ErpCompany.contract_status == "ACTIVE")
            .all()
        )
        active_company_map = {c.company_code: c for c in active_hq_contracts}
        if active_company_map:
            try:
                pg_locations = (
                    B2BCompanyLocation.query
                    .filter(B2BCompanyLocation.partner_id.in_(active_company_map.keys()))
                    .all()
                )
                daehan_items = B2BDaehanMaterial.query.all()
                donghae_workers = B2BDonghaeWorker.query.all()
                hankook_equips = B2BHankookEquipment.query.all()

                for loc in pg_locations:
                    hq_comp = active_company_map.get(loc.partner_id)
                    my_mats = [m for m in daehan_items if m.partner_id == loc.partner_id]
                    my_workers = [w for w in donghae_workers if w.partner_id == loc.partner_id]
                    my_equips = [e for e in hankook_equips if e.partner_id == loc.partner_id]

                    if my_mats:
                        shared_info = f"가용 자재 {len(my_mats)}종 (" + ", ".join(
                            f"{m.item_name} {int(m.available_qty)}{m.unit}" for m in my_mats[:2]
                        ) + ")"
                    elif my_workers:
                        shared_info = f"파견 가능 인력 {len(my_workers)}명 (" + ", ".join(
                            f"{w.worker_name}[{w.cert_code}]" for w in my_workers[:2]
                        ) + ")"
                    elif my_equips:
                        shared_info = f"대기 장비 {len(my_equips)}대 (" + ", ".join(
                            f"{e.asset_name}" for e in my_equips[:2]
                        ) + ")"
                    else:
                        shared_info = f"전문 분야: {loc.specialty_desc}"

                    partner_markers.append({
                        "company_code": loc.partner_id,
                        "company_name": loc.company_name,
                        "company_type": loc.specialty_desc,
                        "representative": loc.representative,
                        "phone": hq_comp.contact_phone if hq_comp else "-",
                        "contract_period": loc.contract_period,
                        "contract_status": hq_comp.contract_status if hq_comp else "ACTIVE",
                        "address": loc.address,
                        "lat": float(loc.latitude),
                        "lng": float(loc.longitude),
                        "shared_b2b_resources": shared_info,
                    })
            except Exception as e:
                print("[협력사 지도 조회 참고]:", e)

    return jsonify({
        "status": "SUCCESS",
        "can_see_partners": can_see_partners,
        "hq_branches": hq_branches,
        "partners": partner_markers,
    })


# ---------------------------------------------------------------------
# 3. RunPod 영상 스트림 & 실시간 JSON 탐지 프록시 (가짜 폴백 없음)
# ---------------------------------------------------------------------
@vision_bp.route("/video_feed")
def video_feed():
    if _use_serverless():
        try:
            stage = request.args.get("stage") or _RUNPOD_STATE.get("stage") or "assembly"
            bay_id = request.args.get("bay_id") or _get_default_camera_code()
            force = request.args.get("refresh") == "1"
            state = _fetch_serverless_stage(stage=stage, bay_id=bay_id, force=force)
            if state["image_bytes"]:
                resp = Response(state["image_bytes"], mimetype="image/jpeg")
                resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                return resp
        except Exception as e:
            print("[RunPod Serverless 이미지 로드 에러]:", e)
        return Response(status=503)

    if not RUNPOD_STREAM_URL:
        return Response(status=503)

    def proxy_stream():
        req = None
        try:
            req = requests.get(RUNPOD_STREAM_URL, headers=_runpod_headers(), stream=True, timeout=(3, 15), verify=False)
            for chunk in req.iter_content(chunk_size=4096):
                if not chunk:
                    break
                yield chunk
        except GeneratorExit:
            pass
        except Exception as e:
            print("[RunPod 스트림 연결 종료]:", e)
        finally:
            if req is not None:
                try:
                    req.close()
                except Exception:
                    pass

    resp = Response(proxy_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@vision_bp.route("/detections")
def get_detections():
    """RunPod에서 들어온 실시간 JSON을 블로킹 없이 웹 탐지 객체 카드에 그대로 전달"""
    if _use_serverless():
        with _RUNPOD_STATE_LOCK:
            if _RUNPOD_STATE["is_fetching"] and not _RUNPOD_STATE["image_bytes"]:
                return jsonify({
                    "image": {"width": 960, "height": 540},
                    "source": "LOADING",
                    "stage": _RUNPOD_STATE["stage"],
                    "predictions": [],
                })
            det = _RUNPOD_STATE["detection_data"] or {
                "image": {"width": 960, "height": 540},
                "source": "WAIT",
                "stage": _RUNPOD_STATE["stage"],
                "predictions": [],
            }
            if _RUNPOD_STATE["last_error"] and not det.get("predictions"):
                det = dict(det, error=_RUNPOD_STATE["last_error"])
            return jsonify(det)

    if not RUNPOD_JSON_URL:
        return jsonify({"image": {"width": 960, "height": 540}, "source": "OFFLINE", "predictions": []})

    try:
        res = requests.get(RUNPOD_JSON_URL, headers=_runpod_headers(), timeout=2.0, verify=False)
        if res.status_code == 200:
            return jsonify(_normalize_incoming_json(res.json()))
        return jsonify({"image": {"width": 960, "height": 540}, "source": "ERR", "predictions": []})
    except Exception:
        return jsonify({"image": {"width": 960, "height": 540}, "source": "OFFLINE", "predictions": []})


@vision_bp.route("/change_video/<filename>")
def change_video(filename):
    bay_id = request.args.get("bay_id") or _get_default_camera_code()
    if _use_serverless():
        try:
            stage = _resolve_stage(filename)
            state = _fetch_serverless_stage(stage=stage, bay_id=bay_id, force=True)
            with _TRACKER_LOCK:
                _SEEN_JOINT_TRACKER_KEYS.clear()
                _SEEN_OCR_KEYS.clear()
            return jsonify({
                "status": "ok",
                "stage": stage,
                "image_data_url": state.get("image_data_url", ""),
                "video_data_url": state.get("video_data_url", ""),
                "detection_data": state["detection_data"],
            })
        except Exception as e:
            print("[RunPod 스테이지 전환 에러]:", e)
            return jsonify({"status": "error", "message": str(e)}), 500

    if not RUNPOD_CHANGE_VIDEO_URL:
        return jsonify({"status": "error", "message": "RUNPOD_BASE가 설정되지 않았습니다."}), 503

    try:
        mapped_name = VIDEO_FILENAME_ALIASES.get(filename.lower(), filename)
        res = requests.get(f"{RUNPOD_CHANGE_VIDEO_URL}{mapped_name}", headers=_runpod_headers(), timeout=3, verify=False)
        with _TRACKER_LOCK:
            _SEEN_JOINT_TRACKER_KEYS.clear()
            _SEEN_OCR_KEYS.clear()
        return jsonify(res.json())
    except Exception as e:
        return jsonify({"status": "error", "video": filename, "message": str(e)}), 500


@vision_bp.route("/pause")
def toggle_pause():
    if _use_serverless() or not RUNPOD_PAUSE_URL:
        return jsonify({"status": "ok", "is_paused": False})
    try:
        res = requests.get(RUNPOD_PAUSE_URL, headers=_runpod_headers(), timeout=3, verify=False)
        return jsonify(res.json())
    except Exception:
        return jsonify({"status": "error", "is_paused": False})


# ---------------------------------------------------------------------
# 4. DB 중복 방지 기록 저장 시나리오 (그대로 유지 — DB 과적재 차단)
# ---------------------------------------------------------------------
def _insert_vision_records(
    *,
    work_id: int,
    camera_code: str,
    source_uri: str,
    orig_img_uri: str,
    vis_img_uri: str,
    width: int,
    height: int,
    inference_ms: float,
    predictions: list[dict],
) -> int:
    has_arc_flame = any(
        str(p.get("class", "")).lower() in ("flame", "arc", "welding_arc", "spark")
        for p in predictions
    )
    welding_state = "IN_WELDING" if has_arc_flame else "BEFORE_WELD"

    new_analysis = VisionAnalysis(
        work_id=int(work_id) if work_id else 1,
        camera_code=camera_code,
        source_stream_uri=source_uri[:500],
        original_image_uri=orig_img_uri[:500],
        visualized_image_uri=vis_img_uri[:500],
        image_width=int(width or 960),
        image_height=int(height or 540),
        inference_time_ms=float(inference_ms or 0.0),
        welding_state=welding_state,
        status_code="COMPLETED",
    )
    db.session.add(new_analysis)
    db.session.flush()

    interfering = any(
        str(p.get("class", "")).lower() in ("worker", "person", "obstacle")
        for p in predictions
    )
    safety_yn = "N" if interfering else "Y"
    frame_ocr_text = _extract_valid_ocr_text(predictions)

    first_det_id = None
    first_waypoints = None
    det_has_ocr_col = hasattr(VisionDetection, "ocr_text")
    insp_has_ocr_col = hasattr(VisionInspection, "ocr_text")

    for pred in predictions:
        cls_name = str(pred.get("class", "unknown"))
        cls_id = int(pred.get("class_id", 0))
        conf = round(float(pred.get("confidence", 0.0)), 4)
        trk_id = pred.get("tracker_id")
        box = pred.get("box") or {}

        raw_ocr = str(pred.get("block_id") or pred.get("ocr_text") or "").strip()
        pred_ocr = raw_ocr[:100] if raw_ocr and raw_ocr.lower() not in ("scanning...", "none", "null") else None

        waypoints = pred.get("centerline_waypoints")
        waypoints_json_str = (
            json.dumps(waypoints, ensure_ascii=False)
            if cls_name == "weld_joint" and waypoints
            else None
        )

        det_kwargs = dict(
            analysis_id=new_analysis.analysis_id,
            class_id=cls_id,
            class_name=cls_name,
            confidence=conf,
            tracker_id=int(trk_id) if trk_id is not None else None,
            bbox_x=round(float(box.get("x", 0.0)), 2),
            bbox_y=round(float(box.get("y", 0.0)), 2),
            bbox_width=round(float(box.get("width", 0.0)), 2),
            bbox_height=round(float(box.get("height", 0.0)), 2),
            centerline_waypoints_json=waypoints_json_str,
        )
        if det_has_ocr_col:
            det_kwargs["ocr_text"] = pred_ocr

        det_row = VisionDetection(**det_kwargs)
        db.session.add(det_row)
        db.session.flush()

        if first_det_id is None:
            first_det_id = det_row.detection_id

        if cls_name == "weld_joint" and waypoints and len(waypoints) >= 2:
            if trk_id is not None:
                with _TRACKER_LOCK:
                    _SEEN_JOINT_TRACKER_KEYS.add((camera_code, int(trk_id)))
            insp_kwargs = dict(
                analysis_id=new_analysis.analysis_id,
                detection_id=det_row.detection_id,
                work_id=int(work_id) if work_id else 1,
                start_point_json=json.dumps(waypoints[0]),
                end_point_json=json.dumps(waypoints[-1]),
                safety_clearance_yn=safety_yn,
                inspection_status="BEFORE_WELD",
                defect_detail_json=None,
            )
            if insp_has_ocr_col:
                insp_kwargs["ocr_text"] = pred_ocr or frame_ocr_text
            db.session.add(VisionInspection(**insp_kwargs))
            first_waypoints = waypoints

    if first_waypoints is None and first_det_id is not None:
        insp_kwargs = dict(
            analysis_id=new_analysis.analysis_id,
            detection_id=first_det_id,
            work_id=int(work_id) if work_id else 1,
            start_point_json="[0, 0]",
            end_point_json="[0, 0]",
            safety_clearance_yn=safety_yn,
            inspection_status="BEFORE_WELD",
            defect_detail_json=None,
        )
        if insp_has_ocr_col:
            insp_kwargs["ocr_text"] = frame_ocr_text
        db.session.add(VisionInspection(**insp_kwargs))

    if frame_ocr_text:
        with _TRACKER_LOCK:
            _SEEN_OCR_KEYS.add((camera_code, frame_ocr_text))

    db.session.commit()
    return new_analysis.analysis_id


@vision_bp.route("/api/save-snapshot", methods=["POST"])
def save_vision_snapshot():
    """동일 tracker_id / OCR 중복 저장 및 쿨다운 차단 시나리오 유지"""
    payload = request.get_json(silent=True) or {}
    work_id = payload.get("work_id") or _get_default_work_id()
    camera_code = payload.get("camera_code") or "CAM-UNASSIGNED"
    video_name = payload.get("video_name", "live")
    auto_mode = payload.get("auto_mode", True)
    detection_data = payload.get("detection_data") or {}

    if not detection_data or not detection_data.get("predictions"):
        if _use_serverless():
            with _RUNPOD_STATE_LOCK:
                detection_data = _RUNPOD_STATE.get("detection_data") or {}
        elif RUNPOD_JSON_URL:
            try:
                det_res = requests.get(RUNPOD_JSON_URL, headers=_runpod_headers(), timeout=1.5, verify=False)
                if det_res.status_code == 200:
                    detection_data = _normalize_incoming_json(det_res.json())
            except Exception:
                pass

    image_meta = detection_data.get("image") or {"width": 960, "height": 540}
    predictions = detection_data.get("predictions") or []
    if not predictions:
        return jsonify({"status": "SKIPPED", "message": "현재 화면에 탐지된 대상이 없습니다."})

    has_arc_flame = any(
        str(p.get("class", "")).lower() in ("flame", "arc", "welding_arc", "spark")
        for p in predictions
    )
    if auto_mode and has_arc_flame:
        return jsonify({"status": "LOCKED", "welding_state": "IN_WELDING"})

    if not auto_mode:
        now_ts = time.time()
        with _CAMERA_AUTOSAVE_LOCK:
            if now_ts - _LAST_MANUAL_SAVE_TS.get(camera_code, 0.0) < MANUAL_SAVE_COOLDOWN_SEC:
                return jsonify({"status": "THROTTLED", "message": "방금 기록을 남겼습니다. 3초 뒤에 다시 시도해 주세요."})
            _LAST_MANUAL_SAVE_TS[camera_code] = now_ts

    if auto_mode:
        joint_preds = [
            p for p in predictions
            if str(p.get("class", "")).lower() == "weld_joint" and p.get("tracker_id") is not None
        ]
        ocr_text = _extract_valid_ocr_text(predictions)
        with _TRACKER_LOCK:
            has_new_joint = any((camera_code, int(p["tracker_id"])) not in _SEEN_JOINT_TRACKER_KEYS for p in joint_preds)
            has_new_ocr = bool(ocr_text and (camera_code, ocr_text) not in _SEEN_OCR_KEYS)
        if not has_new_joint and not has_new_ocr:
            return jsonify({"status": "DUPLICATE", "message": "이미 기록된 동일 대상입니다."})

        now_ts = time.time()
        with _CAMERA_AUTOSAVE_LOCK:
            if now_ts - _LAST_CAMERA_AUTOSAVE_TS.get(camera_code, 0.0) < AUTOSAVE_COOLDOWN_SEC:
                return jsonify({"status": "THROTTLED", "message": "자동 저장 쿨다운 대기 중"})
            _LAST_CAMERA_AUTOSAVE_TS[camera_code] = now_ts

    VISION_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    snap_name = f"snap_{camera_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    snap_path = VISION_CAPTURE_DIR / snap_name
    img_uri = "/static/captures/placeholder.jpg"

    if _use_serverless():
        with _RUNPOD_STATE_LOCK:
            cached_bytes = _RUNPOD_STATE.get("image_bytes") or b""
        if cached_bytes:
            snap_path.write_bytes(cached_bytes)
            _prune_old_disk_files(VISION_CAPTURE_DIR)
            img_uri = f"/static/captures/{snap_name}"

    try:
        analysis_id = _insert_vision_records(
            work_id=work_id,
            camera_code=camera_code,
            source_uri=f"{RUNPOD_SERVERLESS_URL or RUNPOD_STREAM_URL} ({video_name})",
            orig_img_uri=img_uri,
            vis_img_uri=img_uri,
            width=int(image_meta.get("width") or 960),
            height=int(image_meta.get("height") or 540),
            inference_ms=0.0,
            predictions=predictions,
        )
        return jsonify({"status": "SAVED", "analysis_id": analysis_id})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "ERROR", "message": str(e)}), 500


@vision_bp.route("/api/db-records", methods=["GET"])
def get_vision_db_records():
    if "user_id" not in session:
        return jsonify({"status": "ERROR"}), 401
    page = request.args.get("page", 1, type=int)
    camera = request.args.get("camera", "ALL")
    date_str = request.args.get("date", "")
    keyword = request.args.get("keyword", "")
    branch = request.args.get("branch", "ALL")
    return jsonify(_query_inspections_paginated(
        page=page,
        camera_filter=camera,
        date_filter=date_str,
        keyword=keyword,
        branch_filter=branch,
    ))
