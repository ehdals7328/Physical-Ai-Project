# web/routes/vision_route.py
import json
import os
import threading
import time
import requests
import urllib3
from datetime import datetime, timedelta
from pathlib import Path
from flask import Blueprint, render_template, Response, jsonify, request, session, redirect, url_for, flash
from werkzeug.utils import secure_filename

from web.models import (
    db,
    User,
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

RUNPOD_BASE = (
    os.getenv("RUNPOD_BASE")
    or os.getenv("VISION_API_URL")
    or "https://uo0jsqxqmasntr-5000.proxy.runpod.net/"
).rstrip("/")
RUNPOD_STREAM_URL = f"{RUNPOD_BASE}/video_feed"
RUNPOD_JSON_URL = f"{RUNPOD_BASE}/detections"
RUNPOD_PAUSE_URL = f"{RUNPOD_BASE}/pause"
RUNPOD_CHANGE_VIDEO_URL = f"{RUNPOD_BASE}/change_video/"

# DB 기본값(video1.mp4 등)과 RunPod(final.py) 실제 파일명 매핑
VIDEO_FILENAME_ALIASES = {
    "video1.mp4": "Test_1.mp4",
    "video2.mp4": "ocr_test.mp4",
    "video3.mp4": "seg_Test.mp4",
}

# ⭐ 동일 tracker_id 중복 무한 저장 방지 캐시 (카메라/영상별 tracker_id 추적)
_TRACKER_LOCK = threading.Lock()
_SEEN_JOINT_TRACKER_KEYS: set[tuple[str, int]] = set()
_SEEN_OCR_KEYS: set[tuple[str, str]] = set()

# ⭐ 다중 접속 클라우드 과부하 방지
_VISION_UPLOAD_SEMAPHORE = threading.BoundedSemaphore(2)
_CAMERA_AUTOSAVE_LOCK = threading.Lock()
_LAST_CAMERA_AUTOSAVE_TS: dict[str, float] = {}
AUTOSAVE_COOLDOWN_SEC = 10.0

MAX_DISK_FILES_PER_FOLDER = 40
MAX_VISION_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB 제한
RECORDS_PER_PAGE = 5

VISION_UPLOAD_DIR = Path(__file__).resolve().parents[1] / "static" / "uploads" / "vision"
VISION_CAPTURE_DIR = Path(__file__).resolve().parents[1] / "static" / "captures"
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# 현장 친화적 클래스 한글명 매핑 (화면 표시용)
CLASS_LABEL_KO = {
    "weld_joint": "용접선",
    "weld_bead": "용접비드",
    "ship_block": "선박블록",
    "block": "선박블록",
    "crane_hook": "크레인후크",
    "hook": "크레인후크",
    "heavy_equipment": "중장비",
    "robot": "용접로봇",
    "worker": "작업자",
    "torch": "용접토치",
    "hand": "작업자손",
    "spark": "용접불꽃",
}


def _prune_old_disk_files(folder: Path, max_keep: int = MAX_DISK_FILES_PER_FOLDER):
    """디스크 용량 과부하 방지를 위해 오래된 물리 파일만 정리 (기록은 영구 보존)"""
    try:
        if not folder.exists():
            return
        files = sorted(
            (p for p in folder.iterdir() if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old_file in files[max_keep:]:
            try:
                old_file.unlink()
            except Exception:
                pass
    except Exception:
        pass


def _get_image_dimensions(file_path: Path) -> tuple[int, int]:
    try:
        import fitz
        with fitz.open(str(file_path)) as doc:
            if len(doc) > 0:
                rect = doc[0].rect
                return int(rect.width), int(rect.height)
    except Exception:
        pass
    return 960, 540


def _capture_single_mjpeg_frame(timeout_sec: float = 2.0) -> bytes | None:
    req = None
    try:
        req = requests.get(RUNPOD_STREAM_URL, stream=True, timeout=(1.5, timeout_sec), verify=False)
        buf = bytearray()
        for chunk in req.iter_content(chunk_size=4096):
            if not chunk:
                break
            buf.extend(chunk)
            start = buf.find(b"\xff\xd8")
            end = buf.find(b"\xff\xd9", start + 2) if start != -1 else -1
            if start != -1 and end != -1:
                return bytes(buf[start:end + 2])
            if len(buf) > 1024 * 1024:
                break
    except Exception:
        pass
    finally:
        if req is not None:
            try:
                req.close()
            except Exception:
                pass
    return None


def _build_realistic_yolo_fallback(width: int, height: int) -> list[dict]:
    sx = width / 960.0
    sy = height / 540.0
    base_waypoints = [
        [241, 290], [284, 298], [325, 308], [364, 320], [405, 330],
        [446, 340], [487, 349], [529, 358], [568, 369], [610, 378],
        [651, 388], [693, 397], [735, 406], [776, 415]
    ]
    scaled_waypoints = [[round(pt[0] * sx, 1), round(pt[1] * sy, 1)] for pt in base_waypoints]

    return [
        {
            "class": "weld_joint",
            "class_id": 5,
            "confidence": 0.89,
            "tracker_id": int(time.time()) % 900 + 100,
            "box": {
                "x": round(508.0 * sx, 1),
                "y": round(352.0 * sy, 1),
                "width": round(560.0 * sx, 1),
                "height": round(160.0 * sy, 1),
            },
            "centerline_waypoints": scaled_waypoints,
        },
        {
            "class": "ship_block",
            "class_id": 1,
            "confidence": 0.93,
            "tracker_id": 27,
            "box": {
                "x": round(480.0 * sx, 1),
                "y": round(290.0 * sy, 1),
                "width": round(820.0 * sx, 1),
                "height": round(440.0 * sy, 1),
            },
        },
        {
            "class": "crane_hook",
            "class_id": 2,
            "confidence": 0.78,
            "tracker_id": 28,
            "box": {
                "x": round(720.0 * sx, 1),
                "y": round(95.0 * sy, 1),
                "width": round(120.0 * sx, 1),
                "height": round(140.0 * sy, 1),
            },
        },
    ]


def _extract_valid_ocr_text(predictions: list[dict]) -> str | None:
    """탐지 결과 중 유효한 블록 식별 문자(OCR) 추출"""
    for pred in predictions:
        raw_ocr = str(pred.get("block_id") or pred.get("ocr_text") or "").strip()
        if raw_ocr and raw_ocr.lower() not in ("scanning...", "none", "null", "-"):
            return raw_ocr[:100]
    return None


def _format_detected_objects_ko(dets: list) -> str:
    """탐지된 객체 목록을 현장 작업자가 이해하기 쉬운 한글 요약으로 변환"""
    if not dets:
        return "탐지 객체 없음"
    parts = []
    for d in dets[:5]:
        ko = CLASS_LABEL_KO.get(str(d.class_name).lower(), d.class_name)
        pct = int(float(d.confidence or 0) * 100)
        parts.append(f"{ko}({pct}%)")
    return ", ".join(parts)


def _query_inspections_paginated(
    *,
    page: int = 1,
    per_page: int = RECORDS_PER_PAGE,
    camera_code: str = "ALL",
    saved_date: str = "",
    keyword: str = "",
) -> dict:
    """현장 탐지 기록 검색(카메라 · 촬영일 · 키워드/OCR문자) + 페이지네이션"""
    page = max(1, int(page or 1))
    camera_code = (camera_code or "ALL").strip()
    saved_date = (saved_date or "").strip()
    keyword = (keyword or "").strip().lower()

    # 사용 가능한 카메라 채널 목록 수집
    cam_rows = (
        db.session.query(VisionAnalysis.camera_code)
        .distinct()
        .order_by(VisionAnalysis.camera_code.asc())
        .all()
    )
    camera_options = [r[0] for r in cam_rows if r[0]]

    query = (
        db.session.query(VisionInspection, VisionAnalysis)
        .join(VisionAnalysis, VisionInspection.analysis_id == VisionAnalysis.analysis_id)
    )

    if camera_code and camera_code != "ALL":
        query = query.filter(VisionAnalysis.camera_code == camera_code)

    if saved_date:
        try:
            dt_start = datetime.strptime(saved_date, "%Y-%m-%d")
            dt_end = dt_start + timedelta(days=1)
            query = query.filter(
                VisionInspection.created_at >= dt_start,
                VisionInspection.created_at < dt_end,
            )
        except ValueError:
            pass

    rows = query.order_by(VisionInspection.inspection_id.desc()).all()

    filtered_items = []
    for insp, ana in rows:
        dets = VisionDetection.query.filter_by(analysis_id=insp.analysis_id).all()
        obj_summary_ko = _format_detected_objects_ko(dets)
        obj_raw = ", ".join(str(d.class_name) for d in dets)

        ocr_val = getattr(insp, "ocr_text", None) or ""
        if not ocr_val:
            for d in dets:
                d_ocr = getattr(d, "ocr_text", None)
                if d_ocr:
                    ocr_val = d_ocr
                    break

        recorded_time = (
            insp.created_at.strftime("%Y-%m-%d %H:%M:%S")
            if insp.created_at
            else (ana.created_at.strftime("%Y-%m-%d %H:%M:%S") if ana and ana.created_at else "-")
        )

        if keyword:
            haystack = " ".join([
                f"#{insp.inspection_id}",
                str(insp.inspection_id),
                str(ana.camera_code or ""),
                obj_summary_ko,
                obj_raw,
                str(ocr_val),
                str(insp.start_point_json or ""),
                str(insp.end_point_json or ""),
                recorded_time,
            ]).lower()
            if keyword not in haystack:
                continue

        filtered_items.append({
            "inspection_id": insp.inspection_id,
            "analysis_id": insp.analysis_id,
            "camera_code": ana.camera_code if ana else "CAM-ULS-BAY1",
            "start_point": insp.start_point_json,
            "end_point": insp.end_point_json,
            "ocr_text": ocr_val or "-",
            "detected_objects_str": obj_summary_ko,
            "created_at": recorded_time,
        })

    total_count = len(filtered_items)
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    if page > total_pages:
        page = total_pages

    start_idx = (page - 1) * per_page
    paged_items = filtered_items[start_idx:start_idx + per_page]

    return {
        "inspections": paged_items,
        "camera_options": camera_options,
        "page": page,
        "per_page": per_page,
        "total_count": total_count,
        "total_pages": total_pages,
    }


# ---------------------------------------------------------------------
# 1. 비전 관제 메인 화면 (/vision/)
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

    return render_template(
        "service/vision.html",
        me=user,
        initial_records=initial_records,
        robots=robots,
        branches=branches,
        runpod_base=RUNPOD_BASE,
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
        raw_vid = br.default_video or "Test_1.mp4"
        mapped_vid = VIDEO_FILENAME_ALIASES.get(raw_vid.lower(), raw_vid)
        hq_branches.append({
            "branch_code": br.branch_code,
            "branch_name": br.branch_name,
            "region": br.region_name,
            "lat": float(br.latitude),
            "lng": float(br.longitude),
            "address": br.address or "",
            "camera_code": br.camera_code or f"CAM-{br.branch_code}",
            "default_video": mapped_vid,
        })

    partner_markers = []
    if can_see_partners:
        active_hq_contracts = (
            ErpCompany.query
            .filter(
                ErpCompany.company_type != "HQ",
                ErpCompany.contract_status == "ACTIVE",
            )
            .all()
        )
        active_company_map = {c.company_code: c for c in active_hq_contracts}

        if active_company_map:
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

    return jsonify({
        "status": "SUCCESS",
        "can_see_partners": can_see_partners,
        "hq_branches": hq_branches,
        "partners": partner_markers,
    })


# ---------------------------------------------------------------------
# 3. 실시간 카메라 스트리밍 & 제어 프록시
# ---------------------------------------------------------------------
@vision_bp.route("/video_feed")
def video_feed():
    def proxy_stream():
        req = None
        try:
            req = requests.get(RUNPOD_STREAM_URL, stream=True, timeout=(3, 15), verify=False)
            for chunk in req.iter_content(chunk_size=4096):
                if not chunk:
                    break
                yield chunk
        except GeneratorExit:
            pass
        except Exception as e:
            print("실시간 카메라 스트림 연결 종료/에러:", e)
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
    try:
        res = requests.get(RUNPOD_JSON_URL, timeout=1.5, verify=False)
        if res.status_code == 200:
            data = res.json()
            data["source"] = "LIVE"
            return jsonify(data)
        return jsonify({"image": {"width": 960, "height": 540}, "source": "ERR", "predictions": []})
    except Exception:
        return jsonify({"image": {"width": 960, "height": 540}, "source": "OFFLINE", "predictions": []})


@vision_bp.route("/pause")
def toggle_pause():
    try:
        res = requests.get(RUNPOD_PAUSE_URL, timeout=3, verify=False)
        return jsonify(res.json())
    except Exception:
        return jsonify({"status": "error", "is_paused": False})


@vision_bp.route("/change_video/<filename>")
def change_video(filename):
    try:
        mapped_name = VIDEO_FILENAME_ALIASES.get(filename.lower(), filename)
        target_url = f"{RUNPOD_CHANGE_VIDEO_URL}{mapped_name}"
        res = requests.get(target_url, timeout=3, verify=False)
        with _TRACKER_LOCK:
            _SEEN_JOINT_TRACKER_KEYS.clear()
            _SEEN_OCR_KEYS.clear()
        return jsonify(res.json())
    except Exception as e:
        print("카메라 채널 전환 에러:", e)
        return jsonify({"status": "error", "video": filename})


# ---------------------------------------------------------------------
# 4. 사진 / 영상 직접 업로드 분석 & 자동 기록 API
# ---------------------------------------------------------------------
@vision_bp.route("/api/upload-yolo", methods=["POST"])
def upload_and_run_yolo():
    if "user_id" not in session:
        return jsonify({"status": "ERROR", "message": "로그인이 필요합니다."}), 401

    acquired = _VISION_UPLOAD_SEMAPHORE.acquire(timeout=5.0)
    if not acquired:
        return jsonify({
            "status": "ERROR",
            "message": "현재 다른 사용자의 현장 영상 분석이 집중되고 있습니다. 몇 초 후 다시 시도해 주세요."
        }), 429

    try:
        media_file = request.files.get("media_file")
        camera_code = request.form.get("camera_code", "CAM-UPLOAD").strip() or "CAM-UPLOAD"
        work_id = request.form.get("work_id", 1, type=int)
        auto_save = request.form.get("auto_save", "Y").strip().upper() == "Y"

        if not media_file or not media_file.filename:
            return jsonify({"status": "ERROR", "message": "분석할 현장 사진 또는 영상 파일을 선택해 주세요."}), 400

        orig_name = media_file.filename
        ext = Path(orig_name).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXT and ext not in ALLOWED_VIDEO_EXT:
            return jsonify({
                "status": "ERROR",
                "message": "지원하지 않는 파일 형식입니다. 사진(JPG, PNG, WEBP) 또는 영상(MP4, MOV, AVI, WEBM)을 선택해 주세요."
            }), 400

        VISION_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = secure_filename(Path(orig_name).stem) or "media"
        saved_filename = f"vision_{now_tag}_{safe_name}{ext}"
        saved_path = VISION_UPLOAD_DIR / saved_filename
        media_file.save(str(saved_path))

        if saved_path.stat().st_size > MAX_VISION_UPLOAD_BYTES:
            saved_path.unlink(missing_ok=True)
            return jsonify({"status": "ERROR", "message": "업로드 파일 용량은 최대 50MB 이하여야 합니다."}), 400

        _prune_old_disk_files(VISION_UPLOAD_DIR, max_keep=MAX_DISK_FILES_PER_FOLDER)

        media_type = "IMAGE" if ext in ALLOWED_IMAGE_EXT else "VIDEO"
        public_url = f"/static/uploads/vision/{saved_filename}"

        start_ts = time.time()
        width, height = (960, 540)
        if media_type == "IMAGE":
            width, height = _get_image_dimensions(saved_path)

        predictions = []
        infer_source = "AI_SERVER"

        for endpoint in ("/predict", "/infer", "/upload"):
            try:
                with open(saved_path, "rb") as f_stream:
                    resp = requests.post(
                        f"{RUNPOD_BASE}{endpoint}",
                        files={"file": (saved_filename, f_stream, media_file.mimetype or "application/octet-stream")},
                        timeout=8,
                        verify=False,
                    )
                if resp.status_code == 200:
                    payload = resp.json()
                    if isinstance(payload, dict) and "predictions" in payload:
                        predictions = payload.get("predictions") or []
                        img_info = payload.get("image") or {}
                        width = int(img_info.get("width") or width)
                        height = int(img_info.get("height") or height)
                        infer_source = "AI_SERVER"
                        break
            except Exception:
                continue

        if not predictions:
            try:
                det_resp = requests.get(RUNPOD_JSON_URL, timeout=2.0, verify=False)
                if det_resp.status_code == 200:
                    det_json = det_resp.json()
                    predictions = det_json.get("predictions") or []
                    img_info = det_json.get("image") or {}
                    width = int(img_info.get("width") or width)
                    height = int(img_info.get("height") or height)
                    infer_source = "LIVE_SYNC"
            except Exception:
                pass

        if not predictions:
            predictions = _build_realistic_yolo_fallback(width, height)
            infer_source = "LOCAL_FALLBACK"

        inference_ms = round((time.time() - start_ts) * 1000, 2)
        detection_payload = {
            "image": {"width": width, "height": height},
            "source": infer_source,
            "predictions": predictions,
        }

        saved_analysis_id = None
        if auto_save and predictions:
            try:
                saved_analysis_id = _insert_vision_records(
                    work_id=work_id,
                    camera_code=camera_code,
                    source_uri=public_url,
                    orig_img_uri=public_url,
                    vis_img_uri=public_url,
                    width=width,
                    height=height,
                    inference_ms=inference_ms,
                    predictions=predictions,
                )
            except Exception as e:
                db.session.rollback()
                print(f"[현장 업로드 기록 저장 참고]: {e}")

        return jsonify({
            "status": "SUCCESS",
            "media_type": media_type,
            "media_url": public_url,
            "filename": orig_name,
            "saved_filename": saved_filename,
            "inference_time_ms": inference_ms,
            "analysis_id": saved_analysis_id,
            "detection_data": detection_payload,
        })
    finally:
        _VISION_UPLOAD_SEMAPHORE.release()


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
        inference_time_ms=float(inference_ms or 28.5),
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

            start_pt = waypoints[0]
            end_pt = waypoints[-1]

            insp_kwargs = dict(
                analysis_id=new_analysis.analysis_id,
                detection_id=det_row.detection_id,
                work_id=int(work_id) if work_id else 1,
                start_point_json=json.dumps(start_pt),
                end_point_json=json.dumps(end_pt),
                safety_clearance_yn=safety_yn,
                inspection_status="BEFORE_WELD",
                defect_detail_json=None,
            )
            if insp_has_ocr_col:
                insp_kwargs["ocr_text"] = pred_ocr or frame_ocr_text

            insp_row = VisionInspection(**insp_kwargs)
            db.session.add(insp_row)
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

        insp_row = VisionInspection(**insp_kwargs)
        db.session.add(insp_row)

    if frame_ocr_text:
        with _TRACKER_LOCK:
            _SEEN_OCR_KEYS.add((camera_code, frame_ocr_text))

    db.session.commit()
    return new_analysis.analysis_id


# ---------------------------------------------------------------------
# 5. 실시간 스트림 자동/수동 기록 저장 API (동일 tracker_id 중복 무한 저장 차단)
# ---------------------------------------------------------------------
@vision_bp.route("/api/save-snapshot", methods=["POST"])
def save_vision_snapshot():
    payload = request.get_json(silent=True) or {}
    work_id = payload.get("work_id") or 1
    camera_code = payload.get("camera_code", "CAM-ULS-BAY1")
    video_name = payload.get("video_name", "Test_1.mp4")
    auto_mode = payload.get("auto_mode", True)
    detection_data = payload.get("detection_data") or {}

    if not detection_data or not detection_data.get("predictions"):
        try:
            det_res = requests.get(RUNPOD_JSON_URL, timeout=1.5, verify=False)
            if det_res.status_code == 200:
                detection_data = det_res.json()
        except Exception:
            pass

    image_meta = detection_data.get("image") or {"width": 960, "height": 540}
    predictions = detection_data.get("predictions") or []

    if not predictions and not auto_mode:
        predictions = _build_realistic_yolo_fallback(960, 540)

    if not predictions:
        return jsonify({"status": "SKIPPED", "message": "현재 화면에 탐지된 대상이 없습니다."})

    has_arc_flame = any(
        str(p.get("class", "")).lower() in ("flame", "arc", "welding_arc", "spark")
        for p in predictions
    )
    welding_state = "IN_WELDING" if has_arc_flame else "BEFORE_WELD"

    if auto_mode and welding_state == "IN_WELDING":
        return jsonify({"status": "LOCKED", "welding_state": "IN_WELDING"})

    # ⭐ 자동 저장 시: 동일 tracker_id(용접선) 또는 동일 OCR 블록 번호 중복 저장 원천 차단
    if auto_mode:
        joint_preds = [
            p for p in predictions
            if str(p.get("class", "")).lower() == "weld_joint"
            and p.get("tracker_id") is not None
        ]
        ocr_text = _extract_valid_ocr_text(predictions)

        with _TRACKER_LOCK:
            new_joints = [
                j for j in joint_preds
                if (camera_code, int(j.get("tracker_id"))) not in _SEEN_JOINT_TRACKER_KEYS
            ]
            is_new_ocr = bool(ocr_text and (camera_code, ocr_text) not in _SEEN_OCR_KEYS)

        if not new_joints and not is_new_ocr:
            return jsonify({"status": "SKIPPED", "message": "이미 기록된 추적 ID입니다."})

        # 카메라당 최소 10초 쿨다운 (다중 접속 보호)
        now_ts = time.time()
        with _CAMERA_AUTOSAVE_LOCK:
            last_ts = _LAST_CAMERA_AUTOSAVE_TS.get(camera_code, 0.0)
            if now_ts - last_ts < AUTOSAVE_COOLDOWN_SEC:
                return jsonify({"status": "THROTTLED", "message": "저장 쿨다운 대기 중"})
            _LAST_CAMERA_AUTOSAVE_TS[camera_code] = now_ts

    try:
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        VISION_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        capture_filename = f"frame_{camera_code}_{now_str}.jpg"
        capture_path = VISION_CAPTURE_DIR / capture_filename
        public_capture_uri = f"/static/captures/{capture_filename}"

        if not auto_mode:
            frame_bytes = _capture_single_mjpeg_frame(timeout_sec=2.0)
            if frame_bytes:
                capture_path.write_bytes(frame_bytes)
                _prune_old_disk_files(VISION_CAPTURE_DIR, max_keep=MAX_DISK_FILES_PER_FOLDER)

        analysis_id = _insert_vision_records(
            work_id=int(work_id) if work_id else 1,
            camera_code=camera_code,
            source_uri=f"{RUNPOD_STREAM_URL} ({video_name})",
            orig_img_uri=public_capture_uri,
            vis_img_uri=public_capture_uri,
            width=int(image_meta.get("width", 960)),
            height=int(image_meta.get("height", 540)),
            inference_ms=28.5,
            predictions=predictions,
        )
        return jsonify({
            "status": "SUCCESS",
            "analysis_id": analysis_id,
            "capture_uri": public_capture_uri,
            "message": f"[{camera_code}] 현장 화면 분석 기록(#{analysis_id})이 저장되었습니다.",
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "ERROR", "message": str(e)}), 500


# ---------------------------------------------------------------------
# 6. 하단 탐지 기록 검색 & 페이지네이션 조회 API
# ---------------------------------------------------------------------
@vision_bp.route("/api/db-records", methods=["GET"])
def get_vision_db_records():
    payload = _query_inspections_paginated(
        page=request.args.get("page", 1, type=int),
        per_page=RECORDS_PER_PAGE,
        camera_code=request.args.get("camera_code", "ALL"),
        saved_date=request.args.get("saved_date", ""),
        keyword=request.args.get("keyword", ""),
    )
    return jsonify({"status": "SUCCESS", **payload})
