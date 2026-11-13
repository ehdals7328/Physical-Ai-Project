# web/routes/vision_route.py
import json
import math
import requests
import urllib3
from datetime import datetime
from flask import Blueprint, render_template, Response, jsonify, request, session, redirect, url_for, flash

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

# RunPod Proxy URL
RUNPOD_BASE = "https://amu7e870l49mun-5000.proxy.runpod.net"
RUNPOD_STREAM_URL = f"{RUNPOD_BASE}/video_feed"
RUNPOD_JSON_URL = f"{RUNPOD_BASE}/detections"
RUNPOD_PAUSE_URL = f"{RUNPOD_BASE}/pause"
RUNPOD_CHANGE_VIDEO_URL = f"{RUNPOD_BASE}/change_video/"

_SEEN_JOINT_TRACKER_IDS = set()


# ---------------------------------------------------------------------
# 1. 비전 관제 메인 화면 (/vision/)
# ---------------------------------------------------------------------
@vision_bp.route("/")
def index():
    if "user_id" not in session:
        flash("해당 서비스는 로그인 후 이용하실 수 있습니다.", "warning")
        return redirect(url_for("auth.login"))

    user = User.query.get_or_404(session["user_id"])
    recent_inspections = VisionInspection.query.order_by(VisionInspection.inspection_id.desc()).limit(10).all()
    robots = RobotExecution.query.order_by(RobotExecution.robot_exec_id.asc()).all()
    branches = ErpBranch.query.filter_by(status_code="ACTIVE").order_by(ErpBranch.branch_code.asc()).all()

    return render_template(
        "service/vision.html",
        me=user,
        recent_inspections=recent_inspections,
        robots=robots,
        branches=branches,
        runpod_base=RUNPOD_BASE,
    )


# ---------------------------------------------------------------------
# 2. ⭐ 하드코딩 0%! Oracle SHIP_ERP + PostgreSQL B2B View 교차 조회 지도 API
# ---------------------------------------------------------------------
@vision_bp.route("/api/map-markers", methods=["GET"])
def get_map_markers():
    """
    [100% DB 기반 마커 로드]
    1) 파란 마커(본사 야드): Oracle SHIP_ERP.ERP_BRANCHES 에서 STATUS_CODE='ACTIVE' 인 사업장만 조회
    2) 빨간 마커(외주 협력사):
       - Step A: Oracle SHIP_ERP.ERP_COMPANIES 에서 CONTRACT_STATUS='ACTIVE' (계약 중)인 업체 코드 추출
       - Step B: PostgreSQL v_b2b_company_locations 공개 View에서 해당 계약 업체의 좌표/계약기간 조회
       - Step C: PostgreSQL 각 업체별 공개 View(자재/인력/장비)에서 실시간 가용 자원 요약 결합
    """
    if "user_id" not in session:
        return jsonify({"status": "ERROR"}), 401

    user = User.query.get_or_404(session["user_id"])
    can_see_partners = (user.can_access_erp == "Y" or user.role in ("ADMIN", "MANAGER"))

    # 1) 본사 야드 조회 (Oracle SHIP_ERP.ERP_BRANCHES)
    active_branches = (
        ErpBranch.query
        .filter(ErpBranch.status_code == "ACTIVE", ErpBranch.latitude.isnot(None))
        .order_by(ErpBranch.branch_code.asc())
        .all()
    )

    hq_branches = []
    for br in active_branches:
        hq_branches.append({
            "branch_code": br.branch_code,
            "branch_name": br.branch_name,
            "region": br.region_name,
            "lat": float(br.latitude),
            "lng": float(br.longitude),
            "address": br.address or "",
            "camera_code": br.camera_code or f"CAM-{br.branch_code}",
            "default_video": br.default_video or "video1.mp4",
        })

    # 2) 외주 협력사 조회 (Oracle ERP 계약 상태 검증 + PostgreSQL 공개 View 조회)
    partner_markers = []
    if can_see_partners:
        # Step A: 본사 Oracle ERP에서 현재 'ACTIVE'(계약 중)인 외부 협력사 목록만 필터링
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
            # Step B: 외부 PostgreSQL 공개 View(v_b2b_company_locations)에서 계약 중인 업체 위치만 SELECT
            pg_locations = (
                B2BCompanyLocation.query
                .filter(B2BCompanyLocation.partner_id.in_(active_company_map.keys()))
                .all()
            )

            # Step C: 외부 회사가 공개한 가용 자원 View(자재/인력/장비)에서 실시간 가용 데이터 조회
            daehan_items = B2BDaehanMaterial.query.all()
            donghae_workers = B2BDonghaeWorker.query.all()
            hankook_equips = B2BHankookEquipment.query.all()

            for loc in pg_locations:
                hq_comp = active_company_map.get(loc.partner_id)

                # 해당 partner_id가 보유한 자재 / 인력 / 장비 건수를 동적으로 매칭 (하드코딩 0%!)
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
# 3. 페이지 이동/일시정지 시 스레드 렉(Hang) 방지 처리된 스트리밍 프록시
# ---------------------------------------------------------------------
@vision_bp.route("/video_feed")
def video_feed():
    def proxy_stream():
        req = None
        try:
            req = requests.get(RUNPOD_STREAM_URL, stream=True, timeout=(3, 5), verify=False)
            for chunk in req.iter_content(chunk_size=2048):
                if not chunk:
                    break
                yield chunk
        except GeneratorExit:
            pass
        except Exception as e:
            print("RunPod 영상 스트림 연동 종료/에러:", e)
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
            data["source"] = "RUNPOD_LIVE"
            return jsonify(data)
        return jsonify({"image": {"width": 960, "height": 540}, "source": "RUNPOD_ERR", "predictions": []})
    except Exception:
        return jsonify({"image": {"width": 960, "height": 540}, "source": "OFFLINE", "predictions": []})


@vision_bp.route("/pause")
def toggle_pause():
    try:
        res = requests.get(RUNPOD_PAUSE_URL, timeout=2, verify=False)
        return jsonify(res.json())
    except Exception:
        return jsonify({"status": "error", "is_paused": False})


@vision_bp.route("/change_video/<filename>")
def change_video(filename):
    try:
        target_url = f"{RUNPOD_CHANGE_VIDEO_URL}{filename}"
        res = requests.get(target_url, timeout=2, verify=False)
        _SEEN_JOINT_TRACKER_IDS.clear()
        return jsonify(res.json())
    except Exception:
        return jsonify({"status": "error", "video": filename})


# ---------------------------------------------------------------------
# 4. 백그라운드 필수 자동 DB 적재 API (7, 8, 9번 테이블)
# ---------------------------------------------------------------------
@vision_bp.route("/api/save-snapshot", methods=["POST"])
def save_vision_snapshot():
    payload = request.get_json(silent=True) or {}
    work_id = payload.get("work_id") or 1
    camera_code = payload.get("camera_code", "CAM-ULS-BAY1")
    video_name = payload.get("video_name", "video1.mp4")
    auto_mode = payload.get("auto_mode", True)
    detection_data = payload.get("detection_data") or {}

    image_meta = detection_data.get("image") or {"width": 960, "height": 540}
    predictions = detection_data.get("predictions") or []

    if not predictions:
        return jsonify({"status": "SKIPPED"})

    has_arc_flame = any(
        str(p.get("class", "")).lower() in ("flame", "arc", "welding_arc", "spark")
        for p in predictions
    )
    welding_state = "IN_WELDING" if has_arc_flame else "BEFORE_WELD"

    if auto_mode and welding_state == "IN_WELDING":
        return jsonify({"status": "LOCKED", "welding_state": "IN_WELDING"})

    joint_preds = [p for p in predictions if str(p.get("class", "")).lower() == "weld_joint"]
    if auto_mode:
        new_joints = [j for j in joint_preds if j.get("tracker_id") not in _SEEN_JOINT_TRACKER_IDS]
        if not new_joints:
            return jsonify({"status": "SKIPPED"})

    try:
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        new_analysis = VisionAnalysis(
            work_id=int(work_id) if work_id else None,
            camera_code=camera_code,
            source_stream_uri=f"{RUNPOD_STREAM_URL} ({video_name})",
            original_image_uri=f"/static/captures/frame_{now_str}.jpg",
            visualized_image_uri=f"/static/captures/result_{now_str}.jpg",
            image_width=int(image_meta.get("width", 960)),
            image_height=int(image_meta.get("height", 540)),
            inference_time_ms=28.5,
            welding_state=welding_state,
            status_code="COMPLETED",
        )
        db.session.add(new_analysis)
        db.session.flush()

        for pred in predictions:
            cls_name = str(pred.get("class", "unknown"))
            cls_id = int(pred.get("class_id", 0))
            conf = round(float(pred.get("confidence", 0.0)), 4)
            trk_id = pred.get("tracker_id")
            box = pred.get("box") or {}

            waypoints = pred.get("centerline_waypoints")
            waypoints_json_str = (
                json.dumps(waypoints, ensure_ascii=False)
                if cls_name == "weld_joint" and waypoints
                else None
            )

            det_row = VisionDetection(
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
            db.session.add(det_row)
            db.session.flush()

            if cls_name == "weld_joint" and waypoints and len(waypoints) >= 2:
                if trk_id is not None:
                    _SEEN_JOINT_TRACKER_IDS.add(trk_id)

                start_pt = waypoints[0]
                end_pt = waypoints[-1]
                px_dist = math.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
                est_len_mm = round(px_dist * 1.95, 2)

                insp_row = VisionInspection(
                    analysis_id=new_analysis.analysis_id,
                    detection_id=det_row.detection_id,
                    work_id=int(work_id) if work_id else 1,
                    start_point_json=json.dumps(start_pt),
                    end_point_json=json.dumps(end_pt),
                    estimated_length_mm=est_len_mm,
                    safety_clearance_yn="Y",
                    inspection_status="BEFORE_WELD",
                    defect_detail_json=None,
                )
                db.session.add(insp_row)

        db.session.commit()
        return jsonify({"status": "SUCCESS", "analysis_id": new_analysis.analysis_id})

    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "ERROR", "message": str(e)}), 500


# ---------------------------------------------------------------------
# 5. 하단 기록 조회 API (안전판정 제외 & 촬영 날짜/시간 YYYY-MM-DD HH:MM:SS 포함)
# ---------------------------------------------------------------------
@vision_bp.route("/api/db-records", methods=["GET"])
def get_vision_db_records():
    inspections = VisionInspection.query.order_by(VisionInspection.inspection_id.desc()).limit(10).all()

    inspection_list = []
    for i in inspections:
        analysis = VisionAnalysis.query.get(i.analysis_id)
        dets = VisionDetection.query.filter_by(analysis_id=i.analysis_id).all()
        detected_classes = [f"{d.class_name}({int(float(d.confidence)*100)}%)" for d in dets]

        recorded_time = (
            i.created_at.strftime("%Y-%m-%d %H:%M:%S")
            if i.created_at
            else (analysis.created_at.strftime("%Y-%m-%d %H:%M:%S") if analysis and analysis.created_at else "-")
        )

        inspection_list.append({
            "inspection_id": i.inspection_id,
            "analysis_id": i.analysis_id,
            "camera_code": analysis.camera_code if analysis else "CAM-ULS-BAY1",
            "start_point": i.start_point_json,
            "end_point": i.end_point_json,
            "length_mm": float(i.estimated_length_mm or 0),
            "detected_objects_str": ", ".join(detected_classes[:5]) if detected_classes else "weld_joint",
            "created_at": recorded_time,
        })

    return jsonify({
        "status": "SUCCESS",
        "inspections": inspection_list,
    })