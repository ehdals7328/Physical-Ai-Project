import base64
import os
import threading
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from flask import Blueprint, Response, jsonify, render_template, request

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path, override=True)

RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "olau5y89ddpw64").strip()
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "").strip()
print(f"[DEBUG] 로드된 ENDPOINT: {RUNPOD_ENDPOINT_ID}, API_KEY 앞자리: {RUNPOD_API_KEY[:8]}...")

bp = Blueprint("vision", __name__, url_prefix="/vision")

RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "olau5y89ddpw64")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_RUNSYNC_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"

# 기존 DB의 파일명(Test_1.mp4 등)과 신규 샘플명(test1~3) 모두 매핑
STAGE_MAP = {
    "test1.png": "transport",
    "test1": "transport",
    "transport": "transport",
    "test_1.mp4": "assembly",
    "test2.png": "assembly",
    "test2.mp4": "assembly",
    "test2": "assembly",
    "assembly": "assembly",
    "test3.png": "welding",
    "test3.mp4": "welding",
    "test3": "welding",
    "welding": "welding",
}

_state_lock = threading.Lock()
_vision_state = {
    "stage": "transport",
    "is_paused": False,
    "latest_detection": {"source": "WAIT", "predictions": []},
    "latest_jpeg": None,
}


def convert_payload_to_frontend_format(payload: dict) -> dict:
    """Serverless payload를 vision.html과 save-snapshot이 기대하는 predictions 구조로 변환"""
    stage = payload.get("stage", "TRANSPORT")
    predictions = []
    width, height = 1920, 1080

    if stage == "TRANSPORT":
        td = payload.get("transport_data", {})
        res = td.get("image_resolution", [1920, 1080])
        width, height = res[0], res[1]
        for blk in td.get("blocks", []):
            predictions.append({
                "class": "ship_block",
                "confidence": blk.get("conf", 0.0),
                "bbox": blk.get("bbox", []),
                "ocr_text": blk.get("ocr_id") or td.get("block_id") or "",
            })

    elif stage == "ASSEMBLY":
        ad = payload.get("assembly_data", {})
        res = ad.get("image_resolution", [1920, 1080])
        width, height = res[0], res[1]
        for det in ad.get("detections", []):
            predictions.append({
                "class": det.get("class", "object"),
                "confidence": det.get("conf", 0.0),
                "bbox": det.get("bbox", []),
            })
        if ad.get("crane_lifting"):
            predictions.append({"class": "crane_lifting (인양 중)", "confidence": 1.0})

    elif stage == "WELDING":
        wd = payload.get("welding_data", {})
        res = wd.get("image_resolution", [1920, 1080])
        width, height = res[0], res[1]
        if wd.get("seam_detected"):
            predictions.append({
                "class": f"weld_seam (중심점 {wd.get('point_count', 0)}개)",
                "confidence": 1.0,
                "centerline": wd.get("weld_seam_centerline", []),
            })
        if wd.get("torch_detected"):
            predictions.append({
                "class": "torch",
                "confidence": 1.0,
                "bbox": wd.get("torch_bbox", []),
            })
        if wd.get("spark_detected"):
            predictions.append({"class": "spark", "confidence": 1.0})

    return {
        "source": "LIVE",
        "stage": stage,
        "bay_id": payload.get("bay_id", "BAY-01"),
        "timestamp": payload.get("timestamp"),
        "image": {"width": width, "height": height},
        "predictions": predictions,
        "raw_payload": payload,  # DB 및 AI Agent용 원본 좌표 데이터 보존
    }


def call_runpod_serverless(stage: str, image_base64: str = None, bay_id: str = "BAY-01") -> dict:
    if not RUNPOD_API_KEY:
        raise RuntimeError(".env 파일에 RUNPOD_API_KEY가 설정되지 않았습니다.")

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "input": {
            "stage": stage,
            "bay_id": bay_id,
            "image_base64": image_base64,
        }
    }

    res = requests.post(RUNPOD_RUNSYNC_URL, headers=headers, json=body, timeout=90)
    res.raise_for_status()
    result = res.json()

    if result.get("status") != "COMPLETED":
        raise RuntimeError(f"RunPod 추론 실패: {result}")

    output = result.get("output", {})
    if "error" in output:
        raise RuntimeError(output["error"])

    payload = output.get("payload", {})
    img_b64 = output.get("annotated_image_base64", "")
    jpeg_bytes = base64.b64decode(img_b64) if img_b64 else None
    frontend_data = convert_payload_to_frontend_format(payload)

    with _state_lock:
        _vision_state["stage"] = stage
        _vision_state["latest_detection"] = frontend_data
        if jpeg_bytes:
            _vision_state["latest_jpeg"] = jpeg_bytes

    return {
        "status": "ok",
        "detection": frontend_data,
        "annotated_image_base64": img_b64,
    }


@bp.route("/")
def index():
    return render_template("vision.html")


@bp.route("/infer", methods=["POST"])
def infer_vision():
    """기본 샘플 버튼 클릭 또는 사용자 이미지 업로드 추론 엔드포인트"""
    try:
        image_b64 = None
        if request.files and "file" in request.files:
            file_storage = request.files["file"]
            raw_stage = request.form.get("stage", "welding").lower()
            bay_id = request.form.get("bay_id", "BAY-01")
            image_b64 = base64.b64encode(file_storage.read()).decode("utf-8")
        else:
            data = request.get_json(silent=True) or {}
            raw_stage = data.get("stage", "transport").lower()
            bay_id = data.get("bay_id", "BAY-01")
            image_b64 = data.get("image_base64")

        stage = STAGE_MAP.get(raw_stage, "transport")
        result = call_runpod_serverless(stage=stage, image_base64=image_b64, bay_id=bay_id)
        return jsonify(result)
    except Exception as e:
        print("❌ RunPod Serverless 추론 에러:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/change_video/<filename>")
def change_video(filename):
    """기존 야드 탭/채널 전환 라우트 호환"""
    try:
        stage = STAGE_MAP.get(filename.lower(), "transport")
        result = call_runpod_serverless(stage=stage)
        return jsonify({
            "status": "ok",
            "video": filename,
            "stage": stage,
            "detection": result["detection"],
        })
    except Exception as e:
        print("❌ RunPod 단계 전환 추론 에러:", e)
        return jsonify({"status": "error", "video": filename, "message": str(e)}), 500


@bp.route("/detections")
def get_detections():
    with _state_lock:
        return jsonify(_vision_state["latest_detection"])


@bp.route("/video_feed")
def video_feed():
    def frame_stream():
        while True:
            with _state_lock:
                jpeg = _vision_state["latest_jpeg"]
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            time.sleep(0.5)

    return Response(
        frame_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@bp.route("/pause")
def toggle_pause():
    with _state_lock:
        _vision_state["is_paused"] = not _vision_state["is_paused"]
        paused = _vision_state["is_paused"]
    return jsonify({"status": "ok", "is_paused": paused})