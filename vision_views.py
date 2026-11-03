import requests
import urllib3
from flask import Blueprint, render_template, Response, jsonify

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

bp = Blueprint('vision', __name__, url_prefix='/vision')

# 🔥 본인의 정확한 RunPod Proxy URL
RUNPOD_STREAM_URL = "https://t7zszs9gsjl8t5-5000.proxy.runpod.net/video_feed"
RUNPOD_JSON_URL = "https://t7zszs9gsjl8t5-5000.proxy.runpod.net/detections"
RUNPOD_PAUSE_URL = "https://t7zszs9gsjl8t5-5000.proxy.runpod.net/pause"
RUNPOD_CHANGE_VIDEO_URL = "https://t7zszs9gsjl8t5-5000.proxy.runpod.net/change_video/"  # 🔥 추가

@bp.route('/')
def index():
    return render_template('vision.html')

@bp.route('/video_feed')
def video_feed():
    def proxy_stream():
        try:
            req = requests.get(RUNPOD_STREAM_URL, stream=True, timeout=5, verify=False)
            for chunk in req.iter_content(chunk_size=1024):
                yield chunk
        except Exception as e:
            print("RunPod 영상 스트림 연동 에러:", e)

    return Response(
        proxy_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@bp.route('/detections')
def get_detections():
    try:
        res = requests.get(RUNPOD_JSON_URL, timeout=3, verify=False)

        if res.status_code == 200:
            return jsonify(res.json())
        else:
            print(f"❌ RunPod 상태 코드 에러 ({res.status_code}):", res.text[:100])
            return jsonify({"image": {"width": 960, "height": 540}, "predictions": []})

    except Exception as e:
        print("❌ RunPod JSON 연동 에러:", e)
        return jsonify({"image": {"width": 960, "height": 540}, "predictions": []})

@bp.route('/pause')
def toggle_pause():
    try:
        res = requests.get(RUNPOD_PAUSE_URL, timeout=3, verify=False)
        return jsonify(res.json())
    except Exception as e:
        print("RunPod Pause 연동 에러:", e)
        return jsonify({"status": "error", "is_paused": False})

# 🔥 [추가] 영상 채널 변경 프록시 라우트
@bp.route('/change_video/<filename>')
def change_video(filename):
    try:
        target_url = f"{RUNPOD_CHANGE_VIDEO_URL}{filename}"
        res = requests.get(target_url, timeout=3, verify=False)
        return jsonify(res.json())
    except Exception as e:
        print("RunPod Change Video 연동 에러:", e)
        return jsonify({"status": "error", "video": filename})