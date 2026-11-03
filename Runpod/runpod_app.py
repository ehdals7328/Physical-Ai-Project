# 기존 라이브러리 임포트 유지
import cv2
import threading
import time
import supervision as sv
from flask import Flask, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)

MODEL_PATH = "/workspace/runs/detect/welding_vision_project/sub_assembly_v1-2/weights/best.pt"
# 기본 영상을 Test_1.mp4로 설정
current_video_path = "/workspace/Test_1.mp4" 

model = YOLO(MODEL_PATH)
box_annotator = sv.BoxAnnotator(thickness=5)
label_annotator = sv.LabelAnnotator(
    text_scale=0.6, 
    text_padding=6, 
    text_position=sv.Position.TOP_LEFT
)

# 전역 상태 변수
latest_frame_bytes = None
latest_json_data = {
    "image": {"width": 960, "height": 540},
    "predictions": []
}
is_paused = False
change_requested = False # 🔥 영상 교체 요청 플래그 추가

# 🔄 백그라운드 추론 워커 스레드
def video_inference_worker():
    global latest_frame_bytes, latest_json_data, is_paused, current_video_path, change_requested
    cap = cv2.VideoCapture(current_video_path)

    while True:
        # 🔥 영상 교체 요청이 들어오면 기존 캡처를 닫고 새 영상으로 재시작
        if change_requested:
            cap.release()
            cap = cv2.VideoCapture(current_video_path)
            change_requested = False
            is_paused = False # 영상이 바뀌면 정지 상태 해제
            time.sleep(0.5)
            continue

        if is_paused:
            time.sleep(0.05)
            continue

        if not cap.isOpened():
            cap = cv2.VideoCapture(current_video_path)
            time.sleep(0.5)
            continue

        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        try:
            # 기존 모델 추론 및 supervision 어노테이션 로직 그대로 유지
            results = model.track(
                frame, persist=True, conf=0.45, iou=0.45, tracker="bytetrack.yaml", verbose=False
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            predictions = []
            has_tracker = detections.tracker_id is not None

            for idx, (bbox, conf, class_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                cx = x1 + (w / 2)
                cy = y1 + (h / 2)

                class_name = model.names[int(class_id)] if int(class_id) in model.names else str(class_id)
                track_id = int(detections.tracker_id[idx]) if (has_tracker and idx < len(detections.tracker_id)) else None

                predictions.append({
                    "x": round(float(cx), 1),
                    "y": round(float(cy), 1),
                    "width": round(float(w), 1),
                    "height": round(float(h), 1),
                    "confidence": round(float(conf), 2),
                    "class": class_name,
                    "class_id": int(class_id),
                    "tracker_id": track_id
                })

            latest_json_data = {
                "image": {"width": 960, "height": 540},
                "predictions": predictions
            }

            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence in zip(detections.data['class_name'], detections.confidence)
            ]

            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            display_frame = cv2.resize(annotated_frame, (960, 540))
            _, buffer = cv2.imencode('.jpg', display_frame)
            latest_frame_bytes = buffer.tobytes()

        except Exception as e:
            print("추론 오류:", e)
            
        time.sleep(0.01)

threading.Thread(target=video_inference_worker, daemon=True).start()

def generate_stream():
    while True:
        if latest_frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    return jsonify(latest_json_data)

@app.route('/pause')
def toggle_pause():
    global is_paused
    is_paused = not is_paused
    return jsonify({"status": "success", "is_paused": is_paused})

# 🔥 영상 채널 교체를 위한 새로운 API 엔드포인트
@app.route('/change_video/<filename>')
def change_video(filename):
    global current_video_path, change_requested
    if filename in ['Test_1.mp4', 'Test_2.mp4']:
        current_video_path = f"/workspace/{filename}"
        change_requested = True
        return jsonify({"status": "success", "video": filename})
    return jsonify({"status": "error"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)