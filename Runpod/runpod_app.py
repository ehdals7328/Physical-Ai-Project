import os
import cv2
import time
import threading
import numpy as np
from flask import Flask, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# === 1. 모델 및 비디오 경로 설정 ===
MODEL_PATH = "/workspace/runs/segment/VAE_Vision/yolo11m_seg_vae_v1/weights/best.pt"
CURRENT_VIDEO = "/workspace/Video/Test_3.mp4"

model = YOLO(MODEL_PATH)

# 전역 상태 변수
latest_frame_bytes = None
latest_json_data = {
    "image": {"width": 960, "height": 540},
    "predictions": []
}
is_paused = False
cap = None
lock = threading.Lock()


# === 2. 이음부 중심선(Path Tracing + 45px 유클리드 거리 샘플링) 추출 함수 ===
def extract_centerline_waypoints(mask_single_crop, min_distance_px=45):
    """
    1. 노이즈 제거 (Morphology Opening/Closing)
    2. 골격화 (Thinning)
    3. 8방향 이웃 추적 (Path Tracing으로 수직/수평 선 꼬임 방지)
    4. 실제 유클리드 픽셀 거리(min_distance_px) 기준 다운샘플링
    """
    if mask_single_crop is None or np.sum(mask_single_crop) == 0:
        return []

    # 1. 마스크 미세 노이즈 제거 및 매끄럽게 Smoothing
    binary_mask = (mask_single_crop * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # 2. 골격화 (Thinning)
    try:
        skeleton = cv2.ximgproc.thinning(binary_mask)
    except AttributeError:
        skeleton = np.zeros(binary_mask.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = binary_mask.copy()
        while True:
            eroded = cv2.erode(temp, element)
            temp_open = cv2.dilate(eroded, element)
            temp_diff = cv2.subtract(temp, temp_open)
            skeleton = cv2.bitwise_or(skeleton, temp_diff)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break

    y_coords, x_coords = np.where(skeleton > 0)
    if len(x_coords) == 0:
        return []

    points = list(zip(x_coords, y_coords))
    if not points:
        return []

    # 3. 8방향 이웃 추적 (Path Tracing)으로 꼬임 방지
    visited = set()
    ordered_path = []

    # 임의의 시작점 선택 (좌상단에 가장 가까운 픽셀)
    start_pt = min(points, key=lambda p: p[0] + p[1])
    current_pt = start_pt
    visited.add(current_pt)
    ordered_path.append(current_pt)

    point_set = set(points)

    while True:
        cx, cy = current_pt
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in point_set and (nx, ny) not in visited:
                    neighbors.append((nx, ny))

        if not neighbors:
            unvisited = [p for p in points if p not in visited]
            if not unvisited:
                break
            next_pt = min(unvisited, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
            if (next_pt[0] - cx) ** 2 + (next_pt[1] - cy) ** 2 > 2500:
                break
            current_pt = next_pt
        else:
            current_pt = neighbors[0]

        visited.add(current_pt)
        ordered_path.append(current_pt)

    if not ordered_path:
        return []

    # 4. 유클리드 거리 기준 일정 간격(min_distance_px) 다운샘플링
    waypoints = [ordered_path[0]]
    last_pt = ordered_path[0]

    for pt in ordered_path[1:]:
        dist = np.sqrt((pt[0] - last_pt[0]) ** 2 + (pt[1] - last_pt[1]) ** 2)
        if dist >= min_distance_px:
            waypoints.append(pt)
            last_pt = pt

    return [[float(p[0]), float(p[1])] for p in waypoints]


# === 3. 백그라운드 추론 워커 스레드 ===
def video_inference_worker():
    global latest_frame_bytes, latest_json_data, is_paused, cap, CURRENT_VIDEO

    cap = cv2.VideoCapture(CURRENT_VIDEO)

    while True:
        if is_paused:
            time.sleep(0.05)
            continue

        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(CURRENT_VIDEO)
            time.sleep(0.5)
            continue

        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        try:
            # YOLOv11-seg 추론 진행
            results = model.track(
                frame,
                imgsz=(384, 640),
                persist=True,
                conf=0.35,
                iou=0.5,
                tracker="bytetrack.yaml",
                verbose=False
            )[0]

            # 🎯 일반 객체 내부 색상 채우기 제거 (투명 바운딩 박스 시각화)
            annotated_frame = results.plot(masks=False)

            predictions = []
            has_masks = results.masks is not None

            if results.boxes is not None:
                for idx, box in enumerate(results.boxes):
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    confidence = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    track_id = int(box.id[0]) if box.id is not None else None

                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    cx = xyxy[0] + (w / 2)
                    cy = xyxy[1] + (h / 2)

                    # 기본 바운딩 박스 정보
                    pred_data = {
                        "class": class_name,
                        "class_id": cls_id,
                        "confidence": round(float(confidence), 2),
                        "tracker_id": track_id,
                        "box": {
                            "x": round(float(cx), 1),
                            "y": round(float(cy), 1),
                            "width": round(float(w), 1),
                            "height": round(float(h), 1)
                        }
                    }

                    # 🎯 오직 weld_joint 일 때만 중심선 좌표(centerline_waypoints) 추출 및 추가
                    if class_name == "weld_joint" and has_masks and idx < len(results.masks.data):
                        mask_single = results.masks.data[idx].cpu().numpy()
                        mask_resized = cv2.resize(mask_single, (frame.shape[1], frame.shape[0]))

                        # 중심선 경유점 좌표 계산
                        centerline_waypoints = extract_centerline_waypoints(mask_resized, min_distance_px=45)
                        pred_data["centerline_waypoints"] = centerline_waypoints

                        # weld_joint 마스크 영역 자줏빛 투과 시각화
                        mask_bool = mask_resized > 0.5
                        annotated_frame[mask_bool] = (annotated_frame[mask_bool] * 0.6 + np.array([255, 0, 255]) * 0.4).astype(np.uint8)

                        # 중심선 경유점(Yellow Dots) 화면 렌더링
                        for wx, wy in centerline_waypoints:
                            cv2.circle(annotated_frame, (int(wx), int(wy)), radius=4, color=(0, 255, 255), thickness=-1)
                            cv2.circle(annotated_frame, (int(wx), int(wy)), radius=4, color=(0, 0, 255), thickness=1)

                    predictions.append(pred_data)

            display_frame = cv2.resize(annotated_frame, (960, 540))
            _, buffer = cv2.imencode('.jpg', display_frame)

            with lock:
                latest_frame_bytes = buffer.tobytes()
                latest_json_data = {
                    "image": {"width": 960, "height": 540},
                    "predictions": predictions
                }

        except Exception as e:
            print("추론 오류:", e)

        time.sleep(0.01)

threading.Thread(target=video_inference_worker, daemon=True).start()


# === 4. Flask 엔드포인트 라우팅 ===
def generate_stream():
    while True:
        with lock:
            frame_bytes = latest_frame_bytes
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    with lock:
        return jsonify(latest_json_data)

@app.route('/pause')
def toggle_pause():
    global is_paused
    is_paused = not is_paused
    return jsonify({"status": "success", "is_paused": is_paused})

@app.route('/change_video/<filename>')
def change_video(filename):
    global CURRENT_VIDEO, cap
    target_path = f"/workspace/{filename}" if not filename.startswith("/") else filename
    CURRENT_VIDEO = target_path
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(CURRENT_VIDEO)
    return jsonify({"status": "success", "video": filename})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)