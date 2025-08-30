from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def yolo_findings(video_path: str, fps_target: float = 3.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(fps / fps_target)))
    findings = []
    fidx = 0
    while True:
        ret = cap.grab()
        if not ret: break
        if fidx % step == 0:
            ret, frame = cap.retrieve()
            if not ret: break
            ms = int(1000 * (fidx / fps))

            res = model.predict(frame, verbose=False)[0]
            for b in res.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                cls  = int(b.cls[0])
                label = model.names[cls]

                if label in ["person", "car", "face", "license-plate"]:
                  findings.append({
                      "id": f"v_{fidx}_{label}",
                      "modality": "visual",
                      "type": label,
                      "start_ms": ms,
                      "end_ms": ms + int(1000/fps),
                      "bboxes": [[x1, y1, x2-x1, y2-y1]],
                      "confidence": conf,
                      "severity": 80 if label=="license_plate" else 60,
                      "action": "blur",
                      "selected": True
                  })
        fidx += 1
    cap.release()
    return findings