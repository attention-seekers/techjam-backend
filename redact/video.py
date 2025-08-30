import cv2
import subprocess
import os

def blur_region(frame, x,y,w,h, mode="blur", k=51, pix_size=20):
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0: return frame
    if mode == "pixelate":
        small = cv2.resize(roi, (max(1,w//pix_size), max(1,h//pix_size)), interpolation=cv2.INTER_LINEAR)
        roi2 = cv2.resize(small, (w,h), interpolation=cv2.INTER_NEAREST)
    else:
        roi2 = cv2.GaussianBlur(roi, (k|1, k|1), 0)
    frame[y:y+h, x:x+w] = roi2
    return frame

def apply_video_actions(video_path: str, actions: list, out_path: str):
    # ...existing OpenCV code...
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp_out = out_path + ".noaudio.mp4"
    out = cv2.VideoWriter(tmp_out, fourcc, fps, (w,h))

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        t_ms = int(1000 * (fidx / fps))
        for a in actions:
            if t_ms >= a["start_ms"] and t_ms <= a["end_ms"]:
                for (x,y,bw,bh) in a.get("bboxes", []):
                    blur_region(frame, x,y,bw,bh, mode="pixelate" if a["action"]=="pixelate" else "blur")
        out.write(frame)
        fidx += 1

    cap.release(); out.release()

    # --- Add audio back using ffmpeg ---
    # out_path: final output with audio
    # tmp_out: video without audio
    # video_path: original video with audio
    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_out,
        "-i", video_path,
        "-c:v", "copy",
        "-c:a", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        out_path
    ]
    subprocess.run(cmd, check=True)
    os.remove(tmp_out)
