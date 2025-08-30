from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import os, uuid, subprocess, json

from scan.asr import transcribe, audio_findings
from scan.vision import yolo_findings
from scan.fuse import fuse
from scan.pii_function import pii_from_text
from redact.audio import apply_audio_actions
from redact.video import apply_video_actions

app = FastAPI()
WORK = "./tmp/privscan"
os.makedirs(WORK, exist_ok=True)

def extract_audio(mp4, wav):
    subprocess.run(["ffmpeg","-y","-i", mp4, "-vn","-ac","1","-ar","16000", wav], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

@app.post("/scan")
async def scan(video: UploadFile = File(...)):
    vid_id = str(uuid.uuid4())
    vpath = os.path.join(WORK, f"{vid_id}.mp4")
    with open(vpath, "wb") as f: f.write(await video.read())

    # 1) Audio
    apath = os.path.join(WORK, f"{vid_id}.wav")
    extract_audio(vpath, apath)
    segs = transcribe(apath)
    audio = audio_findings(segs)

    # 3) Vision
    vis = yolo_findings(vpath)

    findings = fuse(audio + vis)
    # NOTE: for overlay scaling, include original dimensions
    return JSONResponse({
        "video_width": None, "video_height": None, "video_fps": None,
        "findings": findings
    })

@app.post("/redact")
async def redact(
    video: UploadFile = File(...),
    findings_json: str = Form(...)
):
    vid_id = str(uuid.uuid4())
    vpath = os.path.join(WORK, f"{vid_id}.mp4")
    with open(vpath, "wb") as f: f.write(await video.read())

    data = json.loads(findings_json)
    video_actions, audio_actions = [], []
    for f in data["findings"]:
        if not f.get("selected", True): continue
        if f["modality"] == "visual" and f["action"] in ("blur","pixelate"):
            video_actions.append({
                "start_ms": f["start_ms"], "end_ms": f["end_ms"],
                "bboxes": f.get("bboxes", []), "action": f["action"]
            })
        if f["modality"] == "audio" and f["action"] in ("bleep","mute"):
            audio_actions.append({
                "start_ms": f["start_ms"], "end_ms": f["end_ms"], "action": f["action"]
            })

    # 1) Video edits
    edited_video = os.path.join(WORK, f"{vid_id}.edited.mp4")
    apply_video_actions(vpath, video_actions, edited_video)

    # 2) Audio edits
    apath = os.path.join(WORK, f"{vid_id}.wav")
    extract_audio(edited_video, apath)
    edited_audio = os.path.join(WORK, f"{vid_id}.edited.wav")
    apply_audio_actions(apath, audio_actions, edited_audio)

    # 3) Remux
    out = os.path.join(WORK, f"{vid_id}.out.mp4")
    subprocess.run([
        "ffmpeg","-y","-i", edited_video, "-i", edited_audio,
        "-c:v","libx264","-c:a","aac","-shortest", out
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return FileResponse(out, media_type="video/mp4", filename="redacted.mp4")
