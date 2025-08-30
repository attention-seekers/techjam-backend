from typing import List, Dict
# Choose one: faster_whisper (fast) or openai-whisper (simple)
from faster_whisper import WhisperModel
from scan.pii_function import pii_from_text

model = WhisperModel("small", device="cpu", compute_type="int8")  # or "cuda"

def transcribe(audio_path: str) -> List[Dict]:
    # returns [{"start_ms":..,"end_ms":..,"text":..}, {"start_ms":..,"end_ms":..,"text":..}, ...]
    segments, _ = model.transcribe(audio_path, word_timestamps=False)
    out = []
    for seg in segments:
        out.append({
            "start_ms": int(seg.start * 1000),
            "end_ms": int(seg.end * 1000),
            "text": seg.text.strip()
        })
    return out

def audio_findings(transcript_segments):
    findings = []
    for i, seg in enumerate(transcript_segments):
        hits = pii_from_text(seg["text"])
        for t, conf in hits:
            findings.append({
                "id": f"a_{i}_{t}",
                "modality": "audio",
                "type": t if t not in ("gpe","loc") else "location",
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "span_text": seg["text"],
                "confidence": conf,
                "severity": 80 if t in ("phone_number","email") else 50,
                "action": "bleep",
                "selected": True
            })
    return findings


# Outputs an array of {start_ms, end_ms, text, }