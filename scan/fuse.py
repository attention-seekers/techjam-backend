from typing import List, Dict

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    ua = aw*ah + bw*bh - inter
    return inter / ua if ua>0 else 0.0

def fuse(findings: List[Dict]) -> List[Dict]:
    # Simple dedupe: if a text_pii bbox sits inside a license_plate bbox at similar time, drop the text one
    visuals = [f for f in findings if f["modality"]=="visual"]
    audios  = [f for f in findings if f["modality"]=="audio"]

    pruned = []
    for f in visuals:
        if f["type"]=="text_pii":
            overlapped = any(
                (g["type"]=="license_plate" and
                 abs(f["start_ms"]-g["start_ms"])<500 and
                 any(iou(f["bboxes"][0], gb) > 0.4 for gb in g["bboxes"]))
                for g in visuals
            )
            if overlapped: continue
        pruned.append(f)

    # sort by severity then time
    pruned.extend(audios)
    pruned.sort(key=lambda x: (-x.get("severity",50), x["start_ms"]))
    return pruned
