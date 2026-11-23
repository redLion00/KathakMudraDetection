"""Step 1 - quality control. Writes VideoReport.csv (one row per clip, every clip, with reasons)."""
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import *

FLAGS = ["blur_ok", "brightness_ok", "fps_ok", "resolution_ok", "duration_ok"]


def sha1_file(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def video_key(path: Path) -> str:
    """Platform-independent id, e.g. '1-Ardhpatka/IMG_1246.mp4' (always forward slash)."""
    return f"{path.parent.name}/{path.name}"


def analyze_video(path_str):
    cv2.setNumThreads(1)
    path = Path(path_str)
    class_dir = path.parent.name
    label = class_dir.split("-", 1)[1] if "-" in class_dir else class_dir
    row = {"video": video_key(path), "class_dir": class_dir, "mudra_label": label,
           "status": "OK", "sha1": None, "width": None, "height": None, "frame_count": None,
           "duration_sec": None, "fps": None, "blur": None, "brightness": None, "stability": None}
    row.update({f: False for f in FLAGS})
    try:
        row["sha1"] = sha1_file(path)
        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            row["status"] = "FAILED"
            return finalize(row)
        fps = cap.get(cv2.CAP_PROP_FPS)
        row.update(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                   frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
        row["fps"] = round(fps, 2)
        row["duration_sec"] = round(row["frame_count"] / fps, 2) if fps > 0 else 0

        blurs, brights, motions, first_hw, prev, n = [], [], [], None, None, 0
        cap = cv2.VideoCapture(path_str)
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if n % SAMPLE_STEP == 0:                      # streaming: no list of frames in RAM
                if first_hw is None:
                    first_hw = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blurs.append(cv2.Laplacian(gray, cv2.CV_64F).var())
                brights.append(float(gray.mean()))
                if COMPUTE_STABILITY and prev is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # type: ignore
                    motions.append(float(np.mean(np.hypot(flow[..., 0], flow[..., 1]))))
                prev = gray
            n += 1
        cap.release()
        if not blurs:
            row["status"] = "FAILED"
            return finalize(row)

        h, w = first_hw
        row.update(blur=round(float(np.mean(blurs)), 2), brightness=round(float(np.mean(brights)), 2),
                   stability=round(float(np.mean(motions)), 2) if motions else None,
                   blur_ok=bool(np.mean(blurs) >= BLUR_MIN),
                   brightness_ok=bool(np.mean(brights) >= BRIGHTNESS_MIN),
                   fps_ok=bool(fps >= FPS_MIN),
                   resolution_ok=bool(w >= WIDTH_MIN and h >= HEIGHT_MIN),
                   duration_ok=bool(row["duration_sec"] >= DURATION_MIN_S))
    except Exception as e:                                  # keep a full row, never a partial one
        row["status"] = f"ERROR: {e}"
    return finalize(row)


def finalize(row):
    reasons = []
    if row["status"] != "OK":
        reasons.append("unreadable")
    else:
        for flag in FLAGS:
            if not row[flag] and (flag != "blur_ok" or ENFORCE_BLUR):
                reasons.append(flag.replace("_ok", ""))
    row["qc_pass"] = len(reasons) == 0
    row["exclusion_reason"] = ";".join(reasons)
    row["blur_overridden"] = bool(row["status"] == "OK" and not row["blur_ok"] and not ENFORCE_BLUR)
    return row


def main():
    paths = sorted(str(p) for d in VIDEO_ROOT.iterdir() if d.is_dir()
                   for p in d.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    with ProcessPoolExecutor(max_workers=min(8, __import__("os").cpu_count() or 1)) as ex:
        rows = list(ex.map(analyze_video, paths))
    df = pd.DataFrame(rows)
    df["dup_sha1"] = df["sha1"].notna() & df["sha1"].duplicated(keep=False)   # exact duplicate files
    df.to_csv(REPORT_CSV, index=False)
    print(f"{len(df)} clips | qc_pass {int(df.qc_pass.sum())} | exact duplicates {int(df.dup_sha1.sum())}")
    if (~df.qc_pass).any():
        print(df.loc[~df.qc_pass, "exclusion_reason"].value_counts().to_string())
    if (df.blur_overridden).any():
        print(f"blur threshold overridden for {int(df.blur_overridden.sum())} clips (ENFORCE_BLUR=False)")


if __name__ == "__main__":
    main()