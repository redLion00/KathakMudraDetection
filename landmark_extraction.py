"""Step 3 - pose estimation. Saves RAW landmarks only (no normalisation, no frame dropping), so every
normalisation / missing-data / resampling variant can be rebuilt in step 4 without re-running the pose model.

Per clip -> landmarks_raw/<class>/<clip>.npz
    raw        (T, 59, 3)  x/width, y/height, confidence   (T frames at ~15 fps, timestamp-sampled)
    src_frame  (T,)        index of the source frame each row came from
    n_persons  (T,)        persons detected in that frame
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import *


def pick_person(kps, scores):
    if kps is None or len(kps) == 0:
        return None, None
    i = int(np.argmax(scores[:, :17].mean(axis=1)))     # highest mean body confidence (as in the paper)
    return kps[i], scores[i]


def frame_to_raw(frame, wholebody):
    h, w = frame.shape[:2]
    kps, scores = wholebody(frame)
    k, s = pick_person(kps, scores)
    out = np.zeros((len(KEYPOINT_IDX), 3), np.float32)
    if k is None:
        return out, 0
    out[:, 0] = k[KEYPOINT_IDX, 0] / w
    out[:, 1] = k[KEYPOINT_IDX, 1] / h
    out[:, 2] = s[KEYPOINT_IDX]
    return out, len(kps)


def extract_video(path, wholebody):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("cannot open")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("fps<=0")
    step, next_pick, i = fps / TARGET_FPS, 0.0, 0     # timestamp-based: 24/25/30/60 fps all -> ~15 fps
    raws, src, persons = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i + 1e-9 >= next_pick:
            r, n = frame_to_raw(frame, wholebody)
            raws.append(r); src.append(i); persons.append(n)
            next_pick += step
        i += 1
    cap.release()
    return np.stack(raws), np.array(src), np.array(persons), fps, i


def main(overwrite, limit):
    from rtmlib import Wholebody
    import rtmlib
    LOG_DIR.mkdir(exist_ok=True)
    wholebody = Wholebody(**POSE)
    cfg = {"rtmlib_version": getattr(rtmlib, "__version__", "unknown"), **POSE,
           "pose_model": str(getattr(getattr(wholebody, "pose_model", None), "onnx_model", "inspect wholebody.pose_model")),
           "det_model": str(getattr(getattr(wholebody, "det_model", None), "onnx_model", "inspect wholebody.det_model")),
           "keypoints_used": len(KEYPOINT_IDX), "target_fps": TARGET_FPS}
    (LOG_DIR / "pose_config.json").write_text(json.dumps(cfg, indent=2))   # cite this in Sec 2.3

    rep = pd.read_csv(REPORT_CSV)
    on_disk = {f"{p.parent.name}/{p.name}" for p in VIDEO_ROOT.rglob("*") if p.suffix.lower() in VIDEO_EXTS}
    rows = []
    for v in sorted(on_disk - set(rep.video)):
        rows.append({"video": v, "status": "not_in_report", "reason": "run video_filtering.py again"})

    todo = rep[rep.qc_pass] if "qc_pass" in rep else rep
    for _, r in rep[~rep.qc_pass].iterrows():
        rows.append({"video": r.video, "class_dir": r.class_dir, "status": "skipped_qc", "reason": r.exclusion_reason})
    if limit:
        todo = todo.head(limit)

    for _, r in todo.iterrows():
        path = VIDEO_ROOT / r.class_dir / Path(r.video).name
        out = RAW_DIR / r.class_dir / (Path(r.video).stem + ".npz")
        row = {"video": r.video, "class_dir": r.class_dir, "out_path": out.as_posix()}
        if not path.exists():
            rows.append({**row, "status": "missing_file", "reason": str(path)}); continue
        if out.exists() and not overwrite:
            rows.append({**row, "status": "ok", "reason": "cached"}); continue
        try:
            t0 = time.perf_counter()
            raw, src, persons, fps, n_src = extract_video(path, wholebody)
            dt = time.perf_counter() - t0
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, raw=raw, src_frame=src, n_persons=persons, video=r.video, fps_src=fps)
            detected = (raw[:, :, 2] > 0).any(axis=1)
            rows.append({**row, "status": "ok", "reason": "", "fps_src": round(fps, 2), "n_src_frames": n_src,
                         "n_sampled": len(raw), "pct_person": round(100 * detected.mean(), 1),
                         "pct_multi_person": round(100 * (persons > 1).mean(), 1),
                         "ms_per_sampled_frame": round(1000 * dt / len(raw), 1)})   # pose latency for R1.7
            print(f"ok {r.video}  {len(raw)} frames  {1000*dt/len(raw):.0f} ms/frame")
        except Exception as e:
            rows.append({**row, "status": "error", "reason": str(e)})
        pd.DataFrame(rows).to_csv(LOG_DIR / "extraction_log.csv", index=False)   # crash-safe
    pd.DataFrame(rows).to_csv(LOG_DIR / "extraction_log.csv", index=False)
    print(pd.DataFrame(rows).status.value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.overwrite, a.limit)