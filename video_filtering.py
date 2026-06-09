import pandas as pd
import cv2
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

PATH = r'data\Videos'

def sample_frame(path, step=10):
    frame = cv2.VideoCapture(path)
    frames = []
    count = 0
    while True:
        ret, img = frame.read()
        if not ret:
            break
        if count % step == 0:
            frames.append(img)
        count += 1
    frame.release()
    return frames

def blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

def brightness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def stability_score(frames):
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    motions = []
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None,  # type: ignore
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
        motions.append(magnitude)
        prev_gray = gray
    return np.mean(motions)

def fps_check(cap):
    return cap.get(cv2.CAP_PROP_FPS)

def analyze_video(path):
    cv2.setNumThreads(1)
    print("Processing: ", path)
    relative_path = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))
    mudra_label = os.path.basename(os.path.dirname(path)).split("-", 1)[1]

    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"  Could not open: {path}")
            return {"video": relative_path, "mudra_label": mudra_label, "status": "FAILED"}

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = fps_check(cap)
        cap.release()

        duration_sec = round(frame_count / fps, 2) if fps > 0 else 0

        frames = sample_frame(path, 5)
        if len(frames) == 0:
            print(f"  No frames extracted: {path}")
            return {"video": relative_path, "mudra_label": mudra_label, "status": "FAILED"}
        
        h, w = frames[0].shape[:2]
        avg_blur = np.mean([blur_score(f) for f in frames])
        avg_brightness = np.mean([brightness_score(f) for f in frames])
        stability = stability_score(frames)

        blur_ok = True if avg_blur >= 18 else False
        brightness_ok = True if avg_brightness >= 50 else False
        fps_ok = True if fps >= 15 else False
        resolution_ok = True if (w >= 480 and h >= 640) else False
        duration_ok = True if duration_sec >= 5 else False

        return {
            "video": relative_path,
            "mudra_label": mudra_label,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration_sec": duration_sec,
            "fps": round(fps, 2),
            "blur": round(avg_blur, 2),
            "brightness": round(avg_brightness, 2),
            "stability": round(stability, 2),

            # flags #fuck u IMG_1246.mp4
            "blur_ok": blur_ok,
            "brightness_ok": brightness_ok,
            "fps_ok": fps_ok,
            "resolution_ok": resolution_ok,
            "duration_ok": duration_ok
        }

    except Exception as e:
        print(f"  ERROR on {path}: {e}")
        return {"video": relative_path, "mudra_label": mudra_label, "status": f"ERROR: {e}"}


def main(path):
    formats = ('.mp4', '.mov')
    video_paths = []
    max_workers = min(8, os.cpu_count() or 1)

    for dirs in os.listdir(path):
        dir_path = os.path.join(path, dirs)
        if not os.path.isdir(dir_path):
            continue
        for file in os.listdir(dir_path):
            if file.lower().endswith(formats):
                video_paths.append(os.path.join(dir_path, file))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(analyze_video, video_paths))

    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    df.to_csv('VideoReport.csv', index=False)
    print(f"\nReport saved. {len(df)} videos analyzed.")

if __name__ == "__main__":
    main(PATH)