"""
landmark_standardization.py - same processing as landmark_standardization.ipynb (downsample -> linear
interpolation to 172 frames), plus:
  * a per-clip log (original duration, fps, frames before/after, temporal stretch factor) written to
    plots/clip_durations.csv and plotted in plots/fig_clip_duration_distribution.png/.pdf
    (Reviewer 2, comment 3: duration distribution + effect of fixed-length interpolation)
  * the ThreadPoolExecutor job is now awaited (.result()), so errors are raised instead of silently lost
Run from the project root:  python landmark_standardization.py
"""
import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import warnings
warnings.filterwarnings('ignore')

VIDEO_PATH = r'data/Videos/'
LANDMARK_PATH = r'landmarks/'
PROCESSED_LANDMARK_PATH = r'landmarks_processed/'
REPORT_PATH = r'VideoReport_updated.csv'
FIG_SAVE_PATH = r'plots/'

TARGET_FRAMES = 172
TARGET_FPS = 15

os.makedirs(PROCESSED_LANDMARK_PATH, exist_ok=True)
os.makedirs(FIG_SAVE_PATH, exist_ok=True)


def downsample(sequence, target_fps, fps):
    stride = max(1, int(round(fps / target_fps)))
    return sequence[::stride]


def interpolate(sequence, target_frames=150):
    T = sequence.shape[0]

    if T == target_frames:
        return sequence.astype(np.float32)

    old_idx = np.linspace(0, 1, T)
    new_idx = np.linspace(0, 1, target_frames)

    out = np.zeros((target_frames, sequence.shape[1], sequence.shape[2]), dtype=np.float32)

    for landmark in range(sequence.shape[1]):
        for feature in range(sequence.shape[2]):
            f = interp1d(
                old_idx,
                sequence[:, landmark, feature],
                kind="linear",
                fill_value="extrapolate"
            )

            out[:, landmark, feature] = f(new_idx)
    return out


def process_landmarks_folder(landmarks_root, output_root, fps_lookup, duration_lookup, target_fps, target_frames):
    records = []
    for root, _, files in os.walk(landmarks_root):
        for file in files:
            if not file.endswith(".npy"):
                continue

            in_path = os.path.join(root, file)
            rel_path = os.path.relpath(in_path, landmarks_root)
            out_path = os.path.join(output_root, rel_path)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            landmarks = np.load(in_path)

            video_name = rel_path.replace("\\", "/").replace(".npy", ".mp4")
            fps = fps_lookup.get(video_name)

            if fps is None:
                print(f"FPS not found for {video_name}")
                continue

            kept_frames = landmarks.shape[0]  # frames with both hands (after extraction filtering)
            if fps >= target_fps:
                landmarks = downsample(landmarks, target_fps, fps)
            after_ds = landmarks.shape[0]

            landmarks = interpolate(landmarks, target_frames)

            np.save(out_path, landmarks)
            records.append({
                "video": video_name, "fps": fps,
                "original_duration_s": duration_lookup.get(video_name),
                "kept_frames": kept_frames,
                "frames_after_downsample": after_ds,
                "effective_duration_s": after_ds / target_fps,  # duration represented by the pre-interpolation clip
                # >1: clip stretched (motion slowed), <1: clip compressed (motion sped up)
                "temporal_stretch_factor": target_frames / after_ds,
            })
    print("Done:", output_root)
    return records


def plot_duration_distribution(df):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    d = df["original_duration_s"].dropna()
    axes[0].hist(d, bins=25, color="#4C72B0")
    axes[0].axvline(d.median(), color="red", linestyle="--", label=f"Median = {d.median():.1f} s")
    axes[0].set_xlabel("Original clip duration (s)")
    axes[0].set_ylabel("Number of clips")
    axes[0].set_title("Clip duration")
    axes[0].legend()

    s = df["temporal_stretch_factor"]
    axes[1].hist(s, bins=25, color="#DD8452")
    axes[1].axvline(1.0, color="black", linestyle=":", label="No distortion (1.0)")
    axes[1].axvline(s.median(), color="red", linestyle="--", label=f"Median = {s.median():.2f}")
    axes[1].set_xlabel(f"Stretch factor ({TARGET_FRAMES} / frames at {TARGET_FPS} fps)")
    axes[1].set_ylabel("Number of clips")
    axes[1].set_title("Effect of fixed-length interpolation")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_SAVE_PATH, "fig_clip_duration_distribution.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_SAVE_PATH, "fig_clip_duration_distribution.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(df[["original_duration_s", "kept_frames", "frames_after_downsample", "temporal_stretch_factor"]]
          .describe().round(2).to_string())


def landmarks_main(landmarks_root, output_root, report_path, target_fps, target_frames):
    report_df = pd.read_csv(report_path)
    norm = lambda v: str(v).replace("\\", "/")  # report keys may use either slash style
    fps_lookup = dict(zip(report_df["video"].map(norm), report_df["fps"]))
    duration_lookup = dict(zip(report_df["video"].map(norm), report_df["duration_sec"]))

    max_threads = max(1, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_threads) as ex:
        records = ex.submit(
            process_landmarks_folder,
            landmarks_root, output_root, fps_lookup, duration_lookup, target_fps, target_frames
        ).result()  # wait for the job (and re-raise any exception)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(FIG_SAVE_PATH, "clip_durations.csv"), index=False)
    plot_duration_distribution(df)

    #process_landmarks_folder(landmarks_root, output_root, fps_lookup, duration_lookup, target_fps, target_frames)


if __name__ == "__main__":
    landmarks_main(LANDMARK_PATH, PROCESSED_LANDMARK_PATH, REPORT_PATH, TARGET_FPS, TARGET_FRAMES)
