"""Single source of truth for every data-pipeline constant. Edit here only."""
from pathlib import Path

# ---- paths (relative to where you run the scripts) ----
VIDEO_ROOT   = Path("data/Videos")          # <root>/<N-ClassName>/<clip>.mp4|.mov
REPORT_CSV   = Path("VideoReport.csv")      # written by video_filtering.py
METADATA_CSV = Path("metadata.csv")         # source/performer IDs (manual), see build_metadata.py
RAW_DIR      = Path("landmarks")        # written by landmark_extraction.py
PROC_DIR     = Path("landmarks_processed")  # written by landmark_standardization.py
LOG_DIR      = Path("logs")
SPLIT_DIR    = Path("splits")

# ---- video QC (Section 2.2) ----
VIDEO_EXTS      = (".mp4", ".mov")
SAMPLE_STEP     = 5          # every 5th frame is used for blur/brightness
BLUR_MIN        = 18
BRIGHTNESS_MIN  = 50
FPS_MIN         = 15
WIDTH_MIN       = 480
HEIGHT_MIN      = 640
DURATION_MIN_S  = 5
ENFORCE_BLUR    = False      # your original run ignored blur (manual check). True = enforce it.
COMPUTE_STABILITY = False    # optical-flow score; was computed but never used

# ---- pose / landmarks (Sections 2.3-2.4) ----
TARGET_FPS    = 15
TARGET_FRAMES = 172
BODY_POINTS   = 17
HAND_POINTS   = 21
KEYPOINT_IDX  = list(range(0, 17)) + list(range(91, 133))   # 17 body + 21 L hand + 21 R hand = 59
CONF_THR      = 0.30
MIN_VISIBLE   = 10           # landmarks above CONF_THR for a hand to count as visible
MAX_GAP       = 1            # frames (at 15 fps) of missing hand that are linearly interpolated
MIN_USABLE_FRAMES = 30       # clip needs >= this many frames with BOTH hands valid
POSE = dict(mode="performance", backend="onnxruntime", device="cuda", to_openpose=False)

# Feature layout of a 59-keypoint array
BODY   = slice(0, 17)
L_HAND = slice(17, 38)
R_HAND = slice(38, 59)

SEED = 42