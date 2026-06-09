import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from scipy.interpolate import interp1d
from rtmlib import Wholebody
from torch.amp import autocast
import pandas as pd

df = pd.read_csv("C:\\Users\\Vivek\\projects\\KathakMudraDetection\\VideoReport_updated.csv")  # for class names

# ── CONFIG ────────────────────────────────────────────────────────────────
CHECKPOINT_PATH  = "models/saved/best_tcn.pt"
TARGET_FRAMES    = 172
CONFIDENCE_THR   = 0.60        # below this → show "uncertain"
CAMERA_IDX       = 0
BODY_POINTS      = 17
HAND_POINTS      = 21
CONF_THR         = 0.3
MIN_VISIBLE      = 10
LEFT_HAND_SLICE  = slice(91, 112)
RIGHT_HAND_SLICE = slice(112, 133)
CLASSES          = list(set(list(df['mudra_label'])))
TARGET_FPS    = 15       # must match training
TARGET_FRAMES = 172  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualTCNBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv1  = nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size,
                                dilation=dilation, padding=self.padding)
        self.bn1    = nn.BatchNorm1d(out_channels)
        self.relu1  = nn.ReLU(inplace=True)
        self.drop1  = nn.Dropout(p=dropout)

        self.conv2  = nn.Conv1d(out_channels, out_channels,
                                kernel_size=kernel_size,
                                dilation=dilation, padding=self.padding)
        self.bn2    = nn.BatchNorm1d(out_channels)

        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
        self.relu_out = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(self.skip, nn.Conv1d):
            nn.init.kaiming_normal_(self.skip.weight, mode="fan_out", nonlinearity="relu")

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        """Conv then trim right-side causal padding to preserve T."""
        return conv(x)[:, :, :x.size(2)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C_in, T)
        residual = self.skip(x)                          # (B, C_out, T)

        out = self._causal_conv(self.conv1, x)           # (B, C_out, T)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self._causal_conv(self.conv2, out)         # (B, C_out, T)
        out = self.bn2(out)

        out = out + residual                             # residual add
        return self.relu_out(out)                        # (B, C_out, T)

class TCNActionRecognizer(nn.Module):

    def __init__(
        self,
        num_classes: int,
        input_dim:   int   = 236,
        proj_dim:    int   = 128,
        kernel_size: int   = 3,
        dilations:   tuple = (1, 2, 4, 8),
        num_heads:   int   = 4,
        attn_dropout: float = 0.1,
        tcn_dropout:  float = 0.2,
        head_dropout: float = 0.4,
    ) -> None:
        super().__init__()

        # 1. Input projection  (B, T, 236) → (B, T, 128)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )

        # 2. Four residual TCN blocks  (B, 128, T) → (B, 128, T)
        self.tcn_blocks = nn.ModuleList([
            ResidualTCNBlock(proj_dim, proj_dim,
                             kernel_size=kernel_size,
                             dilation=d, dropout=tcn_dropout)
            for d in dilations
        ])
        # Receptive field ≈ 2 × (kernel-1) × Σ(dilations)
        #                 = 2 × 2 × 15 = 60 frames

        # 3. Temporal self-attention  (B, T, 128) → (B, T, 128)
        self.attn_norm    = nn.LayerNorm(proj_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads,
            dropout=attn_dropout, batch_first=True,
        )

        # 4. Global average pool  (B, T, 128) → (B, 128)  [implicit]

        # 5. Classification head  (B, 128) → (B, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x – (B, T=172, F=236)
        Returns: logits – (B, num_classes)
        """
        # ── Input projection ──────────────────────────────────
        x = self.input_proj(x)                       # (B, 172, 128)

        # ── TCN backbone ─────────────────────────────────────
        x = x.transpose(1, 2)                        # (B, 128, 172)
        for block in self.tcn_blocks:
            x = block(x)                             # (B, 128, 172)
        x = x.transpose(1, 2)                        # (B, 172, 128)

        # ── Temporal attention ────────────────────────────────
        residual = x
        x_norm   = self.attn_norm(x)                 # (B, 172, 128)  pre-norm
        attn_out, _ = self.temporal_attn(x_norm, x_norm, x_norm)  # (B, 172, 128)
        x = attn_out + residual                      # (B, 172, 128)

        # ── Global average pool ───────────────────────────────
        x = x.mean(dim=1)                            # (B, 128)

        # ── Classification ────────────────────────────────────
        return self.classifier(x)                    # (B, num_classes)

# ── LOAD MODEL ────────────────────────────────────────────────────────────
def load_tcn(checkpoint_path, num_classes, device):
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TCNActionRecognizer(
        num_classes  = num_classes,
        input_dim    = 236,
        proj_dim     = 128,
        kernel_size  = 3,
        dilations    = (1, 2, 4, 8),
        num_heads    = 4,
        attn_dropout = 0.1,
        tcn_dropout  = 0.2,
        head_dropout = 0.4,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint  (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.2%})")
    return model

# ── LANDMARK HELPERS (same as extraction notebook) ────────────────────────
def normalize_hand(points: np.ndarray) -> np.ndarray:
    points = points.copy()
    points -= points[0]
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points /= scale
    return points

def _pick_person(keypoints, scores):
    if keypoints.shape[0] == 0:
        return None, None
    idx = int(np.argmax(scores[:, :17].mean(axis=1)))
    return keypoints[idx], scores[idx]

def _body_array(kps, sc, frame_w, frame_h):
    body = np.zeros((BODY_POINTS, 3), dtype=np.float32)
    body[:, 0] = kps[:BODY_POINTS, 0] / frame_w
    body[:, 1] = kps[:BODY_POINTS, 1] / frame_h
    body[:, 2] = sc[:BODY_POINTS]
    return body

def _hand_array(kps, sc, hand_slice, frame_w, frame_h):
    hand_kps = kps[hand_slice]
    hand_sc  = sc[hand_slice]
    if (hand_sc > CONF_THR).sum() < MIN_VISIBLE:
        return np.zeros((HAND_POINTS, 3), dtype=np.float32)
    pts = np.zeros((HAND_POINTS, 3), dtype=np.float32)
    pts[:, 0] = hand_kps[:, 0] / frame_w
    pts[:, 1] = hand_kps[:, 1] / frame_h
    pts[:, 2] = hand_sc
    pts[hand_sc <= CONF_THR, :2] = 0.0
    return normalize_hand(pts)

def extract_frame_landmarks(frame, wholebody):
    """Extract (59, 3) landmarks from a single frame."""
    frame_h, frame_w = frame.shape[:2]
    keypoints, scores = wholebody(frame)
    kps, sc = _pick_person(keypoints, scores)
    if kps is None:
        return np.zeros((59, 3), dtype=np.float32)
    body       = _body_array(kps, sc, frame_w, frame_h)
    left_hand  = _hand_array(kps, sc, LEFT_HAND_SLICE,  frame_w, frame_h)
    right_hand = _hand_array(kps, sc, RIGHT_HAND_SLICE, frame_w, frame_h)
    return np.concatenate([body, left_hand, right_hand], axis=0)  # (59, 3)

# ── PREPROCESSING (mirrors SkeletonDataset.__getitem__) ───────────────────
def preprocess_buffer(buffer: np.ndarray) -> torch.Tensor:
    """
    Args:
        buffer : (T, 59, 3) raw landmark buffer
    Returns:
        tensor : (1, 172, 236) ready for model
    """
    arr = buffer.astype(np.float32)          # (T, 59, 3)

    # Interpolate to TARGET_FRAMES if buffer isn't full yet
    if arr.shape[0] != TARGET_FRAMES:
        old_idx = np.linspace(0, 1, arr.shape[0])
        new_idx = np.linspace(0, 1, TARGET_FRAMES)
        out = np.zeros((TARGET_FRAMES, 59, 3), dtype=np.float32)
        for lm in range(59):
            for ft in range(3):
                f = interp1d(old_idx, arr[:, lm, ft],
                             kind="linear", fill_value="extrapolate")
                out[:, lm, ft] = f(new_idx)
        arr = out

    arr = arr[:, :, :2]                      # (172, 59, 2) — drop confidence

    vel = np.zeros_like(arr)
    vel[1:] = arr[1:] - arr[:-1]
    arr = np.concatenate([arr, vel], axis=-1) # (172, 59, 4)

    pos = arr[:, :, :2]
    mu  = pos.mean(axis=(0, 1), keepdims=True)
    std = pos.std(axis=(0, 1), keepdims=True) + 1e-6
    arr[:, :, :2] = (pos - mu) / std

    seq = arr.reshape(TARGET_FRAMES, -1)     # (172, 236)
    return torch.from_numpy(seq).unsqueeze(0)  # (1, 172, 236)

# ── INFERENCE ─────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, buffer, device):
    """Returns (class_name, confidence, top3)."""
    tensor = preprocess_buffer(buffer).to(device)
    with autocast(device_type=device.type):
        logits = model(tensor)
    probs = F.softmax(logits, dim=-1).squeeze(0)
    top3_vals, top3_idxs = probs.topk(3)
    top3 = [
        {"class": CLASSES[i.item()], "confidence": round(v.item(), 4)}
        for v, i in zip(top3_vals, top3_idxs)
    ]
    return top3[0]["class"], top3[0]["confidence"], top3

# ── DRAW HUD ──────────────────────────────────────────────────────────────
def draw_hud(frame, pred_class, confidence, top3, buffer_len, collecting):
    h, w = frame.shape[:2]

    # Background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (360, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Buffer progress bar
    bar_w  = int(340 * min(buffer_len / TARGET_FRAMES, 1.0))
    status = "COLLECTING" if collecting else "PREDICTING"
    color  = (0, 255, 100) if collecting else (0, 200, 255)
    cv2.rectangle(frame, (10, 130), (350, 148), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, 130), (10 + bar_w, 148), color, -1)
    cv2.putText(frame, f"{status}  {buffer_len}/{TARGET_FRAMES}",
                (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if pred_class is None:
        cv2.putText(frame, "Collecting frames...",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return frame

    # Top prediction
    conf_color = (0, 255, 100) if confidence >= CONFIDENCE_THR else (0, 100, 255)
    label = pred_class if confidence >= CONFIDENCE_THR else f"? ({pred_class})"
    cv2.putText(frame, label,
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, conf_color, 2)
    cv2.putText(frame, f"{confidence:.1%}",
                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, conf_color, 2)

    # Top 3
    for i, entry in enumerate(top3):
        bar_full = int(200 * entry["confidence"])
        y = 82 + i * 22
        cv2.rectangle(frame, (10, y), (10 + bar_full, y + 14), (80, 80, 80), -1)
        cv2.putText(frame, f"{entry['class']}  {entry['confidence']:.1%}",
                    (14, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

    # Controls hint
    cv2.putText(frame, "R: reset  Q: quit",
                (w - 160, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    return frame

# ── MAIN LOOP ─────────────────────────────────────────────────────────────
def run_live_classification():
    model = load_tcn(CHECKPOINT_PATH, len(CLASSES), DEVICE)

    wholebody = Wholebody(
        mode    = "performance",
        backend = "onnxruntime",
        device  = "cuda" if torch.cuda.is_available() else "cpu",
        to_openpose = False,
    )

    cap = cv2.VideoCapture(CAMERA_IDX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_IDX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Rolling buffer — always keep last TARGET_FRAMES landmarks
    buffer     = deque(maxlen=TARGET_FRAMES)
    pred_class = None
    confidence = 0.0
    top3       = []

    print("Camera open. Perform a mudra — press R to reset, Q to quit.")

    cap_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_stride = max(1, int(round(cap_fps / TARGET_FPS)))
    frame_idx    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break
        frame_idx += 1
        if frame_idx % frame_stride != 0:
            cv2.imshow("Mudra Live Classification", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ── Extract landmarks for this frame ──────────────────
        lm = extract_frame_landmarks(frame, wholebody)  # (59, 3)
        buffer.append(lm)

        collecting = len(buffer) < TARGET_FRAMES

        # ── Run inference once buffer is full ─────────────────
        if not collecting:
            buf_arr    = np.stack(buffer, axis=0)        # (172, 59, 3)
            pred_class, confidence, top3 = run_inference(model, buf_arr, DEVICE)

        # ── Draw skeleton overlay ──────────────────────────────
        # Draw hand landmarks on frame for visual feedback
        if len(buffer) > 0:
            last_lm   = buffer[-1]                       # (59, 3)
            fh, fw    = frame.shape[:2]
            for idx in range(17, 59):                    # hands only
                x = int(last_lm[idx, 0] * fw)
                y = int(last_lm[idx, 1] * fh)
                if x > 0 or y > 0:
                    cv2.circle(frame, (x, y), 3, (0, 255, 200), -1)

        # ── Draw HUD ──────────────────────────────────────────
        frame = draw_hud(frame, pred_class, confidence,
                         top3, len(buffer), collecting)

        cv2.imshow("Mudra Live Classification", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            buffer.clear()
            pred_class = None
            print("Buffer reset.")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

# ── RUN ───────────────────────────────────────────────────────────────────
run_live_classification()