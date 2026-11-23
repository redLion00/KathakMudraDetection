"""
pipeline_diagram.py - end-to-end workflow figure (Reviewer 3). Saves plots/fig_workflow.png/.pdf.
Run from the project root:  python pipeline_diagram.py
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_SAVE_PATH = "plots/"
os.makedirs(FIG_SAVE_PATH, exist_ok=True)
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})

PRE, SHARED, TCN, LSTM = "#DCE6F4", "#E9E3F3", "#DDEFD9", "#F6DCDC"
fig, ax = plt.subplots(figsize=(7.2, 8.6))
ax.set_xlim(0, 100); ax.set_ylim(-10, 102); ax.axis("off")


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=color, ec="#444444", lw=0.9))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontweight="bold", fontsize=8)
    ax.text(x + w / 2, y + h - 7.2, body, ha="center", va="top", fontsize=6.8, linespacing=1.25)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.1))


W, H = 21.5, 17
H2 = 28
# --- row 1: acquisition -> landmarks (offline preprocessing)
r1 = 82
xs = [2, 27, 52, 77]
box(xs[0], r1, W, H, "1  Video acquisition", "Instructional +\nperformance clips\n.mov/.mp4 -> H.264 .mp4\n1080 x 1920, 26 classes", PRE)
box(xs[1], r1, W, H, "2  Quality filtering", "blur, brightness, fps,\nresolution, duration\n(VideoReport.csv;\nblur check manual)", PRE)
box(xs[2], r1, W, H, "3  Pose estimation", "RTMPose Wholebody\n(rtmlib, performance)\nframe -> 133 x (x, y, score)", PRE)
box(xs[3], r1, W, H, "4  Landmark selection", "17 body + 21 left hand\n+ 21 right hand\n-> T x 59 x 3\n(x, y, confidence)", PRE)
for i in range(3):
    arrow(xs[i] + W + 0.6, r1 + H / 2, xs[i + 1] - 0.6, r1 + H / 2)

# --- row 2: normalisation -> features
r2 = 48
box(xs[3], r2, W, H2, "5  Normalisation", "body: shoulder-midpoint\nand -width scaling;\nhand: wrist-centred,\nmax-norm scaling;\n< 10 pts > 0.3 -> zeros;\n1-frame gaps interpolated;\nframes without both\nhands dropped", PRE)
box(xs[2], r2, W, H2, "6  Resampling", "downsample to ~15 fps;\nlinear interpolation\nto 172 frames\n172 x 59 x 3", PRE)
box(xs[1], r2, W, H2, "7  Features", "drop confidence;\nadd velocity (x_t - x_t-1);\nz-score positions\n172 x 59 x 4 -> 172 x 236", SHARED)
box(xs[0], r2, W, H2, "8  Data split", "stratified clip-level split\n(85 / 15, seed 42);\nclass-weighted loss;\nlabel smoothing 0.1", SHARED)
arrow(xs[3] + W / 2, r1 - 0.6, xs[3] + W / 2, r2 + H2 + 0.6)
for i in (3, 2, 1):
    arrow(xs[i] - 0.6, r2 + H2 / 2, xs[i - 1] + W + 0.6, r2 + H / 2)

# --- row 3: models
r3 = 8
mw = 34
box(6, r3, mw, 28, "9a  TCN (proposed)", "Linear 236 -> 128 + LayerNorm\n4 residual dilated blocks\n(d = 1, 2, 4, 8; k = 3)\ncausal-trim convolutions\nself-attention (4 heads)\nglobal avg. pool -> 128\nMLP head -> 26", TCN)
box(52, r3, mw, 28, "9b  BiLSTM (baseline)", "Linear 236 -> 128\n4-layer BiLSTM (512 / dir.)\nLinear 1024 -> 128\nself-attention (4 heads)\nglobal avg. pool -> 128\nMLP head -> 26", LSTM)
arrow(xs[0] + W / 2, r2 - 0.6, 23, r3 + 28 + 0.6)
arrow(xs[0] + W / 2 + 8, r2 - 0.6, 69, r3 + 28 + 0.6)
ax.text(40, 41.5, "input 172 x 236", fontsize=7, ha="center", style="italic", bbox=dict(fc="white", ec="none", pad=1))

# --- output
ax.add_patch(FancyBboxPatch((28, -5), 44, 7.5, boxstyle="round,pad=0.4,rounding_size=1.2", fc="#FFF3CD", ec="#444444", lw=0.9))
ax.text(50, -1.2, "10  Softmax over 26 mudra classes -> predicted label", ha="center", va="center", fontweight="bold", fontsize=8)
arrow(23, r3 - 0.6, 40, 3.2)
arrow(69, r3 - 0.6, 60, 3.2)

# --- legend
for x0, c, t in [(0, PRE, "Offline preprocessing"), (27, SHARED, "Shared by both models"),
                 (56, TCN, "TCN branch"), (75, LSTM, "BiLSTM branch")]:
    ax.add_patch(plt.Rectangle((x0, -13), 2.2, 2.2, fc=c, ec="#444444", lw=0.7, clip_on=False))
    ax.text(x0 + 3.2, -11.9, t, fontsize=6.8, va="center")

fig.savefig(os.path.join(FIG_SAVE_PATH, "fig_workflow.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_SAVE_PATH, "fig_workflow.pdf"), bbox_inches="tight")
print("saved plots/fig_workflow.png / .pdf")
