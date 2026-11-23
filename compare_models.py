"""
compare_models.py - figures/statistics that need BOTH models. Run after tcn_model.py and bilstm_model.py.
Reads only the files they saved in plots/ (history, predictions, reports, benchmarks), so every number
comes from the saved experiment logs.
Run from the project root:  python compare_models.py
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import common as C

K = C.MODELS
hist = {k: C.load_history(k) for k in K}


def first_epoch_reaching(acc, thr):
    hit = [i + 1 for i, a in enumerate(acc) if a >= thr]
    return hit[0] if hit else None


# ---------------------------------------------------------------- Figure 4: learning trajectories
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
for k, cfg in K.items():
    h = hist[k]
    ep = np.arange(1, len(h["val_acc"]) + 1)
    axes[0].plot(ep, [a * 100 for a in h["val_acc"]], color=cfg["color"], label=cfg["label"])
    axes[1].plot(ep, h["val_loss"], color=cfg["color"], label=cfg["label"])
    axes[0].scatter([h["best_epoch"]], [h["val_acc"][h["best_epoch"] - 1] * 100], marker="*", s=120,
                    color="red", zorder=5)
    axes[1].scatter([h["best_epoch"]], [h["val_loss"][h["best_epoch"] - 1]], marker="*", s=120,
                    color="red", zorder=5)
    for ax in axes:
        ax.axvline(h["last_epoch"], color=cfg["color"], linestyle=":", linewidth=1.5)
for ax, yl in zip(axes, ("Validation accuracy (%)", "Validation loss")):
    ax.set_xlim(0, C.NUM_EPOCHS)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(yl)
axes[0].set_ylim(0, 102)
axes[1].set_ylim(*C.LOSS_YLIM)
axes[0].set_title("Validation accuracy")
axes[1].set_title("Validation loss")
axes[0].legend(loc="lower right")
fig.tight_layout()
C.save_fig(fig, "fig4_training_comparison",
           "Learning trajectories of the TCN (green) and BiLSTM (red) on the validation partition: accuracy (left) and "
           "cross-entropy loss (right). Red stars mark each model's selected checkpoint; dotted vertical lines in the "
           "model's colour mark the last epoch before early stopping. Axis ranges match Figures 2 and 3.")
for k in K:
    a = hist[k]["val_acc"]
    print(f"{K[k]['label']:<7}: first >=80% val acc at epoch {first_epoch_reaching(a, .8)}, "
          f">=90% at epoch {first_epoch_reaching(a, .9)}, best checkpoint epoch {hist[k]['best_epoch']}, "
          f"last epoch {hist[k]['last_epoch']}")

# ---------------------------------------------------------------- Figure 5c: per-class F1 comparison
rep = {k: pd.read_csv(C.out_path(f"per_class_metrics_{k}.csv"), index_col=0) for k in K}
order = rep["tcn"]["f1"].sort_values().index
fig, ax = plt.subplots(figsize=(7.2, 7))
y, h = np.arange(len(order)), 0.38
ax.barh(y + h / 2, rep["tcn"].loc[order, "f1"], height=h, color=K["tcn"]["color"], label="TCN")
ax.barh(y - h / 2, rep["bilstm"].loc[order, "f1"], height=h, color=K["bilstm"]["color"], label="BiLSTM")
ax.set_yticks(y)
ax.set_yticklabels(order, fontsize=8)
ax.set_xlim(0, 1.05)
ax.set_xlabel("F1 score")
ax.set_title("Per-class F1 score: TCN vs. BiLSTM (validation set)")
ax.legend(loc="lower right")
fig.tight_layout()
C.save_fig(fig, "fig5c_f1_comparison",
           "Per-class F1 score of the TCN (green) and the BiLSTM (red) on the same validation partition, sorted by the "
           "TCN's F1 score.")

# ---------------------------------------------------------------- Figure 7: latency
bench = {k: json.load(open(C.out_path(f"inference_benchmark_{k}.json"))) for k in K}
fig, ax = plt.subplots(figsize=(3.6, 3.4))
means = [bench[k]["mean_ms"] for k in K]
stds = [bench[k]["std_ms"] for k in K]
ax.bar([K[k]["label"] for k in K], means, yerr=stds, color=[K[k]["color"] for k in K], capsize=8)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + max(means) * 0.02, f"{m:.1f} ms", ha="center")
ax.set_ylabel("Inference time per batch (ms)")
ax.set_title("Inference latency")
C.save_fig(fig, "fig7_latency",
           f"Forward-pass latency per batch of {bench['tcn']['batch_size']} clips (172 x 236 input), mean +- standard "
           f"deviation over {bench['tcn']['n_runs']} timed runs after {bench['tcn']['n_warmup']} warm-up runs, on "
           f"{bench['tcn']['device']} ({bench['tcn']['precision']}, CUDA-synchronised timing). Pose estimation and "
           f"preprocessing time are not included.")
for k in K:
    b = bench[k]
    print(f"{b['model']:<7}: {b['mean_ms']:.2f} +- {b['std_ms']:.2f} ms/batch | {b['per_clip_ms']:.3f} ms/clip | "
          f"{b['throughput_clips_per_s']:.0f} clips/s | peak mem {b['peak_gpu_mem_MB']} MB")
print(f"Speed-up (BiLSTM/TCN): {bench['bilstm']['mean_ms'] / bench['tcn']['mean_ms']:.2f}x")

# ---------------------------------------------------------------- paired statistics on the same validation clips
pr = {k: pd.read_csv(C.out_path(f"predictions_{k}.csv")) for k in K}
assert (pr["tcn"].video == pr["bilstm"].video).all(), "models were not evaluated on the same clips"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


rows = []
for k in K:
    n, c = len(pr[k]), int(pr[k].correct.sum())
    lo, hi = wilson(c, n)
    rows.append(dict(model=K[k]["label"], n_val=n, correct=c, accuracy=c / n, wilson95_low=lo, wilson95_high=hi))
b01 = int(((pr["tcn"].correct) & (~pr["bilstm"].correct)).sum())  # TCN right, BiLSTM wrong
b10 = int(((~pr["tcn"].correct) & (pr["bilstm"].correct)).sum())
p = stats.binomtest(b01, b01 + b10, 0.5).pvalue if (b01 + b10) > 0 else 1.0
df = pd.DataFrame(rows)
df["mcnemar_exact_p_tcn_vs_bilstm"] = p
df["tcn_only_correct"], df["bilstm_only_correct"] = b01, b10
df.to_csv(C.out_path("model_comparison_stats.csv"), index=False)
print(df.to_string(index=False))
