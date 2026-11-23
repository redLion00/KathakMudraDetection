"""
tcn_model.py - TCN model, training, evaluation, figures, latency, multi-seed runs and the
temporal-modelling ablations. Architecture and hyper-parameters are exactly those of model.ipynb.
Run from the project root:  python tcn_model.py
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

import common as C

RUN_MULTI_SEED = True  # 5 extra trainings of the TCN (mean +- std, 95% CI)
RUN_ABLATIONS = True  # position-only TCN, pooled MLP, static-frame MLP (5 seeds each)


# ----------------------------------------------------------------------------- model (unchanged)
class ResidualTCNBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               dilation=dilation, padding=self.padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               dilation=dilation, padding=self.padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

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
        residual = self.skip(x)

        out = self._causal_conv(self.conv1, x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self._causal_conv(self.conv2, out)
        out = self.bn2(out)

        out = out + residual
        return self.relu_out(out)


class TCNActionRecognizer(nn.Module):

    def __init__(self, num_classes: int, input_dim: int = 236, proj_dim: int = 128, kernel_size: int = 3,
                 dilations: tuple = (1, 2, 4, 8), num_heads: int = 4, attn_dropout: float = 0.1,
                 tcn_dropout: float = 0.2, head_dropout: float = 0.4) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )

        self.tcn_blocks = nn.ModuleList([
            ResidualTCNBlock(proj_dim, proj_dim, kernel_size=kernel_size, dilation=d, dropout=tcn_dropout)
            for d in dilations
        ])

        self.attn_norm = nn.LayerNorm(proj_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)

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
        x = self.input_proj(x)

        x = x.transpose(1, 2)
        for block in self.tcn_blocks:
            x = block(x)
        x = x.transpose(1, 2)

        residual = x
        x_norm = self.attn_norm(x)
        attn_out, _ = self.temporal_attn(x_norm, x_norm, x_norm)
        x = attn_out + residual

        x = x.mean(dim=1)
        return self.classifier(x)


def build_tcn(input_dim: int = 236, num_classes: int = 26) -> TCNActionRecognizer:
    return TCNActionRecognizer(num_classes=num_classes, input_dim=input_dim, proj_dim=128, kernel_size=3,
                               dilations=(1, 2, 4, 8), num_heads=4, attn_dropout=0.1,
                               tcn_dropout=0.2, head_dropout=0.4)


TRAIN_KW = dict(num_epochs=100, lr=3e-4, weight_decay=1e-4, grad_clip=1.0, label_smoothing=0.1, patience=15)


# ----------------------------------------------------------------------------- ablation helpers (Reviewer 2, comment 1)
POS_COLS = [i for i in range(236) if i % 4 < 2]  # x, y of the 59 landmarks (feature layout: x, y, vx, vy)


class FeatureView(Dataset):
    """Wraps SkeletonDataset and applies fn(x) to every (172, 236) sample, e.g. to drop the velocity channels."""

    def __init__(self, base, fn):
        self.base, self.fn, self.classes = base, fn, base.classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return self.fn(x), y


class MLPBaseline(nn.Module):
    """pool=True : mean over the 172 frames, then MLP (temporally pooled MLP)
       pool=False: input is already a single frame (static-frame MLP)."""

    def __init__(self, input_dim, num_classes, pool=True, hidden=256, dropout=0.4):
        super().__init__()
        self.pool = pool
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes))

    def forward(self, x):
        return self.net(x.mean(dim=1) if self.pool else x)


def run_ablations(full_dataset, train_idx, val_idx, class_weights, full_summary):
    classes, nc = full_dataset.classes, len(full_dataset.classes)
    # (key, display name, feature transform, model factory)
    variants = [
        ("tcn_pos_only", "TCN, position only", lambda x: x[:, POS_COLS], lambda: build_tcn(len(POS_COLS), nc)),
        ("mlp_pooled", "Pooled MLP (mean over time, pos + vel)", lambda x: x, lambda: MLPBaseline(236, nc, pool=True)),
        ("mlp_static", "Static MLP (middle frame, position only)", lambda x: x[x.shape[0] // 2, POS_COLS],
         lambda: MLPBaseline(len(POS_COLS), nc, pool=False)),
    ]
    rows = [dict(full_summary, model="TCN, position + velocity (proposed)")]
    for key, name, fn, build in variants:
        tl, vl = C.make_loaders(FeatureView(full_dataset, fn), train_idx, val_idx)
        print(f"\n--- Ablation: {name}")
        rows.append(dict(C.run_seeds(key, build, tl, vl, classes, class_weights, TRAIN_KW), model=name))
    df = pd.DataFrame(rows)
    df.to_csv(C.out_path("ablation_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(df))
    ax.bar(x, df.accuracy_mean * 100, yerr=df.accuracy_std * 100, capsize=5, color="#55A868")
    for i, r in df.iterrows():
        ax.text(i, r.accuracy_mean * 100 + r.accuracy_std * 100 + 1, f"{r.accuracy_mean * 100:.1f}", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(", ", "\n").replace(" (", "\n(") for m in df.model], fontsize=8)
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Ablation of temporal modelling and velocity features")
    C.save_fig(fig, "fig8_ablation_tcn",
               f"Validation accuracy (mean +- standard deviation over {len(C.SEEDS)} random seeds, same {C.SPLIT_MODE} "
               f"split) of the proposed TCN and three simpler baselines: a TCN without velocity channels, an MLP on "
               f"temporally pooled features and an MLP on a single middle frame.")


# ----------------------------------------------------------------------------- main
def main():
    C.set_seed(C.SEED)
    full_dataset, train_idx, val_idx, train_loader, val_loader, class_weights = C.prepare_data()
    nc = len(full_dataset.classes)
    C.plot_class_distribution(full_dataset, train_idx, val_idx)

    model = build_tcn(num_classes=nc)
    assert model(torch.randn(2, 172, 236)).shape == (2, nc), "Shape mismatch!"
    print(f"Forward-pass shape check passed: (2, 172, 236) -> (2, {nc})")

    ck = C.ckpt_path(C.MODELS["tcn"]["ckpt"])
    history = C.train(model, train_loader, val_loader, num_classes=nc, class_weights=class_weights,
                      checkpoint_path=ck, device=C.DEVICE, **TRAIN_KW)
    C.save_history(history, "tcn")
    C.plot_training(history, "tcn")

    eval_model = C.load_checkpoint(ck, build_tcn(num_classes=nc), device=C.DEVICE)
    C.evaluate_and_plot(eval_model, "tcn", full_dataset, val_idx, val_loader)
    C.measure_inference_latency(eval_model, "tcn")

    summary = None
    if RUN_MULTI_SEED or RUN_ABLATIONS:
        print("\n=== Multi-seed TCN")
        summary = C.run_seeds("tcn", lambda: build_tcn(num_classes=nc), train_loader, val_loader,
                              full_dataset.classes, class_weights, TRAIN_KW)
        pd.DataFrame([summary]).to_csv(C.out_path("seed_summary_tcn.csv"), index=False)
    if RUN_ABLATIONS:
        run_ablations(full_dataset, train_idx, val_idx, class_weights, summary)


if __name__ == "__main__":
    main()
