"""
bilstm_model.py - BiLSTM model, training, evaluation, figures, latency and multi-seed runs.
Architecture and hyper-parameters are exactly those used for the BiLSTM in model.ipynb
(hidden_dim=512, 4 layers, lr=1e-3, grad_clip=0.5, head_dropout=0.3).
Run from the project root:  python bilstm_model.py
"""
import torch
import torch.nn as nn
import pandas as pd

import common as C

RUN_MULTI_SEED = True  # 5 extra trainings of the BiLSTM (mean +- std, 95% CI)


# ----------------------------------------------------------------------------- model (unchanged)
class BiLSTMActionRecognizer(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = 236, proj_dim: int = 128, hidden_dim: int = 256,
                 num_layers: int = 2, num_heads: int = 4, attn_dropout: float = 0.1,
                 lstm_dropout: float = 0.3, head_dropout: float = 0.4):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
        )

        self.bilstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        self.proj = nn.Linear(hidden_dim * 2, proj_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.bilstm(x)
        x = self.proj(x)

        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out

        x = x.mean(dim=1)
        return self.classifier(x)


def build_bilstm(num_classes: int = 26) -> BiLSTMActionRecognizer:
    return BiLSTMActionRecognizer(num_classes=num_classes, input_dim=236, proj_dim=128, hidden_dim=512,
                                  num_layers=4, num_heads=4, attn_dropout=0.1, lstm_dropout=0.2,
                                  head_dropout=0.3)


TRAIN_KW = dict(num_epochs=100, lr=1e-3, weight_decay=1e-4, grad_clip=0.5, label_smoothing=0.1, patience=15)


# ----------------------------------------------------------------------------- main
def main():
    C.set_seed(C.SEED)
    full_dataset, train_idx, val_idx, train_loader, val_loader, class_weights = C.prepare_data()
    nc = len(full_dataset.classes)

    model = build_bilstm(nc)
    assert model(torch.randn(2, 172, 236)).shape == (2, nc), "Shape mismatch!"
    print(f"Forward-pass check passed: (2, 172, 236) -> (2, {nc})")

    ck = C.ckpt_path(C.MODELS["bilstm"]["ckpt"])
    history = C.train(model, train_loader, val_loader, num_classes=nc, class_weights=class_weights,
                      checkpoint_path=ck, device=C.DEVICE, **TRAIN_KW)
    C.save_history(history, "bilstm")
    C.plot_training(history, "bilstm")

    eval_model = C.load_checkpoint(ck, build_bilstm(nc), device=C.DEVICE)
    C.evaluate_and_plot(eval_model, "bilstm", full_dataset, val_idx, val_loader)
    C.measure_inference_latency(eval_model, "bilstm")

    if RUN_MULTI_SEED:
        print("\n=== Multi-seed BiLSTM")
        summary = C.run_seeds("bilstm", lambda: build_bilstm(nc), train_loader, val_loader,
                              full_dataset.classes, class_weights, TRAIN_KW)
        pd.DataFrame([summary]).to_csv(C.out_path("seed_summary_bilstm.csv"), index=False)


if __name__ == "__main__":
    main()
