"""
common.py - shared code for tcn_model.py and bilstm_model.py
(data loading, split + audit files, training loop, evaluation, plotting, benchmarking)

Everything that was copy-pasted between the TCN and BiLSTM sections of model.ipynb lives here.
Hyper-parameters, model definitions and the training recipe are UNCHANGED (they stay in the two model files).
Run the scripts from the project root (KathakMudraDetection/), same as the notebooks.
"""
import json
import os
import platform
import random
import time
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- paths / config
PROCESSED_LANDMARK_PATH = r"landmarks_processed/"
MODEL_SAVE_PATH = r"models/saved/"
REPORT_PATH = r"VideoReport_updated.csv"
FIG_SAVE_PATH = r"plots/"

BATCH_SIZE = 32
VAL_SPLIT = 0.15
NUM_WORKERS = 0  # windows
SEED = 42
NUM_EPOCHS = 100  # used only to fix the x-axis range of all training figures

# "random" = original clip-level stratified split (paper's current result)
# "group"  = source/performer-disjoint split (needs METADATA_PATH, see group_split()).
#            All outputs get a "_group" suffix so the two protocols never overwrite each other.
SPLIT_MODE = "random"
METADATA_PATH = r"clip_metadata.csv"  # columns: video (e.g. 1-Ardhpatka/IMG_1246.mp4), source
GROUP_COLUMN = "source"  # recording session / performer id

SEEDS = [11, 22, 33, 44, 55]  # multi-seed experiments (Reviewer 1, comment 6)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAG = "" if SPLIT_MODE == "random" else f"_{SPLIT_MODE}"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(FIG_SAVE_PATH, exist_ok=True)

MODELS = {
    "tcn": dict(label="TCN", color="#55A868", cmap="Blues", ckpt="best_tcn.pt",
                fig_train="fig2_tcn_training", fig_f1="fig5a_f1_tcn", fig_cm="fig6a_confusion_tcn"),
    "bilstm": dict(label="BiLSTM", color="#C44E52", cmap="Reds", ckpt="best_bilstm.pt",
                   fig_train="fig3_bilstm_training", fig_f1="fig5b_f1_bilstm", fig_cm="fig6b_confusion_bilstm"),
}
TRAIN_COLOR, VAL_COLOR = "#4C72B0", "#DD8452"
LOSS_YLIM = (0, 4.0)  # identical axis ranges in every training figure (Reviewer 3)


def setup_style():
    """One publication style for all figures: 7.2 in (full page width) at 300 dpi, 9-11 pt text."""
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "figure.dpi": 100, "savefig.dpi": 300,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


setup_style()


def out_path(filename: str) -> str:
    """plots/<name><TAG>.<ext>  (TAG is '' for the random split)."""
    base, ext = os.path.splitext(filename)
    return os.path.join(FIG_SAVE_PATH, base + TAG + ext)


def ckpt_path(filename: str) -> str:
    base, ext = os.path.splitext(filename)
    return os.path.join(MODEL_SAVE_PATH, base + TAG + ext)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_fig(fig, name: str, caption: Optional[str] = None) -> None:
    """Saves PNG (300 dpi) + vector PDF, and stores a suggested caption in plots/figure_captions.json."""
    fig.savefig(out_path(name + ".png"), dpi=300)
    fig.savefig(out_path(name + ".pdf"))
    plt.close(fig)
    if caption:
        p = out_path("figure_captions.json")
        caps = json.load(open(p)) if os.path.exists(p) else {}
        caps[name + TAG] = caption
        json.dump(caps, open(p, "w"), indent=2)


def save_json(obj, name: str) -> None:
    json.dump(obj, open(out_path(name), "w"), indent=2, default=float)


# ----------------------------------------------------------------------------- dataset
class SkeletonDataset(Dataset):

    def __init__(self, root: str) -> None:
        self.samples: list[tuple[str, int]] = []
        self.classes: list[str] = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            for fname in sorted(os.listdir(cls_dir)):  # sorted -> split reproducible on every OS
                if fname.endswith(".npy"):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls])
                    )

        print(f"  Dataset: {len(self.samples)} clips, {len(self.classes)} classes")
        print(f"  Classes: {self.classes}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        arr = np.load(path).astype(np.float32)  # (172, 59, 3)
        arr = arr[:, :, :2]  # remove conf

        vel = np.zeros_like(arr)
        vel[1:] = arr[1:] - arr[:-1]
        arr = np.concatenate([arr, vel], axis=-1)

        # normalization
        pos = arr[:, :, :2]
        mu = pos.mean(axis=(0, 1), keepdims=True)
        std = pos.std(axis=(0, 1), keepdims=True) + 1e-6
        arr[:, :, :2] = (pos - mu) / std

        seq = arr.reshape(172, -1)

        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)


def video_key(path: str) -> str:
    """landmarks_processed/1-Ardhpatka/IMG_1246.npy -> '1-Ardhpatka/IMG_1246.mp4' (same key as VideoReport)."""
    rel = os.path.relpath(path, PROCESSED_LANDMARK_PATH)
    return rel.replace("\\", "/").replace(".npy", ".mp4")


def report_lookup(column: str) -> dict:
    """VideoReport column as {video_key: value}; keys normalised to forward slashes."""
    rep = pd.read_csv(REPORT_PATH)
    return dict(zip(rep["video"].astype(str).str.replace("\\", "/", regex=False), rep[column]))


def make_loaders(dataset, train_idx, val_idx):
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader


# ----------------------------------------------------------------------------- split + audit files (Reviewer 1: 1, 2, 3)
def group_split(full_dataset, labels):
    """Source/performer-disjoint split. Needs clip_metadata.csv with columns [video, source];
    without it only exact duplicate files (sha1 from VideoReport) are kept together."""
    keys = [video_key(p) for p, _ in full_dataset.samples]
    if os.path.exists(METADATA_PATH):
        meta = pd.read_csv(METADATA_PATH)
        lookup = dict(zip(meta["video"], meta[GROUP_COLUMN].astype(str)))
        missing = [k for k in keys if k not in lookup]
        if missing:
            print(f"  WARNING: {len(missing)} clips missing in {METADATA_PATH}; treated as their own group")
        groups = [lookup.get(k, k) for k in keys]
    else:
        print(f"  WARNING: {METADATA_PATH} not found -> groups = exact-duplicate files only (NOT source-disjoint)")
        sha = report_lookup("sha1")
        groups = [str(sha.get(k, k)) for k in keys]
    n_splits = max(2, int(round(1 / VAL_SPLIT)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    tr, va = next(sgkf.split(np.zeros(len(labels)), labels, groups))
    return tr.tolist(), va.tolist()


def audit_split(full_dataset, train_idx, val_idx):
    """Writes the partition files + per-class counts, and checks for train/val leakage."""
    classes, samples = full_dataset.classes, full_dataset.samples
    rows = [{"video": video_key(samples[i][0]), "class": classes[samples[i][1]], "subset": s}
            for s, idx in (("train", train_idx), ("val", val_idx)) for i in idx]
    part = pd.DataFrame(rows)
    part.to_csv(out_path("partition_files.csv"), index=False)

    counts = (part.pivot_table(index="class", columns="subset", values="video", aggfunc="count", fill_value=0)
              .reindex(classes).fillna(0).astype(int))
    for c in ("train", "val"):
        if c not in counts:
            counts[c] = 0
    counts["total"] = counts["train"] + counts["val"]
    counts[["total", "train", "val"]].to_csv(out_path("class_counts_by_split.csv"))
    print("\n  Per-class clip counts (total / train / val):")
    print(counts[["total", "train", "val"]].to_string())
    empty = counts.index[counts["train"] == 0].tolist()
    if empty:
        print(f"  WARNING: classes with NO training clip: {empty}")

    # exact duplicate files shared between train and val
    try:
        sha = report_lookup("sha1")
        tr_sha = {sha.get(v) for v in part.loc[part.subset == "train", "video"]} - {None}
        leaked = [v for v in part.loc[part.subset == "val", "video"] if sha.get(v) in tr_sha]
        print(f"\n  Exact-duplicate leakage (same sha1 in train and val): {len(leaked)} val clip(s)")
        pd.Series(leaked, name="video").to_csv(out_path("leakage_exact_duplicates.csv"), index=False)
    except Exception as e:  # report missing / no sha1 column
        print(f"  (duplicate check skipped: {e})")

    # near-duplicate check: nearest TRAIN neighbour of every val clip (cosine similarity of normalised positions)
    def feat(i):
        x = full_dataset[i][0].numpy().reshape(172, 59, 4)[:, :, :2].ravel()
        x = x - x.mean()
        return x / (np.linalg.norm(x) + 1e-8)

    F_tr = np.stack([feat(i) for i in train_idx])
    F_va = np.stack([feat(i) for i in val_idx])
    sim = F_va @ F_tr.T
    nn_i = sim.argmax(1)
    near = pd.DataFrame({
        "val_video": [video_key(samples[i][0]) for i in val_idx],
        "nearest_train_video": [video_key(samples[train_idx[j]][0]) for j in nn_i],
        "cosine_similarity": sim.max(1),
        "same_class": [samples[i][1] == samples[train_idx[j]][1] for i, j in zip(val_idx, nn_i)],
    }).sort_values("cosine_similarity", ascending=False)
    near.to_csv(out_path("near_duplicates_val_vs_train.csv"), index=False)
    print("  Most similar val->train pairs (inspect manually for same recording / take):")
    print(near.head(5).to_string(index=False))


def prepare_data():
    full_dataset = SkeletonDataset(PROCESSED_LANDMARK_PATH)
    num_classes = len(full_dataset.classes)

    try:
        _report_n = pd.read_csv(REPORT_PATH).shape[0]
        print(f"Clips listed in {REPORT_PATH}      : {_report_n}")
        print(f"Clips actually usable (processed landmarks): {len(full_dataset)}  "
              f"(dropped: {_report_n - len(full_dataset)})")
    except Exception as e:
        print(f"Could not read {REPORT_PATH}: {e}")

    all_indices = list(range(len(full_dataset)))
    all_labels_list = [full_dataset.samples[i][1] for i in all_indices]

    # split is made AFTER the unusable clips were removed (only processed landmarks are listed)
    if SPLIT_MODE == "group":
        train_idx, val_idx = group_split(full_dataset, all_labels_list)
    else:
        train_idx, val_idx = train_test_split(
            all_indices, test_size=VAL_SPLIT, stratify=all_labels_list, random_state=SEED)

    train_loader, val_loader = make_loaders(full_dataset, train_idx, val_idx)

    print(f"\nSplit mode: {SPLIT_MODE}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Number of Classes: {num_classes}")
    print(f"Train clips: {len(train_idx)} | Val clips: {len(val_idx)}")

    all_labels = torch.tensor(all_labels_list)
    class_counts = torch.bincount(all_labels, minlength=num_classes).float()
    class_weights = (class_counts.sum() / (num_classes * class_counts)).clamp(max=10.0)
    print(f"Class weights: {class_weights.tolist()}")

    audit_split(full_dataset, train_idx, val_idx)
    return full_dataset, train_idx, val_idx, train_loader, val_loader, class_weights


# ----------------------------------------------------------------------------- training
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min", verbose: bool = True) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score < self.best_score - self.min_delta if self.mode == "min"
            else score > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"    EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_summary(model: nn.Module) -> None:
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    print("=" * 52)
    print(f"  Model          : {model.__class__.__name__}")
    print(f"  Total params   : {total:,}")
    print(f"  Trainable      : {trainable:,}")
    print(f"  Non-trainable  : {total - trainable:,}")
    print("=" * 52)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, grad_clip=1.0) -> dict:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.perf_counter()

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += inputs.size(0)

    elapsed = time.perf_counter() - t0
    return {"loss": total_loss / total, "accuracy": correct / total, "samples_per_sec": total / elapsed}


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits, labels)

        total_loss += loss.item() * inputs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += inputs.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


def train(model, train_loader, val_loader, *, num_classes: int, num_epochs: int = 100, lr: float = 3e-4,
          weight_decay: float = 1e-4, grad_clip: float = 1.0, label_smoothing: float = 0.1,
          class_weights: Optional[torch.Tensor] = None, patience: int = 15,
          checkpoint_path: str = "best.pt", device: Optional[torch.device] = None,
          verbose: bool = True) -> dict:
    """AdamW + CosineAnnealingLR + AMP + early stopping on validation accuracy.
    Returns history (train_loss, train_acc, val_loss, val_acc) + best_epoch / last_epoch / patience."""
    if device is None:
        device = DEVICE

    model = model.to(device)
    if verbose:
        model_summary(model)
        print(f"  Device: {device}\n")

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 1e-2)
    scaler = GradScaler(device=device.type, enabled=device.type == "cuda")
    early_stop = EarlyStopping(patience=patience, mode="max", verbose=verbose)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch = 0.0, 0

    if verbose:
        hdr = f"{'Epoch':>6} | {'TrLoss':>8} | {'TrAcc':>7} | {'VaLoss':>8} | {'VaAcc':>7} | {'LR':>9}"
        print(hdr)
        print("-" * len(hdr))

    for epoch in range(1, num_epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, grad_clip)
        va = validate(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["accuracy"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["accuracy"])

        if verbose:
            print(f"{epoch:>6} | {tr['loss']:>8.4f} | {tr['accuracy']:>6.2%} | "
                  f"{va['loss']:>8.4f} | {va['accuracy']:>6.2%} | {lr_now:>9.2e}")

        if va["accuracy"] > best_val_acc:
            best_val_acc, best_epoch = va["accuracy"], epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "classes": train_loader.dataset.dataset.classes
                if hasattr(train_loader.dataset, "dataset") else [],
            }, checkpoint_path)

        if early_stop(va["accuracy"]):
            if verbose:
                print(f"\n  Early stopping at epoch {epoch} (best checkpoint: epoch {best_epoch}).")
            break

    history.update(best_epoch=best_epoch, last_epoch=epoch, patience=patience)
    if verbose:
        print(f"\n  Best val accuracy: {best_val_acc:.2%} (epoch {best_epoch}) ->  '{checkpoint_path}'")
    return history


def load_checkpoint(checkpoint_path: str, model: nn.Module, device: Optional[torch.device] = None):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint '{checkpoint_path}' "
          f"(epoch {ckpt.get('epoch', '?')}, val_acc={ckpt.get('val_acc', float('nan')):.2%})")
    return model


def save_history(history: dict, key: str) -> None:
    df = pd.DataFrame({k: v for k, v in history.items() if isinstance(v, list)})
    df.insert(0, "epoch", range(1, len(df) + 1))
    for k in ("best_epoch", "last_epoch", "patience"):
        df[k] = history[k]
    df.to_csv(out_path(f"history_{key}.csv"), index=False)


def load_history(key: str) -> dict:
    df = pd.read_csv(out_path(f"history_{key}.csv"))
    h = {c: df[c].tolist() for c in ("train_loss", "train_acc", "val_loss", "val_acc")}
    h.update({k: int(df[k].iloc[0]) for k in ("best_epoch", "last_epoch", "patience")})
    return h


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, classes: list, device=None, verbose: bool = True) -> dict:
    """Full evaluation pass. Returns accuracy, loss, class_report, confusion_matrix, y_true, y_pred."""
    if device is None:
        device = DEVICE
    num_classes = len(classes)
    criterion = nn.CrossEntropyLoss()
    model.to(device).eval()

    all_preds, all_labels, total_loss, total = [], [], 0.0, 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits, labels)
        total_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = (all_preds == all_labels).float().mean().item()

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(all_labels.tolist(), all_preds.tolist()):
        confusion[true][pred] += 1

    class_report = {}
    for i, cls in enumerate(classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        class_report[cls] = {"precision": round(precision, 4), "recall": round(recall, 4),
                             "f1": round(f1, 4), "support": int(confusion[i, :].sum())}

    if verbose:
        print(f"\n{'=' * 58}")
        print(f"  Overall accuracy : {accuracy:.2%}  ({int((all_preds == all_labels).sum())}/{total})")
        print(f"  Average loss     : {total_loss / total:.4f}")
        print(f"{'=' * 58}")
        print(f"  {'Class':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
        print(f"  {'-' * 47}")
        for cls, m in class_report.items():
            print(f"  {cls:<20} {m['precision']:>6.2%} {m['recall']:>6.2%} {m['f1']:>6.2%} {m['support']:>5}")
        print(f"{'=' * 58}\n")

    return {"accuracy": accuracy, "avg_loss": round(total_loss / total, 4), "class_report": class_report,
            "confusion_matrix": confusion, "y_true": all_labels.numpy(), "y_pred": all_preds.numpy()}


def macro_f1(results: dict) -> float:
    return float(np.mean([m["f1"] for m in results["class_report"].values()]))


def save_predictions_and_report(results, full_dataset, val_idx, key):
    """Saves per-clip predictions and recomputes EVERY aggregate metric from that saved file
    (Reviewer 1, comments 2 and 4). Also splits accuracy by hand-landmark confidence (Reviewer 2)."""
    classes = full_dataset.classes
    paths = [full_dataset.samples[i][0] for i in val_idx]  # val_loader is not shuffled -> same order
    df = pd.DataFrame({
        "video": [video_key(p) for p in paths],
        "true_idx": results["y_true"], "true_class": [classes[i] for i in results["y_true"]],
        "pred_idx": results["y_pred"], "pred_class": [classes[i] for i in results["y_pred"]],
    })
    df["correct"] = df.true_idx == df.pred_idx
    df["mean_hand_conf"] = [float(np.load(p)[:, 17:, 2].mean()) for p in paths]
    df.to_csv(out_path(f"predictions_{key}.csv"), index=False)

    # ---- everything below is recomputed from the saved file
    df = pd.read_csv(out_path(f"predictions_{key}.csv"))
    n, k = len(df), int(df.correct.sum())
    print(f"[{key}] validation samples = {n} | correct = {k} | accuracy = {k / n:.4%}")
    rep = pd.DataFrame(classification_report(df.true_idx, df.pred_idx, labels=list(range(len(classes))),
                                             target_names=classes, output_dict=True, zero_division=0)).T
    rep.to_csv(out_path(f"classification_report_{key}.csv"))
    per_class = rep.loc[classes, ["precision", "recall", "f1-score", "support"]].rename(columns={"f1-score": "f1"})
    per_class["support"] = per_class["support"].astype(int)
    per_class.to_csv(out_path(f"per_class_metrics_{key}.csv"))
    print(f"[{key}] macro-P {rep.loc['macro avg', 'precision']:.4f} | macro-R {rep.loc['macro avg', 'recall']:.4f} | "
          f"macro-F1 {rep.loc['macro avg', 'f1-score']:.4f} | errors = {n - k}")
    print("  misclassified:", [(r.video, r.true_class, r.pred_class) for r in df[~df.correct].itertuples()])

    # accuracy by hand-landmark confidence tercile
    df["hand_conf_level"] = pd.qcut(df.mean_hand_conf, 3, labels=["low", "medium", "high"], duplicates="drop")
    vis = df.groupby("hand_conf_level").agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                           mean_conf=("mean_hand_conf", "mean"))
    vis.to_csv(out_path(f"accuracy_by_hand_confidence_{key}.csv"))
    print(vis.to_string())
    return df, per_class, float(rep.loc["macro avg", "f1-score"])


# ----------------------------------------------------------------------------- figures
def plot_class_distribution(full_dataset, train_idx, val_idx):
    labels = np.array([s[1] for s in full_dataset.samples])
    C = len(full_dataset.classes)
    tr = np.bincount(labels[train_idx], minlength=C)
    va = np.bincount(labels[val_idx], minlength=C)
    tot = tr + va
    order = np.argsort(tot)[::-1]
    names = [full_dataset.classes[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(C)
    ax.bar(x, tr[order], color="#4C72B0", label="Training")
    ax.bar(x, va[order], bottom=tr[order], color="#DD8452", label="Validation")
    ax.axhline(tot.mean(), color="red", linestyle="--", linewidth=1, label=f"Mean = {tot.mean():.1f} clips/class")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Number of clips")
    ax.set_title("Per-class clip distribution")
    ax.legend()
    save_fig(fig, "fig1_class_distribution",
             f"Number of usable clips per class ({tot.sum()} clips, {C} classes) after quality filtering and landmark "
             f"extraction, split into training ({tr.sum()}) and validation ({va.sum()}) clips ({SPLIT_MODE} split). "
             f"Classes are sorted by total count; the dashed line is the mean clips per class. "
             f"Min = {tot.min()} ({full_dataset.classes[tot.argmin()]}), max = {tot.max()} "
             f"({full_dataset.classes[tot.argmax()]}).")
    print(f"Min clips/class: {tot.min()} ({full_dataset.classes[tot.argmin()]}) | "
          f"Max clips/class: {tot.max()} ({full_dataset.classes[tot.argmax()]})")


def plot_training(history: dict, key: str):
    cfg = MODELS[key]
    ep = np.arange(1, len(history["train_loss"]) + 1)
    best, last = history["best_epoch"], history["last_epoch"]
    best_acc = history["val_acc"][best - 1] * 100

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, tr, va, ylab in ((axes[0], history["train_loss"], history["val_loss"], "Loss"),
                             (axes[1], [a * 100 for a in history["train_acc"]],
                              [a * 100 for a in history["val_acc"]], "Accuracy (%)")):
        ax.plot(ep, tr, color=TRAIN_COLOR, label="Train")
        ax.plot(ep, va, color=VAL_COLOR, label="Validation")
        ax.axvline(last, color="grey", linestyle=":", linewidth=1.5, label=f"Training stopped (epoch {last})")
        ax.set_xlim(0, NUM_EPOCHS)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
    axes[0].set_ylim(*LOSS_YLIM)
    axes[1].set_ylim(0, 102)
    axes[0].scatter([best], [history["val_loss"][best - 1]], marker="*", s=120, color="red", zorder=5)
    axes[1].scatter([best], [best_acc], marker="*", s=120, color="red", zorder=5,
                    label=f"Best checkpoint (epoch {best}, {best_acc:.2f}%)")
    axes[0].set_title(f"{cfg['label']}: loss")
    axes[1].set_title(f"{cfg['label']}: accuracy")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, cfg["fig_train"],
             f"Training history of the {cfg['label']} (left: cross-entropy loss; right: accuracy). Blue = training set, "
             f"orange = validation set. The red star marks the checkpoint kept for evaluation (epoch {best}, validation "
             f"accuracy {best_acc:.2f}%); the dotted vertical line marks the last epoch run (epoch {last}), at which "
             f"early stopping (patience {history['patience']} epochs on validation accuracy) ended training. "
             f"Axis ranges are identical for all models.")


def plot_f1(per_class: pd.DataFrame, key: str, n_val: int):
    cfg = MODELS[key]
    df = per_class.sort_values("f1")
    macro = df["f1"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 6))
    y = np.arange(len(df))
    ax.barh(y, df["f1"], color=cfg["color"])
    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=8)
    ax.axvline(macro, color="red", linestyle="--", label=f"Macro-avg F1 = {macro:.3f}")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 score")
    ax.set_title(f"Per-class F1 score - {cfg['label']} (validation set, n = {n_val})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, cfg["fig_f1"],
             f"Per-class F1 score of the {cfg['label']} on the validation partition (n = {n_val} clips, {SPLIT_MODE} "
             f"split), sorted in ascending order. The dashed red line is the macro-averaged F1 ({macro:.3f}), i.e. the "
             f"unweighted mean over the {len(df)} classes.")


def plot_confusion(cm: np.ndarray, classes: list, key: str):
    cfg = MODELS[key]
    C = len(classes)
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[3.6, 0.09, 1.0], wspace=0.28)
    ax, cax, axl = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    axl.axis("off")
    im = ax.imshow(cm, cmap=cfg["cmap"], vmin=0)
    ids = [str(i + 1) for i in range(C)]
    ax.set_xticks(range(C)); ax.set_xticklabels(ids, rotation=90, fontsize=7)
    ax.set_yticks(range(C)); ax.set_yticklabels(ids, fontsize=7)
    ax.set_xticks(np.arange(-.5, C, 1), minor=True)
    ax.set_yticks(np.arange(-.5, C, 1), minor=True)
    ax.grid(which="minor", color="#BBBBBB", linewidth=0.4)
    ax.tick_params(which="minor", length=0)
    for i in range(C):
        for j in range(C):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > 0.6 * cm.max() else "black")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix - {cfg['label']} (validation set, n = {int(cm.sum())})")
    cb = fig.colorbar(im, cax=cax)
    cax.set_title("Count", fontsize=8)
    if cm.max() <= 10:
        cb.set_ticks(range(0, int(cm.max()) + 1))
    axl.text(0.0, 1.0, "\n".join(f"{i + 1:>2}  {c}" for i, c in enumerate(classes)),
             va="top", ha="left", family="monospace", fontsize=7, transform=axl.transAxes)
    save_fig(fig, cfg["fig_cm"],
             f"Confusion matrix of the {cfg['label']} on the validation partition (n = {int(cm.sum())} clips, "
             f"{SPLIT_MODE} split). Rows are true classes, columns predicted classes; non-zero cells show the clip "
             f"count. Axis numbers refer to the class legend on the right; {int(cm.sum() - np.trace(cm))} clip(s) "
             f"are misclassified (off-diagonal cells).")


def evaluate_and_plot(model, key, full_dataset, val_idx, val_loader):
    """Evaluate -> save predictions/reports -> per-class F1 figure + confusion-matrix figure."""
    classes = full_dataset.classes
    results = evaluate(model, val_loader, classes=classes, device=DEVICE)
    pd.DataFrame(results["confusion_matrix"], index=classes, columns=classes).to_csv(
        out_path(f"confusion_matrix_{key}.csv"))
    _, per_class, _ = save_predictions_and_report(results, full_dataset, val_idx, key)
    plot_f1(per_class, key, n_val=len(val_idx))
    plot_confusion(results["confusion_matrix"], classes, key)

    cm = results["confusion_matrix"].copy()
    np.fill_diagonal(cm, 0)
    pairs = sorted(((classes[i], classes[j], int(cm[i, j])) for i in range(len(classes))
                    for j in range(len(classes)) if cm[i, j] > 0), key=lambda t: -t[2])
    print(f"Top confused (true -> predicted) pairs ({MODELS[key]['label']}):")
    for t, p, n in pairs[:10]:
        print(f"  {t:<20} -> {p:<20} : {n} clip(s)")
    return results


# ----------------------------------------------------------------------------- latency (Reviewer 1, comment 7)
@torch.no_grad()
def measure_inference_latency(model, key, batch_size=32, seq_len=172, feat_dim=236, device=None,
                              n_warmup=20, n_runs=100):
    """Documented protocol: FP32, random input, n_warmup untimed passes, n_runs timed passes,
    torch.cuda.synchronize() before starting and before stopping every timer."""
    if device is None:
        device = DEVICE
    cuda = device.type == "cuda"
    model.to(device).eval()
    dummy = torch.randn(batch_size, seq_len, feat_dim, device=device)

    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(n_warmup):
        model(dummy)
    times = []
    for _ in range(n_runs):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(dummy)
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)

    info = {
        "model": MODELS[key]["label"], "device": torch.cuda.get_device_name(device) if cuda else platform.processor(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "precision": "FP32 (no autocast)",
        "batch_size": batch_size, "input_shape": [batch_size, seq_len, feat_dim],
        "n_warmup": n_warmup, "n_runs": n_runs, "sync": "torch.cuda.synchronize() around every timed forward pass",
        "mean_ms": float(times.mean()), "std_ms": float(times.std()), "median_ms": float(np.median(times)),
        "per_clip_ms": float(times.mean() / batch_size),
        "throughput_clips_per_s": float(batch_size / (times.mean() / 1000)),
        "peak_gpu_mem_MB": float(torch.cuda.max_memory_allocated(device) / 2 ** 20) if cuda else None,
        "note": "Model forward pass only; excludes video decoding, pose estimation and preprocessing.",
    }
    save_json(info, f"inference_benchmark_{key}.json")
    print(f"{info['model']:<7}: {info['mean_ms']:.2f} +/- {info['std_ms']:.2f} ms/batch "
          f"({info['per_clip_ms']:.3f} ms/clip, {info['throughput_clips_per_s']:.0f} clips/s, "
          f"peak mem {info['peak_gpu_mem_MB']} MB) on {info['device']}")
    return info


# ----------------------------------------------------------------------------- multi-seed (Reviewer 1, comment 6)
def run_seeds(name, build_fn, train_loader, val_loader, classes, class_weights, train_kwargs, seeds=SEEDS):
    """Re-trains `build_fn()` once per seed (same split, different initialisation/shuffling) and
    reports mean, std and 95% t-interval of accuracy and macro-F1."""
    os.makedirs(os.path.join(MODEL_SAVE_PATH, "seeds"), exist_ok=True)
    rows = []
    for s in seeds:
        set_seed(s)
        ck = ckpt_path(os.path.join("seeds", f"{name}_seed{s}.pt"))
        hist = train(build_fn(), train_loader, val_loader, num_classes=len(classes), class_weights=class_weights,
                     checkpoint_path=ck, device=DEVICE, verbose=False, **train_kwargs)
        model = load_checkpoint(ck, build_fn(), device=DEVICE)
        res = evaluate(model, val_loader, classes, DEVICE, verbose=False)
        rows.append(dict(seed=s, accuracy=res["accuracy"], macro_f1=macro_f1(res),
                         best_epoch=hist["best_epoch"], last_epoch=hist["last_epoch"]))
        print(f"  [{name}] seed {s}: acc {res['accuracy']:.4f} | macro-F1 {rows[-1]['macro_f1']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(out_path(f"seed_results_{name}.csv"), index=False)
    n = len(df)
    summ = {"model": name, "n_seeds": n}
    for m in ("accuracy", "macro_f1"):
        mean, sd = df[m].mean(), df[m].std(ddof=1)
        h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else float("nan")
        summ.update({f"{m}_mean": mean, f"{m}_std": sd, f"{m}_ci95_low": mean - h, f"{m}_ci95_high": mean + h})
    print(f"  [{name}] accuracy {summ['accuracy_mean']:.4f} +/- {summ['accuracy_std']:.4f} "
          f"(95% CI {summ['accuracy_ci95_low']:.4f}-{summ['accuracy_ci95_high']:.4f}) | "
          f"macro-F1 {summ['macro_f1_mean']:.4f} +/- {summ['macro_f1_std']:.4f}")
    return summ
