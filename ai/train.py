"""
Train the shot-prediction model.

Usage:
  python ai/train.py

Reads all shots_*.jsonl from ai/dataset/
Saves model to ai/shot_model.pt
"""
import glob
import json
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import ShotNet, encode, augment

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "shot_model.pt")

EPOCHS      = 600
LR          = 3e-4
BATCH_SIZE  = 16
VAL_FRAC    = 0.2
SEED        = 42


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_records() -> list[dict]:
    files = glob.glob(os.path.join(DATASET_DIR, "selfplay_*.jsonl"))
    records = []
    for path in sorted(files):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    print(f"Loaded {len(records)} raw shots from {len(files)} file(s)")
    return records


# ── 2. Augment + encode ───────────────────────────────────────────────────────

def build_dataset(records: list[dict]):
    random.seed(SEED)
    all_x, all_y = [], []

    for rec in records:
        for aug in augment(rec):
            all_x.append(encode(aug))
            all_y.append(aug["ghost_n"])

    print(f"After augmentation: {len(all_x)} samples")

    # shuffle
    combined = list(zip(all_x, all_y))
    random.shuffle(combined)
    all_x, all_y = zip(*combined)

    X = torch.tensor(all_x, dtype=torch.float32)
    Y = torch.tensor(all_y, dtype=torch.float32)

    split = int(len(X) * (1 - VAL_FRAC))
    return (X[:split], Y[:split]), (X[split:], Y[split:])


# ── 3. Train ──────────────────────────────────────────────────────────────────

def train():
    records = load_records()
    if not records:
        print("No data found in", DATASET_DIR)
        return

    (X_tr, Y_tr), (X_val, Y_val) = build_dataset(records)
    print(f"Train: {len(X_tr)}  Val: {len(X_val)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model  = ShotNet().to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_tr.to(device), Y_tr.to(device)),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    best_val = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        # validate
        if epoch % 50 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val.to(device))
                val_loss = loss_fn(val_pred, Y_val.to(device)).item()
                tr_pred  = model(X_tr.to(device))
                tr_loss  = loss_fn(tr_pred,  Y_tr.to(device)).item()

            # mean pixel error (table = 1.0 unit)
            val_px = (val_pred - Y_val.to(device)).pow(2).sum(-1).sqrt().mean().item()
            print(f"Epoch {epoch:4d}  train={tr_loss:.5f}  val={val_loss:.5f}"
                  f"  val_err={val_px*100:.1f}% of table")

            if val_loss < best_val:
                best_val  = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # save best
    torch.save({"state_dict": best_state, "val_loss": best_val}, MODEL_PATH)
    print(f"\nBest val_loss: {best_val:.5f}")
    print(f"Model saved: {MODEL_PATH}")

    # show sample predictions
    model.load_state_dict(best_state)
    model.eval()
    print("\nSample predictions (first 5 val samples):")
    print(f"{'Predicted':30s}  {'Target':30s}  {'Error%':8s}")
    with torch.no_grad():
        preds = model(X_val[:5].to(device)).cpu()
    for i in range(min(5, len(X_val))):
        px, py = preds[i].tolist()
        tx, ty = Y_val[i].tolist()
        err = ((px-tx)**2 + (py-ty)**2)**0.5 * 100
        print(f"({px:.3f}, {py:.3f})                  "
              f"({tx:.3f}, {ty:.3f})                  "
              f"{err:.1f}%")


if __name__ == "__main__":
    train()
