"""
Trains the fall-PREDICTION TCN on Oops! skeletons.
  risky  = window ends within HORIZON seconds before a fall
  normal = window ends earlier than that, or comes from a no-fall clip
  windows after the fall are left out (they belong to the alarm stage)
Splits: Oops train -> TRAIN. Oops val -> split by YouTube compilation into
VAL (choosing the best epoch + threshold) and TEST (used once, at the end).
Run from ~/acc-ai-mvp:
    python src/train_oops.py
Outputs in data/oops/results/: tcn_oops.pt, config.json, test_report.txt, risk_curves.png
"""
import json
import os
import random
import re
import sys
import time
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fallnet import FallTCN, fill_missing, window_features, FEAT_DIM

FALL_CSV = "data/oops/prepared_clips.csv"
FALL_DIR = "data/oops/features"
NORM_CSV = "data/oops/normal_clips.csv"
NORM_DIR = "data/oops/features_normal"
OUT_DIR = "data/oops/results"

FPS = 15
WINDOW = 15
HORIZON = 3.0
TRAIN_STRIDE = 2
EVAL_STRIDE = 1
EPOCHS = 25
PATIENCE = 6
BATCH = 256
LR = 1e-3
SEED = 42


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def comp_key(filename):
    return re.sub(r"\d+$", "", str(filename)).strip()


def load_clips():
    clips = []
    for csv_path, feat_dir, kind in ((FALL_CSV, FALL_DIR, "fall"), (NORM_CSV, NORM_DIR, "normal")):
        df = pd.read_csv(csv_path)
        df = df[df["keep"] == True]
        for r in df.itertuples():
            d = np.load(os.path.join(feat_dir, f"{r.clip_id}.npz"))
            det = d["detected"].astype(bool)
            split = r.split
            if split == "val":
                split = "val" if zlib.crc32(comp_key(r.filename).encode()) % 2 == 0 else "test"
            clips.append({"id": r.clip_id, "kind": kind, "split": split, "child": bool(r.child),
                          "poses": fill_missing(d["poses"], det), "det": det,
                          "times": d["times"].astype(np.float32),
                          "t_fail": float(r.t_fail) if kind == "fall" else None})
    return clips


def make_index(clips, stride):
    items = []
    for ci, c in enumerate(clips):
        for end in range(WINDOW, len(c["times"]) + 1, stride):
            t = float(c["times"][end - 1])
            if c["kind"] == "fall":
                ttf = c["t_fail"] - t
                if ttf <= 0:
                    continue
                label = 1.0 if ttf <= HORIZON else 0.0
            else:
                ttf, label = float("inf"), 0.0
            items.append((ci, end, label, ttf))
    return items


class WindowDataset(Dataset):
    def __init__(self, clips, items, augment):
        self.clips, self.items, self.augment = clips, items, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ci, end, label, _ = self.items[i]
        c = self.clips[ci]
        flip = self.augment and random.random() < 0.5
        x = window_features(c["poses"][end - WINDOW:end], c["det"][end - WINDOW:end], flip=flip)
        if self.augment:
            x[:, :66] += np.random.normal(0, 0.02, (x.shape[0], 66)).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.float32)


@torch.no_grad()
def predict(model, clips, items, device):
    model.eval()
    dl = DataLoader(WindowDataset(clips, items, False), batch_size=1024)
    return np.concatenate([torch.sigmoid(model(xb.to(device))).cpu().numpy() for xb, _ in dl])


def group_by_clip(items):
    groups = {}
    for k, it in enumerate(items):
        groups.setdefault(it[0], []).append(k)
    return groups


def clip_metrics(clips, items, scores, thr, only=lambda c: True):
    leads, max_leads = [], []
    n_fall = n_warned = n_norm = n_norm_alarm = crossings = norm_windows = 0
    for ci, ks in group_by_clip(items).items():
        c = clips[ci]
        if not only(c):
            continue
        above = scores[ks] >= thr
        if c["kind"] == "fall":
            n_fall += 1
            ttf = np.array([items[k][3] for k in ks])
            max_leads.append(ttf.max())
            if above.any():
                n_warned += 1
                leads.append(float(ttf[above].max()))
        else:
            n_norm += 1
            norm_windows += len(ks)
            rises = int(above[0]) + int(np.sum(above[1:] & ~above[:-1]))
            crossings += rises
            n_norm_alarm += int(rises > 0)
    hours = norm_windows * EVAL_STRIDE / FPS / 3600
    return {"n_fall": n_fall, "n_warned": n_warned, "leads": np.array(leads),
            "max_lead": float(np.mean(max_leads)) if max_leads else float("nan"),
            "n_norm": n_norm, "n_norm_alarm": n_norm_alarm,
            "alarms_per_hour": crossings / hours if hours > 0 else float("nan"),
            "normal_minutes": hours * 60}


def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    clips = load_clips()
    parts = {s: [c for c in clips if c["split"] == s] for s in ("train", "val", "test")}
    idx = {s: make_index(parts[s], TRAIN_STRIDE if s == "train" else EVAL_STRIDE) for s in parts}
    labels = {s: np.array([it[2] for it in idx[s]], dtype=np.float32) for s in idx}
    for s in parts:
        nf = sum(c["kind"] == "fall" for c in parts[s])
        print(f"{s:<5}: {nf} fall clips + {len(parts[s]) - nf} normal clips -> "
              f"{len(labels[s])} windows ({int(labels[s].sum())} risky)")

    ytr = labels["train"]
    pos_weight = torch.tensor((len(ytr) - ytr.sum()) / max(ytr.sum(), 1.0), device=device)
    train_dl = DataLoader(WindowDataset(parts["train"], idx["train"], True), batch_size=BATCH, shuffle=True)

    model = FallTCN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model_path = os.path.join(OUT_DIR, "tcn_oops.pt")
    best_ap, bad = -1.0, 0

    print("\nTraining...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, total = time.time(), 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        sval = predict(model, parts["val"], idx["val"], device)
        ap = average_precision_score(labels["val"], sval)
        auc = roc_auc_score(labels["val"], sval)
        note = ""
        if ap > best_ap:
            best_ap, bad = ap, 0
            torch.save(model.state_dict(), model_path)
            note = "  <- best so far, saved"
        else:
            bad += 1
        print(f"Epoch {epoch:2d}  train loss {total/len(ytr):.4f}  val AUC {auc:.3f}  "
              f"val AP {ap:.3f}  ({time.time()-t0:.0f}s){note}", flush=True)
        if bad >= PATIENCE:
            print("No improvement for several epochs, stopping early.")
            break

    model.load_state_dict(torch.load(model_path, map_location=device))
    sval = predict(model, parts["val"], idx["val"], device)
    prec, rec, thr = precision_recall_curve(labels["val"], sval)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    threshold = float(thr[int(np.argmax(f1[:-1]))])

    stest = predict(model, parts["test"], idx["test"], device)
    ytest = labels["test"]
    m = clip_metrics(parts["test"], idx["test"], stest, threshold)
    mc = clip_metrics(parts["test"], idx["test"], stest, threshold, only=lambda c: c["child"])

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("\n" + "=" * 64)
    out("TEST RESULTS  (clips never used for training or model choice)")
    out("=" * 64)
    out(f"Window level:  AUC {roc_auc_score(ytest, stest):.3f}   AP {average_precision_score(ytest, stest):.3f}"
        f"   (AP of random guessing = {ytest.mean():.3f})")
    out(f"Warning threshold (chosen on validation, best F1): {threshold:.3f}")
    out("")
    out(f"Falls warned before impact:    {m['n_warned']}/{m['n_fall']}  ({100*m['n_warned']/max(m['n_fall'],1):.0f}%)")
    if len(m["leads"]):
        out(f"Lead time (mTTA):              mean {m['leads'].mean():.2f}s, median {np.median(m['leads']):.2f}s")
        for s in (1, 2, 3):
            out(f"  warned at least {s}s before:   {100*(m['leads'] >= s).sum()/max(m['n_fall'],1):.0f}% of falls")
    out(f"  (longest measurable lead with this footage: about {m['max_lead']:.1f}s on average)")
    out("")
    out(f"Normal clips with a false warning: {m['n_norm_alarm']}/{m['n_norm']}  ({100*m['n_norm_alarm']/max(m['n_norm'],1):.0f}%)")
    out(f"False warnings per hour of normal footage: {m['alarms_per_hour']:.0f}  "
        f"(measured on {m['normal_minutes']:.1f} min)")
    out("")
    out(f"Child clips: warned {mc['n_warned']}/{mc['n_fall']} falls"
        + (f", mean lead {mc['leads'].mean():.2f}s" if len(mc["leads"]) else "")
        + f"; false warnings on {mc['n_norm_alarm']}/{mc['n_norm']} normal clips")

    with open(os.path.join(OUT_DIR, "test_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump({"fps": FPS, "window": WINDOW, "horizon_s": HORIZON, "feat_dim": FEAT_DIM,
                   "threshold": threshold, "best_val_ap": best_ap,
                   "features": "fallnet.window_features on fill_missing MediaPipe poses"}, f, indent=2)

    groups = group_by_clip(idx["test"])
    rng = random.Random(0)
    falls = [ci for ci in groups if parts["test"][ci]["kind"] == "fall"]
    norms = [ci for ci in groups if parts["test"][ci]["kind"] == "normal"]
    picks = rng.sample(falls, min(6, len(falls))) + rng.sample(norms, min(2, len(norms)))
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    for ax, ci in zip(axes.flat, picks):
        c, ks = parts["test"][ci], groups[ci]
        s = stest[ks]
        if c["kind"] == "fall":
            x = -np.array([idx["test"][k][3] for k in ks])
            ax.axvspan(-HORIZON, 0, color="orange", alpha=0.15, label="risky zone")
            ax.axvline(0, color="red", lw=1.5, label="fall")
            ax.set_xlabel("seconds relative to fall")
            ax.set_title(f"FALL {c['id']}" + (" (child)" if c["child"] else ""))
        else:
            x = np.array([c["times"][idx["test"][k][1] - 1] for k in ks])
            ax.set_xlabel("seconds into clip")
            ax.set_title(f"NORMAL {c['id']}" + (" (child)" if c["child"] else ""))
        ax.plot(x, s, lw=2)
        ax.axhline(threshold, color="gray", ls="--", label="warning threshold")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("risk score")
    axes.flat[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "risk_curves.png"), dpi=110)
    print(f"\nSaved model, config, report and risk_curves.png to {OUT_DIR}/")


if __name__ == "__main__":
    main()
