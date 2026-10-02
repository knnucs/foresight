"""
Sanity check: loads the trained TCN, runs it over full held-out clips
(both fall and ADL/normal) frame-by-frame, and plots predicted risk score
over time. For fall clips, the labeled risk window is shaded red.
For ADL clips, the score should stay low throughout — this checks the
false-alarm side of the model, not just the anticipation side.

Saves one PNG per clip to data/features/plots/. Run from ~/acc-ai-mvp.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

FEATURES_DIR = "data/features"
PLOTS_DIR = os.path.join(FEATURES_DIR, "plots")
WINDOW_SIZE = 30
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
os.makedirs(PLOTS_DIR, exist_ok=True)

manifest = pd.read_csv(os.path.join(FEATURES_DIR, "manifest.csv"))
train_clips, val_clips = train_test_split(
    manifest, test_size=0.2, stratify=manifest["label"], random_state=42
)
val_falls = val_clips[val_clips["label"] == "fall"]
val_adl = val_clips[val_clips["label"] == "normal"]
print(f"Held-out fall clips: {list(val_falls['clip_name'])}")
print(f"Held-out ADL clips: {list(val_adl['clip_name'])}")

class TCN(nn.Module):
    def __init__(self, input_dim=132, channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=3, padding=1, dilation=1), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = self.pool(x).squeeze(-1)
        return torch.sigmoid(self.fc(x)).squeeze(-1)

model = TCN().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(FEATURES_DIR, "tcn_model.pt"), map_location=DEVICE))
model.eval()

def plot_clip(row, clip_type):
    clip_name = row["clip_name"]
    poses = np.load(os.path.join(FEATURES_DIR, row["poses_file"]))
    labels = np.load(os.path.join(FEATURES_DIR, row["labels_file"]))

    if len(poses) <= WINDOW_SIZE:
        print(f"Skipping {clip_name}: too short ({len(poses)} frames)")
        return

    scores, frame_idx = [], []
    with torch.no_grad():
        for end in range(WINDOW_SIZE, len(poses)):
            window = poses[end - WINDOW_SIZE:end]
            xb = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            scores.append(model(xb).item())
            frame_idx.append(end - 1)

    scores = np.array(scores)
    frame_idx = np.array(frame_idx)
    true_labels = labels[frame_idx]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame_idx, scores, label="Predicted risk score", color="tab:blue")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="0.5 threshold")

    risk_mask = true_labels == 1
    if risk_mask.any():
        ax.axvspan(frame_idx[risk_mask][0], frame_idx[risk_mask][-1],
                   color="red", alpha=0.15, label="Labeled risk window")

    ax.set_title(f"{clip_name} ({clip_type}) — predicted risk score over time")
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Risk score")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=8)

    out_path = os.path.join(PLOTS_DIR, f"{clip_name}_risk_curve.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

for _, row in val_falls.iterrows():
    plot_clip(row, "fall")
for _, row in val_adl.iterrows():
    plot_clip(row, "ADL")

print(f"\nDone. Check {PLOTS_DIR}/ for plots.")
