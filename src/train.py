"""
Loads the pose feature sequences, builds sliding windows, trains a small TCN
to predict risk score. Run from ~/child-accident-mvp after extract_features.py.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

FEATURES_DIR = "data/features"
WINDOW_SIZE = 30          # ~1 second of context at 30fps
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------- Split clips first (stratified by label), before building any windows ----------
manifest = pd.read_csv(os.path.join(FEATURES_DIR, "manifest.csv"))

train_clips, val_clips = train_test_split(
    manifest, test_size=0.2, stratify=manifest["label"], random_state=42
)

def build_windows(clip_rows):
    X, y = [], []
    for _, row in clip_rows.iterrows():
        poses = np.load(os.path.join(FEATURES_DIR, row["poses_file"]))   # (num_frames, 132)
        labels = np.load(os.path.join(FEATURES_DIR, row["labels_file"])) # (num_frames,)
        for end in range(WINDOW_SIZE, len(poses)):
            window = poses[end - WINDOW_SIZE:end]      # last WINDOW_SIZE frames
            window_label = labels[end - 1]             # label of the last frame in the window
            X.append(window)
            y.append(window_label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train, y_train = build_windows(train_clips)
X_val, y_val = build_windows(val_clips)
print(f"Train: {len(X_train)} windows ({int(y_train.sum())} risk) | Val: {len(X_val)} windows ({int(y_val.sum())} risk)")

class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=32)

# ---------- A small TCN ----------
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
        # x comes in as (batch, time, features) — Conv1d wants (batch, features, time)
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = self.pool(x).squeeze(-1)
        return torch.sigmoid(self.fc(x)).squeeze(-1)

model = TCN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

# ---------- Training loop ----------
EPOCHS = 15
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            val_loss += criterion(pred, yb).item()

    avg_val_loss = val_loss / len(val_loader)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "data/features/tcn_model.pt")

    print(f"Epoch {epoch+1}/{EPOCHS} — train loss: {total_loss/len(train_loader):.4f}  val loss: {avg_val_loss:.4f}")

print(f"\nBest val loss: {best_val_loss:.4f} — best checkpoint saved to data/features/tcn_model.pt")
