"""
Auto-labels the UR Fall Detection dataset.
- Clip-level label (fall/normal) comes from folder naming, as before.
- For fall clips, the PRECISE impact frame is found from the synchronization CSV
  (data/sync/fall-XX-data.csv), by locating the peak of the SV_total column —
  the accelerometer spike at the moment of impact.
- risk_start_frame = impact_frame - RISK_WINDOW_FRAMES (the "leading up to the fall" window)

Falls back to the old fixed-offset heuristic (last 60 frames) if a sync file
is missing for a given fall clip.

Run this from ~/child-accident-mvp

Output: data/labels/labels.csv
"""

import os
import pandas as pd

RAW_DIR = "data/raw_clips"
SYNC_DIR = "data/sync"
OUT_CSV = "data/labels/labels.csv"
RISK_WINDOW_FRAMES = 60  # ~2 seconds at 30fps
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

rows = []

for folder_name in sorted(os.listdir(RAW_DIR)):
    folder_path = os.path.join(RAW_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue

    frame_files = [f for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTENSIONS)]
    total_frames = len(frame_files)
    if total_frames == 0:
        print(f"Warning: no images found in {folder_path}, skipping")
        continue

    if folder_name.lower().startswith("fall"):
        label = "fall"
        sync_path = os.path.join(SYNC_DIR, f"{folder_name}-data.csv")

        if os.path.exists(sync_path):
            # Columns: frame_number, time_ms, sv_total  (no header row in this dataset)
            sync_df = pd.read_csv(sync_path, header=None, names=["frame_number", "time_ms", "sv_total"])
            impact_row = sync_df.loc[sync_df["sv_total"].idxmax()]
            impact_frame = int(impact_row["frame_number"])
            risk_start_frame = max(0, impact_frame - RISK_WINDOW_FRAMES)
            print(f"{folder_name}: impact detected at frame {impact_frame} (sv_total={impact_row['sv_total']:.3f})")
        else:
            # Fallback: fixed-offset heuristic
            risk_start_frame = max(0, total_frames - RISK_WINDOW_FRAMES)
            print(f"{folder_name}: no sync file found, using fixed-offset fallback")
    else:
        label = "normal"
        risk_start_frame = -1

    rows.append({
        "folder_path": folder_path,
        "label": label,
        "total_frames": total_frames,
        "risk_start_frame": risk_start_frame
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nWrote {len(df)} labeled clips to {OUT_CSV}")
print(df["label"].value_counts())
