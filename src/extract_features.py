"""
Reads labels.csv, runs MediaPipe pose (new Tasks API) on every frame of every
clip, saves the resulting skeleton sequences as .npy files for training.

Requires the pose_landmarker.task model file, downloaded via curl earlier.
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

LABELS_CSV = "data/labels/labels.csv"
FEATURES_DIR = "data/features"
MODEL_PATH = "pose_landmarker.task"
os.makedirs(FEATURES_DIR, exist_ok=True)

base_options = mp_python.BaseOptions(
    model_asset_path=MODEL_PATH,
    delegate=mp_python.BaseOptions.Delegate.CPU
)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    output_segmentation_masks=False
)

landmarker = mp_vision.PoseLandmarker.create_from_options(options)

df = pd.read_csv(LABELS_CSV)
manifest_rows = []

for idx, row in df.iterrows():
    folder = row["folder_path"]
    label = row["label"]
    risk_start = row["risk_start_frame"]

    frame_files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    clip_landmarks = []
    for fname in frame_files:
        img_bgr = cv2.imread(os.path.join(folder, fname))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            coords = []
            for lm in result.pose_landmarks[0]:
                coords.extend([lm.x, lm.y, lm.z, lm.visibility])
            clip_landmarks.append(coords)
        else:
            clip_landmarks.append([0.0] * 132)

    clip_array = np.array(clip_landmarks, dtype=np.float32)

    frame_labels = np.zeros(len(clip_array), dtype=np.float32)
    if label == "fall" and risk_start >= 0:
        frame_labels[int(risk_start):] = 1.0

    clip_name = os.path.basename(folder)
    np.save(os.path.join(FEATURES_DIR, f"{clip_name}_poses.npy"), clip_array)
    np.save(os.path.join(FEATURES_DIR, f"{clip_name}_labels.npy"), frame_labels)

    manifest_rows.append({
        "clip_name": clip_name,
        "label": label,
        "num_frames": len(clip_array),
        "poses_file": f"{clip_name}_poses.npy",
        "labels_file": f"{clip_name}_labels.npy"
    })

    print(f"[{idx+1}/{len(df)}] Processed {clip_name}: {len(clip_array)} frames")

pd.DataFrame(manifest_rows).to_csv(os.path.join(FEATURES_DIR, "manifest.csv"), index=False)
print(f"\nDone. Feature files saved to {FEATURES_DIR}/")
