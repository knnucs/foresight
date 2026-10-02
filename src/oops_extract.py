"""
Runs MediaPipe pose (VIDEO mode) on the selected Oops! clips, only on the
window from PRE_SECONDS before the fall to POST_SECONDS after, sampled at
TARGET_FPS. Run from ~/acc-ai-mvp:
    python src/oops_extract.py --limit 20     (timing test on 20 random clips)
    python src/oops_extract.py                (full run; skips finished clips)
Output: data/oops/features/oops_XXXXX.npz + data/oops/features/manifest.csv
"""
import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

SELECTED = "data/oops/selected_clips.csv"
CLIPS_DIR = "data/oops/clips"
OUT_DIR = "data/oops/features"
MANIFEST = os.path.join(OUT_DIR, "manifest.csv")
MODEL_PATH = "pose_landmarker.task"

TARGET_FPS = 15.0
PRE_SECONDS = 5.0
POST_SECONDS = 1.0

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=0, help="only process this many random clips (timing test)")
args = parser.parse_args()

os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(SELECTED)
df["clip_id"] = [f"oops_{i:05d}" for i in range(len(df))]
todo = df.sample(n=min(args.limit, len(df)), random_state=1) if args.limit else df


def make_landmarker():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=mp_python.BaseOptions.Delegate.CPU),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1)
    return mp_vision.PoseLandmarker.create_from_options(opts)


def process_clip(path, t_fail):
    start = max(0.0, t_fail - PRE_SECONDS)
    end = t_fail + POST_SECONDS
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:
        fps = 30.0

    poses, times, detected = [], [], []
    step = 1.0 / TARGET_FPS
    next_t = start
    frame_idx = -1
    lm = make_landmarker()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            t = frame_idx / fps
            if t > end:
                break
            if t + 1e-6 < next_t:
                continue
            while next_t <= t + 1e-6:
                next_t += step
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = lm.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
            if result.pose_landmarks:
                coords = []
                for p in result.pose_landmarks[0]:
                    coords.extend([p.x, p.y, p.z, p.visibility])
                poses.append(coords)
                detected.append(True)
            else:
                poses.append([0.0] * 132)
                detected.append(False)
            times.append(t)
    finally:
        lm.close()
        cap.release()

    if not poses:
        return None
    return (np.array(poses, dtype=np.float32), np.array(times, dtype=np.float32),
            np.array(detected, dtype=bool), fps)


def write_manifest(rows):
    pd.DataFrame(rows).sort_values("clip_id").to_csv(MANIFEST, index=False)


rows = []
failed = []
t0 = time.time()
n_new = 0

for i, r in enumerate(todo.itertuples(), 1):
    out_path = os.path.join(OUT_DIR, f"{r.clip_id}.npz")
    if os.path.exists(out_path):
        d = np.load(out_path)
        det, fps = d["detected"], float(d["fps"])
        n = len(det)
    else:
        video_path = os.path.join(CLIPS_DIR, r.split, f"{r.filename}.mp4")
        c0 = time.time()
        res = process_clip(video_path, float(r.t_fail))
        if res is None:
            failed.append(r.filename)
            print(f"[{i}/{len(todo)}] {r.clip_id} FAILED to read", flush=True)
            continue
        poses, times, det, fps = res
        np.savez_compressed(out_path, poses=poses, times=times, detected=det,
                            t_fail=float(r.t_fail), fps=fps)
        n = len(det)
        n_new += 1
        if args.limit or i % 25 == 0:
            print(f"[{i}/{len(todo)}] {r.clip_id}: {n} frames, pose found in "
                  f"{det.mean()*100:.0f}%, {time.time()-c0:.1f}s", flush=True)

    rows.append({"clip_id": r.clip_id, "filename": r.filename, "split": r.split,
                 "t_fail": r.t_fail, "child": r.child, "num_frames": n,
                 "detect_rate": round(float(det.mean()), 3), "video_fps": round(fps, 2)})
    if i % 100 == 0:
        write_manifest(rows)

write_manifest(rows)
elapsed = time.time() - t0
m = pd.DataFrame(rows)

print(f"\nProcessed {n_new} new clips in {elapsed/60:.1f} min "
      f"({elapsed/max(n_new,1):.1f} s per clip). Failed: {len(failed)}")
print(f"Average pose detection rate: {m['detect_rate'].mean()*100:.0f}%")
print(f"Clips with pose found in < 50% of frames: {int((m['detect_rate'] < 0.5).sum())}/{len(m)}")
if args.limit and n_new:
    total_h = elapsed / n_new * len(df) / 3600
    print(f"\nPROJECTED full run for all {len(df)} clips: about {total_h:.1f} hours")
