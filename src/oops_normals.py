"""
Builds the NORMAL (no-fall) examples from Oops! clips whose failure is not a fall.
Uses only footage BEFORE the failure moment: ordinary activity, same YouTube style.
Stages (each skips work already done, so it is safe to rerun):
  1. select   -> data/oops/normal_selected.csv
  2. unpack   -> data/oops/clips_normal/{train,val}/*.mp4
  3. extract  -> data/oops/features_normal/normal_XXXXX.npz
  4. clean    -> data/oops/normal_clips.csv  (keep flag + reason)
Run from ~/acc-ai-mvp:
    python src/oops_normals.py
"""
import json
import os
import re
import shutil
import statistics
import tarfile
import time
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

OOPS_DIR = "data/oops"
ANN_DIR = os.path.join(OOPS_DIR, "oops_dataset", "annotations")
LANG_DIR = os.path.join(OOPS_DIR, "lang", "svos")
ARCHIVE = os.path.join(OOPS_DIR, "oops_dataset", "video.tar.gz")
SEL_CSV = os.path.join(OOPS_DIR, "normal_selected.csv")
CLIPS_DIR = os.path.join(OOPS_DIR, "clips_normal")
FEAT_DIR = os.path.join(OOPS_DIR, "features_normal")
OUT_CSV = os.path.join(OOPS_DIR, "normal_clips.csv")
MODEL_PATH = "pose_landmarker.task"

MAX_CLIPS = 1200
MIN_PRE_SECONDS = 2.5
MARGIN_SECONDS = 0.5
MAX_SECONDS = 8.0
MIN_USABLE_SECONDS = 2.0
TARGET_FPS = 15.0
MAX_LABEL_STDEV = 1.0
MIN_DETECT = 0.5
CUT_JUMP = 1.0
CUT_SCALE = 2.0
MAX_GAP_S = 0.2
L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24

EXCLUDE_RE = re.compile(
    r"\b(fall|falls|fell|falling|fallen|slip|slips|slipped|slipping|"
    r"trip|trips|tripped|tripping|tumble|tumbles|tumbled|tumbling|"
    r"stumble|stumbles|stumbled|stumbling|faceplant|faceplants|"
    r"wipe out|wipes out|wiped out|crash|crashes|crashed|crashing|"
    r"collapse|collapses|collapsed|knocked over|knocked down|"
    r"hit the ground|hits the ground|on the ground|on the floor|"
    r"flip|flips|flipped|flipping|land on|lands on|landed on)\b", re.I)
CHILD_RE = re.compile(
    r"\b(kid|kids|child|children|baby|babies|boy|boys|girl|girls|"
    r"toddler|toddlers|son|daughter)\b", re.I)


def norm(name):
    name = name.strip()
    return name[:-4] if name.endswith(".mp4") else name


def read_list(filename):
    with open(os.path.join(ANN_DIR, filename)) as f:
        return set(norm(line) for line in f if line.strip())


# ---------------- Stage 1: select ----------------
def stage_select():
    if os.path.exists(SEL_CSV):
        df = pd.read_csv(SEL_CSV)
        print(f"[1/4] Using existing selection: {len(df)} clips")
        return df

    with open(os.path.join(ANN_DIR, "transition_times.json")) as f:
        times = {norm(k): v for k, v in json.load(f).items()}
    train_set = read_list("train_filtered.txt")
    val_set = read_list("val_filtered.txt")
    lang = {}
    for part in ("train.json", "val.json"):
        with open(os.path.join(LANG_DIR, part)) as f:
            for k, v in json.load(f).items():
                lang[norm(k)] = v

    stats = Counter()
    rows = []
    for name, anns in lang.items():
        if name not in times:
            stats["no failure-time entry"] += 1
            continue
        if name in val_set:
            split = "val"
        elif name in train_set:
            split = "train"
        else:
            stats["failed dataset quality control"] += 1
            continue
        all_text = " ".join((a.get("goal", "") or "") + " " + (a.get("wentwrong", "") or "") for a in anns)
        if EXCLUDE_RE.search(all_text):
            stats["mentions a fall-like word"] += 1
            continue
        valid = [x for x in times[name]["t"] if x >= 0]
        if len(valid) < 2:
            stats["fewer than 2 failure labels"] += 1
            continue
        t_fail = statistics.median(valid)
        if statistics.pstdev(valid) > MAX_LABEL_STDEV:
            stats["labelers disagreed on time"] += 1
            continue
        if t_fail < MIN_PRE_SECONDS:
            stats["too little footage before failure"] += 1
            continue
        rows.append({"filename": name, "split": split, "t_fail": round(t_fail, 3),
                     "child": bool(CHILD_RE.search(all_text)),
                     "wentwrong": (anns[0].get("wentwrong", "") or "").replace("\n", " ")})

    df = pd.DataFrame(rows)
    print("[1/4] Selection")
    for reason, n in stats.most_common():
        print(f"      removed - {reason:<36}{n}")
    print(f"      eligible no-fall clips: {len(df)}")
    if len(df) > MAX_CLIPS:
        df = df.sample(n=MAX_CLIPS, random_state=0)
    df = df.sort_values(["split", "filename"]).reset_index(drop=True)
    df.insert(0, "clip_id", [f"normal_{i:05d}" for i in range(len(df))])
    df.to_csv(SEL_CSV, index=False)
    print(f"      selected {len(df)} ({df['split'].value_counts().to_dict()}) -> {SEL_CSV}")
    return df


# ---------------- Stage 2: unpack ----------------
def stage_unpack(df):
    for split in ("train", "val"):
        os.makedirs(os.path.join(CLIPS_DIR, split), exist_ok=True)
    wanted = {}
    for r in df.itertuples():
        if not os.path.exists(os.path.join(CLIPS_DIR, r.split, f"{r.filename}.mp4")):
            wanted[r.filename] = r.split
    if not wanted:
        print("[2/4] All clips already unpacked")
        return
    print(f"[2/4] Unpacking {len(wanted)} clips (reads the whole archive once, ~15 min)...", flush=True)
    found = 0
    scanned = 0
    with tarfile.open(ARCHIVE, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            scanned += 1
            base = os.path.basename(member.name)
            key = base[:-4] if base.endswith(".mp4") else None
            if key in wanted:
                out_path = os.path.join(CLIPS_DIR, wanted.pop(key), base)
                with open(out_path, "wb") as f:
                    shutil.copyfileobj(tar.extractfile(member), f)
                found += 1
                if not wanted:
                    break
            if scanned % 2000 == 0:
                print(f"      scanned {scanned}, unpacked {found}", flush=True)
    print(f"      unpacked {found}, missing {len(wanted)}")


# ---------------- Stage 3: extract ----------------
def make_landmarker():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=MODEL_PATH, delegate=mp_python.BaseOptions.Delegate.CPU),
        running_mode=mp_vision.RunningMode.VIDEO, num_poses=1)
    return mp_vision.PoseLandmarker.create_from_options(opts)


def extract_clip(path, start, end):
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


def stage_extract(df):
    os.makedirs(FEAT_DIR, exist_ok=True)
    todo = [r for r in df.itertuples()
            if not os.path.exists(os.path.join(FEAT_DIR, f"{r.clip_id}.npz"))]
    if not todo:
        print("[3/4] All clips already extracted")
        return
    print(f"[3/4] Extracting skeletons for {len(todo)} clips...", flush=True)
    t0 = time.time()
    failed = 0
    for i, r in enumerate(todo, 1):
        path = os.path.join(CLIPS_DIR, r.split, f"{r.filename}.mp4")
        t_end = float(r.t_fail) - MARGIN_SECONDS
        start = max(0.0, t_end - MAX_SECONDS)
        res = extract_clip(path, start, t_end) if os.path.exists(path) else None
        if res is None:
            failed += 1
            continue
        poses, times, det, fps = res
        np.savez_compressed(os.path.join(FEAT_DIR, f"{r.clip_id}.npz"),
                            poses=poses, times=times, detected=det,
                            t_end=t_end, t_fail=float(r.t_fail), fps=fps)
        if i % 50 == 0:
            rate = (time.time() - t0) / i
            print(f"      [{i}/{len(todo)}] ~{rate*(len(todo)-i)/60:.0f} min left", flush=True)
    print(f"      done in {(time.time()-t0)/60:.1f} min, failed {failed}")


# ---------------- Stage 4: clean ----------------
def xy(poses, idx):
    return poses[:, idx * 4: idx * 4 + 2]


def has_scene_cut(times, det, hip, torso):
    idx = np.where(det)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if times[b] - times[a] > MAX_GAP_S:
            continue
        ref = max(min(torso[a], torso[b]), 1e-3)
        if (np.linalg.norm(hip[b] - hip[a]) / ref > CUT_JUMP or
                max(torso[a], torso[b]) / ref > CUT_SCALE):
            return True
    return False


def stage_clean(df):
    rows = []
    for r in df.itertuples():
        path = os.path.join(FEAT_DIR, f"{r.clip_id}.npz")
        reason, det_rate, dur = "ok", np.nan, np.nan
        if not os.path.exists(path):
            reason = "extraction failed"
        else:
            d = np.load(path)
            poses, times, det = d["poses"], d["times"], d["detected"]
            det_rate = float(det.mean())
            dur = float(times[-1] - times[0]) if len(times) > 1 else 0.0
            sh = (xy(poses, L_SH) + xy(poses, R_SH)) / 2
            hip = (xy(poses, L_HIP) + xy(poses, R_HIP)) / 2
            torso = np.linalg.norm(sh - hip, axis=1)
            if dur < MIN_USABLE_SECONDS:
                reason = "too short"
            elif det_rate < MIN_DETECT:
                reason = "low pose detection"
            elif np.median(torso[det]) < 1e-3:
                reason = "person too small"
            elif has_scene_cut(times, det, hip, torso):
                reason = "scene cut or tracker jump"
        rows.append({**r._asdict(), "detect_rate": round(det_rate, 3),
                     "duration": round(dur, 2), "keep": reason == "ok", "reason": reason})

    out = pd.DataFrame(rows).drop(columns=["Index"])
    out.to_csv(OUT_CSV, index=False)
    kept = out[out["keep"]]
    print("[4/4] Cleaning")
    print(out["reason"].value_counts().to_string())
    print(f"\nKEPT NORMAL CLIPS: {len(kept)}  ->  {OUT_CSV}")
    print(kept["split"].value_counts().to_string())
    print(f"Kept involving a child: {int(kept['child'].sum())}")
    print(f"Total normal footage kept: {kept['duration'].sum()/60:.1f} min")


if __name__ == "__main__":
    selection = stage_select()
    stage_unpack(selection)
    stage_extract(selection)
    stage_clean(selection)
