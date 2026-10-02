"""
Cleans the extracted Oops! skeletons and checks that a person actually fell.
v2: folded torso angle, "ends horizontal" rule, scene-cut / identity-jump check.
Run from ~/acc-ai-mvp:
    python src/oops_prepare.py
Output: data/oops/prepared_clips.csv (all clips, with keep flag + reason)
        data/oops/spot_check.txt     (random clips per category to watch)
"""
import os

import numpy as np
import pandas as pd

FEAT_DIR = "data/oops/features"
MANIFEST = os.path.join(FEAT_DIR, "manifest.csv")
SELECTED = "data/oops/selected_clips.csv"
CLIPS_DIR = "data/oops/clips"
OUT_CSV = "data/oops/prepared_clips.csv"
CHECK_TXT = "data/oops/spot_check.txt"

MIN_DETECT_OVERALL = 0.5
MIN_DETECT_NEAR_FALL = 0.5
DROP_THRESH = 0.5
ANGLE_AFTER = 45.0
ANGLE_CHANGE = 25.0
ANGLE_HORIZONTAL = 60.0
CUT_JUMP = 1.0
CUT_SCALE = 2.0
MAX_GAP_S = 0.2

L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24


def xy(poses, idx):
    return poses[:, idx * 4: idx * 4 + 2]


def torso_angle(sh, hip):
    v = sh - hip
    return np.degrees(np.arctan2(np.abs(v[:, 0]), np.abs(v[:, 1])))


def has_scene_cut(times, det, hip, torso, t_fail):
    idx = np.where(det & (times <= t_fail - 0.2))[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if times[b] - times[a] > MAX_GAP_S:
            continue
        ref = max(min(torso[a], torso[b]), 1e-3)
        jump = np.linalg.norm(hip[b] - hip[a]) / ref
        scale = max(torso[a], torso[b]) / ref
        if jump > CUT_JUMP or scale > CUT_SCALE:
            return True
    return False


def check_clip(path, t_fail):
    d = np.load(path)
    poses, times, det = d["poses"], d["times"], d["detected"]
    out = {"detect_rate": float(det.mean()), "detect_near_fall": np.nan,
           "hip_drop": np.nan, "angle_before": np.nan, "angle_after": np.nan}

    near = (times >= t_fail - 0.5) & (times <= t_fail + 0.5)
    out["detect_near_fall"] = float(det[near].mean()) if near.any() else 0.0

    if out["detect_rate"] < MIN_DETECT_OVERALL:
        return out, "low pose detection overall"
    if out["detect_near_fall"] < MIN_DETECT_NEAR_FALL:
        return out, "low pose detection near fall"

    sh = (xy(poses, L_SH) + xy(poses, R_SH)) / 2
    hip = (xy(poses, L_HIP) + xy(poses, R_HIP)) / 2
    torso = np.linalg.norm(sh - hip, axis=1)
    before = det & (times >= t_fail - 1.5) & (times <= t_fail - 0.3)
    after = det & (times >= t_fail - 0.2) & (times <= t_fail + 1.0)
    if before.sum() < 2 or after.sum() < 2:
        return out, "no pose just before/after fall"

    torso_len = float(np.median(torso[before]))
    if torso_len < 1e-3:
        return out, "person too small"

    if has_scene_cut(times, det, hip, torso, t_fail):
        return out, "scene cut or tracker jump before fall"

    angles = torso_angle(sh, hip)
    out["hip_drop"] = float((np.percentile(hip[after, 1], 80) - np.median(hip[before, 1])) / torso_len)
    out["angle_before"] = float(np.median(angles[before]))
    out["angle_after"] = float(np.percentile(angles[after], 80))

    fell = (out["hip_drop"] >= DROP_THRESH
            or (out["angle_after"] >= ANGLE_AFTER and out["angle_after"] - out["angle_before"] >= ANGLE_CHANGE)
            or out["angle_after"] >= ANGLE_HORIZONTAL)
    return out, ("ok" if fell else "no fall motion in skeleton")


manifest = pd.read_csv(MANIFEST)
text = pd.read_csv(SELECTED)[["filename", "wentwrong", "pre_seconds"]]
manifest = manifest.merge(text, on="filename", how="left")

rows = []
for r in manifest.itertuples():
    metrics, reason = check_clip(os.path.join(FEAT_DIR, f"{r.clip_id}.npz"), float(r.t_fail))
    rows.append({**r._asdict(), **{k: round(v, 3) for k, v in metrics.items()},
                 "keep": reason == "ok", "reason": reason})

df = pd.DataFrame(rows).drop(columns=["Index"])
df.to_csv(OUT_CSV, index=False)
kept = df[df["keep"]]

print(f"Clips checked: {len(df)}")
print("\nResult by reason:")
print(df["reason"].value_counts().to_string())
print(f"\nKEPT: {len(kept)} clips  ->  {OUT_CSV}")
print(kept["split"].value_counts().to_string())
print(f"Kept clips involving a child: {int(kept['child'].sum())}")
for s in (3, 4, 5):
    print(f"Kept with >= {s}s before the fall: {int((kept['pre_seconds'] >= s).sum())}")


def describe(r):
    path = os.path.join(CLIPS_DIR, r["split"], f"{r['filename']}.mp4")
    return (f"  {r['clip_id']}  fall at {r['t_fail']:.1f}s  | hip_drop={r['hip_drop']}  "
            f"angle {r['angle_before']}->{r['angle_after']}\n"
            f"    \"{str(r['wentwrong'])[:100]}\"\n"
            f"    vlc \"{path}\"\n")


sections = [
    ("KEPT (should be real falls, skeleton on the person who fell)", kept),
    ("REJECTED: scene cut or tracker jump (should show a cut or a switch between people)",
     df[df["reason"] == "scene cut or tracker jump before fall"]),
    ("REJECTED: no fall motion (should mostly NOT show the tracked person falling)",
     df[df["reason"] == "no fall motion in skeleton"]),
]
with open(CHECK_TXT, "w") as f:
    f.write("Run 'set +H' in the terminal first so filenames with ! open correctly.\n\n")
    for title, part in sections:
        f.write(f"=== {title} ===\n")
        for _, r in part.sample(min(10, len(part)), random_state=0).iterrows():
            f.write(describe(r))
        f.write("\n")
print(f"\nSpot-check list written to {CHECK_TXT}")
