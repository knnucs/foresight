"""
Collects footage for the demo video from the Oops! clips you already have.
  - AI clips: child falls run through Foresight (skeleton, risk bar, warning/alarm drawn in)
  - B-roll: raw clips whose descriptions mention kitchens/stoves, water, stairs, climbing
Everything goes to ~/acc-ai-mvp/video_assets/ as small 720p .mp4 files, plus list.txt.
Run from ~/acc-ai-mvp (venv active, ffmpeg installed):
    python src/video_assets.py
"""
import os
import re
import subprocess
import sys

import pandas as pd

OUT = "video_assets"
TMP = os.path.join(OUT, "tmp")
os.makedirs(TMP, exist_ok=True)

N_AI_FALLS = 5
N_BROLL_EACH = 3
THEMES = {
    "kitchen": r"\b(stove|kitchen|cook\w*|oven|pan|pot|counter)\b",
    "water": r"\b(pool|bath\w*|tub|water|swim\w*)\b",
    "stairs": r"\b(stair\w*|steps)\b",
    "climbing": r"\b(climb\w*|crib|shelf|dresser|table|chair|couch|sofa|bed)\b",
}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def to_mp4(src, dst, start=None, dur=None):
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{max(start, 0):.2f}"]
    cmd += ["-i", src]
    if dur is not None:
        cmd += ["-t", f"{dur:.2f}"]
    cmd += ["-vf", "scale=-2:720", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", dst]
    return run(cmd)


def safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name)[:40].strip("_")


lines = []
falls = pd.read_csv("data/oops/prepared_clips.csv")
falls = falls[(falls["keep"] == True) & (falls["detect_rate"] >= 0.8) & (falls["pre_seconds"] >= 3.5)]

picks = falls[falls["child"] == True].sort_values("detect_rate", ascending=False).head(N_AI_FALLS)
for theme, pat in THEMES.items():
    m = falls[falls["wentwrong"].astype(str).str.contains(pat, flags=re.I, regex=True) & ~falls.index.isin(picks.index)]
    picks = pd.concat([picks, m.head(1)])

print(f"Rendering {len(picks)} AI clips (skeleton + risk bar)...", flush=True)
for i, r in enumerate(picks.itertuples(), 1):
    src = os.path.join("data/oops/clips", r.split, f"{r.filename}.mp4")
    if not os.path.exists(src):
        continue
    avi = os.path.join(TMP, f"ai_{i}.avi")
    ok = run([sys.executable, "src/live.py", "--no-server", "--no-display", "--fast",
              "--video", src, "--save", avi])
    dst = os.path.join(OUT, f"ai_{i:02d}_{safe(str(r.wentwrong))}.mp4")
    if ok and to_mp4(avi, dst):
        tag = "child" if r.child else "adult"
        lines.append(f"{os.path.basename(dst)} | AI overlay | {tag} | fall at {r.t_fail:.1f}s | {r.wentwrong}")
        print(" ", lines[-1], flush=True)

normals = pd.read_csv("data/oops/normal_clips.csv")
normals = normals[normals["keep"] == True]
pools = [("data/oops/clips_normal", normals), ("data/oops/clips", falls)]
print("\nCollecting b-roll (raw clips by theme)...", flush=True)
for theme, pat in THEMES.items():
    n = 0
    for clip_dir, df in pools:
        for r in df[df["wentwrong"].astype(str).str.contains(pat, flags=re.I, regex=True)].itertuples():
            if n >= N_BROLL_EACH:
                break
            src = os.path.join(clip_dir, r.split, f"{r.filename}.mp4")
            if not os.path.exists(src):
                continue
            start = float(r.t_fail) - 4.0
            dst = os.path.join(OUT, f"broll_{theme}_{n + 1}.mp4")
            if to_mp4(src, dst, start=start, dur=5.0):
                n += 1
                lines.append(f"{os.path.basename(dst)} | raw b-roll | {theme} | {r.wentwrong}")
                print(" ", lines[-1], flush=True)

with open(os.path.join(OUT, "list.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
for name in os.listdir(TMP):
    os.remove(os.path.join(TMP, name))
os.rmdir(TMP)
total = sum(os.path.getsize(os.path.join(OUT, n)) for n in os.listdir(OUT)) / 1e6
print(f"\nDone: {len(lines)} clips in {OUT}/ ({total:.0f} MB). Watch them, then send me the ones you like + list.txt.")
