"""
Copies only the clips listed in data/oops/selected_clips.csv out of
video.tar.gz, without unpacking the rest. Run from ~/acc-ai-mvp:
    python src/oops_unpack.py
Output: data/oops/clips/train/*.mp4, data/oops/clips/val/*.mp4
"""
import os
import shutil
import tarfile

import pandas as pd

ARCHIVE = "data/oops/oops_dataset/video.tar.gz"
SELECTED = "data/oops/selected_clips.csv"
OUT_DIR = "data/oops/clips"
MISSING_TXT = "data/oops/missing_clips.txt"

df = pd.read_csv(SELECTED)
wanted = dict(zip(df["filename"], df["split"]))
for split in ("train", "val"):
    os.makedirs(os.path.join(OUT_DIR, split), exist_ok=True)

found = set()
scanned = 0
print(f"Looking for {len(wanted)} clips in {ARCHIVE} (this reads the whole archive once)...", flush=True)

with tarfile.open(ARCHIVE, "r|gz") as tar:
    for member in tar:
        if not member.isfile():
            continue
        scanned += 1
        if scanned <= 3:
            print(f"  example name inside archive: {member.name}", flush=True)
        base = os.path.basename(member.name)
        if not base.endswith(".mp4"):
            continue
        key = base[:-4]
        if key in wanted and key not in found:
            out_path = os.path.join(OUT_DIR, wanted[key], base)
            if not (os.path.exists(out_path) and os.path.getsize(out_path) == member.size):
                src = tar.extractfile(member)
                with open(out_path, "wb") as f:
                    shutil.copyfileobj(src, f)
            found.add(key)
            if len(found) == len(wanted):
                print("All selected clips found, stopping early.", flush=True)
                break
        if scanned % 1000 == 0:
            print(f"  scanned {scanned} files, extracted {len(found)}/{len(wanted)}", flush=True)

missing = sorted(k for k in wanted if k not in found)
with open(MISSING_TXT, "w") as f:
    f.write("\n".join(missing))

print(f"\nDone. Extracted {len(found)}/{len(wanted)} clips into {OUT_DIR}/")
print(f"Missing: {len(missing)} (listed in {MISSING_TXT})")
