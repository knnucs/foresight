"""
Selects fall clips from the Oops! dataset using the natural-language
"what went wrong" descriptions and the labeled failure times.
No video is read. Run from ~/acc-ai-mvp:
    python src/oops_select.py
Output: data/oops/selected_clips.csv
"""
import json
import os
import re
import statistics
from collections import Counter

import pandas as pd

OOPS_DIR = "data/oops"
ANN_DIR = os.path.join(OOPS_DIR, "oops_dataset", "annotations")
LANG_DIR = os.path.join(OOPS_DIR, "lang", "svos")
OUT_CSV = os.path.join(OOPS_DIR, "selected_clips.csv")

MIN_PRE_SECONDS = 2.0
MAX_LABEL_STDEV = 1.0

FALL_RE = re.compile(
    r"\b(fall|falls|fell|falling|fallen|slip|slips|slipped|slipping|"
    r"trip|trips|tripped|tripping|tumble|tumbles|tumbled|tumbling|"
    r"stumble|stumbles|stumbled|stumbling|faceplant|faceplants|"
    r"wipes out|wiped out|wipe out)\b", re.I)
NEG_RE = re.compile(
    r"(\bnot|n't|\bnever|\bwithout|\balmost|\bnearly)\s+(\w+\s+)?"
    r"(fall|falls|fell|falling|slip|slipped|trip|tripped|tumble|stumble)", re.I)
CHILD_RE = re.compile(
    r"\b(kid|kids|child|children|baby|babies|boy|boys|girl|girls|"
    r"toddler|toddlers|son|daughter)\b", re.I)


def norm(name):
    name = name.strip()
    return name[:-4] if name.endswith(".mp4") else name


def read_list(filename):
    with open(os.path.join(ANN_DIR, filename)) as f:
        return set(norm(line) for line in f if line.strip())


with open(os.path.join(ANN_DIR, "transition_times.json")) as f:
    times = {norm(k): v for k, v in json.load(f).items()}

train_set = read_list("train_filtered.txt")
val_set = read_list("val_filtered.txt")

lang = {}
for part in ("train.json", "val.json"):
    with open(os.path.join(LANG_DIR, part)) as f:
        for k, v in json.load(f).items():
            lang[norm(k)] = v

print(f"Clips with failure times:   {len(times)}")
print(f"Clips with descriptions:    {len(lang)}")

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

    texts = [a.get("wentwrong", "") or "" for a in anns]
    fall_votes = sum(1 for t in texts if FALL_RE.search(t) and not NEG_RE.search(t))
    if fall_votes == 0:
        stats["not described as a fall"] += 1
        continue

    valid = [x for x in times[name]["t"] if x >= 0]
    if len(valid) < 2:
        stats["fewer than 2 failure labels"] += 1
        continue
    t_fail = statistics.median(valid)
    spread = statistics.pstdev(valid)
    if spread > MAX_LABEL_STDEV:
        stats["labelers disagreed on time"] += 1
        continue
    if t_fail < MIN_PRE_SECONDS:
        stats["too little footage before fall"] += 1
        continue

    all_text = " ".join((a.get("goal", "") or "") + " " + (a.get("wentwrong", "") or "") for a in anns)
    rows.append({
        "filename": name,
        "split": split,
        "t_fail": round(t_fail, 3),
        "label_spread": round(spread, 3),
        "video_len": round(times[name]["len"], 3),
        "pre_seconds": round(t_fail, 3),
        "fall_votes": fall_votes,
        "child": bool(CHILD_RE.search(all_text)),
        "wentwrong": texts[0].replace("\n", " "),
    })

df = pd.DataFrame(rows).sort_values(["split", "filename"]).reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)

print("\nRemoved:")
for reason, n in stats.most_common():
    print(f"  {reason:<35}{n}")

print(f"\nSELECTED: {len(df)} fall clips  ->  {OUT_CSV}")
print(df["split"].value_counts().to_string())
print(f"\nInvolving a child:          {int(df['child'].sum())}")
for s in (2, 3, 4, 5):
    print(f"With >= {s}s before the fall:  {int((df['pre_seconds'] >= s).sum())}")

print("\nRandom examples (check these look like real falls):")
for _, r in df.sample(min(10, len(df)), random_state=0).iterrows():
    print(f"  [{r['split']}, child={r['child']}, {r['pre_seconds']}s before] {r['wentwrong'][:90]}")
