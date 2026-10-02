"""
Picks the live warning threshold by SIMULATING the device's warning logic
(smoothing + must-stay-high + hysteresis, from alert.py) on the validation clips,
then reports honest numbers on the untouched test clips.
Rule: lowest threshold where at most MAX_FALSE_CLIP_RATE of normal val clips get a false warning.
Run from ~/acc-ai-mvp after train_oops.py:
    python src/calibrate.py
Saves "live_threshold" into data/oops/results/config.json, full table in calibration.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fallnet import FallTCN
from alert import RiskSmoother, WarningTrigger
from train_oops import load_clips, make_index, predict, group_by_clip, set_seed, OUT_DIR, EVAL_STRIDE, FPS, SEED

MAX_FALSE_CLIP_RATE = 0.15
THRESHOLDS = np.round(np.arange(0.30, 0.96, 0.02), 2)


def simulate(clips, items, scores, thr):
    leads = []
    n_fall = n_norm = n_norm_warned = warnings = norm_windows = 0
    for ci, ks in group_by_clip(items).items():
        c = clips[ci]
        smoother, trigger = RiskSmoother(), WarningTrigger(thr)
        first_lead, count = None, 0
        for k in ks:
            _, end, _, ttf = items[k]
            if trigger.update(smoother.update(float(scores[k])), float(c["times"][end - 1])) == "warn":
                count += 1
                if first_lead is None:
                    first_lead = ttf
        if c["kind"] == "fall":
            n_fall += 1
            if first_lead is not None:
                leads.append(first_lead)
        else:
            n_norm += 1
            norm_windows += len(ks)
            warnings += count
            n_norm_warned += int(count > 0)
    leads = np.array(leads)
    hours = norm_windows * EVAL_STRIDE / FPS / 3600
    return {"threshold": float(thr),
            "falls_warned_%": 100 * len(leads) / max(n_fall, 1),
            "warned_1s_before_%": 100 * (leads >= 1).sum() / max(n_fall, 1),
            "warned_2s_before_%": 100 * (leads >= 2).sum() / max(n_fall, 1),
            "median_lead_s": float(np.median(leads)) if len(leads) else float("nan"),
            "normal_clips_false_%": 100 * n_norm_warned / max(n_norm, 1),
            "false_per_hour": warnings / hours if hours > 0 else float("nan")}


def main():
    set_seed(SEED)
    cfg_path = os.path.join(OUT_DIR, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = FallTCN()
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "tcn_oops.pt"), map_location="cpu"))

    clips = load_clips()
    parts = {s: [c for c in clips if c["split"] == s] for s in ("val", "test")}
    idx = {s: make_index(parts[s], EVAL_STRIDE) for s in parts}
    scores = {s: predict(model, parts[s], idx[s], torch.device("cpu")) for s in parts}

    val = pd.DataFrame([simulate(parts["val"], idx["val"], scores["val"], t) for t in THRESHOLDS])
    test = pd.DataFrame([simulate(parts["test"], idx["test"], scores["test"], t) for t in THRESHOLDS])
    ok = val[val["normal_clips_false_%"] <= 100 * MAX_FALSE_CLIP_RATE]
    chosen = float(ok["threshold"].min()) if len(ok) else float(val.loc[val["normal_clips_false_%"].idxmin(), "threshold"])

    pd.set_option("display.width", 160)
    print("VALIDATION trade-off (with smoothing + half-second hold):")
    print(val.iloc[::3].round(1).to_string(index=False))
    row = test[np.isclose(test["threshold"], chosen)].iloc[0]
    print(f"\nChosen live threshold: {chosen:.2f}  (at most {MAX_FALSE_CLIP_RATE*100:.0f}% of normal val clips falsely warned)")
    print("\nTEST RESULTS at that threshold (clips never used for training or choosing):")
    print(f"  Falls warned before impact:  {row['falls_warned_%']:.0f}%")
    print(f"  Warned at least 1s before:   {row['warned_1s_before_%']:.0f}%")
    print(f"  Warned at least 2s before:   {row['warned_2s_before_%']:.0f}%")
    print(f"  Median lead time:            {row['median_lead_s']:.2f}s")
    print(f"  Normal clips falsely warned: {row['normal_clips_false_%']:.0f}%")
    print(f"  False warnings per hour:     {row['false_per_hour']:.0f}")

    cfg["live_threshold"] = chosen
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    pd.concat([val.assign(split="val"), test.assign(split="test")]).to_csv(
        os.path.join(OUT_DIR, "calibration.csv"), index=False)
    print(f"\nSaved live_threshold to {cfg_path} and the full table to {OUT_DIR}/calibration.csv")


if __name__ == "__main__":
    main()
