import pandas as pd

df = pd.read_csv("data/labels/labels.csv")
falls = df[df["label"] == "fall"].copy()
falls["pre_risk_frames"] = falls["risk_start_frame"]  # frames available before risk window starts
falls["testable"] = falls["pre_risk_frames"] >= 30     # needs >= WINDOW_SIZE to plot any pre-window score
print(falls[["folder_path", "risk_start_frame", "testable"]].to_string(index=False))
print(f"\n{falls['testable'].sum()} of {len(falls)} fall clips have enough pre-risk buffer to test anticipation")
