# -*- coding: utf-8 -*-
"""
Simplest YOLO + EAR fusion script:
- Reads avg_ear from ear_log.csv
- Reads yolo_closed_prob from yolo_vote_log.csv
- Fuses to create used_closed_flag
- Computes PERCLOS and LongestClosed
- Outputs CSV and plots for each sequence
"""

import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== Paths ==========
BASE_DIR = r"C:\TUB\Projekt\Version2"
OUT_DIR  = os.path.join(BASE_DIR, "out")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

EAR_CSV  = os.path.join(OUT_DIR, "ear_log.csv")
YOLO_CSV = os.path.join(OUT_DIR, "yolo_vote_log.csv")

# ========== Parameters ==========
FPS = 10                          # Frames per second
CALIB_SEC = 10                   # Calibration time (first 10 sec)
WINDOW_SEC = 30                  # Sliding window for PERCLOS
PERCLOS_THRESH = 0.10            # Threshold for fatigue alert (PERCLOS)
LONGEST_FRAMES = 10              # Continuous closed threshold (frames)
EAR_RATIO = 0.75                 # Threshold = median_ear * ratio
YOLO_THRESH = 0.6                # YOLO probability threshold

# ========== Helper Functions ==========

def to_int_frame(x):
    """
    Extract numeric frame index from string (e.g., 'frame_003.png' → 3)
    """
    s = str(x)
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else int(x)

def longest_run(flags):
    """
    Count longest continuous True values in a list
    """
    cur, best = 0, 0
    for f in flags:
        if f:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

def moving_perclos(flags, win):
    """
    Compute moving average of 'closed' flags over sliding window
    """
    arr = np.array(flags, int)
    out = []
    for i in range(len(arr)):
        start = max(0, i-win+1)
        out.append(arr[start:i+1].mean())
    return out

# ========== Read CSVs ==========
print("=== Reading CSV files ===")
df_ear = pd.read_csv(EAR_CSV)
df_yolo = pd.read_csv(YOLO_CSV)

# Rename columns if needed
if "avg_ear" in df_ear.columns:
    df_ear = df_ear.rename(columns={"avg_ear": "ear_avg"})
df_ear["frame"] = df_ear["frame"].apply(to_int_frame)

if "yolo_closed_prob" in df_yolo.columns:
    df_yolo = df_yolo.rename(columns={"yolo_closed_prob": "p_yolo"})
if "p_avg" in df_yolo.columns:
    df_yolo = df_yolo.drop(columns=["p_avg"])  # Avoid duplication
df_yolo["frame"] = df_yolo["frame"].apply(to_int_frame)

# ========== Merge DataFrames ==========
df = pd.merge(df_ear[["seq","frame","ear_avg"]],
              df_yolo[["seq","frame","p_yolo"]],
              on=["seq","frame"], how="inner")
print("Rows after merge:", len(df))

# ========== EAR-based closed flag ==========
df["ear_closed"] = False
for seq, g in df.groupby("seq"):
    thr = g.head(CALIB_SEC*FPS)["ear_avg"].median() * EAR_RATIO
    df.loc[df["seq"]==seq, "ear_closed"] = df.loc[df["seq"]==seq, "ear_avg"] < thr
    df.loc[df["seq"]==seq, "ear_thr"] = thr

# ========== YOLO-based closed flag ==========
df["yolo_closed"] = df["p_yolo"] >= YOLO_THRESH

# ========== Final Fusion: either one detects closed ==========
df["used_closed"] = df["ear_closed"] | df["yolo_closed"]

# ========== Per-sequence Analysis ==========
summary = []
for seq, g in df.groupby("seq"):
    flags = g["used_closed"].values
    win_size = WINDOW_SEC * FPS
    perclos_win = moving_perclos(flags, win_size)
    g["perclos_win"] = perclos_win

    perclos_max = max(perclos_win) if len(perclos_win) > 0 else 0
    longest = longest_run(flags)
    alert = (perclos_max >= PERCLOS_THRESH) or (longest >= LONGEST_FRAMES)

    summary.append({
        "seq": seq,
        "perclos_max": round(perclos_max, 3),
        "longest_frames": longest,
        "longest_sec": round(longest / FPS, 2),
        "alert": alert
    })

    # ========== Plot for this sequence ==========
    t = (g["frame"] - g["frame"].min()) / FPS
    plt.figure(figsize=(10, 4))
    plt.plot(t, g["ear_avg"], label="EAR")
    plt.plot(t, g["p_yolo"], label="YOLO prob")
    plt.axhline(YOLO_THRESH, color="g", linestyle="--")
    plt.axhline(g["ear_thr"].iloc[0], color="r", linestyle="--")
    plt.title(f"{seq} | PERCLOS={perclos_max:.2f} | Longest={longest}f | ALERT={alert}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{seq}_fusion.png"))
    plt.close()

# ========== Save Outputs ==========
df.to_csv(os.path.join(OUT_DIR, "fused_log.csv"), index=False)
pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, "perclos_fused_summary.csv"), index=False)

print("=== Fusion complete ===")

