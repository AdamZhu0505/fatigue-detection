# -*- coding: utf-8 -*-
# YOLO-only vs Fusion (YOLO OR EAR) comparison — simple, beginner style

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Paths --------
BASE_DIR = r"C:\TUB\Projekt\Version2"
OUT_DIR  = os.path.join(BASE_DIR, "out")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

EAR_CSV  = os.path.join(OUT_DIR, "ear_log.csv")
YOLO_CSV = os.path.join(OUT_DIR, "yolo_vote_log.csv")

# -------- Params --------
FPS = 10
WIN_SEC = 30
CALIB_SEC = 10
PERCLOS_T = 0.10          # 10%
LONG_FR = 10              # 10 frames = 1s
YOLO_T = 0.60             # YOLO closed threshold
EAR_RATIO = 0.90          # EAR threshold = baseline * EAR_RATIO

# -------- Small utils --------
def to_int_frame(x):
    m = re.search(r'(\d+)', str(x))
    return int(m.group(1)) if m else int(x)

def longest_run(flags):
    cur = 0
    best = 0
    for v in flags:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best

def mov_perclos(flags, w):
    arr = np.array(flags, dtype=int)
    out = []
    for i in range(len(arr)):
        s = max(0, i - w + 1)
        out.append(arr[s:i+1].mean())
    return out

# -------- Read CSVs --------
df_ear = pd.read_csv(EAR_CSV)
if "avg_ear" in df_ear.columns:
    df_ear = df_ear.rename(columns={"avg_ear": "ear_avg"})
df_ear["frame"] = df_ear["frame"].apply(to_int_frame)

df_yolo = pd.read_csv(YOLO_CSV)
# keep only one prob column -> p_yolo
if "yolo_closed_prob" in df_yolo.columns and "p_avg" in df_yolo.columns:
    df_yolo = df_yolo.drop(columns=["p_avg"])
if "yolo_closed_prob" in df_yolo.columns:
    df_yolo = df_yolo.rename(columns={"yolo_closed_prob": "p_yolo"})
elif "p_avg" in df_yolo.columns:
    df_yolo = df_yolo.rename(columns={"p_avg": "p_yolo"})
else:
    raise RuntimeError("yolo_vote_log.csv needs yolo_closed_prob or p_avg column")

df_yolo["frame"] = df_yolo["frame"].apply(to_int_frame)

# -------- Merge --------
df = pd.merge(
    df_ear[["seq", "frame", "ear_avg"]],
    df_yolo[["seq", "frame", "p_yolo"]],
    on=["seq", "frame"],
    how="inner"
).sort_values(["seq", "frame"]).reset_index(drop=True)

rows = []
W = WIN_SEC * FPS

# -------- Per-seq compute and plots --------
for seq, g in df.groupby("seq"):
    g = g.sort_values("frame").copy()

    # YOLO-only flags
    yolo_flag = (g["p_yolo"] >= YOLO_T).values

    # EAR threshold from first CALIB_SEC seconds (median)
    base = g.head(CALIB_SEC * FPS)["ear_avg"].median()
    if np.isnan(base):
        base = g["ear_avg"].median()
    ear_thr = base * EAR_RATIO
    ear_flag = (g["ear_avg"] < ear_thr).values

    # Fusion OR
    fuse_flag = yolo_flag | ear_flag

    # YOLO-only stats
    y_per = mov_perclos(yolo_flag, W)
    y_per_max = max(y_per) if len(y_per) > 0 else 0.0
    y_long = longest_run(yolo_flag)
    y_alert = (y_per_max >= PERCLOS_T) or (y_long >= LONG_FR)

    # Fusion stats
    f_per = mov_perclos(fuse_flag, W)
    f_per_max = max(f_per) if len(f_per) > 0 else 0.0
    f_long = longest_run(fuse_flag)
    f_alert = (f_per_max >= PERCLOS_T) or (f_long >= LONG_FR)

    rows.append({
        "seq": seq,
        "yolo_perclos_max": round(y_per_max, 3),
        "yolo_longest": int(y_long),
        "yolo_alert": y_alert,
        "fuse_perclos_max": round(f_per_max, 3),
        "fuse_longest": int(f_long),
        "fuse_alert": f_alert,
        "delta_perclos": round(f_per_max - y_per_max, 3),
        "delta_longest": int(f_long - y_long),
        "alert_change": f"{y_alert}->{f_alert}"
    })

    # ---- Plots (EN only to avoid font warnings) ----
    t = (g["frame"] - g["frame"].min()) / float(FPS)

    # YOLO-only plot
    plt.figure(figsize=(10, 4))
    plt.plot(t, g["p_yolo"], label="YOLO prob")
    plt.axhline(YOLO_T, ls="--", label="YOLO threshold")
    plt.title(f"{seq} | YOLO-only | PERCLOS={y_per_max:.2f} | Longest={y_long}f | ALERT={y_alert}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{seq}_yolo_only.png"))
    plt.close()

    # Fusion plot
    plt.figure(figsize=(10, 4))
    plt.plot(t, g["ear_avg"], label="EAR")
    plt.axhline(ear_thr, ls="--", label="EAR threshold")
    plt.plot(t, g["p_yolo"], label="YOLO prob")
    plt.axhline(YOLO_T, ls="--", color="orange",label="YOLO threshold")
    plt.title(f"{seq} | FUSION | PERCLOS={f_per_max:.2f} | Longest={f_long}f | ALERT={f_alert}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{seq}_fusion.png"))
    plt.close()

# -------- Save compare table --------
cmp_path = os.path.join(OUT_DIR, "compare_yolo_vs_fuse.csv")
pd.DataFrame(rows).to_csv(cmp_path, index=False)
print("OK ->", cmp_path, "| plots in", PLOT_DIR)
