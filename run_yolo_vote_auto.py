# -*- coding: utf-8 -*-
"""
YOLO Dual-Model × Left/Right Eye Voting Fusion (Supports Both Classification and Detection)
This script outputs `yolo_vote_log.csv`, compatible with downstream fusion logic.
Style: beginner-friendly, with redundant comments and debug prints.
"""

from ultralytics import YOLO
import os, re, csv, sys
from tqdm import tqdm
import numpy as np
import cv2
import torch

# ========== Paths and Parameters ==========
images_root = r"C:\TUB\Projekt\Version2\data\images"     # Root folder containing all seq_xxx
weights_a   = r"C:\TUB\Projekt\Version2\models\best_0.pt"
weights_b   = r"C:\TUB\Projekt\Version2\models\best_1.pt"
out_dir     = r"C:\TUB\Projekt\Version2\out"
os.makedirs(out_dir, exist_ok=True)
csv_path    = os.path.join(out_dir, "yolo_vote_log.csv")

# Thresholds for decision
prob_thresh       = 0.6   # If average of 4 scores ≥ this → vote_or = 1
conf_min_for_and  = 0.8   # For vote_and = 1 → both eyes must have at least one score ≥ this

# Optional image sizes (normally no need to change)
imgsz_cls = 224   # For classification models
imgsz_det = 640   # For detection models

# ========== Class name keywords ==========
CLOSED_GENERIC = ["closed", "eye_closed", "closed_eye", "shut"]
OPEN_GENERIC   = ["open", "eye_open", "opened"]
LEFT_MARKERS   = ["left", "l_", "l-"]
RIGHT_MARKERS  = ["right", "r_", "r-"]

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def get_side_closed_idx(model, side: str):
    """
    Try to find the class index for left/right closed eyes in model.names.
    1) Look for both side and closed in name
    2) Look for generic 'closed'
    3) If only open/closed, return the non-open
    4) If single class, return it
    5) Otherwise, return None
    """
    names = getattr(model, "names", {})
    if not isinstance(names, dict) or len(names) == 0:
        return None
    items = {int(k): _norm(v) for k, v in names.items()}
    side_keys = LEFT_MARKERS if side == "left" else RIGHT_MARKERS

    for k, v in items.items():
        if any(sk in v for sk in side_keys) and any(c in v for c in CLOSED_GENERIC):
            return k
    for k, v in items.items():
        if any(c in v for c in CLOSED_GENERIC):
            return k
    if len(items) == 2 and any(any(o in v for o in OPEN_GENERIC) for v in items.values()):
        for k, v in items.items():
            if not any(o in v for o in OPEN_GENERIC):
                return k
    if len(items) == 1:
        return list(items.keys())[0]
    return None

# ========== Match left/right image pairs ==========
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def list_pairs(seq_dir):
    """
    List all valid (left, right) image pairs in a given folder.
    Returns a list of tuples: (frame_id, left_path, right_path)
    """
    files = [f for f in os.listdir(seq_dir) if f.lower().endswith(IMG_EXTS)]
    lefts, rights = {}, {}
    for f in files:
        m_left  = re.match(r'^(left)_(\d+)\.[^.]+$', f, re.I)
        m_right = re.match(r'^(right)_(\d+)\.[^.]+$', f, re.I)
        if m_left:
            lefts[m_left.group(2)] = f
        if m_right:
            rights[m_right.group(2)] = f
    ids = sorted(set(lefts) & set(rights), key=lambda x: int(x))
    return [(fid, os.path.join(seq_dir, lefts[fid]), os.path.join(seq_dir, rights[fid])) for fid in ids]

# ========== Compute closed-eye confidence ==========
def closed_score(model, img_bgr, side_idx):
    """
    Return confidence score (0~1) for closed eyes.
    Handles both classification and detection models.
    """
    task = getattr(model, 'task', None)
    imgsz = imgsz_cls if task == 'classify' else imgsz_det
    res = model.predict(img_bgr, imgsz=imgsz, conf=0.001, verbose=False)[0]

    if getattr(res, "probs", None):
        probs = res.probs.data
        probs = probs.detach().float().cpu().numpy() if hasattr(probs, "detach") else np.array(probs)
        probs = np.clip(probs, 0, 1)
        return float(probs[int(side_idx)]) if side_idx is not None and int(side_idx) < len(probs) else float(np.max(probs))

    if getattr(res, "boxes", None) and res.boxes.conf is not None:
        cls  = res.boxes.cls.detach().cpu().numpy().astype(int)
        conf = res.boxes.conf.detach().cpu().numpy()
        if len(cls) == 0:
            return 0.0
        if side_idx is not None:
            mask = (cls == int(side_idx))
            return float(conf[mask].max()) if mask.any() else 0.0
        items = {int(k): _norm(v) for k, v in getattr(model, "names", {}).items()}
        non_open_ids = [k for k, v in items.items() if not any(o in v for o in OPEN_GENERIC)]
        if len(non_open_ids) > 0:
            mask = np.isin(cls, non_open_ids)
            return float(conf[mask].max()) if mask.any() else 0.0
        return float(conf.max())
    return 0.0

# ========== Load YOLO model ==========
def load_model(path):
    print(f"🔹 Loading model: {path}")
    m = YOLO(path)
    if torch.cuda.is_available():
        try:
            m.to('cuda')
            print("⚡ Using CUDA")
        except Exception as e:
            print(f"[CUDA warning] {e}")
    print(f"🧭 Task={getattr(m, 'task', 'unknown')}, Classes={getattr(m, 'names', {})}")
    return m

model_a = load_model(weights_a)
model_b = load_model(weights_b)

# Get class indices
left_idx_a  = get_side_closed_idx(model_a, "left")
right_idx_a = get_side_closed_idx(model_a, "right")
left_idx_b  = get_side_closed_idx(model_b, "left")
right_idx_b = get_side_closed_idx(model_b, "right")
print(f"Class indices - A: left={left_idx_a}, right={right_idx_a} | B: left={left_idx_b}, right={right_idx_b}")

# ========== Process all sequences ==========
header = [
    "seq", "frame",
    "left_closed_a","left_closed_b",
    "right_closed_a","right_closed_b",
    "p_left","p_right",
    "yolo_closed_prob", "p_avg",
    "vote_or", "vote_and"
]
rows = []

if not os.path.isdir(images_root):
    print(f"[Error] Folder not found: {images_root}")
    sys.exit(1)

seq_list = [d for d in sorted(os.listdir(images_root)) if os.path.isdir(os.path.join(images_root, d))]
if not seq_list:
    print(f"[Warning] No subfolders found in {images_root}")

for seq in seq_list:
    seq_dir = os.path.join(images_root, seq)
    pairs = list_pairs(seq_dir)
    print(f"📂 {seq}: {len(pairs)} image pairs found")
    for fid, lp, rp in tqdm(pairs, desc=f"{seq}", leave=False):
        lbgr, rbgr = cv2.imread(lp), cv2.imread(rp)
        if lbgr is None or rbgr is None:
            print(f"[Warning] Failed to read: {lp} or {rp}")
            continue

        la = closed_score(model_a, lbgr, left_idx_a)
        lb = closed_score(model_b, lbgr, left_idx_b)
        ra = closed_score(model_a, rbgr, right_idx_a)
        rb = closed_score(model_b, rbgr, right_idx_b)

        la, lb, ra, rb = map(lambda x: float(np.clip(x, 0.0, 1.0)), [la, lb, ra, rb])

        p_left  = max(la, lb)
        p_right = max(ra, rb)
        avg_prob = float(np.mean([la, lb, ra, rb]))
        vote_or = int(avg_prob >= prob_thresh)
        vote_and = int((p_left >= conf_min_for_and) and (p_right >= conf_min_for_and))

        rows.append([
            seq, int(fid),
            round(la,4), round(lb,4),
            round(ra,4), round(rb,4),
            round(p_left,4), round(p_right,4),
            round(avg_prob,4), round(avg_prob,4),
            vote_or, vote_and
        ])

# Save to CSV
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"✅ Saved voting results to: {csv_path}")
print("🏁 Done: all sequences processed.")
