import os
import cv2
import numpy as np
import pandas as pd

# ========== Parameters ==========
images_root = "data/images"     # Root path to all sequences
out_dir = "out"                 # Output directory for CSV files
fps = 10                        # Frames per second
window_seconds = 30            # Sliding window size in seconds
ear_threshold = 0.2            # EAR below this is considered closed
perclos_threshold = 0.1        # If >10% of frames are closed → alarm
longest_closed_limit = 10      # If eye closed for ≥10 consecutive frames → alarm

# ========== Simplified EAR Calculation ==========
def dummy_ear(image):
    """
    Approximate the eye aspect ratio (EAR) using edge height.
    This is a fast alternative to landmark detection.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 150)

    ys, xs = np.where(edges > 0)
    if len(ys) < 2:
        return 0.0  # No edge detected

    eye_height = ys.max() - ys.min()
    ear = eye_height / w
    return min(ear, 1.0)

# ========== Initialize ==========
ear_log = []

# Loop through each sequence folder
for seq_name in sorted(os.listdir(images_root)):
    seq_path = os.path.join(images_root, seq_name)
    if not os.path.isdir(seq_path):
        continue

    print(f"📂 Processing sequence: {seq_name}")

    # List all left eye images and sort by frame index
    left_imgs = sorted([f for f in os.listdir(seq_path) if f.startswith("left")])
    frame_indices = [f.split("_")[1].split(".")[0] for f in left_imgs]

    for frame_idx in frame_indices:
        left_path = os.path.join(seq_path, f"left_{frame_idx}.png")
        right_path = os.path.join(seq_path, f"right_{frame_idx}.png")

        if not os.path.exists(left_path) or not os.path.exists(right_path):
            print(f" Missing image pair: {frame_idx}, skipping")
            continue

        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        if left_img is None or right_img is None:
            print(f" Failed to read image: {frame_idx}")
            continue

        # Compute dummy EAR for both eyes
        left_ear = dummy_ear(left_img)
        right_ear = dummy_ear(right_img)
        avg_ear = (left_ear + right_ear) / 2

        # Save to log
        ear_log.append({
            "seq": seq_name,
            "frame": frame_idx,
            "left_ear": round(left_ear, 4),
            "right_ear": round(right_ear, 4),
            "avg_ear": round(avg_ear, 4)
        })

# ========== Save EAR Log ==========
os.makedirs(out_dir, exist_ok=True)
ear_df = pd.DataFrame(ear_log)
ear_df.to_csv(os.path.join(out_dir, "ear_log.csv"), index=False)
print(" Saved ear_log.csv with total frames:", len(ear_df))

# ========== PERCLOS Calculation ==========
summary = []
window_size = fps * window_seconds  # Convert window from seconds to frame count

for seq in ear_df["seq"].unique():
    seq_df = ear_df[ear_df["seq"] == seq].sort_values("frame")
    ear_values = seq_df["avg_ear"].values
    n_frames = len(ear_values)

    print(f" Analyzing sequence: {seq}, total frames: {n_frames}")

    for start in range(0, n_frames - window_size + 1):
        window = ear_values[start:start + window_size]
        is_closed = window < ear_threshold

        perclos = np.mean(is_closed)  # Ratio of closed frames
        longest_closed = 0
        count = 0

        # Count longest continuous closed segment
        for closed in is_closed:
            if closed:
                count += 1
                longest_closed = max(longest_closed, count)
            else:
                count = 0

        # Determine if alarm should be raised
        alarm = perclos > perclos_threshold or longest_closed >= longest_closed_limit

        summary.append({
            "seq": seq,
            "start_frame": start,
            "end_frame": start + window_size - 1,
            "perclos": round(perclos, 4),
            "longest_closed": longest_closed,
            "alarm": int(alarm)
        })

# ========== Save PERCLOS Summary ==========
pd.DataFrame(summary).to_csv(os.path.join(out_dir, "perclos_summary.csv"), index=False)
print(" Saved perclos_summary.csv")
print(" Script completed")
