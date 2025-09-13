from ultralytics import YOLO
import os
import shutil
from tqdm import tqdm

# ✅ Load two YOLO models
model0 = YOLO(r'C:\TUB\Projekt\Version2\models\best_0.pt')
model1 = YOLO(r'C:\TUB\Projekt\Version1\models\best_1.pt')

# ✅ Class index and confidence threshold
left_eye_closed_class = 2         # Class index for "left eye closed"
conf_threshold = 0.8              # Minimum confidence for "strong" decision

# ✅ Input image directory
image_dir = r'C:\TUB\Projekt\Version1\left_eye\images'
image_paths = sorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith(('.jpg', '.png')) and f.lower().startswith("left_")
])

# ✅ Output directories
base_output = r'C:\TUB\Projekt\Version1\fusion_results'
closed_dir = os.path.join(base_output, 'voted_closed')           # Strongly voted closed eye
uncertain_dir = os.path.join(base_output, 'voted_uncertain')     # Uncertain (only one model says closed)
non_closed_dir = os.path.join(base_output, 'voted_non_closed')   # Both models say not closed

os.makedirs(closed_dir, exist_ok=True)
os.makedirs(uncertain_dir, exist_ok=True)
os.makedirs(non_closed_dir, exist_ok=True)

# ✅ Counters
final_closed = []   # Strong closed eye images
uncertain = []      # Only one model says closed
non_closed = []     # Neither model says closed

# ✅ Loop through all images
for path in tqdm(image_paths, desc="Fusion Voting"):
    filename = os.path.basename(path)

    # Inference: model0
    res0 = model0(path, verbose=False, conf=0.25)[0]
    cls0 = res0.boxes.cls.cpu().numpy() if res0.boxes.cls is not None else []
    conf0 = res0.boxes.conf.cpu().numpy() if res0.boxes.conf is not None else []
    pred0 = [(c, s) for c, s in zip(cls0, conf0) if int(c) == left_eye_closed_class]

    # Inference: model1
    res1 = model1(path, verbose=False, conf=0.25)[0]
    cls1 = res1.boxes.cls.cpu().numpy() if res1.boxes.cls is not None else []
    conf1 = res1.boxes.conf.cpu().numpy() if res1.boxes.conf is not None else []
    pred1 = [(c, s) for c, s in zip(cls1, conf1) if int(c) == left_eye_closed_class]

    # Determine if each model detected closed eyes
    is_closed_0 = len(pred0) > 0
    is_closed_1 = len(pred1) > 0
    max_conf_0 = max([s for _, s in pred0], default=0)
    max_conf_1 = max([s for _, s in pred1], default=0)

    # ✅ Fusion voting logic
    if is_closed_0 and is_closed_1 and (max_conf_0 >= conf_threshold or max_conf_1 >= conf_threshold):
        # Both models detect closed eye, and at least one is confident
        final_closed.append(filename)
        dst_path = os.path.join(closed_dir, filename)
    elif is_closed_0 or is_closed_1:
        # Only one model detects closed eye → uncertain
        uncertain.append(filename)
        dst_path = os.path.join(uncertain_dir, filename)
    else:
        # No model detects closed eye
        non_closed.append(filename)
        dst_path = os.path.join(non_closed_dir, filename)

    # ✅ Copy image to corresponding folder
    shutil.copy2(path, dst_path)

# ✅ Print summary
print("\n=========== Fusion Voting Summary ===========")
print(f"✅ Strong Closed (both models agree): {len(final_closed)} images")
print(f"❓ Uncertain (only one model says closed): {len(uncertain)} images")
print(f"❌ Not Closed (neither model): {len(non_closed)} images")
print(f"\nImages saved to: {base_output}")
