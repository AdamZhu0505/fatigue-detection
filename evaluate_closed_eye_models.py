from ultralytics import YOLO
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

# Models to evaluate
model_paths = [
    r'C:\TUB\Projekt\Version2\models\best_0.pt',
    r'C:\TUB\Projekt\Version2\models\best_1.pt',
    r'C:\TUB\Projekt\Version2\models\best_2.pt'
]

# Test images (only left eye will be evaluated)
image_dir = r'C:\TUB\Projekt\Version2\data\crossval\test'

# Class ID for left eye closed
left_eye_closed_class = 2
conf_thresh = 0.8

# Gather image files
image_paths = sorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith(('.jpg', '.png')) and f.lower().startswith("left_")
])

print(f"Found {len(image_paths)} left-eye images.\n")

# Evaluate each model
for model_path in model_paths:
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"Evaluating {model_name}...")

    model = YOLO(model_path)

    # Output: prediction + plot
    output_dir = os.path.join(os.path.dirname(image_dir), f'closed_eye_{model_name}')
    os.makedirs(output_dir, exist_ok=True)

    # Output: high-confidence closed eyes (global)
    high_conf_dir = r'C:\TUB\Projekt\Version2\out\closed_eye_high_conf'
    os.makedirs(high_conf_dir, exist_ok=True)

    total = 0
    closed = 0
    confidence_scores = []
    high_conf_count = 0

    for path in tqdm(image_paths, desc=f"{model_name}"):
        total += 1
        name = os.path.basename(path).lower()
        results = model(path, verbose=False, conf=0.25)[0]

        found = False
        if results.boxes and results.boxes.cls is not None:
            classes = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for cls, conf in zip(classes, confs):
                if int(cls) == left_eye_closed_class:
                    confidence_scores.append(conf)

                    if conf >= conf_thresh:
                        if not found:
                            # 保存到当前模型输出文件夹
                            shutil.copy2(path, os.path.join(output_dir, name))
                            closed += 1
                            found = True  # 只复制一张

                        # 再复制一份到全局高置信度图像文件夹
                        try:
                            shutil.copy2(path, os.path.join(high_conf_dir, name))
                            high_conf_count += 1
                        except Exception as e:
                            print(f"Failed to copy {name} to high-conf dir: {e}")

    # Summary
    print("\nSummary")
    if total:
        print(f"Closed (≥ {conf_thresh}): {closed} / {total} ({closed / total:.2%})")
        print(f"Also copied {high_conf_count} high-confidence closed-eye images to:")
        print(f"{high_conf_dir}")
    else:
        print("No images found.")

    # Plot
    if confidence_scores:
        plt.figure(figsize=(8, 5))
        plt.hist(confidence_scores, bins=20, color='steelblue', edgecolor='black')
        plt.axvline(conf_thresh, color='red', linestyle='--', label=f'Thresh = {conf_thresh}')
        plt.title(f'Confidence Distribution ({model_name})')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"{model_name}_confidence_hist.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved plot: {fig_path}")
    else:
        print("No predictions for left eye closed.")

    print("-" * 40)
