from ultralytics import YOLO
import os

# --------------------------
# Basic settings
# --------------------------
model_path = 'yolov8n.pt'   # You can replace this with your own .pt model
base_data_path = r'C:\TUB\Projekt\Version2\Dataset\crossval'

epochs = 50
imgsz = 640
batch = 16
device = 0  # 0 = first GPU; use 'cpu' if needed

# --------------------------
# Train across 3 folds
# --------------------------
for fold in range(3):
    print(f"\nFold {fold}...")

    # Path to YAML file for this fold
    data_yaml = os.path.join(base_data_path, f"fold_{fold}", f"data_fold_{fold}.yaml")

    # Load model
    model = YOLO(model_path)

    # Start training
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name=f"fold{fold}"
    )

print("\nDone.")
