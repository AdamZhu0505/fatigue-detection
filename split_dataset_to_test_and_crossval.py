import os
import shutil
import random
from tqdm import tqdm
from sklearn.model_selection import KFold

# ------------------------
# Paths and parameters
# ------------------------
dataset_root = r'C:\TUB\Projekt\Version2\Dataset'

images_path = os.path.join(dataset_root, 'images', 'train')
labels_path = os.path.join(dataset_root, 'labels', 'train')

test_images_path = os.path.join(dataset_root, 'test', 'images')
test_labels_path = os.path.join(dataset_root, 'test', 'labels')

crossval_root = os.path.join(dataset_root, 'crossval')

num_folds = 3
test_ratio = 0.1

# ------------------------
# Make sure output folders exist
# ------------------------
os.makedirs(test_images_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)
os.makedirs(crossval_root, exist_ok=True)

print("Setting up...")

# ------------------------
# Gather all images
# ------------------------
all_images = []
for file in os.listdir(images_path):
    if file.endswith('.jpg') or file.endswith('.png'):
        all_images.append(file)

# Corresponding label files
all_labels = []
for img in all_images:
    if img.endswith('.jpg'):
        label = img.replace('.jpg', '.txt')
    elif img.endswith('.png'):
        label = img.replace('.png', '.txt')
    else:
        label = None
    all_labels.append(label)

all_samples = list(zip(all_images, all_labels))

# Shuffle
random.seed(42)
random.shuffle(all_samples)

# ------------------------
# Split test set
# ------------------------
num_test = int(len(all_samples) * test_ratio)
test_samples = all_samples[:num_test]
trainval_samples = all_samples[num_test:]

print(f"Total: {len(all_samples)} images")
print(f"Test:  {len(test_samples)}")
print(f"Train+Val: {len(trainval_samples)}")

# Copy test set
print("Copying test set...")
for img, lbl in tqdm(test_samples, desc="test"):
    shutil.copy(os.path.join(images_path, img), os.path.join(test_images_path, img))
    shutil.copy(os.path.join(labels_path, lbl), os.path.join(test_labels_path, lbl))

# ------------------------
# Cross-validation
# ------------------------
print(f"Cross-validation ({num_folds}-fold)...")

trainval_images = [img for img, _ in trainval_samples]

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(trainval_images)):
    print(f"Fold {fold_idx}")

    fold_dir = os.path.join(crossval_root, f'fold_{fold_idx}')
    train_img_dir = os.path.join(fold_dir, 'images', 'train')
    val_img_dir = os.path.join(fold_dir, 'images', 'val')
    train_lbl_dir = os.path.join(fold_dir, 'labels', 'train')
    val_lbl_dir = os.path.join(fold_dir, 'labels', 'val')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # Copy training files
    for i in train_idx:
        img = trainval_images[i]
        lbl = img.replace('.jpg', '.txt').replace('.png', '.txt')
        shutil.copy(os.path.join(images_path, img), os.path.join(train_img_dir, img))
        shutil.copy(os.path.join(labels_path, lbl), os.path.join(train_lbl_dir, lbl))

    # Copy validation files
    for i in val_idx:
        img = trainval_images[i]
        lbl = img.replace('.jpg', '.txt').replace('.png', '.txt')
        shutil.copy(os.path.join(images_path, img), os.path.join(val_img_dir, img))
        shutil.copy(os.path.join(labels_path, lbl), os.path.join(val_lbl_dir, lbl))

    print(f"  done")

print("Split complete.")
