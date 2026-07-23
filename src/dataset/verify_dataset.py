from pathlib import Path
from PIL import Image

DATASET_PATH = Path("data/imagenet_subset/ILSVRC2012_img_val_subset")

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

print("✓ Dataset folder found.")

# Get all class folders
class_folders = [folder for folder in DATASET_PATH.iterdir() if folder.is_dir()]

print(f"Number of class folders: {len(class_folders)}")

EXPECTED_CLASSES = 1000

if len(class_folders) != EXPECTED_CLASSES:
    print("❌ Warning: Unexpected number of class folders.")
else:
    print("✓ Correct number of class folders found.")

total_images = 0
empty_folders = []

for folder in class_folders:
    images = list(folder.glob("*.JPEG"))

    if len(images) == 0:
        empty_folders.append(folder.name)

    total_images += len(images)

print(f"Total images found: {total_images}")

if empty_folders:
    print(f"❌ Empty folders found: {len(empty_folders)}")
else:
    print("✓ Every class folder contains images.")

corrupted_images = []

for folder in class_folders:
    images = list(folder.glob("*.JPEG"))

    for image_path in images:
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            corrupted_images.append(str(image_path))

if corrupted_images:
    print(f"❌ Corrupted images found: {len(corrupted_images)}")
else:
    print("✓ All images were verified successfully.")

print("\nDataset verification completed successfully.")