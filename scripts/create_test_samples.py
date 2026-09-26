from pathlib import Path
import pandas as pd
import shutil
# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "data/raw/pcam/train"
LABEL_FILE = PROJECT_ROOT / "data/raw/pcam/train_labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/test_samples"
# Number of images from each class
NUM_SAMPLES = 5
# Create output folders
normal_dir = OUTPUT_DIR / "normal"
tumor_dir = OUTPUT_DIR / "tumor"
normal_dir.mkdir(parents=True, exist_ok=True)
tumor_dir.mkdir(parents=True, exist_ok=True)
# Read labels
df = pd.read_csv(LABEL_FILE)
# Select available images
normal_count = 0
tumor_count = 0
for _, row in df.iterrows():
    image_id = str(row["id"])
    label = int(row["label"])
    image_path = TRAIN_DIR / f"{image_id}.tif"
    # Skip if image does not exist
    if not image_path.exists():
        continue
    if label == 0 and normal_count < NUM_SAMPLES:
        destination = normal_dir / f"normal_{normal_count + 1:02d}.tif"
        shutil.copy2(image_path, destination)
        normal_count += 1
    elif label == 1 and tumor_count < NUM_SAMPLES:
        destination = tumor_dir / f"tumor_{tumor_count + 1:02d}.tif"
        shutil.copy2(image_path, destination)
        tumor_count += 1
    if normal_count == NUM_SAMPLES and tumor_count == NUM_SAMPLES:
        break
print("\nTest samples created successfully!")
print(f"Normal images : {normal_count}")
print(f"Tumor images  : {tumor_count}")
print(f"\nLocation:")
print(OUTPUT_DIR)
