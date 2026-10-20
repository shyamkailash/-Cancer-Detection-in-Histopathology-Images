from pathlib import Path
import shutil
import sys
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "data/raw/pcam/train"
LABEL_FILE = PROJECT_ROOT / "data/raw/pcam/train_labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/test_samples"
NUM_SAMPLES = 5


def create_test_samples(
    train_dir: Path = TRAIN_DIR,
    label_file: Path = LABEL_FILE,
    output_dir: Path = OUTPUT_DIR,
    num_samples: int = NUM_SAMPLES,
) -> int:
    """Create sample normal and tumor histopathology images from raw PCam dataset."""
    if not label_file.exists():
        print(f"Error: Label file not found: {label_file}")
        return 1

    if not train_dir.exists():
        print(f"Error: Train directory not found: {train_dir}")
        return 1

    normal_dir = output_dir / "normal"
    tumor_dir = output_dir / "tumor"
    normal_dir.mkdir(parents=True, exist_ok=True)
    tumor_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(label_file)
    normal_count = 0
    tumor_count = 0

    for _, row in df.iterrows():
        image_id = str(row["id"])
        label = int(row["label"])
        image_path = train_dir / f"{image_id}.tif"

        if not image_path.exists():
            continue

        if label == 0 and normal_count < num_samples:
            destination = normal_dir / f"normal_{normal_count + 1:02d}.tif"
            shutil.copy2(image_path, destination)
            normal_count += 1
        elif label == 1 and tumor_count < num_samples:
            destination = tumor_dir / f"tumor_{tumor_count + 1:02d}.tif"
            shutil.copy2(image_path, destination)
            tumor_count += 1

        if normal_count == num_samples and tumor_count == num_samples:
            break

    print("\nTest samples created successfully!")
    print(f"Normal images : {normal_count}")
    print(f"Tumor images  : {tumor_count}")
    print(f"\nLocation:\n{output_dir}")
    return 0


def main():
    sys.exit(create_test_samples())


if __name__ == "__main__":
    main()
