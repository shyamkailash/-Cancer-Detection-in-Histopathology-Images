from pathlib import Path
import csv
import sys
import json

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.pipeline import InferencePipeline


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SAMPLE_DIR = PROJECT_ROOT / "data" / "test_samples"
RESULT_DIR = PROJECT_ROOT / "results" / "labeled_test"

MODELS = {
    "centralized": "Centralized ResNet-18",
    "centralized_finetuned": "Centralized Fine-Tuned ResNet-18",
    "fedavg": "Federated ResNet-18 (FedAvg)",
    "fedprox": "Federated ResNet-18 (FedProx)",
    "fedbn": "Federated ResNet-18 (FedBN)",
    "dp_fedavg": "Privacy-Preserving Federated ResNet-18 (DP-FedAvg)",
}

MODEL_NAME = "centralized"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def actual_label_from_path(path: Path):
    """
    Determine ground-truth label from the directory name.
    """
    folder = path.parent.name.lower()

    if folder == "normal":
        return 0, "NORMAL"

    if folder == "tumor":
        return 1, "TUMOR"

    raise ValueError(
        f"Unknown label folder '{folder}' for image: {path}"
    )


def find_images():
    """
    Automatically discover all labeled images.
    """
    images = []

    for folder in ["normal", "tumor"]:
        directory = SAMPLE_DIR / folder

        if not directory.exists():
            continue

        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
                images.append(path)

    return images


def create_thumbnail(image_path, prediction, actual, confidence, output):
    """
    Save an annotated copy of the image.
    """
    image = Image.open(image_path).convert("RGB")

    # Enlarge for easier viewing.
    image = image.resize((384, 384))

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            18
        )
    except Exception:
        font = ImageFont.load_default()

    correct = actual == prediction

    text = (
        f"Actual: {actual}\n"
        f"Prediction: {prediction}\n"
        f"Confidence: {confidence * 100:.2f}%\n"
        f"Result: {'CORRECT' if correct else 'WRONG'}"
    )

    # Background rectangle.
    bbox = draw.multiline_textbbox((10, 10), text, font=font)
    draw.rectangle(
        (
            bbox[0] - 6,
            bbox[1] - 6,
            bbox[2] + 6,
            bbox[3] + 6,
        ),
        fill="white",
    )

    draw.multiline_text(
        (10, 10),
        text,
        fill="black",
        font=font,
        spacing=5,
    )

    image.save(output)


def create_contact_sheet(records, output):
    """
    Create one image containing all tested samples.
    """
    if not records:
        return

    thumb_w = 320
    thumb_h = 360

    columns = 2
    rows = (len(records) + columns - 1) // columns

    sheet = Image.new(
        "RGB",
        (columns * thumb_w, rows * thumb_h),
        "white",
    )

    for index, record in enumerate(records):

        image = Image.open(record["image"]).convert("RGB")
        image.thumbnail((300, 300))

        x = (index % columns) * thumb_w + 10
        y = (index // columns) * thumb_h + 10

        sheet.paste(image, (x, y))

        draw = ImageDraw.Draw(sheet)

        text = (
            f'{record["name"]}\n'
            f'Actual: {record["actual"]}\n'
            f'Prediction: {record["prediction"]}\n'
            f'Confidence: {record["confidence"] * 100:.2f}%\n'
            f'{"CORRECT" if record["correct"] else "WRONG"}'
        )

        draw.multiline_text(
            (x, y + 305),
            text,
            fill="black",
            spacing=3,
        )

    sheet.save(output)


def create_confusion_matrix(records, output):
    """
    Generate a simple confusion matrix without external plotting
    dependencies.
    """
    matrix = np.zeros((2, 2), dtype=int)

    for record in records:
        actual = record["actual_id"]
        predicted = record["prediction_id"]
        matrix[actual][predicted] += 1

    image = Image.new("RGB", (600, 500), "white")
    draw = ImageDraw.Draw(image)

    try:
        font_big = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            26,
        )
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            20,
        )
    except Exception:
        font_big = ImageFont.load_default()
        font = ImageFont.load_default()

    draw.text(
        (180, 20),
        "Confusion Matrix",
        fill="black",
        font=font_big,
    )

    # Labels
    draw.text((260, 80), "Predicted", fill="black", font=font)
    draw.text((100, 125), "NORMAL", fill="black", font=font)
    draw.text((370, 125), "TUMOR", fill="black", font=font)

    draw.text(
        (20, 260),
        "Actual",
        fill="black",
        font=font,
    )

    draw.text(
        (70, 210),
        "NORMAL",
        fill="black",
        font=font,
    )

    draw.text(
        (70, 350),
        "TUMOR",
        fill="black",
        font=font,
    )

    # Matrix cells
    positions = [
        (130, 170, 300, 300),
        (300, 170, 470, 300),
        (130, 300, 300, 430),
        (300, 300, 470, 430),
    ]

    values = [
        matrix[0][0],
        matrix[0][1],
        matrix[1][0],
        matrix[1][1],
    ]

    for pos, value in zip(positions, values):
        draw.rectangle(pos, outline="black", width=3)

        x1, y1, x2, y2 = pos

        text = str(value)

        bbox = draw.textbbox((0, 0), text, font=font_big)

        tx = (x1 + x2 - (bbox[2] - bbox[0])) / 2
        ty = (y1 + y2 - (bbox[3] - bbox[1])) / 2

        draw.text(
            (tx, ty),
            text,
            fill="black",
            font=font_big,
        )

    image.save(output)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print(" AUTOMATED PCAM LABELED SAMPLE TEST")
    print("=" * 70)

    print(f"Project     : {PROJECT_ROOT}")
    print(f"Samples     : {SAMPLE_DIR}")
    print(f"Model       : {MODEL_NAME}")
    print(f"Device      : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()

    images = find_images()

    if not images:
        print("ERROR: No labeled images found.")
        print()
        print("Expected:")
        print("data/test_samples/normal/*.tif")
        print("data/test_samples/tumor/*.tif")
        return 1

    print(f"Found {len(images)} labeled images.")
    print()

    # --------------------------------------------------------------
    # Load your existing inference pipeline
    # --------------------------------------------------------------
    print("Loading project inference pipeline...")

    pipeline = InferencePipeline()

    print("Model loaded successfully.")
    print()

    records = []

    # --------------------------------------------------------------
    # Run inference
    # --------------------------------------------------------------
    for index, image_path in enumerate(images, start=1):

        actual_id, actual_name = actual_label_from_path(image_path)

        print(
            f"[{index:02d}/{len(images):02d}] "
            f"{image_path.name:<55}",
            end="",
            flush=True,
        )

        try:
            result = pipeline.predict(
                image_input=image_path,
                model_name=MODEL_NAME,
                include_gradcam=True,
            )

            prediction = result["prediction"]

            prediction_id = prediction["class_id"]

            # Your project calls class 1 "metastasis".
            # For this labeled sample tester, display it as TUMOR.
            prediction_name = (
                "NORMAL"
                if prediction_id == 0
                else "TUMOR"
            )

            confidence = float(prediction["confidence"])

            probabilities = prediction["probabilities"]

            correct = actual_id == prediction_id

            record = {
                "image": str(image_path),
                "name": image_path.name,
                "actual_id": actual_id,
                "actual": actual_name,
                "prediction_id": prediction_id,
                "prediction": prediction_name,
                "confidence": confidence,
                "normal_probability": probabilities["normal"],
                "tumor_probability": probabilities["metastasis"],
                "correct": correct,
                "inference_time_ms": result["inference_time_ms"],
            }

            records.append(record)

            # ------------------------------------------------------
            # Save original + annotated image
            # ------------------------------------------------------
            category = "correct" if correct else "wrong"

            annotated_dir = (
                RESULT_DIR /
                "annotated" /
                category
            )

            annotated_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            create_thumbnail(
                image_path,
                prediction_name,
                actual_name,
                confidence,
                annotated_dir / image_path.name.replace(
                    image_path.suffix,
                    ".png",
                ),
            )

            # ------------------------------------------------------
            # Save Grad-CAM overlay
            # ------------------------------------------------------
            explainability = result.get("explainability")

            if (
                explainability
                and explainability.get("overlay_base64")
            ):
                import base64

                gradcam_dir = (
                    RESULT_DIR /
                    "gradcam" /
                    category
                )

                gradcam_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                encoded = explainability["overlay_base64"]

                if "," in encoded:
                    encoded = encoded.split(",", 1)[1]

                output_path = (
                    gradcam_dir /
                    image_path.name.replace(
                        image_path.suffix,
                        "_gradcam.png",
                    )
                )

                output_path.write_bytes(
                    base64.b64decode(encoded)
                )

            print(
                f" Actual={actual_name:<7} "
                f"Predicted={prediction_name:<7} "
                f"{confidence * 100:6.2f}% "
                f"{'✓' if correct else '✗'}"
            )

        except Exception as exc:

            print(" ERROR")

            print(f"      {type(exc).__name__}: {exc}")

            record = {
                "image": str(image_path),
                "name": image_path.name,
                "actual_id": actual_id,
                "actual": actual_name,
                "prediction_id": -1,
                "prediction": "ERROR",
                "confidence": 0.0,
                "normal_probability": 0.0,
                "tumor_probability": 0.0,
                "correct": False,
                "inference_time_ms": 0.0,
                "error": str(exc),
            }

            records.append(record)

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------
    valid = [
        r for r in records
        if r["prediction_id"] in (0, 1)
    ]

    correct_count = sum(
        r["correct"]
        for r in valid
    )

    total = len(valid)

    accuracy = (
        correct_count / total
        if total
        else 0.0
    )

    normal_records = [
        r for r in valid
        if r["actual_id"] == 0
    ]

    tumor_records = [
        r for r in valid
        if r["actual_id"] == 1
    ]

    normal_correct = sum(
        r["correct"]
        for r in normal_records
    )

    tumor_correct = sum(
        r["correct"]
        for r in tumor_records
    )

    normal_recall = (
        normal_correct / len(normal_records)
        if normal_records
        else 0.0
    )

    tumor_recall = (
        tumor_correct / len(tumor_records)
        if tumor_records
        else 0.0
    )

    # --------------------------------------------------------------
    # Console report
    # --------------------------------------------------------------
    print()
    print("=" * 70)
    print(" RESULTS")
    print("=" * 70)

    print()

    print(
        f"{'Image':<45}"
        f"{'Actual':<10}"
        f"{'Prediction':<12}"
        f"{'Confidence':<12}"
        f"Result"
    )

    print("-" * 95)

    for r in records:

        print(
            f"{r['name']:<45}"
            f"{r['actual']:<10}"
            f"{r['prediction']:<12}"
            f"{r['confidence'] * 100:>7.2f}%     "
            f"{'✓ CORRECT' if r['correct'] else '✗ WRONG'}"
        )

    print()
    print("-" * 70)

    print(f"Total tested       : {total}")
    print(f"Correct            : {correct_count}")
    print(f"Wrong              : {total - correct_count}")
    print(f"Accuracy           : {accuracy * 100:.2f}%")
    print(f"Normal recall      : {normal_recall * 100:.2f}%")
    print(f"Tumor recall       : {tumor_recall * 100:.2f}%")

    print("-" * 70)

    # --------------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------------
    csv_path = RESULT_DIR / "predictions.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "image",
            "actual",
            "prediction",
            "confidence",
            "normal_probability",
            "tumor_probability",
            "correct",
            "inference_time_ms",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for r in records:
            writer.writerow({
                "image": r["name"],
                "actual": r["actual"],
                "prediction": r["prediction"],
                "confidence": r["confidence"],
                "normal_probability": r["normal_probability"],
                "tumor_probability": r["tumor_probability"],
                "correct": r["correct"],
                "inference_time_ms": r["inference_time_ms"],
            })

    # --------------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------------
    summary = {
        "model": MODEL_NAME,
        "model_display_name": MODELS[MODEL_NAME],
        "total_images": total,
        "correct": correct_count,
        "wrong": total - correct_count,
        "accuracy": accuracy,
        "normal_recall": normal_recall,
        "tumor_recall": tumor_recall,
        "device": (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
        "note": (
            "These samples are extracted from the PCam training "
            "dataset. Results are for labeled-sample sanity checking "
            "and demonstration, not unbiased test-set evaluation."
        ),
    }

    summary_path = RESULT_DIR / "summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------------
    create_confusion_matrix(
        valid,
        RESULT_DIR / "confusion_matrix.png",
    )

    # --------------------------------------------------------------
    # Contact sheet
    # --------------------------------------------------------------
    create_contact_sheet(
        valid,
        RESULT_DIR / "all_samples.png",
    )

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------
    print()
    print("Generated automatically:")
    print()
    print(f"  {RESULT_DIR}/predictions.csv")
    print(f"  {RESULT_DIR}/summary.json")
    print(f"  {RESULT_DIR}/confusion_matrix.png")
    print(f"  {RESULT_DIR}/all_samples.png")
    print(f"  {RESULT_DIR}/annotated/")
    print(f"  {RESULT_DIR}/gradcam/")
    print()

    print("=" * 70)
    print(" TEST COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
