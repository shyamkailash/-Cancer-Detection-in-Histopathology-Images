#!/usr/bin/env python
"""
CLI tool for Histopathology Cancer Detection Prediction & Grad-CAM Visual Explainability.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.inference.pipeline import get_inference_pipeline
from ml.inference.gradcam import base64_to_pil


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict cancer status on a histopathology patch and generate Grad-CAM explainability overlay."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to histopathology image file (PNG/JPG/TIFF).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="centralized",
        choices=["centralized", "fedavg", "dp_fedavg"],
        help="Model architecture: 'centralized', 'fedavg', or 'dp_fedavg' (default: centralized).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/predictions",
        help="Directory to save original, heatmap, and overlay images (default: artifacts/predictions).",
    )
    parser.add_argument(
        "--no-gradcam",
        action="store_true",
        help="Disable Grad-CAM heatmap generation.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        choices=[0, 1],
        help="Specific class to explain (0=normal, 1=metastasis). Default: predicted class.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image file not found: {image_path.resolve()}", file=sys.stderr)
        return 1

    pipeline = get_inference_pipeline()

    print("=" * 65)
    print("Histopathology Cancer Detection - Inference & Explainability")
    print("=" * 65)
    print(f"Input Image:   {image_path.resolve()}")
    print(f"Model:         {args.model}")
    print(f"Grad-CAM:      {'Disabled' if args.no_gradcam else 'Enabled'}")

    result = pipeline.predict(
        image_input=image_path,
        model_name=args.model,
        include_gradcam=not args.no_gradcam,
        target_class=args.target_class,
    )

    pred = result["prediction"]
    probs = pred["probabilities"]

    print("\n" + "-" * 40)
    print("Prediction Results:")
    print("-" * 40)
    print(f"  Predicted Class:   {pred['class_name'].upper()} (ID: {pred['class_id']})")
    print(f"  Confidence:        {pred['confidence'] * 100:.2f}%")
    print(f"  Normal Prob:       {probs['normal'] * 100:.2f}%")
    print(f"  Metastasis Prob:   {probs['metastasis'] * 100:.2f}%")
    print(f"  Inference Time:    {result['inference_time_ms']} ms")
    print(f"  Device:            {result['model']['device']}")

    # Save output visuals if Grad-CAM was generated
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.get("explainability") and "heatmap_base64" in result["explainability"]:
        expl = result["explainability"]
        orig_pil = base64_to_pil(expl["original_base64"])
        heat_pil = base64_to_pil(expl["heatmap_base64"])
        over_pil = base64_to_pil(expl["overlay_base64"])

        orig_pil.save(out_dir / "input_patch.png")
        heat_pil.save(out_dir / "gradcam_heatmap.png")
        over_pil.save(out_dir / "gradcam_overlay.png")

        print(f"\n[SAVED] Visuals saved to: {out_dir.resolve()}/")
        print(f"  - input_patch.png")
        print(f"  - gradcam_heatmap.png")
        print(f"  - gradcam_overlay.png")

    # Save JSON report (without large base64 strings for compact size)
    json_result = dict(result)
    if "explainability" in json_result and json_result["explainability"]:
        json_result["explainability"] = {
            k: v for k, v in json_result["explainability"].items()
            if not k.endswith("_base64")
        }
        json_result["explainability"]["saved_visuals_dir"] = str(out_dir.resolve())

    with open(out_dir / "prediction_result.json", "w", encoding="utf-8") as f:
        json.dump(json_result, f, indent=2)

    print(f"[SAVED] Result JSON: {out_dir.resolve()}/prediction_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
