"""
Experiment Visualization and Plot Generation for Federated Histopathology Cancer Detection.
Generates:
1. Validation Accuracy Curves across Federated Rounds (FedAvg, FedProx, FedBN, DP-FedAvg)
2. Validation Loss Curves across Federated Rounds
3. Client Data Distribution (Non-IID label skew across simulated sites)
4. Comprehensive Metric Comparison (Centralized vs FedAvg vs FedProx vs FedBN vs DP-FedAvg)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_json(filepath: Path) -> Optional[Dict[str, Any]]:
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def draw_bar_chart(
    data: Dict[str, Dict[str, float]],
    title: str,
    output_path: Path,
    width: int = 900,
    height: int = 500,
):
    """Draw a clean multi-model comparison bar chart using PIL."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    margin_top, margin_bottom, margin_left, margin_right = 70, 70, 80, 50
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    draw.text((width // 2 - 200, 25), title, fill=(30, 30, 30))

    for y_val in [0.0, 0.25, 0.50, 0.75, 1.0]:
        y_pos = margin_top + chart_h - int(y_val * chart_h)
        draw.line([(margin_left, y_pos), (margin_left + chart_w, y_pos)], fill=(220, 220, 220), width=1)
        draw.text((margin_left - 45, y_pos - 7), f"{int(y_val * 100)}%", fill=(100, 100, 100))

    metrics = ["accuracy", "sensitivity", "specificity", "precision", "f1_score"]
    models = list(data.keys())
    colors = [
        (52, 152, 219),   # Centralized - Blue
        (46, 204, 113),   # FedAvg - Green
        (155, 89, 182),   # FedProx - Purple
        (241, 196, 15),   # FedBN - Yellow
        (231, 76, 60),    # DP-FedAvg - Red
    ]

    num_metrics = len(metrics)
    num_models = len(models)
    group_width = chart_w / num_metrics
    bar_width = min(22, (group_width * 0.85) / max(1, num_models))

    for m_idx, metric in enumerate(metrics):
        group_center = margin_left + m_idx * group_width + group_width / 2
        start_x = group_center - (num_models * bar_width) / 2

        draw.text((group_center - len(metric) * 3, margin_top + chart_h + 10), metric.replace("_", " ").title(), fill=(50, 50, 50))

        for mod_idx, model_name in enumerate(models):
            val = data[model_name].get(metric, 0.0)
            if val is None:
                val = 0.0
            val_clamped = max(0.0, min(1.0, float(val)))
            bar_h = int(val_clamped * chart_h)
            x0 = start_x + mod_idx * bar_width
            y0 = margin_top + chart_h - bar_h
            x1 = x0 + bar_width - 2
            y1 = margin_top + chart_h

            color = colors[mod_idx % len(colors)]
            draw.rectangle([(x0, y0), (x1, y1)], fill=color)

    # Legend
    legend_x = margin_left
    legend_y = height - 30
    for mod_idx, model_name in enumerate(models):
        color = colors[mod_idx % len(colors)]
        draw.rectangle([(legend_x, legend_y), (legend_x + 12, legend_y + 12)], fill=color)
        draw.text((legend_x + 16, legend_y), model_name, fill=(50, 50, 50))
        legend_x += len(model_name) * 7 + 30

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def draw_line_curves(
    curves: Dict[str, List[float]],
    title: str,
    y_label: str,
    output_path: Path,
    width: int = 800,
    height: int = 450,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
):
    """Draw training / validation metric curves across rounds."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    margin_top, margin_bottom, margin_left, margin_right = 60, 60, 80, 50
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    draw.text((width // 2 - 160, 20), title, fill=(30, 30, 30))

    all_vals = [v for vals in curves.values() for v in vals if v is not None]
    if not all_vals:
        return

    actual_min = min_val if min_val is not None else min(all_vals) * 0.95
    actual_max = max_val if max_val is not None else max(all_vals) * 1.05
    if actual_max <= actual_min:
        actual_max = actual_min + 1.0

    for step in np.linspace(actual_min, actual_max, 5):
        y_pos = margin_top + chart_h - int(((step - actual_min) / (actual_max - actual_min)) * chart_h)
        draw.line([(margin_left, y_pos), (margin_left + chart_w, y_pos)], fill=(230, 230, 230), width=1)
        draw.text((margin_left - 55, y_pos - 7), f"{step:.3f}", fill=(100, 100, 100))

    colors = [
        (41, 128, 185),   # Blue
        (39, 174, 96),    # Green
        (142, 68, 173),   # Purple
        (243, 156, 18),   # Orange
        (231, 76, 60),    # Red
    ]
    legend_x = margin_left
    legend_y = height - 25

    for c_idx, (name, points) in enumerate(curves.items()):
        color = colors[c_idx % len(colors)]
        num_points = len(points)
        if num_points < 2:
            continue

        coords = []
        for i, val in enumerate(points):
            x = margin_left + int((i / (num_points - 1)) * chart_w)
            y = margin_top + chart_h - int(((val - actual_min) / (actual_max - actual_min)) * chart_h)
            coords.append((x, y))

        for i in range(len(coords) - 1):
            draw.line([coords[i], coords[i + 1]], fill=color, width=3)

        for x, y in coords:
            draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], fill=color)

        draw.rectangle([(legend_x, legend_y), (legend_x + 12, legend_y + 12)], fill=color)
        draw.text((legend_x + 16, legend_y), name, fill=(50, 50, 50))
        legend_x += len(name) * 7 + 30

    max_rounds = max(len(p) for p in curves.values())
    for i in range(max_rounds):
        x = margin_left + int((i / max(1, max_rounds - 1)) * chart_w)
        draw.text((x - 12, margin_top + chart_h + 8), f"R{i+1}", fill=(80, 80, 80))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def draw_client_distribution(partitions_path: Path, output_path: Path):
    """Draw stacked horizontal bar chart showing class distribution per simulated site."""
    if not partitions_path.exists():
        return

    with open(partitions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sites = data.get("sites", {})
    if not sites:
        return

    width, height = 800, 450
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((width // 2 - 200, 20), "Simulated Healthcare Sites - Non-IID Class Skew", fill=(30, 30, 30))

    margin_top, margin_left, bar_h, spacing = 70, 180, 40, 20
    max_w = 540

    c_normal = (52, 152, 219)       # Blue for normal
    c_metastasis = (231, 76, 60)    # Red for metastasis

    for idx, (cid, p) in enumerate(sites.items()):
        y = margin_top + idx * (bar_h + spacing)
        site_label = f"{p['site_name']} ({p['sample_count']} samples)"
        draw.text((15, y + 10), site_label, fill=(40, 40, 40))

        norm_pct = p["normal_percentage"] / 100.0
        w_norm = int(norm_pct * max_w)
        w_meta = max_w - w_norm

        draw.rectangle([(margin_left, y), (margin_left + w_norm, y + bar_h)], fill=c_normal)
        draw.rectangle([(margin_left + w_norm, y), (margin_left + max_w, y + bar_h)], fill=c_metastasis)

        if w_norm > 40:
            draw.text((margin_left + w_norm // 2 - 15, y + 12), f"{p['normal_percentage']:.0f}%", fill=(255, 255, 255))
        if w_meta > 40:
            draw.text((margin_left + w_norm + w_meta // 2 - 15, y + 12), f"{p['metastasis_percentage']:.0f}%", fill=(255, 255, 255))

    # Legend
    draw.rectangle([(margin_left, height - 30), (margin_left + 15, height - 15)], fill=c_normal)
    draw.text((margin_left + 22, height - 30), "Normal Tissue (Class 0)", fill=(40, 40, 40))

    draw.rectangle([(margin_left + 200, height - 30), (margin_left + 215, height - 15)], fill=c_metastasis)
    draw.text((margin_left + 222, height - 30), "Metastasis Tissue (Class 1)", fill=(40, 40, 40))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def generate_all_plots(
    reports_dir: Path = Path("artifacts/reports"),
    figures_dir: Path = Path("artifacts/reports/figures"),
):
    """Generate all figures from experiment JSON reports."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    cent_data = load_json(reports_dir / "centralized_metrics.json")
    fed_data = load_json(reports_dir / "federated_metrics.json") or load_json(reports_dir / "fedavg_metrics.json")
    prox_data = load_json(reports_dir / "fedprox_metrics.json")
    bn_data = load_json(reports_dir / "fedbn_metrics.json")
    dp_data = load_json(reports_dir / "privacy_federated_metrics.json") or load_json(reports_dir / "dp_fedavg_metrics.json")

    # 1. Validation Curves across Rounds
    val_acc_curves = {}
    val_loss_curves = {}

    if fed_data and "history" in fed_data:
        val_acc_curves["FedAvg"] = [h["global_val_metrics"]["accuracy"] * 100 for h in fed_data["history"]]
        val_loss_curves["FedAvg"] = [h["global_val_metrics"]["loss"] for h in fed_data["history"]]

    if prox_data and "history" in prox_data:
        val_acc_curves["FedProx"] = [h["global_val_metrics"]["accuracy"] * 100 for h in prox_data["history"]]
        val_loss_curves["FedProx"] = [h["global_val_metrics"]["loss"] for h in prox_data["history"]]

    if bn_data and "history" in bn_data:
        val_acc_curves["FedBN"] = [h["global_val_metrics"]["accuracy"] * 100 for h in bn_data["history"]]
        val_loss_curves["FedBN"] = [h["global_val_metrics"]["loss"] for h in bn_data["history"]]

    if dp_data and "history" in dp_data:
        val_acc_curves["DP-FedAvg"] = [h["global_val_metrics"]["accuracy"] * 100 for h in dp_data["history"]]
        val_loss_curves["DP-FedAvg"] = [h["global_val_metrics"]["loss"] for h in dp_data["history"]]

    if val_acc_curves:
        draw_line_curves(
            val_acc_curves,
            title="Global Validation Accuracy vs Federated Rounds",
            y_label="Accuracy (%)",
            output_path=figures_dir / "val_accuracy_curves.png",
        )
        print(f"[PLOT] Generated: {figures_dir / 'val_accuracy_curves.png'}")

    if val_loss_curves:
        draw_line_curves(
            val_loss_curves,
            title="Global Validation Loss vs Federated Rounds",
            y_label="Cross-Entropy Loss",
            output_path=figures_dir / "val_loss_curves.png",
        )
        print(f"[PLOT] Generated: {figures_dir / 'val_loss_curves.png'}")

    # 2. Client Data Distribution
    draw_client_distribution(
        Path("data/metadata/federated/site_partitions.json"),
        figures_dir / "client_data_distribution.png",
    )
    print(f"[PLOT] Generated: {figures_dir / 'client_data_distribution.png'}")

    # 3. Model Comparison Bar Chart
    comparison_data = {}
    if cent_data and "test_metrics" in cent_data:
        t = cent_data["test_metrics"]
        comparison_data["Centralized"] = {
            "accuracy": t.get("accuracy", 0.0),
            "sensitivity": t.get("sensitivity", 0.0),
            "specificity": t.get("specificity", 0.0),
            "precision": t.get("precision", 0.0),
            "f1_score": t.get("f1_score", 0.0),
        }

    if fed_data and "test_metrics" in fed_data:
        t = fed_data["test_metrics"]
        comparison_data["FedAvg"] = {
            "accuracy": t.get("accuracy", 0.0),
            "sensitivity": t.get("sensitivity", 0.0),
            "specificity": t.get("specificity", 0.0),
            "precision": t.get("precision", 0.0),
            "f1_score": t.get("f1_score", 0.0),
        }

    if prox_data and "test_metrics" in prox_data:
        t = prox_data["test_metrics"]
        comparison_data["FedProx"] = {
            "accuracy": t.get("accuracy", 0.0),
            "sensitivity": t.get("sensitivity", 0.0),
            "specificity": t.get("specificity", 0.0),
            "precision": t.get("precision", 0.0),
            "f1_score": t.get("f1_score", 0.0),
        }

    if bn_data and "test_metrics" in bn_data:
        t = bn_data["test_metrics"]
        comparison_data["FedBN"] = {
            "accuracy": t.get("accuracy", 0.0),
            "sensitivity": t.get("sensitivity", 0.0),
            "specificity": t.get("specificity", 0.0),
            "precision": t.get("precision", 0.0),
            "f1_score": t.get("f1_score", 0.0),
        }

    if dp_data and "test_metrics" in dp_data:
        t = dp_data["test_metrics"]
        comparison_data["DP-FedAvg"] = {
            "accuracy": t.get("accuracy", 0.0),
            "sensitivity": t.get("sensitivity", 0.0),
            "specificity": t.get("specificity", 0.0),
            "precision": t.get("precision", 0.0),
            "f1_score": t.get("f1_score", 0.0),
        }

    if comparison_data:
        draw_bar_chart(
            comparison_data,
            title="Algorithm Performance Comparison on Global Test Set",
            output_path=figures_dir / "model_comparison_metrics.png",
        )
        print(f"[PLOT] Generated: {figures_dir / 'model_comparison_metrics.png'}")


if __name__ == "__main__":
    generate_all_plots()
