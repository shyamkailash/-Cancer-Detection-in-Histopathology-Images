"""
Regression tests for class-index mapping, preprocessing consistency,
checkpoint compatibility, and inference prediction output.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from ml.inference.pipeline import InferencePipeline, CLASS_NAMES, CLASS_DESCRIPTIONS
from ml.inference.model_manager import get_model_manager, MODEL_REGISTRY_CONFIG
from ml.preprocessing.transforms import get_eval_transforms, get_train_transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_class_index_mapping_contract():
    """Verify class-index mapping is consistent between registry, schema, and inference pipeline."""
    # 1. Pipeline class names contract
    assert CLASS_NAMES[0] == "normal"
    assert CLASS_NAMES[1] == "metastasis"
    assert "Normal" in CLASS_DESCRIPTIONS[0]
    assert "Metastatic" in CLASS_DESCRIPTIONS[1] or "Tumor" in CLASS_DESCRIPTIONS[1]

    # 2. Check metadata class_mapping.json
    mapping_file = PROJECT_ROOT / "data" / "metadata" / "class_mapping.json"
    if mapping_file.exists():
        with open(mapping_file) as f:
            data = json.load(f)
        pcam_mapping = data["mappings"]["pcam"]
        assert pcam_mapping["to_standardized_id"]["normal"] == 0
        assert pcam_mapping["to_standardized_id"]["0"] == 0
        assert pcam_mapping["to_standardized_id"]["tumor"] == 1
        assert pcam_mapping["to_standardized_id"]["metastasis"] == 1
        assert pcam_mapping["to_standardized_id"]["1"] == 1
        assert pcam_mapping["to_label_name"]["0"] == "normal"
        assert pcam_mapping["to_label_name"]["1"] == "metastasis"


def test_preprocessing_consistency_pipeline_vs_transforms():
    """Verify inference pipeline preprocessing exactly matches get_eval_transforms()."""
    pipeline = InferencePipeline()
    eval_transform = get_eval_transforms(target_size=(96, 96))

    # Generate synthetic RGB test patch
    raw_array = np.random.randint(50, 200, (96, 96, 3), dtype=np.uint8)
    pil_img = Image.fromarray(raw_array, mode="RGB")

    # Pipeline preprocessing
    loaded_img = pipeline._load_image(pil_img)
    pipeline_transformed = pipeline.transform(loaded_img)
    if isinstance(pipeline_transformed, np.ndarray):
        pipeline_tensor = torch.from_numpy(pipeline_transformed).float()
    else:
        pipeline_tensor = pipeline_transformed.float()

    # Direct eval transform
    direct_transformed = eval_transform(pil_img)
    if isinstance(direct_transformed, np.ndarray):
        direct_tensor = torch.from_numpy(direct_transformed).float()
    else:
        direct_tensor = direct_transformed.float()

    # Verify identical dimensions and values
    assert pipeline_tensor.shape == (3, 96, 96)
    assert direct_tensor.shape == (3, 96, 96)
    assert torch.allclose(pipeline_tensor, direct_tensor, atol=1e-5)


def test_checkpoint_compatibility_all_models():
    """Verify all registered model checkpoints resolve, load cleanly, and perform forward inference."""
    mm = get_model_manager()
    dummy_input = torch.randn(1, 3, 96, 96).to(mm.device)

    for model_id in MODEL_REGISTRY_CONFIG:
        model = mm.get_model(model_id)
        assert model is not None
        assert model.training is False

        # Verify output dimension is 2 classes
        with torch.no_grad():
            output = model(dummy_input)
            assert output.shape == (1, 2)
            assert not torch.isnan(output).any()


def test_federated_checkpoints_best_vs_global_identity():
    """Verify best_model.pt and best_global_model.pt parameter weights are identical for federated models."""
    for alg in ["fedavg", "fedprox", "fedbn", "dp_fedavg"]:
        best_path = PROJECT_ROOT / f"artifacts/federated/{alg}/best_model.pt"
        global_path = PROJECT_ROOT / f"artifacts/federated/{alg}/best_global_model.pt"

        if best_path.exists() and global_path.exists():
            ckpt_b = torch.load(str(best_path), map_location="cpu", weights_only=False)
            ckpt_g = torch.load(str(global_path), map_location="cpu", weights_only=False)

            sd_b = ckpt_b.get("model_state_dict", ckpt_b)
            sd_g = ckpt_g.get("model_state_dict", ckpt_g)

            assert set(sd_b.keys()) == set(sd_g.keys())
            for k in sd_b:
                assert torch.equal(sd_b[k], sd_g[k]), f"Weight mismatch for {alg} on parameter '{k}'"


def test_real_sample_inference_predictions_and_probability_constraints():
    """Verify real inference output constraints, probability sums, and unambiguous classifications."""
    pipeline = InferencePipeline()
    sample_dir = PROJECT_ROOT / "data" / "test_samples"

    tumor_sample = sample_dir / "tumor" / "tumor_02.tif"
    normal_sample = sample_dir / "normal" / "normal_02.tif"

    if not tumor_sample.exists() or not normal_sample.exists():
        pytest.skip("Test sample images not available")

    for model_id in ["centralized", "centralized_finetuned", "fedavg"]:
        # Test tumor patch
        res_tum = pipeline.predict(tumor_sample, model_name=model_id, include_gradcam=False)
        pred_tum = res_tum["prediction"]
        probs_tum = pred_tum["probabilities"]

        assert pred_tum["class_id"] == 1
        assert pred_tum["class_name"] == "metastasis"
        assert probs_tum["metastasis"] > 0.90
        assert abs((probs_tum["normal"] + probs_tum["metastasis"]) - 1.0) < 1e-3
        assert pred_tum["confidence"] == max(probs_tum["normal"], probs_tum["metastasis"])

        # Test normal patch
        res_norm = pipeline.predict(normal_sample, model_name=model_id, include_gradcam=False)
        pred_norm = res_norm["prediction"]
        probs_norm = pred_norm["probabilities"]

        assert pred_norm["class_id"] == 0
        assert pred_norm["class_name"] == "normal"
        assert probs_norm["normal"] > 0.90
        assert abs((probs_norm["normal"] + probs_norm["metastasis"]) - 1.0) < 1e-3
        assert pred_norm["confidence"] == max(probs_norm["normal"], probs_norm["metastasis"])


def test_eval_transforms_are_deterministic():
    """Verify get_eval_transforms does not apply stochastic rotations or color jitter."""
    eval_transform = get_eval_transforms(target_size=(96, 96))
    raw_array = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    pil_img = Image.fromarray(raw_array, mode="RGB")

    t1 = eval_transform(pil_img)
    t2 = eval_transform(pil_img)

    assert np.array_equal(t1, t2)
