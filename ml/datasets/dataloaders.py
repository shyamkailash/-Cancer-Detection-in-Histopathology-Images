"""
PyTorch Dataset and DataLoader utilities for the PCam dataset.
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from ml.data.manifest import DatasetManifest
from ml.preprocessing.transforms import (
    get_train_transforms,
    get_eval_transforms,
)


class PCamDataset(Dataset):
    """PyTorch Dataset backed by the generated PCam manifest."""

    def __init__(
        self,
        manifest: DatasetManifest,
        split: str,
        transform=None,
    ):
        self.samples = [
            sample
            for sample in manifest.samples
            if sample.split == split and sample.is_valid
        ]

        self.transform = transform

        if not self.samples:
            raise ValueError(
                f"No valid samples found for split='{split}'."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image_path = Path(sample.image_path)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image: {image_path}"
            ) from exc

        if self.transform is not None:
            image = self.transform(image)

        # Custom transform returns NumPy array.
        # Convert it to a PyTorch tensor.
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.copy()).float()

        label = torch.tensor(
            sample.standardized_label,
            dtype=torch.long,
        )

        return image, label


def create_pcam_dataloaders(
    manifest_path: str = (
        "data/metadata/manifests/pcam_manifest.jsonl"
    ),
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation and test DataLoaders from the PCam manifest.
    """

    manifest = DatasetManifest.load_jsonl(manifest_path)

    train_dataset = PCamDataset(
        manifest=manifest,
        split="train",
        transform=get_train_transforms(),
    )

    val_dataset = PCamDataset(
        manifest=manifest,
        split="val",
        transform=get_eval_transforms(),
    )

    test_dataset = PCamDataset(
        manifest=manifest,
        split="test",
        transform=get_eval_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader