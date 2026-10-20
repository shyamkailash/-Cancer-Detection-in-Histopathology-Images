"""
PCam PyTorch DataLoader verification script.
"""

import torch

from ml.datasets.dataloaders import create_pcam_dataloaders


def main():
    train_loader, val_loader, test_loader = create_pcam_dataloaders(
        batch_size=32,
        num_workers=0,
    )

    print("Train samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("\nBatch shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Labels:", labels[:10].tolist())
    print("Image dtype:", images.dtype)
    print("Image min:", images.min().item())
    print("Image max:", images.max().item())

    print("\nPyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()