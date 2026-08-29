"""
Train ResNet-18 on the PCam dataset.
"""

import argparse

import torch

from ml.datasets.dataloaders import create_pcam_dataloaders
from ml.models.resnet import create_resnet18
from ml.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet-18 for PCam cancer detection."
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/checkpoints/pcam_resnet18_best.pt",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("PCam ResNet-18 Training")
    print("=" * 60)
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading PCam DataLoaders...")

    train_loader, val_loader, test_loader = create_pcam_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("Train samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    print("\nCreating ResNet-18...")

    model = create_resnet18(
        num_classes=2,
        pretrained=True,
    )

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = trainer.train_one_epoch(train_loader)

        val_metrics = trainer.evaluate(val_loader)

        print(
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Accuracy: {train_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Loss:   {val_metrics['loss']:.4f} | "
            f"Val Accuracy: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]

            trainer.save_checkpoint(
                args.checkpoint,
                epoch=epoch,
                metrics={
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                },
            )

            print("[SAVED] Best checkpoint:", args.checkpoint)

    print("\nEvaluating best/current model on test set...")

    test_metrics = trainer.evaluate(test_loader)

    print(
        f"Test Loss:     {test_metrics['loss']:.4f}"
    )
    print(
        f"Test Accuracy: {test_metrics['accuracy']:.4f}"
    )

    print("\nTraining completed.")


if __name__ == "__main__":
    main()