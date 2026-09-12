"""
Federated Server Coordinator for multi-site federated learning.
Manages global rounds, client selection, aggregation (FedAvg/FedBN), global validation, and testing.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.models.resnet import create_resnet18
from ml.evaluation.metrics import evaluate_model
from .client import FederatedClient
from .aggregation import federated_averaging
from .fedbn import aggregate_fedbn
from .utils import estimate_model_size_mb, estimate_communication_volume


class FederatedServer:
    """
    Central coordinator for Federated Learning.
    Dispatches global model parameters to clients, aggregates updates (FedAvg / FedBN),
    and performs centralized validation and testing on untouched global splits.
    """

    def __init__(
        self,
        clients: List[FederatedClient],
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        initial_model: Optional[nn.Module] = None,
        client_fraction: float = 1.0,
        algorithm: str = "fedavg",
        seed: int = 42,
        checkpoint_dir: str = "artifacts/federated/fedavg",
    ):
        self.clients = clients
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.client_fraction = client_fraction
        self.algorithm = algorithm.lower()
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.rng = np.random.default_rng(seed)

        if initial_model is not None:
            self.global_model = initial_model.to(self.device)
        else:
            self.global_model = create_resnet18(num_classes=2, pretrained=True).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

    def select_clients(self, round_num: int) -> List[FederatedClient]:
        """Select a subset of clients for the round based on client_fraction."""
        num_total = len(self.clients)
        if self.client_fraction >= 1.0:
            return self.clients

        num_select = max(1, int(round(num_total * self.client_fraction)))
        round_rng = np.random.default_rng(self.seed + round_num)
        selected_indices = round_rng.choice(num_total, size=num_select, replace=False)
        return [self.clients[i] for i in selected_indices]

    def aggregate(self, client_updates: List[tuple]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates according to selected algorithm."""
        if self.algorithm == "fedbn":
            base_state = {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
            return aggregate_fedbn(client_updates, base_state_dict=base_state)
        else:
            # Standard FedAvg / FedProx / DP-FedAvg parameter weighting
            return federated_averaging(client_updates)

    def fit(self, rounds: int = 5) -> Dict[str, Any]:
        """
        Execute federated training rounds.
        Monitors global validation set after each round.
        Selects best model checkpoint based on validation ROC-AUC / Accuracy.
        Performs single final evaluation on global held-out test set.
        """
        history: List[Dict[str, Any]] = []
        best_val_auc = -1.0
        best_val_acc = 0.0
        best_val_metrics = None
        best_round = 0
        best_state_dict = None
        start_time = time.time()

        model_size_mb = estimate_model_size_mb(self.global_model)

        print("\n" + "=" * 65)
        print(f"Starting Federated Learning Training ({self.algorithm.upper()})")
        print(f"Rounds: {rounds} | Total Sites: {len(self.clients)} | Fraction: {self.client_fraction}")
        print("=" * 65)

        # Evaluate initial baseline model state before training
        print("\n--- Initial Validation Before Federated Training ---")
        init_val = evaluate_model(self.global_model, self.val_loader, criterion=self.criterion, device=self.device)
        init_auc = init_val.get("roc_auc") or 0.0
        print(f"Initial Val Loss: {init_val['loss']:.4f} | Val Acc: {init_val['accuracy'] * 100:.2f}% | Sensitivity: {init_val['sensitivity'] * 100:.2f}% | ROC-AUC: {init_auc:.4f}")

        best_val_auc = init_auc
        best_val_acc = init_val["accuracy"]
        best_val_metrics = init_val
        best_state_dict = {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}

        for r in range(1, rounds + 1):
            round_start = time.time()
            print(f"\n============================================================")
            print(f"Federated Round {r}/{rounds} [{self.algorithm.upper()}]")
            print(f"============================================================")

            selected_clients = self.select_clients(r)
            selected_ids = [c.client_id for c in selected_clients]
            print(f"Participating Sites ({len(selected_clients)}/{len(self.clients)}): {', '.join(selected_ids)}")

            # Copy global model parameters to CPU state dict for dispatch
            global_state = {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}

            client_updates: List[tuple] = []
            round_client_metrics: List[Dict[str, Any]] = []

            for client in selected_clients:
                print(f"\n  Training on {client.client_id} ({client.site_name})...")
                res = client.train(global_state)
                client_updates.append((res["state_dict"], res["num_samples"]))
                round_client_metrics.append({
                    "client_id": res["client_id"],
                    "site_name": res["site_name"],
                    "num_samples": res["num_samples"],
                    "train_loss": res["train_loss"],
                    "train_accuracy": res["train_accuracy"],
                })
                print(f"    Samples: {res['num_samples']} | Loss: {res['train_loss']:.4f} | Acc: {res['train_accuracy'] * 100:.2f}%")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Perform aggregation
            print(f"\nAggregating {len(client_updates)} client updates using {self.algorithm.upper()}...")
            aggregated_state = self.aggregate(client_updates)

            # Update global model
            self.global_model.load_state_dict(aggregated_state)

            # Global validation
            val_metrics = evaluate_model(
                self.global_model,
                self.val_loader,
                criterion=self.criterion,
                device=self.device,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            round_duration = time.time() - round_start
            val_auc = val_metrics.get("roc_auc") or 0.0
            print(f"Global Validation Loss:        {val_metrics['loss']:.4f}")
            print(f"Global Validation Accuracy:    {val_metrics['accuracy'] * 100:.2f}%")
            print(f"Global Validation Sensitivity: {val_metrics['sensitivity'] * 100:.2f}%")
            print(f"Global Validation Specificity: {val_metrics['specificity'] * 100:.2f}%")
            print(f"Global Validation ROC-AUC:     {val_auc:.4f}")
            print(f"Round Duration: {round_duration:.2f}s")

            round_record = {
                "round": r,
                "duration_seconds": round_duration,
                "participating_clients": selected_ids,
                "client_metrics": round_client_metrics,
                "global_val_metrics": val_metrics,
            }
            history.append(round_record)

            # Validation-only model selection rule (primary: ROC-AUC, secondary: accuracy)
            is_better = False
            if val_auc > best_val_auc + 1e-4:
                is_better = True
            elif abs(val_auc - best_val_auc) <= 1e-4 and val_metrics["accuracy"] > best_val_acc:
                is_better = True

            if is_better or r == 1:
                best_val_auc = val_auc
                best_val_acc = val_metrics["accuracy"]
                best_val_metrics = val_metrics
                best_round = r
                best_state_dict = {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
                best_payload = {
                    "round": r,
                    "algorithm": self.algorithm,
                    "model_state_dict": best_state_dict,
                    "val_metrics": val_metrics,
                }
                best_path = self.checkpoint_dir / "best_global_model.pt"
                torch.save(best_payload, best_path)
                torch.save(best_payload, self.checkpoint_dir / "best_model.pt")
                print(f"[*] New best global model at Round {r}: {best_path}")

        total_training_time = time.time() - start_time

        # Save final global model
        final_path = self.checkpoint_dir / "final_global_model.pt"
        torch.save(
            {
                "round": rounds,
                "algorithm": self.algorithm,
                "model_state_dict": self.global_model.state_dict(),
                "history": history,
            },
            final_path,
        )
        print(f"[SAVED] Final global model: {final_path}")

        # Load best selected model for test evaluation
        if best_state_dict is not None:
            self.global_model.load_state_dict(best_state_dict)

        # Final Global Evaluation on untouched test set
        print("\n" + "=" * 65)
        print(f"Evaluating Best {self.algorithm.upper()} Model (Round {best_round}) on Test Set")
        print("=" * 65)

        test_metrics = evaluate_model(
            self.global_model,
            self.test_loader,
            criterion=self.criterion,
            device=self.device,
        )

        print(f"Global Test Loss:        {test_metrics['loss']:.4f}")
        print(f"Global Test Accuracy:    {test_metrics['accuracy'] * 100:.2f}%")
        print(f"Global Test Sensitivity: {test_metrics['sensitivity'] * 100:.2f}% (Metastasis Recall)")
        print(f"Global Test Specificity: {test_metrics['specificity'] * 100:.2f}% (Normal Tissue)")
        print(f"Global Test Precision:   {test_metrics['precision'] * 100:.2f}%")
        print(f"Global Test F1-Score:    {test_metrics['f1_score'] * 100:.2f}%")
        if test_metrics.get("roc_auc") is not None:
            print(f"Global Test ROC-AUC:     {test_metrics['roc_auc']:.4f}")
        print(f"Total Training Time:     {total_training_time:.2f}s")

        avg_clients_per_round = int(round(len(self.clients) * self.client_fraction))
        comm_stats = estimate_communication_volume(
            num_rounds=rounds,
            num_participating_clients_per_round=avg_clients_per_round,
            model_size_mb=model_size_mb,
        )

        # Save test metrics and confusion matrix in algorithm output directory
        with open(self.checkpoint_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

        if "confusion_matrix" in test_metrics:
            with open(self.checkpoint_dir / "confusion_matrix.json", "w", encoding="utf-8") as f:
                json.dump(test_metrics["confusion_matrix"], f, indent=2)

        with open(self.checkpoint_dir / "round_metrics.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return {
            "algorithm": self.algorithm,
            "rounds": rounds,
            "best_round": best_round,
            "best_val_accuracy": best_val_acc,
            "best_val_metrics": best_val_metrics,
            "init_val_metrics": init_val,
            "history": history,
            "test_metrics": test_metrics,
            "total_training_time": total_training_time,
            "communication_stats": comm_stats,
            "best_checkpoint": str(self.checkpoint_dir / "best_model.pt"),
        }
