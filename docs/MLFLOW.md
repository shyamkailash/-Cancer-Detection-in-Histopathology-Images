# MLflow Experiment Tracking Architecture & Graceful Fallback

## Overview
Phase 6 integrates **MLflow Experiment Tracking** to record hyperparameters, evaluation metrics, model artifacts, and run tags across centralized baseline training, fine-tuning, and federated optimization rounds.

---

## 1. Architectural Design & Graceful Degradation
In environments where `mlflow` is unavailable or tracking servers are offline, standard MLOps platforms crash.

The `BaseExperimentTracker` abstraction and `MLflowTracker` implement **zero-crash graceful fallback**:
- When `mlflow` is installed and reachable, telemetry is logged to the configured URI (`artifacts/mlflow` or remote HTTP).
- When `mlflow` is missing or offline, `MLflowTracker` automatically and seamlessly switches to structured in-memory logging, caching runs, metrics, and parameters without throwing unhandled exceptions.

---

## 2. Configuration Options
Defined in `configs/mlops.yaml`:
```yaml
mlflow:
  enabled: false
  tracking_uri: "artifacts/mlflow"
  experiment_name: "pcam_cancer_detection"
```

Environment variable overrides:
- `MLFLOW_ENABLED=true`
- `MLFLOW_TRACKING_URI=http://localhost:5000`
