"""
REST API Views for Phase 6 Agentic MLOps Orchestration and Telemetry.
Provides endpoints for health, model registry lifecycle, drift assessment, retraining plans, and orchestration.
"""

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status

from mlops.config import load_mlops_config
from mlops.tracking import MLflowTracker
from mlops.storage import MLOpsStorage
from mlops.distributed import DistributedCoordinator
from mlops.drift_monitor import DriftMonitor
from mlops.privacy_accountant import PrivacyAccountant
from mlops.registry_lifecycle import ModelRegistryLifecycle
from mlops.contracts import QualityGateConfig
from agents.orchestrator import MLOpsOrchestrator
from agents.drift import DriftAgent
from agents.retraining import RetrainingAgent

_config = load_mlops_config()
_tracker = MLflowTracker(
    tracking_uri=_config.mlflow.tracking_uri,
    experiment_name=_config.mlflow.experiment_name,
    enabled=_config.mlflow.enabled,
)
_storage = MLOpsStorage()
_coordinator = DistributedCoordinator()
_lifecycle = ModelRegistryLifecycle()
_drift_monitor = DriftMonitor(
    warning_threshold=_config.drift.warning_threshold,
    critical_threshold=_config.drift.critical_threshold,
    min_samples=_config.drift.min_samples,
)
_privacy_accountant = PrivacyAccountant()
_orchestrator = MLOpsOrchestrator(
    config=_config,
    tracker=_tracker,
    storage=_storage,
    coordinator=_coordinator,
    lifecycle=_lifecycle,
    drift_monitor=_drift_monitor,
    privacy_accountant=_privacy_accountant,
)


@api_view(["GET"])
@permission_classes([AllowAny])
def mlops_status_view(request):
    """Return overall MLOps subsystem status and component health."""
    node_statuses = _coordinator.ping_nodes()
    return Response({
        "status": "healthy",
        "mlops_version": "1.0.0",
        "components": {
            "tracking": {
                "engine": "MLflow (fallback-resilient)",
                "is_available": _tracker.is_available(),
                "tracking_uri": _config.mlflow.tracking_uri,
            },
            "drift_monitoring": {
                "method": _config.drift.method,
                "warning_threshold": _config.drift.warning_threshold,
                "critical_threshold": _config.drift.critical_threshold,
            },
            "registry": {
                "total_models": len(_lifecycle.list_models()),
            },
            "distributed_coordination": {
                "total_nodes": len(node_statuses),
                "healthy_nodes": sum(1 for s in node_statuses.values() if s == "HEALTHY"),
            },
            "privacy_accounting": {
                "accountant": "Rényi Differential Privacy (RDP)",
                "default_delta": _config.privacy.delta,
            },
        },
    })


@api_view(["GET"])
@permission_classes([AllowAny])
def mlops_models_view(request):
    """List all models in the lifecycle registry with approval states and history."""
    state_filter = request.query_params.get("state")
    models = _lifecycle.list_models(state_filter=state_filter)
    return Response({
        "count": len(models),
        "models": models,
    })


@api_view(["GET"])
@permission_classes([AllowAny])
def mlops_runs_view(request):
    """List recent orchestration runs."""
    limit = int(request.query_params.get("limit", 50))
    runs = _storage.list_runs(limit=limit)
    return Response({
        "count": len(runs),
        "runs": runs,
    })


@api_view(["GET"])
@permission_classes([AllowAny])
def mlops_drift_view(request):
    """Return current drift status and recent drift telemetry events."""
    recent_events = _storage.list_drift_events(limit=20)
    return Response({
        "current_config": {
            "method": _config.drift.method,
            "warning_threshold": _config.drift.warning_threshold,
            "critical_threshold": _config.drift.critical_threshold,
            "min_samples": _config.drift.min_samples,
        },
        "recent_events": recent_events,
    })


@api_view(["POST"])
@permission_classes([AllowAny])
def mlops_drift_check_view(request):
    """Evaluate statistical drift between baseline and current prediction distributions."""
    data = request.data or {}
    baseline = data.get("baseline")
    current = data.get("current")
    method = data.get("method", "psi")

    drift_agent = DriftAgent(_drift_monitor)
    res = drift_agent.run({
        "baseline": baseline,
        "current": current,
        "method": method,
    })

    if res.details.get("drift_detected"):
        _storage.save_drift_event(res.to_dict())

    return Response(res.to_dict())


@api_view(["GET"])
@permission_classes([AllowAny])
def mlops_retraining_view(request):
    """Return recent retraining plans and audit history."""
    events = _storage.list_retraining_events(limit=20)
    return Response({
        "count": len(events),
        "retraining_plans": events,
    })


@api_view(["POST"])
@permission_classes([AllowAny])
def mlops_retraining_evaluate_view(request):
    """Evaluate retraining triggers and formulate a structured RetrainingPlan."""
    data = request.data or {}
    drift_detected = data.get("drift_detected", False)
    perf_drop = data.get("performance_drop", False)
    strategy = data.get("strategy", "fedavg")
    sample_count = int(data.get("sample_count", 100))

    retraining_agent = RetrainingAgent()
    res = retraining_agent.run({
        "drift_detected": drift_detected,
        "performance_drop": perf_drop,
        "strategy": strategy,
        "sample_count": sample_count,
    })

    if res.details.get("retraining_needed"):
        _storage.save_retraining_event(res.details.get("plan", {}))

    return Response(res.to_dict())


@api_view(["POST"])
@permission_classes([AllowAny])
def mlops_orchestrate_view(request):
    """
    Trigger full 14-stage agentic MLOps orchestration.
    Enforces safe dry-run by default unless explicitly requested otherwise.
    """
    data = request.data or {}
    strategy = data.get("strategy", "fedavg")
    dry_run = data.get("dry_run", True)
    simulate_drift = data.get("simulate_drift", False)

    result = _orchestrator.run_full_lifecycle(
        strategy=strategy,
        dry_run=dry_run,
        simulate_drift=simulate_drift,
    )
    return Response(result)


@api_view(["POST"])
@permission_classes([AllowAny])
def mlops_registry_validate_view(request):
    """Validate a candidate model against quality gates."""
    data = request.data or {}
    model_id = data.get("model_id")
    if not model_id:
        return Response({"error": "model_id is required"}, status=status.HTTP_400_BAD_REQUEST)

    qg_dict = data.get("quality_gate")
    qg = QualityGateConfig(**qg_dict) if qg_dict else None

    try:
        passed, note, record = _lifecycle.validate_candidate(model_id=model_id, quality_gate=qg)
        return Response({"passed": passed, "note": note, "model": record})
    except KeyError as e:
        return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)


@api_view(["POST"])
@permission_classes([AllowAny])
def mlops_registry_approve_view(request):
    """Explicitly approve a validated candidate model for promotion."""
    data = request.data or {}
    model_id = data.get("model_id")
    approver = data.get("approver_id", "admin")

    if not model_id:
        return Response({"error": "model_id is required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        record = _lifecycle.approve_model(model_id=model_id, approver_id=approver)
        return Response({"approved": True, "model": record})
    except (KeyError, ValueError) as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
