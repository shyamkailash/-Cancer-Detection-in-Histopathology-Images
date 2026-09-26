from django.urls import path
from apps.mlops.views import (
    mlops_status_view,
    mlops_models_view,
    mlops_runs_view,
    mlops_drift_view,
    mlops_drift_check_view,
    mlops_retraining_view,
    mlops_retraining_evaluate_view,
    mlops_orchestrate_view,
    mlops_registry_validate_view,
    mlops_registry_approve_view,
)

urlpatterns = [
    path("status/", mlops_status_view, name="mlops-status"),
    path("models/", mlops_models_view, name="mlops-models"),
    path("runs/", mlops_runs_view, name="mlops-runs"),
    path("drift/", mlops_drift_view, name="mlops-drift"),
    path("drift/check/", mlops_drift_check_view, name="mlops-drift-check"),
    path("retraining/", mlops_retraining_view, name="mlops-retraining"),
    path("retraining/evaluate/", mlops_retraining_evaluate_view, name="mlops-retraining-evaluate"),
    path("orchestrate/", mlops_orchestrate_view, name="mlops-orchestrate"),
    path("registry/validate/", mlops_registry_validate_view, name="mlops-registry-validate"),
    path("registry/approve/", mlops_registry_approve_view, name="mlops-registry-approve"),
]
