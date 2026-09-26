# Bounded Continual Learning & Retraining Workflow

## Overview
Automated retraining in healthcare computer vision can introduce model degradation, catastrophic forgetting, or unintended GPU resource consumption if not strictly bounded.

The Phase 6 Retraining Workflow implements **deterministic gating**, **reviewable plans**, and **safe execution boundaries**.

---

## 1. Safety Guardrails & Principles
1. **Zero Unintended Compute (`dry_run=True` Default):**
   Generating a retraining plan formulation does *not* immediately consume GPU cycles. It produces a structured `RetrainingPlan` DTO.
2. **Deterministic Triggering:**
   Retraining is only recommended when:
   - Statistical distribution drift exceeds threshold ($\text{PSI} \ge 0.25$).
   - A minimum sample buffer is accumulated ($N \ge 50$ samples).
3. **Candidate Isolation:**
   Retrained models are saved as `CANDIDATE` checkpoints and are never allowed to overwrite production baseline weights automatically.
4. **Explicit Quality Gate & Approval:**
   Promotion from `VALIDATED` to `APPROVED` requires passing test evaluation gates ($\text{Accuracy} \ge 90\%$, $\text{ROC-AUC} \ge 0.90$) and explicit administrator approval.
