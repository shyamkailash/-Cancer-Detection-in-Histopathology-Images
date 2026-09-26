"""
CLI Demonstration Entrypoint for Phase 6 Agentic MLOps Framework.
Usage:
    python -m mlops.demo [--strategy STRATEGY] [--simulate-drift] [--json]
"""

import argparse
import json
import sys
import time

from agents.orchestrator import MLOpsOrchestrator
from mlops.config import load_mlops_config


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6 Agentic MLOps Lifecycle Demonstration for Histopathology FL"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="fedavg",
        choices=["centralized", "centralized_finetuned", "fedavg", "fedprox", "fedbn", "dp_fedavg"],
        help="Training/orchestration strategy to demonstrate (default: fedavg)",
    )
    parser.add_argument(
        "--simulate-drift",
        action="store_true",
        help="Simulate severe distribution drift in prediction telemetry",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw orchestration JSON result",
    )
    args = parser.parse_args()

    orchestrator = MLOpsOrchestrator()

    if not args.json:
        print("=" * 70)
        print("  PHASE 6: AGENTIC MLOPS ORCHESTRATION FRAMEWORK")
        print("  Privacy-Preserving Federated Learning for Histopathology Images")
        print("=" * 70)
        print(f"Strategy:        {args.strategy}")
        print(f"Mode:            Dry Run (Safe Verification)")
        print(f"Simulate Drift:  {args.simulate_drift}")
        print("-" * 70)

    result = orchestrator.run_full_lifecycle(
        strategy=args.strategy,
        dry_run=True,
        simulate_drift=args.simulate_drift,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("\n[ORCHESTRATION EXECUTION TRACE]")
    for stage in result.get("stages", []):
        num = stage["stage_number"]
        name = stage["stage_name"]
        status = stage["status"]
        print(f"  Stage {num:02d}: {name:<35} [{status}]")

    summary = result.get("summary", {})
    print("\n" + "=" * 70)
    print("  ORCHESTRATION SUMMARY")
    print("=" * 70)
    print(f"Run ID:                {summary.get('run_id')}")
    print(f"Status:                {summary.get('status')}")
    print(f"Execution Time:        {summary.get('execution_time_ms')} ms")
    print(f"Drift Detected:        {summary.get('drift_detected')}")
    print(f"Quality Gate Passed:   {summary.get('quality_gate_passed')}")
    print(f"Staged Candidate ID:   {summary.get('staged_candidate_id')}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
