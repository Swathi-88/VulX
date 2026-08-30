"""
VulX Human Review Interactive Analyst CLI (human_review_cli.py)

Interactive command-line tool for human analyst review during live pitches/demos.
Allows analyst operators to review queued transactions and issue APPROVE or REJECT overrides.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from vulx.config import DEFAULT_DEMO_JSON
from vulx.execution_engine import execute
from vulx.ledger import record_event
from vulx.models.decision_contract import get_decision_contract
from vulx.policy_engine import evaluate


def run_cli():
    parser = argparse.ArgumentParser(description="VulX Human Analyst Review Console")
    parser.add_argument(
        "--demo_id",
        type=str,
        default=None,
        help="Optional transaction ID to review (defaults to first case requiring HUMAN_REVIEW)",
    )
    args = parser.parse_args()

    if not os.path.exists(DEFAULT_DEMO_JSON):
        print(f"Error: Demo cases file '{DEFAULT_DEMO_JSON}' not found.")
        sys.exit(1)

    with open(DEFAULT_DEMO_JSON, "r", encoding="utf-8") as f:
        demo_cases = json.load(f)

    target_case = None
    if args.demo_id:
        target_case = next(
            (c for c in demo_cases if c.get("transaction_id") == args.demo_id), None
        )
        if not target_case:
            print(f"Transaction ID '{args.demo_id}' not found in demo_cases.json.")
            sys.exit(1)
    else:
        # Find first case evaluating to HUMAN_REVIEW
        for c in demo_cases:
            contract = get_decision_contract(c)
            p_res = evaluate(contract)
            if p_res["routing_decision"] == "HUMAN_REVIEW":
                target_case = c
                break

        if not target_case:
            target_case = demo_cases[1]  # Fallback to counterpart case

    contract = get_decision_contract(target_case)
    p_res = evaluate(contract)

    print("\n" + "=" * 80)
    print(" VULX HUMAN ANALYST REVIEW QUEUE — LIVE INTERACTIVE CONSOLE")
    print("=" * 80)

    print(f"\n  Transaction ID:           {target_case['transaction_id']}")
    print(f"  Amount:                   INR {target_case.get('amount', 0.0):,.2f}")
    print(f"  Customer Tenure:          {target_case.get('customer_tenure_days', 0)} days")
    print(f"  Description:              {target_case.get('description', 'N/A')}")
    print(f"  Ground Truth Label:       {target_case.get('ground_truth_label', 'unknown')}")

    print("\n  --- PHASE 2: ML RISK ANALYSIS ---")
    print(f"  Risk Probability:         {contract['risk_probability'] * 100:.1f}%")
    print(f"  Uncertainty StdDev:       {contract['prediction_uncertainty']['std_dev']:.4f} ({contract['prediction_uncertainty']['uncertainty_level']})")
    print(f"  Naive Recommended Action: {contract['naive_recommended_action']}")
    print("  Top SHAP Signals:")
    for sig in contract["top_contributing_signals"]:
        tags_str = f" [tags: {', '.join(sig['tags'])}]" if sig.get("tags") else ""
        print(f"    - {sig['feature']}: {sig['contribution']:+.4f}{tags_str}")

    print("\n  --- PHASE 3: POLICY ENGINE RATIONALE ---")
    print(f"  Routing Decision:         {p_res['routing_decision']}")
    for idx, trace in enumerate(p_res["rationale_trace"], start=1):
        print(f"    {idx}. {trace}")

    print("\n" + "-" * 80)
    print(" [ACTION REQUIRED] Please enter analyst decision:")
    print("   [A] Approve Payment (Complete transaction)")
    print("   [R] Reject Payment  (Block transaction)")
    print("-" * 80)

    user_input = input(" Enter choice (A/R) [Default: R]: ").strip().upper()

    if user_input.startswith("A"):
        analyst_choice = "APPROVE"
    else:
        analyst_choice = "REJECT"

    exec_res = execute(
        routing_decision=p_res["routing_decision"],
        transaction=contract,
        analyst_override=analyst_choice,
    )

    event_payload = {
        **contract,
        **p_res,
        **exec_res,
        "ground_truth_label": target_case.get("ground_truth_label", "legitimate"),
        "timestamp": target_case.get("timestamp", "2026-08-26T15:00:00Z"),
    }

    recorded = record_event(event_payload)

    print("\n" + "=" * 80)
    print(" PIPELINE & LEDGER EXECUTION COMPLETE")
    print("=" * 80)
    print(f"  Analyst Decision:         {analyst_choice}")
    print(f"  Final Payment Outcome:    {recorded['final_outcome'].upper()}")
    print(f"  Ledger Audit Correctness: {recorded['correctness']}")
    print(f"  Legal Basis Tag:          {recorded['legal_basis_tag']}")
    print(f"  Retention Class:          {recorded['retention_class']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_cli()
