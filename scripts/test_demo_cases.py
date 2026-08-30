#!/usr/bin/env python3
"""
Test Demo Cases Runner for Vulcan Decision Simulator (VulX Phase 2)
---------------------------------------------------------------------
Loads demo_cases.json, evaluates each case through get_decision_contract,
prints output contracts, and asserts killer demo false-positive behavior.
"""

import json
import sys
from pathlib import Path

# Add src/ directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vulx.config import DEFAULT_DEMO_JSON
from vulx.models.decision_contract import get_decision_contract


def run_demo_case_evaluations(demo_json_path=None):
    if demo_json_path is None:
        demo_json_path = DEFAULT_DEMO_JSON

    print("=" * 80)
    print(" VULCAN DECISION SIMULATOR - SHOWCASE DEMO CASES EVALUATION")
    print("=" * 80)

    with open(demo_json_path, "r", encoding="utf-8") as f:
        demo_cases = json.load(f)

    contracts = []
    kp_case_contract = None

    for i, case in enumerate(demo_cases, 1):
        contract = get_decision_contract(case)
        contracts.append(contract)

        print(f"\n[Case {i}/8] ID: {case['transaction_id']} ({case.get('category_tag', 'N/A')})")
        print(f"  Description    : {case.get('description', 'N/A')}")
        print(f"  Ground Truth   : {case['ground_truth_label'].upper()}")
        print(f"  Risk Prob      : {contract['risk_probability']:.4f}")
        print(f"  Uncertainty    : {contract['prediction_uncertainty']['uncertainty_level'].upper()} (std={contract['prediction_uncertainty']['std_dev']:.4f})")
        print(f"  Naive Action   : {contract['naive_recommended_action']}")
        print("  Top 3 Signals  :")
        for sig in contract['top_contributing_signals']:
            print(f"    - {sig['feature']:<25} : {sig['contribution']:+.4f}")

        # Check for killer demo false-positive case (amount=45000)
        if case.get("amount") == 45000.0 or case.get("transaction_id") == "demo_fp_001_showcase":
            kp_case_contract = contract

    print("\n" + "=" * 80)
    print(" KILLER DEMO FALSE-POSITIVE CASE ASSERTION CHECK")
    print("=" * 80)

    assert kp_case_contract is not None, "Mandatory showcase false positive transaction (INR 45,000) not found!"

    print(f"Showcase Txn ID         : {kp_case_contract['transaction_id']}")
    print(f"Risk Probability        : {kp_case_contract['risk_probability']}")
    print(f"Naive Recommended Action: {kp_case_contract['naive_recommended_action']}")

    assert kp_case_contract['risk_probability'] > 0.75, (
        f"Expected risk_probability > 0.75, got {kp_case_contract['risk_probability']}"
    )
    assert kp_case_contract['naive_recommended_action'] == "BLOCK", (
        f"Expected naive_recommended_action == 'BLOCK', got {kp_case_contract['naive_recommended_action']}"
    )

    print("ASSERTION PASSED: The INR 45,000 showcase case produces risk_probability > 0.75 and action 'BLOCK'.")
    print("This confirms Vulcan naively flags this false-positive case, setting up VulX's Phase 3 override!\n")
    return contracts


if __name__ == "__main__":
    run_demo_case_evaluations()
