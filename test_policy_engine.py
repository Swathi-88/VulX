"""
VulX Policy Engine Showcase & Demo Runner Script (test_policy_engine.py)

Runs the ₹45,000 killer demo case and true-fraud counterpart through evaluate(),
prints full rationale traces, asserts expected routing decisions, and demonstrates
live policy modification via policies.yaml.
"""

import json
import os
import sys
import tempfile
import yaml

# Ensure src is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from vulx.config import DEFAULT_DEMO_JSON, DEFAULT_POLICIES_PATH
from vulx.models.decision_contract import get_decision_contract
from vulx.policy_engine import evaluate, load_policies


def run_showcase():
    print("=" * 80)
    print(" VULX POLICY ENGINE — SHOWCASE DEMO CASE EVALUATION")
    print("=" * 80)

    if not os.path.exists(DEFAULT_DEMO_JSON):
        print(f"Error: Demo file '{DEFAULT_DEMO_JSON}' not found.")
        sys.exit(1)

    with open(DEFAULT_DEMO_JSON, "r", encoding="utf-8") as f:
        demo_cases = json.load(f)

    # -----------------------------------------------------------------
    # 1. KILLER DEMO CASE (₹45,000 False Positive)
    # -----------------------------------------------------------------
    kp_case = next((c for c in demo_cases if c.get("amount") == 45000.0), None)
    if not kp_case:
        print("Error: Could not find ₹45,000 showcase case in demo_cases.json")
        sys.exit(1)

    print("\n[1] CASE 1: INR 45,000 FALSE-POSITIVE SHOWCASE")
    print(f"    Transaction ID:           {kp_case['transaction_id']}")
    print(f"    Amount:                   INR {kp_case['amount']:,.2f}")
    print(f"    Customer Tenure:          {kp_case['customer_tenure_days']} days")
    print(f"    Description:              {kp_case['description']}")

    kp_contract = get_decision_contract(kp_case)
    print(f"    Phase 2 Naive Action:    {kp_contract['naive_recommended_action']} (Risk Prob: {kp_contract['risk_probability']:.4f})")

    kp_result = evaluate(kp_contract)
    print(f"\n  --> POLICY ENGINE DECISION: {kp_result['routing_decision']}")
    print("  --> FULL RATIONALE TRACE:")
    for idx, trace in enumerate(kp_result["rationale_trace"], start=1):
        print(f"      {idx}. {trace}")

    assert kp_result["routing_decision"] == "VERIFY", f"Expected VERIFY but got {kp_result['routing_decision']}"
    print("\n  [OK] SUCCESS: Naive BLOCK successfully downgraded to VERIFY via policy rules!")

    # -----------------------------------------------------------------
    # 2. TRUE FRAUD COUNTERPART
    # -----------------------------------------------------------------
    tf_case = next((c for c in demo_cases if c.get("transaction_id") == "demo_fraud_001_counterpart"), None)
    if tf_case:
        print("\n" + "-" * 80)
        print("[2] CASE 2: TRUE FRAUD COUNTERPART (Identical Risk Prob Profile)")
        print(f"    Transaction ID:           {tf_case['transaction_id']}")
        print(f"    Amount:                   INR {tf_case['amount']:,.2f}")
        print(f"    Customer Tenure:          {tf_case['customer_tenure_days']} days")
        print(f"    Description:              {tf_case['description']}")

        tf_contract = get_decision_contract(tf_case)
        print(f"    Phase 2 Naive Action:    {tf_contract['naive_recommended_action']} (Risk Prob: {tf_contract['risk_probability']:.4f})")

        tf_result = evaluate(tf_contract)
        print(f"\n  --> POLICY ENGINE DECISION: {tf_result['routing_decision']}")
        print("  --> FULL RATIONALE TRACE:")
        for idx, trace in enumerate(tf_result["rationale_trace"], start=1):
            print(f"      {idx}. {trace}")

        assert tf_result["routing_decision"] in {"HUMAN_REVIEW", "BLOCK"}, "Expected HUMAN_REVIEW or BLOCK"
        print(f"\n  [OK] SUCCESS: Differentiated from FP case! Engine routed true fraud to {tf_result['routing_decision']}.")

    # -----------------------------------------------------------------
    # 3. LIVE YAML THRESHOLD MODIFICATION DEMO
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[3] DEMO: LIVE POLICY CONFIGURATION (YAML THRESHOLD EDITING)")
    print("    Original severity_tiers: medium_max = 10000.0 => INR 45,000 is 'high' severity.")

    pol = load_policies(DEFAULT_POLICIES_PATH)
    pol["severity_tiers"]["medium_max"] = 50000.0  # Make INR 45,000 severity 'medium'

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp_f:
        yaml.safe_dump(pol, tmp_f)
        tmp_policy_path = tmp_f.name

    mod_result = evaluate(kp_contract, policy_path=tmp_policy_path)
    os.remove(tmp_policy_path)

    print("    Modified severity_tiers in YAML: medium_max = 50000.0 => INR 45,000 is now 'medium' severity.")
    print(f"  --> NEW ROUTING DECISION: {mod_result['routing_decision']}")
    print(f"  --> UPDATED RATIONALE:     {mod_result['rationale_trace'][1]}")
    print(f"  --> UPDATED FINAL STEP:    {mod_result['rationale_trace'][-1]}")
    assert mod_result["routing_decision"] in {"BLOCK", "HUMAN_REVIEW"}
    print("\n  [OK] SUCCESS: Proved engine responds live to YAML policy configuration edits without code changes!")

    print("\n" + "=" * 80)
    print(" ALL POLICY ENGINE DEMO ASSERTIONS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_showcase()
