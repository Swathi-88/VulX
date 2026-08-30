"""
PyTest unit & integration tests for VulX Policy Engine (Phase 3)
"""

import json
import tempfile
import yaml
import pytest
from vulx.config import DEFAULT_DEMO_JSON, DEFAULT_POLICIES_PATH
from vulx.models.decision_contract import get_decision_contract
from vulx.policy_engine import evaluate, load_policies


@pytest.fixture
def demo_cases():
    with open(DEFAULT_DEMO_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def test_policies_yaml_structure():
    policies = load_policies()
    assert "severity_tiers" in policies
    assert "false_positive_cost_table" in policies
    assert "reversibility_rules" in policies
    assert "uncertainty_thresholds" in policies
    assert "purpose_constraint_rules" in policies
    assert "routing_rules" in policies


def test_killer_demo_case_routing(demo_cases):
    """
    Test the ₹45,000 killer demo false-positive case:
    Naive recommended action is BLOCK, but Policy Engine routes to VERIFY.
    """
    kp_case = next(c for c in demo_cases if c.get("amount") == 45000.0)
    contract = get_decision_contract(kp_case)

    # Confirm Phase 2 naive action is BLOCK
    assert contract["naive_recommended_action"] == "BLOCK"
    assert contract["risk_probability"] > 0.70

    res = evaluate(contract)

    assert res["transaction_id"] == kp_case["transaction_id"]
    # Killer assertion: Policy Engine MUST route to VERIFY, not BLOCK
    assert res["routing_decision"] == "VERIFY"
    assert len(res["rationale_trace"]) == 6

    trace_text = "\n".join(res["rationale_trace"])
    assert "severity=high" in trace_text
    assert "FP_cost=high" in trace_text
    assert "reversibility=reversible" in trace_text
    assert "final: VERIFY" in trace_text


def test_true_fraud_counterpart_routing(demo_cases):
    """
    Test the true-fraud counterpart with similar risk_probability.
    Demonstrates that Policy Engine differentiates true fraud (HUMAN_REVIEW/BLOCK) from FP (VERIFY).
    """
    fraud_case = next(c for c in demo_cases if c.get("transaction_id") == "demo_fraud_001_counterpart")
    contract = get_decision_contract(fraud_case)

    assert contract["naive_recommended_action"] == "BLOCK"
    assert contract["risk_probability"] > 0.70

    res = evaluate(contract)

    assert res["transaction_id"] == fraud_case["transaction_id"]
    # Must route to HUMAN_REVIEW or BLOCK due to short tenure / lower FP cost profile
    assert res["routing_decision"] in {"HUMAN_REVIEW", "BLOCK"}
    assert res["routing_decision"] != "VERIFY"


def test_purpose_constraint_rule():
    """
    Test purpose constraint rule: signals tagged cross_merchant_derived enforce min VERIFY and forbid silent auto-BLOCK.
    """
    contract = {
        "transaction_id": "test_cm_001",
        "amount": 500.0,
        "customer_tenure_days": 10,
        "risk_probability": 0.85,
        "prediction_uncertainty": {"std_dev": 0.02, "uncertainty_level": "low"},
        "top_contributing_signals": [
            {
                "feature": "merchant_history_score",
                "contribution": 0.45,
                "tags": ["cross_merchant_derived"],
            }
        ],
        "naive_recommended_action": "BLOCK",
    }

    res = evaluate(contract)
    assert res["routing_decision"] in {"VERIFY", "HUMAN_REVIEW", "BLOCK"}
    assert res["routing_decision"] != "ALLOW"
    trace_text = "\n".join(res["rationale_trace"])
    assert "purpose_constraint=triggered" in trace_text


def test_live_yaml_threshold_modification():
    """
    Test that every threshold is dynamically loaded from policies.yaml.
    Modifying YAML numbers changes the routing decision live without code changes.
    """
    base_policies = load_policies()

    # Modify severity tier threshold so 45000 is classified as low/medium severity
    modified_policies = base_policies.copy()
    modified_policies["severity_tiers"] = {
        "low_max": 50000.0,
        "medium_max": 100000.0,
    }

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(modified_policies, f)
        temp_path = f.name

    contract = {
        "transaction_id": "test_dyn_001",
        "amount": 45000.0,
        "customer_tenure_days": 400,
        "risk_probability": 0.88,
        "prediction_uncertainty": {"std_dev": 0.22, "uncertainty_level": "high"},
        "top_contributing_signals": [],
        "naive_recommended_action": "BLOCK",
    }

    # Evaluate with default policies -> severity=high, fp_cost=high -> VERIFY
    res_default = evaluate(contract, policy_path=DEFAULT_POLICIES_PATH)
    assert res_default["routing_decision"] == "VERIFY"

    # Evaluate with modified policies -> severity=low -> fp_cost=low -> BLOCK (automated block)
    res_modified = evaluate(contract, policy_path=temp_path)
    assert res_modified["routing_decision"] in {"BLOCK", "HUMAN_REVIEW"}
