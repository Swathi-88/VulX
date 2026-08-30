"""
PyTest unit test suite for showcase demo cases decision contracts.
"""

import json
import pytest
from vulx.config import DEFAULT_DEMO_JSON
from vulx.models.decision_contract import get_decision_contract


def test_demo_cases_contracts():
    with open(DEFAULT_DEMO_JSON, "r", encoding="utf-8") as f:
        demo_cases = json.load(f)

    assert len(demo_cases) == 8

    kp_contract = None
    for case in demo_cases:
        contract = get_decision_contract(case)
        assert "risk_probability" in contract
        assert "prediction_uncertainty" in contract
        assert "top_contributing_signals" in contract
        assert len(contract["top_contributing_signals"]) <= 3
        assert contract["naive_recommended_action"] in {"ALLOW", "REVIEW", "BLOCK"}

        if case.get("amount") == 45000.0 or case.get("transaction_id") == "demo_fp_001_showcase":
            kp_contract = contract

    assert kp_contract is not None
    assert kp_contract["risk_probability"] > 0.75
    assert kp_contract["naive_recommended_action"] == "BLOCK"
