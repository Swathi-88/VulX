"""
PyTest unit and integration test suite for VulX Phase 4 (Execution Engine, SQLite Ledger, and Pipeline)
"""

import json
import os
import tempfile
import pytest
from vulx.config import DEFAULT_DEMO_JSON
from vulx.execution_engine import execute
from vulx.ledger import (
    compute_correctness,
    compute_legal_basis,
    compute_retention_class,
    init_db,
    query_ledger,
    record_event,
)
from vulx.models.decision_contract import get_decision_contract
from vulx.policy_engine import evaluate


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile("w", suffix=".db", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def demo_cases():
    with open(DEFAULT_DEMO_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def test_execution_engine_allow():
    res = execute("ALLOW", {})
    assert res["action_taken"] == "ALLOW"
    assert res["verification_outcome"] is None
    assert res["human_review_outcome"] is None
    assert res["final_outcome"] == "completed"


def test_execution_engine_verify():
    # Deterministic seed test
    res_success = execute("VERIFY", {}, success_prob=1.0, seed=42)
    assert res_success["action_taken"] == "VERIFY"
    assert res_success["verification_outcome"] == "VERIFIED"
    assert res_success["final_outcome"] == "completed"

    res_fail = execute("VERIFY", {}, success_prob=0.0, seed=42)
    assert res_fail["action_taken"] == "VERIFY"
    assert res_fail["verification_outcome"] == "FAILED"
    assert res_fail["final_outcome"] == "blocked"


def test_execution_engine_human_review():
    # Analyst override approve
    res_app = execute("HUMAN_REVIEW", {}, analyst_override="APPROVE")
    assert res_app["action_taken"] == "HUMAN_REVIEW"
    assert res_app["human_review_outcome"] == "APPROVED"
    assert res_app["final_outcome"] == "completed"

    # Analyst override reject
    res_rej = execute("HUMAN_REVIEW", {}, analyst_override="REJECT")
    assert res_rej["action_taken"] == "HUMAN_REVIEW"
    assert res_rej["human_review_outcome"] == "REJECTED"
    assert res_rej["final_outcome"] == "blocked"

    # Heuristic fallback (low risk -> approve, high risk -> reject)
    res_low = execute("HUMAN_REVIEW", {"risk_probability": 0.2})
    assert res_low["human_review_outcome"] == "APPROVED"
    assert res_low["final_outcome"] == "completed"

    res_high = execute("HUMAN_REVIEW", {"risk_probability": 0.8})
    assert res_high["human_review_outcome"] == "REJECTED"
    assert res_high["final_outcome"] == "blocked"


def test_ledger_correctness_calculation():
    assert compute_correctness("blocked", "fraud") == "TP"
    assert compute_correctness("blocked", "legitimate") == "FP"
    assert compute_correctness("completed", "legitimate") == "TN"
    assert compute_correctness("completed", "fraud") == "FN"


def test_ledger_legal_basis_and_retention():
    # Legal basis tagged as cross_merchant_derived if top signals tagged or features match
    event_cm = {
        "top_contributing_signals": [
            {"feature": "merchant_history_score", "contribution": 0.4, "tags": ["cross_merchant_derived"]}
        ],
        "rationale_trace": ["purpose_constraint=triggered ..."],
    }
    assert compute_legal_basis(event_cm) == "cross_merchant_derived"

    event_own = {
        "top_contributing_signals": [
            {"feature": "device_novelty", "contribution": 0.5, "tags": []}
        ],
        "rationale_trace": ["purpose_constraint=passed ..."],
    }
    assert compute_legal_basis(event_own) == "own_history"

    # Retention class: completed -> financial_record_required, blocked -> short_term_ops
    assert compute_retention_class("completed") == "financial_record_required"
    assert compute_retention_class("blocked") == "short_term_ops"


def test_ledger_record_and_query(temp_db_path):
    init_db(temp_db_path)

    sample_event = {
        "transaction_id": "tx_test_100",
        "risk_probability": 0.75,
        "naive_recommended_action": "BLOCK",
        "routing_decision": "VERIFY",
        "rationale_trace": ["severity=high", "FP_cost=high"],
        "action_taken": "VERIFY",
        "verification_outcome": "VERIFIED",
        "final_outcome": "completed",
        "ground_truth_label": "legitimate",
    }

    rec = record_event(sample_event, db_path=temp_db_path)
    assert rec["correctness"] == "TN"  # Completed legitimate payment
    assert rec["retention_class"] == "financial_record_required"

    # Query back
    all_events = query_ledger(db_path=temp_db_path)
    assert len(all_events) == 1
    assert all_events[0]["transaction_id"] == "tx_test_100"
    assert all_events[0]["correctness"] == "TN"

    # Query with filter
    filtered = query_ledger(filter_dict={"correctness": "TN"}, db_path=temp_db_path)
    assert len(filtered) == 1
    assert filtered[0]["transaction_id"] == "tx_test_100"

    empty_filtered = query_ledger(filter_dict={"correctness": "FP"}, db_path=temp_db_path)
    assert len(empty_filtered) == 0


def test_full_pipeline_integration(demo_cases, temp_db_path):
    """
    Tests end-to-end pipeline execution from Phase 2 contract -> Phase 3 policy -> Phase 4 execution -> SQLite ledger.
    """
    kp_case = next(c for c in demo_cases if c.get("amount") == 45000.0)

    contract = get_decision_contract(kp_case)
    policy_res = evaluate(contract)
    exec_res = execute(policy_res["routing_decision"], contract, seed=42)

    event_payload = {
        **contract,
        **policy_res,
        **exec_res,
        "ground_truth_label": kp_case.get("ground_truth_label", "legitimate"),
    }

    recorded = record_event(event_payload, db_path=temp_db_path)

    assert contract["naive_recommended_action"] == "BLOCK"
    assert policy_res["routing_decision"] == "VERIFY"
    assert exec_res["action_taken"] == "VERIFY"
    assert recorded["transaction_id"] == kp_case["transaction_id"]
    assert recorded["correctness"] in ["TN", "FP"]
