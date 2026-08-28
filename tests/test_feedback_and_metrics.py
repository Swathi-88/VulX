"""
PyTest unit test suite for Anvil Phase 5 (Feedback Pipeline, Retraining Comparison, and System Metrics)
"""

import json
import os
import tempfile
import pytest
import pandas as pd
from anvil.config import DEFAULT_DEMO_JSON
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from feedback_pipeline import extract_feedback
from retrain_and_compare import retrain_and_compare
from metrics_dashboard import run_metrics_dashboard


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_feedback_pipeline_extraction(temp_dir):
    csv_path = os.path.join(temp_dir, "test_corrections.csv")
    n_corr, fp_cnt, fn_cnt, route_map = extract_feedback(output_csv=csv_path)

    assert isinstance(n_corr, int)
    assert isinstance(fp_cnt, int)
    assert isinstance(fn_cnt, int)
    assert isinstance(route_map, dict)
    assert os.path.exists(csv_path)

    if n_corr > 0:
        df = pd.read_csv(csv_path)
        assert len(df) == n_corr
        assert "ground_truth_label" in df.columns
        assert "amount" in df.columns


def test_retrain_and_compare_evaluation(temp_dir):
    corr_csv = os.path.join(temp_dir, "test_corrections.csv")

    # Create dummy correction CSV if empty
    dummy_df = pd.DataFrame([{
        "amount": 45000.0,
        "device_novelty": 0.95,
        "transaction_velocity": 4,
        "location_deviation": 0.88,
        "merchant_history_score": 0.05,
        "customer_tenure_days": 400,
        "payment_method": "UPI",
        "ip_reputation_score": 0.6,
        "ground_truth_label": "legitimate"
    }])
    dummy_df.to_csv(corr_csv, index=False)

    json_path = os.path.join(temp_dir, "comparison.json")
    comp = retrain_and_compare(correction_csv=corr_csv, output_json=json_path)

    assert "baseline_model" in comp
    assert "retrained_model" in comp
    assert "delta" in comp
    assert "precision" in comp["retrained_model"]
    assert "recall" in comp["retrained_model"]
    assert "f1_score" in comp["retrained_model"]
    assert "false_positive_rate" in comp["retrained_model"]
    assert os.path.exists(json_path)


def test_metrics_dashboard_calculation():
    metrics = run_metrics_dashboard()

    assert metrics is not None
    assert "total_events" in metrics
    assert "routing_distribution" in metrics
    assert "verify_analysis" in metrics
    assert "human_review_analysis" in metrics
    assert "naive_baseline" in metrics
    assert "anvil_system" in metrics

    assert metrics["total_events"] > 0
    assert "ALLOW" in metrics["routing_distribution"]
    assert "VERIFY" in metrics["routing_distribution"]
    assert "HUMAN_REVIEW" in metrics["routing_distribution"]
