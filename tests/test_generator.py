"""
Unit tests for VulX synthetic payment event generator package.
"""

import json
import pytest
import pandas as pd
from vulx.generator.event_builder import generate_dataset, compute_category_counts
from vulx.generator.showcase import get_showcase_demo_cases


def test_category_counts():
    counts = compute_category_counts(2000)
    assert sum(counts.values()) == 2000
    assert counts["normal"] == 1400
    assert counts["suspicious"] == 160
    assert counts["borderline"] == 200
    assert counts["fraud"] == 100
    assert counts["legitimate_but_unusual"] == 100
    assert counts["merchant_anomaly"] == 40


def test_generate_dataset_shape_and_columns():
    df = generate_dataset(2000, 42)
    assert len(df) == 2000
    expected_cols = [
        "transaction_id", "timestamp", "amount", "device_novelty",
        "transaction_velocity", "location_deviation", "merchant_history_score",
        "customer_tenure_days", "payment_method", "ip_reputation_score",
        "category_tag", "ground_truth_label"
    ]
    for col in expected_cols:
        assert col in df.columns


def test_value_bounds():
    df = generate_dataset(500, 42)
    assert (df["device_novelty"] >= 0.0).all() and (df["device_novelty"] <= 1.0).all()
    assert (df["location_deviation"] >= 0.0).all() and (df["location_deviation"] <= 1.0).all()
    assert (df["merchant_history_score"] >= 0.0).all() and (df["merchant_history_score"] <= 1.0).all()
    assert (df["ip_reputation_score"] >= 0.0).all() and (df["ip_reputation_score"] <= 1.0).all()
    assert (df["amount"] > 0).all()


def test_showcase_demo_cases():
    cases = get_showcase_demo_cases()
    assert len(cases) == 8

    # Verify false-positive showcase case
    fp_case = cases[0]
    assert fp_case["amount"] == 45000.0
    assert fp_case["device_novelty"] == 0.95
    assert fp_case["transaction_velocity"] == 4
    assert fp_case["location_deviation"] == 0.88
    assert fp_case["merchant_history_score"] == 0.05
    assert fp_case["customer_tenure_days"] == 400
    assert fp_case["ip_reputation_score"] == 0.6
    assert fp_case["ground_truth_label"] == "legitimate"
