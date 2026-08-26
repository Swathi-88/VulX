#!/usr/bin/env python3
"""
Decision Contract Module for Vulcan Decision Simulator (Anvil Phase 2)
---------------------------------------------------------------------
Exposes get_decision_contract(transaction: dict) -> dict for single transaction evaluation.
"""

import sys
from pathlib import Path

# Add src/ directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anvil.config import DEFAULT_MODEL_PATH, DEFAULT_ENSEMBLE_DIR
from anvil.models.decision_contract import get_decision_contract as _get_contract, load_models


def get_decision_contract(transaction: dict, model_path=None, ensemble_dir=None) -> dict:
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if ensemble_dir is None:
        ensemble_dir = DEFAULT_ENSEMBLE_DIR
    return _get_contract(transaction, model_path=model_path, ensemble_dir=ensemble_dir)


__all__ = ["get_decision_contract", "load_models"]

if __name__ == "__main__":
    import json
    sample_transaction = {
        "transaction_id": "sample_test_001",
        "timestamp": "2026-08-23T10:15:30Z",
        "amount": 45000.0,
        "device_novelty": 0.95,
        "transaction_velocity": 4,
        "location_deviation": 0.88,
        "merchant_history_score": 0.05,
        "customer_tenure_days": 400,
        "payment_method": "UPI",
        "ip_reputation_score": 0.6,
        "category_tag": "legitimate_but_unusual",
        "ground_truth_label": "legitimate"
    }
    contract = get_decision_contract(sample_transaction)
    print("SAMPLE DECISION CONTRACT OUTPUT:")
    print(json.dumps(contract, indent=2))
