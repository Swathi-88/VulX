"""
Feature Preprocessor for Vulcan Decision Simulator (Anvil Phase 2)
"""

import pandas as pd

FEATURE_COLUMNS = [
    "amount",
    "device_novelty",
    "transaction_velocity",
    "location_deviation",
    "merchant_history_score",
    "customer_tenure_days",
    "ip_reputation_score",
    "payment_method_UPI",
    "payment_method_card",
    "payment_method_netbanking",
    "payment_method_wallet",
]

PAYMENT_METHODS = ["UPI", "card", "netbanking", "wallet"]


def preprocess_features(data):
    """
    Transforms raw input (dict, list of dicts, or DataFrame) into a standardized
    feature DataFrame X matching trained model columns.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    X = pd.DataFrame()
    X["amount"] = df["amount"].astype(float)
    X["device_novelty"] = df["device_novelty"].astype(float)
    X["transaction_velocity"] = df["transaction_velocity"].astype(float)
    X["location_deviation"] = df["location_deviation"].astype(float)
    X["merchant_history_score"] = df["merchant_history_score"].astype(float)
    X["customer_tenure_days"] = df["customer_tenure_days"].astype(float)
    X["ip_reputation_score"] = df["ip_reputation_score"].astype(float)

    # One-hot encode payment methods
    for method in PAYMENT_METHODS:
        col_name = f"payment_method_{method}"
        X[col_name] = (df["payment_method"] == method).astype(int)

    return X[FEATURE_COLUMNS]
