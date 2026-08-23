"""
Decision Contract & SHAP Explainability Engine for Vulcan Decision Simulator
"""

import os
import pickle
import numpy as np
import shap
import shap.explainers._tree as _shap_tree
from anvil.config import DEFAULT_MODEL_PATH, DEFAULT_ENSEMBLE_DIR
from anvil.models.preprocessor import preprocess_features

# Apply monkeypatch for XGBoost 3.x + SHAP 0.49 base_score UBJSON string compatibility
_orig_decode_ubjson = _shap_tree.decode_ubjson_buffer

def _patched_decode_ubjson(buffer):
    res = _orig_decode_ubjson(buffer)
    try:
        if isinstance(res, dict) and "learner" in res:
            p = res["learner"].get("learner_model_param", {})
            if "base_score" in p and isinstance(p["base_score"], str):
                p["base_score"] = float(p["base_score"].strip("[]"))
    except Exception:
        pass
    return res

_shap_tree.decode_ubjson_buffer = _patched_decode_ubjson


_PRIMARY_MODEL = None
_ENSEMBLE_MODELS = None
_SHAP_EXPLAINER = None


def load_models(model_path=None, ensemble_dir=None):
    """Loads and caches primary and bootstrap ensemble models."""
    global _PRIMARY_MODEL, _ENSEMBLE_MODELS, _SHAP_EXPLAINER

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if ensemble_dir is None:
        ensemble_dir = DEFAULT_ENSEMBLE_DIR

    if _PRIMARY_MODEL is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Primary model file '{model_path}' not found. Please run train_model.py first.")
        with open(model_path, "rb") as f:
            _PRIMARY_MODEL = pickle.load(f)
        _SHAP_EXPLAINER = shap.TreeExplainer(_PRIMARY_MODEL)

    if _ENSEMBLE_MODELS is None:
        _ENSEMBLE_MODELS = []
        if os.path.exists(ensemble_dir):
            for i in range(5):
                ens_path = os.path.join(ensemble_dir, f"model_{i}.pkl")
                if os.path.exists(ens_path):
                    with open(ens_path, "rb") as f:
                        _ENSEMBLE_MODELS.append(pickle.load(f))

    return _PRIMARY_MODEL, _ENSEMBLE_MODELS, _SHAP_EXPLAINER


def get_decision_contract(transaction: dict, model_path=None, ensemble_dir=None) -> dict:
    """
    Computes standard decision contract for a transaction event:
      - Risk probability via primary XGBoost model
      - Prediction uncertainty via standard deviation of 5 bootstrap ensemble models
      - Top 3 SHAP feature contributions
      - Naive recommended action (BLOCK / REVIEW / ALLOW)
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if ensemble_dir is None:
        ensemble_dir = DEFAULT_ENSEMBLE_DIR

    model, ensemble_models, explainer = load_models(model_path, ensemble_dir)

    # Preprocess transaction into 1-row DataFrame X
    X = preprocess_features(transaction)

    # 1. Primary Risk Probability
    risk_prob = float(model.predict_proba(X)[0][1])

    # 2. Bootstrap Ensemble Uncertainty
    if ensemble_models:
        ens_probs = [float(m.predict_proba(X)[0][1]) for m in ensemble_models]
        std_dev = float(np.std(ens_probs))
    else:
        std_dev = 0.0

    if std_dev < 0.05:
        uncertainty_level = "low"
    elif std_dev < 0.15:
        uncertainty_level = "medium"
    else:
        uncertainty_level = "high"

    # 3. Real SHAP Feature Contributions using shap_values
    raw_shap = explainer.shap_values(X)

    # Handle shape formats across SHAP versions (list for binary, or 2D array)
    if isinstance(raw_shap, list):
        vals = raw_shap[1][0]
    elif len(raw_shap.shape) == 3:  # (1, n_features, 2)
        vals = raw_shap[0, :, 1]
    elif len(raw_shap.shape) == 2:  # (1, n_features)
        vals = raw_shap[0]
    else:
        vals = np.array(raw_shap).flatten()

    feature_names = list(X.columns)
    shap_pairs = list(zip(feature_names, vals))

    # Sort by absolute SHAP contribution descending
    shap_pairs.sort(key=lambda item: abs(item[1]), reverse=True)
    top_3_signals = [
        {"feature": name, "contribution": float(round(contrib, 4))}
        for name, contrib in shap_pairs[:3]
    ]

    # 4. Naive Recommended Action Thresholds
    if risk_prob > 0.7:
        action = "BLOCK"
    elif risk_prob < 0.3:
        action = "ALLOW"
    else:
        action = "REVIEW"

    # 5. Standard Decision Contract
    contract = {
        "transaction_id": transaction.get("transaction_id", "unknown_id"),
        "risk_probability": float(round(risk_prob, 4)),
        "prediction_uncertainty": {
            "std_dev": float(round(std_dev, 4)),
            "uncertainty_level": uncertainty_level,
        },
        "top_contributing_signals": top_3_signals,
        "naive_recommended_action": action,
    }

    return contract
