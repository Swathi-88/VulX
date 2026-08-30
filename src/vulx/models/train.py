"""
Model Training & Bootstrap Ensemble Pipeline for Vulcan Decision Simulator
"""

import json
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import resample
from vulx.models.preprocessor import preprocess_features


def train_and_evaluate(events_csv="events.csv", model_path="model.pkl", metrics_path="metrics.json", ensemble_dir="model_ensemble"):
    """
    Loads dataset, trains primary XGBoost classifier, computes test set metrics,
    and trains 5 bootstrap ensemble models for uncertainty estimation.
    """
    print(f"Loading dataset from '{events_csv}'...")
    df = pd.read_csv(events_csv)

    X = preprocess_features(df)
    y = (df["ground_truth_label"] == "fraud").astype(int)

    # 80/20 train/test stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Dataset split: Train ({len(X_train)} samples), Test ({len(X_test)} samples)")

    # Train primary XGBoost Classifier
    primary_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    primary_model.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = primary_model.predict(X_test)
    y_prob = primary_model.predict_proba(X_test)[:, 1]

    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "auc_roc": round(float(auc), 4),
        "confusion_matrix": cm,
        "test_samples": len(y_test),
        "test_fraud_count": sum(y_test),
    }

    # Save primary model
    with open(model_path, "wb") as f:
        pickle.dump(primary_model, f)
    print(f"Saved primary model to '{model_path}'.")

    # Save metrics JSON
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to '{metrics_path}'.")

    # Train 5 Bootstrap Ensemble models
    os.makedirs(ensemble_dir, exist_ok=True)
    print(f"Training 5 bootstrap ensemble models in '{ensemble_dir}/'...")

    for i in range(5):
        seed = 42 + (i + 1) * 10
        res = resample(
            X_train, y_train, replace=True, random_state=seed
        )
        assert res is not None
        X_resampled, y_resampled = res

        ens_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=seed,
            eval_metric="logloss",
        )
        ens_model.fit(X_resampled, y_resampled)

        ens_file = os.path.join(ensemble_dir, f"model_{i}.pkl")
        with open(ens_file, "wb") as f:
            pickle.dump(ens_model, f)

    print("Bootstrap ensemble training complete.")
    return metrics, primary_model
