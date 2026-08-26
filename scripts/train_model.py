#!/usr/bin/env python3
"""
Train Vulcan Fraud Classifier & Bootstrap Ensemble (Anvil Phase 2)
------------------------------------------------------------------
Trains XGBoost model on events.csv, saves primary model to models/model.pkl,
saves metrics to models/metrics.json, and saves 5 ensemble models to models/model_ensemble/*.pkl.
"""

import sys
from pathlib import Path

# Add src/ directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anvil.config import (
    DEFAULT_OUTPUT_CSV,
    DEFAULT_MODEL_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_ENSEMBLE_DIR,
)
from anvil.models.train import train_and_evaluate


def main():
    print("=" * 80)
    print(" VULCAN DECISION SIMULATOR - MODEL TRAINING")
    print("=" * 80)
    metrics, _ = train_and_evaluate(
        events_csv=DEFAULT_OUTPUT_CSV,
        model_path=DEFAULT_MODEL_PATH,
        metrics_path=DEFAULT_METRICS_PATH,
        ensemble_dir=DEFAULT_ENSEMBLE_DIR,
    )

    print("\nMODEL EVALUATION METRICS (Held-out 20% Test Set):")
    print("-" * 50)
    print(f"  Precision       : {metrics['precision']:.4f}")
    print(f"  Recall          : {metrics['recall']:.4f}")
    print(f"  F1-Score        : {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC         : {metrics['auc_roc']:.4f}")
    print("  Confusion Matrix:")
    print(f"    TN: {metrics['confusion_matrix'][0][0]} | FP: {metrics['confusion_matrix'][0][1]}")
    print(f"    FN: {metrics['confusion_matrix'][1][0]} | TP: {metrics['confusion_matrix'][1][1]}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
