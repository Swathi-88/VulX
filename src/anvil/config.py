"""
Configuration parameters and default file paths for Anvil Fraud Detection System
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCS_DIR = BASE_DIR / "docs"
MODELS_DIR = BASE_DIR / "models"
MODEL_ENSEMBLE_DIR = MODELS_DIR / "model_ensemble"

# Ensure canonical directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_N_EVENTS = int(os.getenv("ANVIL_N_EVENTS", "2000"))
DEFAULT_SEED = int(os.getenv("ANVIL_SEED", "42"))

# Canonical default file paths
DEFAULT_OUTPUT_CSV = str(RAW_DATA_DIR / os.getenv("ANVIL_OUTPUT_CSV", "events.csv"))
DEFAULT_DEMO_JSON = str(PROCESSED_DATA_DIR / os.getenv("ANVIL_DEMO_JSON", "demo_cases.json"))
DEFAULT_MODEL_PATH = str(MODELS_DIR / "model.pkl")
DEFAULT_METRICS_PATH = str(MODELS_DIR / "metrics.json")
DEFAULT_ENSEMBLE_DIR = str(MODEL_ENSEMBLE_DIR)

CATEGORY_RATIOS = {
    "normal": 0.70,
    "suspicious": 0.08,
    "borderline": 0.10,
    "fraud": 0.05,
    "legitimate_but_unusual": 0.05,
    "merchant_anomaly": 0.02,
}
