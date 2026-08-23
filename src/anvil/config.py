"""
Configuration parameters for Anvil Fraud Detection Prototype
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"

DEFAULT_N_EVENTS = int(os.getenv("ANVIL_N_EVENTS", "2000"))
DEFAULT_SEED = int(os.getenv("ANVIL_SEED", "42"))
DEFAULT_OUTPUT_CSV = str(BASE_DIR / os.getenv("ANVIL_OUTPUT_CSV", "events.csv"))
DEFAULT_DEMO_JSON = str(BASE_DIR / os.getenv("ANVIL_DEMO_JSON", "demo_cases.json"))

CATEGORY_RATIOS = {
    "normal": 0.70,
    "suspicious": 0.08,
    "borderline": 0.10,
    "fraud": 0.05,
    "legitimate_but_unusual": 0.05,
    "merchant_anomaly": 0.02,
}
