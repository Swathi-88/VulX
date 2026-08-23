# Anvil - Fraud Detection Synthetic Event Generator

Anvil is a synthetic payment event generator designed to benchmark, evaluate, and test real-time fraud detection prototypes and rule engines. It generates payment events with correlated risk signals, 6 distinct statistical category distributions, realistic log-normal payment amount distributions, and reproducible showcase test vectors.

---

## 📁 Repository Folder Structure

```
.
├── .env                      # Environment configuration variables
├── .env.example              # Template environment configuration file
├── .gitignore                # Version control exclusions
├── README.md                 # Project documentation & execution guide
├── SCHEMA.md                 # Complete dataset schema reference
├── requirements.txt          # Python dependencies
├── generate_events.py        # Main CLI entrypoint script
├── demo_cases.json           # Fixed, hand-crafted showcase test vectors (8 cases)
├── events.csv                # Generated synthetic payment event dataset (2000 events)
│
├── config/                   # Configuration settings
│   └── settings.py
│
├── data/                     # Output datasets & raw data directory
│   ├── raw/
│   └── processed/
│
├── docs/                     # Detailed technical documentation
│   └── SCHEMA.md             # Dataset schema & correlation specification
│
├── src/                      # Anvil Python Package
│   └── anvil/
│       ├── __init__.py
│       ├── config.py         # Global project configuration
│       ├── generator/        # Event generation & category distribution logic
│       │   ├── __init__.py
│       │   ├── distributions.py
│       │   ├── event_builder.py
│       │   └── showcase.py
│       └── utils/            # Terminal formatting & output helpers
│           ├── __init__.py
│           └── output_formatter.py
│
└── tests/                    # Automated PyTest test suite
    ├── __init__.py
    └── test_generator.py
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Events
Run generator with default settings (2,000 events, seed 42):
```bash
python generate_events.py
```

Run with custom CLI arguments:
```bash
python generate_events.py --n_events 5000 --seed 100 --output data/events_5k.csv --demo_output data/demo_cases_5k.json
```

### 3. Run Automated Tests
```bash
pytest tests/
```

---

## 📊 Event Categories & Feature Correlations

| Category | Dataset % | Key Feature Correlations | Ground Truth |
| :--- | :--- | :--- | :--- |
| `normal` | 70% | Low novelty, velocity, deviation; clean IP | 100% Legitimate |
| `suspicious` | 8% | High novelty + velocity + location deviation together | ~80% Fraud / 20% Legitimate |
| `borderline` | 10% | Exactly 1 or 2 elevated risk signals (e.g. device upgrade) | 50% Legitimate / 50% Fraud |
| `fraud` | 5% | All risk signals elevated + dirty IP + short tenure | 100% Fraud |
| `legitimate_but_unusual` | 5% | High amount or velocity, but long tenure & clean IP | 100% Legitimate |
| `merchant_anomaly` | 2% | Normal customer behavior, elevated merchant dispute rate | 50% Legitimate / 50% Fraud |

---

## 🔬 Showcase Test Vectors (`demo_cases.json`)

Includes 8 fixed test vectors reproducible across runs:
- **False Positive Showcase**: ₹45,000.00 UPI payment with a new device (`device_novelty=0.95`) and high velocity (`4`), validated by long customer tenure (`400 days`), clean merchant history (`0.05`), and `legitimate` ground truth.
- **True Fraud Counterpart**: ₹48,000.00 card payment with new device (`0.96`), high velocity (`5`), brand-new account (`12 days`), dirty IP (`0.15`), and `fraud` ground truth.
