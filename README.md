# VulX — Policy-as-Code Fraud Decision & Governance Engine

VulX is a policy-as-code routing and audit layer for payment fraud detection systems that bridges raw machine learning risk scores and real-world business execution. It takes Standard Decision Contracts containing risk probabilities, prediction uncertainty, and SHAP explainability signals, and evaluates them against human-editable YAML policy rules. By dynamically routing high-risk/high-false-positive transactions to step-up verification (`VERIFY`) or analyst review (`HUMAN_REVIEW`) instead of silent auto-blocking, VulX prevents costly false positives while maintaining rigorous audit compliance in SQLite.

---

## Quick Start & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Main End-to-End Pipeline
Run the full 4-phase integrated pipeline over the showcase demo vectors:
```bash
python run_pipeline.py
```

### 3. Run Policy Engine & Demo Case Showcase
Run the policy engine decision evaluation and live YAML modification demo:
```bash
python test_policy_engine.py
```

### 4. Continuous Feedback & Model Retraining
Extract false-positive feedback from the SQLite ledger and retrain XGBoost:
```bash
python feedback_pipeline.py
python retrain_and_compare.py
```

### 5. View System Metrics Dashboard
Compute system-wide routing distributions, false-positive prevention stats, and governed vs naive model metrics:
```bash
python metrics_dashboard.py
```

### 6. Interactive Analyst Review Console
Launch the interactive CLI analyst review queue for live pitch demos:
```bash
python human_review_cli.py
```

### 7. Interactive Streamlit Web UI (Optional)
Launch the visual web dashboard:
```bash
streamlit run app.py
```

---

## Repository Structure & Documentation

```
.
├── policies.yaml             # Human-readable policy-as-code rules configuration
├── run_pipeline.py           # Main end-to-end demo execution pipeline (Phases 1-4)
├── policy_engine.py          # Policy engine evaluation module & re-export
├── execution_engine.py       # Verification simulation & human review execution
├── ledger.py                 # SQLite audit ledger layer & compliance tagging
├── feedback_pipeline.py     # Continuous feedback extraction (FP/FN -> correction_increment.csv)
├── retrain_and_compare.py    # XGBoost model retraining & before/after performance comparison
├── metrics_dashboard.py      # System metrics dashboard (VulX governed system vs raw model)
├── human_review_cli.py       # Interactive analyst CLI console for live pitch demo
├── app.py                    # Streamlit visual presentation dashboard
├── ARCHITECTURE.md           # System architecture diagram & stage-by-stage documentation
├── DEMO_SCRIPT.md            # Timed 5-minute pitch video walkthrough script
│
├── data/
│   ├── raw/events.csv        # Raw synthetic payment events dataset (2,000 events)
│   └── processed/
│       ├── demo_cases.json   # Fixed showcase demo test vectors (8 cases)
│       └── vulx_ledger.db   # Persistent SQLite audit ledger database
│
├── models/
│   ├── model.pkl             # Trained primary XGBoost classifier
│   ├── metrics.json          # Baseline model performance metrics
│   ├── model_ensemble/       # 5 bootstrap ensemble models for uncertainty estimation
│   └── retrain_comparison.json # Retraining comparison metrics JSON
│
└── tests/                    # Automated PyTest unit & integration test suite (17 tests)
    ├── test_generator.py
    ├── test_demo_cases.py
    ├── test_policy_engine.py
    ├── test_pipeline.py
    └── test_feedback_and_metrics.py
```

---

## Running Automated Tests

Run the complete PyTest test suite:
```bash
pytest tests/
```
