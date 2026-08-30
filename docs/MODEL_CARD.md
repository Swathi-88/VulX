# Vulcan Decision Simulator - Model Card

This Model Card details the architecture, feature pipeline, evaluation metrics, explainability mechanisms, and uncertainty estimation methods for the **Vulcan Decision Simulator** (VulX Phase 2 tabular fraud model).

---

## 1. Model Overview & Architecture

- **Model Type**: Gradient Boosted Decision Tree Classifier (`XGBClassifier`)
- **Library**: `xgboost` v3.2.0
- **Hyperparameters**:
  - `n_estimators`: 100
  - `max_depth`: 4
  - `learning_rate`: 0.1
  - `eval_metric`: `logloss`
  - `random_state`: 42
- **Artifact Location**: [`model.pkl`](file:///d:/razorpay/model.pkl)
- **Bootstrap Ensemble Artifacts**: [`model_ensemble/model_0.pkl`](file:///d:/razorpay/model_ensemble/model_0.pkl) through `model_4.pkl`

---

## 2. Feature Schema & Inputs

The model evaluates 11 input features derived from the payment event dataset:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `amount` | Float | Payment amount in INR (log-normal distribution). |
| `device_novelty` | Float | Device unrecognized score ($0.0 - 1.0$, $>0.7$ = new device). |
| `transaction_velocity` | Float / Int | Customer transactions executed in last 60 minutes. |
| `location_deviation` | Float | Distance from historical location centroids ($0.0 - 1.0$). |
| `merchant_history_score` | Float | Merchant historical dispute/chargeback rate ($0.0 - 1.0$). |
| `customer_tenure_days` | Float / Int | Account age in days. |
| `ip_reputation_score` | Float | IP address cleanliness score ($1.0$ = clean, $<0.4$ = dirty/proxy). |
| `payment_method_UPI` | Binary | 1 if payment method is UPI, else 0. |
| `payment_method_card` | Binary | 1 if payment method is Card, else 0. |
| `payment_method_netbanking` | Binary | 1 if payment method is Netbanking, else 0. |
| `payment_method_wallet` | Binary | 1 if payment method is Wallet, else 0. |

---

## 3. Held-Out Evaluation Metrics

The dataset (`events.csv`, 2,000 records) was partitioned using a **80/20 train/test stratified split**:
- **Training Set**: 1,600 transactions
- **Test Set**: 400 transactions

Saved metrics artifact: [`metrics.json`](file:///d:/razorpay/metrics.json)

| Metric | Score |
| :--- | :--- |
| **Precision** | `0.7284` |
| **Recall** | `0.8551` |
| **F1-Score** | `0.7867` |
| **AUC-ROC** | `0.9711` |

### Confusion Matrix (Held-out Test Set)

```
                     Predicted Legitimate    Predicted Fraud
Actual Legitimate            309 (TN)               22 (FP)
Actual Fraud                  10 (FN)               59 (TP)
```

---

## 4. Explainability & Standard Decision Contract

The `decision_contract.py` module computes a **Standard Decision Contract** for any transaction event using `shap.TreeExplainer`:

```json
{
  "transaction_id": "demo_fp_001_showcase",
  "risk_probability": 0.7818,
  "prediction_uncertainty": {
    "std_dev": 0.2242,
    "uncertainty_level": "high"
  },
  "top_contributing_signals": [
    { "feature": "ip_reputation_score", "contribution": 1.4671 },
    { "feature": "device_novelty", "contribution": 1.3114 },
    { "feature": "location_deviation", "contribution": 0.7509 }
  ],
  "naive_recommended_action": "BLOCK"
}
```

### Naive Action Threshold Rules:
- `risk_probability > 0.70` $\rightarrow$ **`BLOCK`**
- `risk_probability < 0.30` $\rightarrow$ **`ALLOW`**
- $0.30 \le \text{risk\_probability} \le 0.70$ $\rightarrow$ **`REVIEW`**

---

## 5. Uncertainty Estimation Methodology (Honest Proxy Notice)

> [!IMPORTANT]
> **Bootstrap Ensemble Uncertainty Proxy**:
> Vulcan computes prediction uncertainty by taking the **standard deviation** ($\sigma$) across 5 bootstrap-resampled XGBoost models (`model_ensemble/model_0.pkl` ... `model_4.pkl`) trained on resampled subsets of the training data.
> 
> - **Uncertainty Bucketing**:
>   - $\sigma < 0.05$: `"low"`
>   - $0.05 \le \sigma < 0.15$: `"medium"`
>   - $\sigma \ge 0.15$: `"high"`
> 
> *Honesty Disclaimer*: This bootstrap ensemble standard deviation serves as a computationally efficient **frequentist epistemic uncertainty proxy**. It measures data sampling variance across trees, but is **not a true Bayesian posterior uncertainty** (such as standard deviation from MC Dropout or Bayesian Neural Networks).

---

## 6. Killer Showcase Case Analysis

- **False-Positive Showcase (`demo_fp_001_showcase`)**:
  - Amount: INR 45,000 | Tenure: 400 days | Device Novelty: 0.95 | Velocity: 4
  - **Model Output**: `risk_probability = 0.7818` $\rightarrow$ Naive Action: **`BLOCK`**
  - *Significance*: Proves Vulcan's tabular model naively flags high-value transactions with new devices despite long customer tenure. This sets up VulX's Phase 3 intelligent override mechanism.

- **True-Fraud Counterpart (`demo_fraud_001_counterpart`)**:
  - Amount: INR 48,000 | Tenure: 12 days | Device Novelty: 0.96 | Velocity: 5 | IP Rep: 0.15
  - **Model Output**: `risk_probability = 0.9678` $\rightarrow$ Naive Action: **`BLOCK`**
  - *Significance*: Confirms high true-positive detection when tenure is short and IP reputation is dirty.
