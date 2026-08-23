# Anvil Synthetic Payment Event Dataset Schema

This document details the schema, statistical distributions, feature correlations, and category construction rules for the **Anvil** synthetic payment event dataset generator (`generate_events.py`).

---

## 1. Data Fields Reference Table

| Field Name | Data Type | Range / Domain | Description |
| :--- | :--- | :--- | :--- |
| `transaction_id` | String (UUIDv4) | Standard 36-char UUID | Unique identifier for each transaction event. |
| `timestamp` | String (ISO 8601) | Datetime string (UTC) | Event timestamp formatted as `YYYY-MM-DDTHH:MM:SSZ`. |
| `amount` | Float | ₹10.00 – ₹100,000.00 | Payment amount in Indian Rupees (INR). Log-normally distributed skewed toward small payments. |
| `device_novelty` | Float | 0.0000 – 1.0000 | Likelihood that the transaction originates from an unrecognized device (>0.7 indicates new/unrecognized device). |
| `transaction_velocity` | Integer | 0 – 10 | Count of transactions executed by this customer in the preceding 60 minutes. |
| `location_deviation` | Float | 0.0000 – 1.0000 | Normalized distance score from the customer's historical geolocation centroid (0=home/work, 1=unusual distant location). |
| `merchant_history_score` | Float | 0.0000 – 1.0000 | Historical dispute/chargeback rate of the recipient merchant (0.0=clean, 1.0=high risk/compromised). |
| `customer_tenure_days` | Integer | 1 – 1,500 days | Account age of the customer in days. |
| `payment_method` | Categorical | `UPI`, `card`, `netbanking`, `wallet` | Payment channel utilized for the transaction. |
| `ip_reputation_score` | Float | 0.0000 – 1.0000 | Reputation score of the originating IP address (1.0=clean residential IP, <0.4=proxy/VPN/tor/blacklisted). |
| `category_tag` | Categorical | `normal`, `suspicious`, `borderline`, `fraud`, `legitimate_but_unusual`, `merchant_anomaly` | Statistical distribution category tag used during generation. |
| `ground_truth_label` | Categorical | `legitimate`, `fraud` | Verified label for model evaluation. |

---

## 2. Feature Distributions & Statistical Properties

### Transaction Amount (`amount`)
- **Base Distribution**: Log-Normal distribution skewed toward small daily payments with a long right tail.
- **Log-normal parameters**: $\mu = 6.2$, $\sigma = 1.0$, producing a median of ~₹490 and 95th percentile under ₹8,500.
- **Unusual High-Value Overrides**: For high-value transactions (`legitimate_but_unusual` or `fraud`), amounts range from ₹25,000 to ₹100,000.

### Payment Method (`payment_method`)
Reflects realistic digital payments adoption in India:
- **UPI**: 60%
- **Card (Debit/Credit)**: 25%
- **Netbanking**: 10%
- **Wallet**: 5%

---

## 3. Category Construction & Correlation Rules

The synthetic generator distributes events across 6 distinct categories with realistic feature correlations:

### 1. `normal` (70% of dataset)
- **Concept**: Routine, everyday customer payments.
- **Feature Correlations**:
  - `device_novelty`: Low $[0.0, 0.25]$
  - `transaction_velocity`: Low ($0$ or $1$, $2$ rare)
  - `location_deviation`: Low $[0.0, 0.25]$
  - `merchant_history_score`: Low $[0.0, 0.12]$ (Beta distribution skewed near zero)
  - `ip_reputation_score`: Clean $[0.85, 1.0]$
- **Ground Truth**: `legitimate` (100%)

### 2. `suspicious` (8% of dataset)
- **Concept**: Account takeover or multi-vector attack patterns where all key user risk signals spike simultaneously.
- **Feature Correlations**:
  - `device_novelty` $> 0.72$ **AND** `transaction_velocity` $\ge 3$ **AND** `location_deviation` $> 0.71$ **together**.
  - `ip_reputation_score`: Moderate to low $[0.30, 0.75]$.
  - `customer_tenure_days`: Shorter $[5, 200]$.
- **Ground Truth**: `fraud` (~80%), `legitimate` (~20%).

### 3. `borderline` (10% of dataset)
- **Concept**: Ambiguous cases featuring **exactly ONE or TWO** elevated risk signals (e.g. new phone purchase, or travel velocity) but NOT all three. Designed to test model decision boundary discrimination.
- **Feature Correlations**:
  - 6 distinct sub-patterns swapping novelty, velocity, and deviation elevations.
  - `ip_reputation_score`: $[0.50, 0.90]$.
- **Ground Truth**: 50% `legitimate`, 50% `fraud` (deliberately hard to separate).

### 4. `fraud` (5% of dataset)
- **Concept**: Definite malicious fraud with full feature compromise.
- **Feature Correlations**:
  - `device_novelty` $> 0.75$ **AND** `transaction_velocity` $\ge 4$ **AND** `location_deviation` $> 0.75$.
  - `merchant_history_score`: Elevated $[0.25, 0.95]$.
  - `customer_tenure_days`: Very short $[1, 90]$.
  - `ip_reputation_score`: Poor $[0.05, 0.40]$.
- **Ground Truth**: `fraud` (100%).

### 5. `legitimate_but_unusual` (5% of dataset)
- **Concept**: False-positive showcase cases. Transaction appears surface-risky (high amount $\ge ₹25,000$ or high velocity $\ge 4$) but is legitimate due to contextual trust factors.
- **Feature Correlations**:
  - `customer_tenure_days`: Long $[180, 1500]$.
  - `ip_reputation_score`: High clean $[0.80, 1.0]$.
  - `merchant_history_score`: Clean $[0.0, 0.10]$.
- **Ground Truth**: `legitimate` (100%).

### 6. `merchant_anomaly` (2% of dataset)
- **Concept**: Customer acts completely normal, but the merchant receiving the payment exhibits suspicious dispute history.
- **Feature Correlations**:
  - Customer features (`device_novelty`, `transaction_velocity`, `location_deviation`, `ip_reputation_score`) are all clean/normal.
  - `merchant_history_score`: Spikes to $[0.35, 0.95]$.
- **Ground Truth**: 50% `legitimate`, 50% `fraud`.

---

## 4. Showcase Demo Cases (`demo_cases.json`)

To provide reproducible testing vectors for rule engines and ML prototypes, `generate_events.py` generates 8 fixed demo transactions in `demo_cases.json`.

### Key Showcase Scenarios:
1. **Showcase False Positive (`demo_fp_001_showcase`)**:
   - `amount`: 45,000.00 INR
   - `device_novelty`: 0.95
   - `transaction_velocity`: 4
   - `location_deviation`: 0.88
   - `merchant_history_score`: 0.05
   - `customer_tenure_days`: 400
   - `ip_reputation_score`: 0.60
   - `ground_truth_label`: `"legitimate"`
   - *Purpose*: Prove Anvil can evaluate customer tenure and merchant trust to avoid flagging legitimate high-value purchases.

2. **Showcase True Fraud Counterpart (`demo_fraud_001_counterpart`)**:
   - `amount`: 48,000.00 INR
   - `device_novelty`: 0.96
   - `transaction_velocity`: 5
   - `location_deviation`: 0.91
   - `merchant_history_score`: 0.28
   - `customer_tenure_days`: 12
   - `ip_reputation_score`: 0.15
   - `ground_truth_label`: `"fraud"`
   - *Purpose*: Prove the model is not merely pattern-matching on high amount/device novelty, but correctly factors in low tenure and dirty IP reputation.
