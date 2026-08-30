"""
Reproducible showcase test vectors for VulX rule engines & ML models.
"""


def get_showcase_demo_cases():
    """
    Returns 8 fixed, hand-crafted demo vectors reproducible across runs.
    Includes the mandatory false-positive showcase case and a matching true-fraud counterpart.
    """
    return [
        {
            "transaction_id": "demo_fp_001_showcase",
            "timestamp": "2026-08-23T10:15:30Z",
            "amount": 45000.0,
            "device_novelty": 0.95,
            "transaction_velocity": 4,
            "location_deviation": 0.88,
            "merchant_history_score": 0.05,
            "customer_tenure_days": 400,
            "payment_method": "UPI",
            "ip_reputation_score": 0.6,
            "category_tag": "legitimate_but_unusual",
            "ground_truth_label": "legitimate",
            "description": "Showcase False Positive: High amount & new device by long-tenure customer with clean merchant dispute history."
        },
        {
            "transaction_id": "demo_fraud_001_counterpart",
            "timestamp": "2026-08-23T10:18:45Z",
            "amount": 48000.0,
            "device_novelty": 0.96,
            "transaction_velocity": 5,
            "location_deviation": 0.91,
            "merchant_history_score": 0.28,
            "customer_tenure_days": 12,
            "payment_method": "card",
            "ip_reputation_score": 0.15,
            "category_tag": "fraud",
            "ground_truth_label": "fraud",
            "description": "Showcase True Fraud Counterpart: Similar surface features (amount, device, velocity, deviation) but brand-new account on dirty IP."
        },
        {
            "transaction_id": "demo_norm_001",
            "timestamp": "2026-08-23T08:30:12Z",
            "amount": 350.0,
            "device_novelty": 0.05,
            "transaction_velocity": 0,
            "location_deviation": 0.04,
            "merchant_history_score": 0.02,
            "customer_tenure_days": 620,
            "payment_method": "UPI",
            "ip_reputation_score": 0.98,
            "category_tag": "normal",
            "ground_truth_label": "legitimate",
            "description": "Standard Routine Transaction: Low risk across all feature dimensions."
        },
        {
            "transaction_id": "demo_susp_001",
            "timestamp": "2026-08-23T09:12:00Z",
            "amount": 12500.0,
            "device_novelty": 0.89,
            "transaction_velocity": 6,
            "location_deviation": 0.93,
            "merchant_history_score": 0.08,
            "customer_tenure_days": 45,
            "payment_method": "netbanking",
            "ip_reputation_score": 0.32,
            "category_tag": "suspicious",
            "ground_truth_label": "fraud",
            "description": "Account Takeover Pattern: Simultaneous device change, velocity spike, and geo-jump."
        },
        {
            "transaction_id": "demo_bord_001",
            "timestamp": "2026-08-23T11:05:18Z",
            "amount": 2100.0,
            "device_novelty": 0.82,
            "transaction_velocity": 1,
            "location_deviation": 0.12,
            "merchant_history_score": 0.04,
            "customer_tenure_days": 210,
            "payment_method": "card",
            "ip_reputation_score": 0.75,
            "category_tag": "borderline",
            "ground_truth_label": "legitimate",
            "description": "Borderline Legitimate: Single isolated signal (device upgrade) with normal velocity and location."
        },
        {
            "transaction_id": "demo_bord_002",
            "timestamp": "2026-08-23T11:40:22Z",
            "amount": 1800.0,
            "device_novelty": 0.10,
            "transaction_velocity": 4,
            "location_deviation": 0.08,
            "merchant_history_score": 0.06,
            "customer_tenure_days": 90,
            "payment_method": "UPI",
            "ip_reputation_score": 0.65,
            "category_tag": "borderline",
            "ground_truth_label": "fraud",
            "description": "Borderline Fraud: Single signal (velocity burst) testing card/UPI limits on existing device."
        },
        {
            "transaction_id": "demo_merch_001",
            "timestamp": "2026-08-23T12:20:00Z",
            "amount": 7500.0,
            "device_novelty": 0.08,
            "transaction_velocity": 0,
            "location_deviation": 0.05,
            "merchant_history_score": 0.78,
            "customer_tenure_days": 550,
            "payment_method": "wallet",
            "ip_reputation_score": 0.92,
            "category_tag": "merchant_anomaly",
            "ground_truth_label": "fraud",
            "description": "Merchant Anomaly Fraud: Trusted customer transacting at a compromised/high-dispute merchant site."
        },
        {
            "transaction_id": "demo_legit_unusual_002",
            "timestamp": "2026-08-23T13:10:05Z",
            "amount": 1200.0,
            "device_novelty": 0.02,
            "transaction_velocity": 5,
            "location_deviation": 0.03,
            "merchant_history_score": 0.01,
            "customer_tenure_days": 850,
            "payment_method": "UPI",
            "ip_reputation_score": 0.99,
            "category_tag": "legitimate_but_unusual",
            "ground_truth_label": "legitimate",
            "description": "High Velocity Legitimate: Rapid micro-payments by long-standing power user at local market/event."
        }
    ]
