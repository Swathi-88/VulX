"""
Core event builder implementing the 6 category feature correlation rules.
"""

import uuid
import numpy as np
import pandas as pd
from vulx.generator.distributions import generate_timestamps, sample_payment_methods
from vulx.config import CATEGORY_RATIOS


def compute_category_counts(n_events):
    """Computes exact category allocations based on target percentages."""
    counts = {}
    allocated = 0
    keys = list(CATEGORY_RATIOS.keys())
    for cat in keys[:-1]:
        cnt = int(round(n_events * CATEGORY_RATIOS[cat]))
        counts[cat] = cnt
        allocated += cnt
    counts[keys[-1]] = max(0, n_events - allocated)
    return counts


def build_category_dataset(rng, cat, count):
    """Generates synthetic dataframe for a specific category."""
    if count <= 0:
        return pd.DataFrame()

    txn_ids = [str(uuid.UUID(bytes=bytes(rng.randint(0, 256, size=16, dtype=np.uint8)))) for _ in range(count)]
    timestamps = generate_timestamps(rng, count)
    payment_methods = sample_payment_methods(rng, count)

    if cat == "normal":
        device_novelty = rng.uniform(0.0, 0.25, size=count)
        transaction_velocity = rng.choice([0, 1, 2], size=count, p=[0.75, 0.20, 0.05])
        location_deviation = rng.uniform(0.0, 0.25, size=count)
        merchant_history_score = np.clip(rng.beta(1.0, 25.0, size=count), 0.0, 0.12)
        customer_tenure_days = rng.randint(15, 1200, size=count)
        ip_reputation_score = rng.uniform(0.85, 1.0, size=count)
        amounts = np.clip(rng.lognormal(mean=6.2, sigma=1.0, size=count), 15.0, 15000.0)
        labels = np.array(["legitimate"] * count)

    elif cat == "suspicious":
        device_novelty = rng.uniform(0.72, 0.99, size=count)
        transaction_velocity = rng.choice([3, 4, 5, 6, 7], size=count, p=[0.30, 0.30, 0.20, 0.10, 0.10])
        location_deviation = rng.uniform(0.71, 0.98, size=count)
        merchant_history_score = rng.uniform(0.02, 0.35, size=count)
        customer_tenure_days = rng.randint(5, 200, size=count)
        ip_reputation_score = rng.uniform(0.30, 0.75, size=count)
        amounts = np.clip(rng.lognormal(mean=7.5, sigma=1.2, size=count), 500.0, 45000.0)
        labels = rng.choice(["fraud", "legitimate"], size=count, p=[0.80, 0.20])

    elif cat == "borderline":
        device_novelty = np.zeros(count)
        transaction_velocity = np.zeros(count, dtype=int)
        location_deviation = np.zeros(count)
        patterns = rng.choice(6, size=count)
        for i in range(count):
            p = patterns[i]
            if p == 0:
                device_novelty[i] = rng.uniform(0.72, 0.96)
                transaction_velocity[i] = rng.choice([0, 1])
                location_deviation[i] = rng.uniform(0.0, 0.25)
            elif p == 1:
                device_novelty[i] = rng.uniform(0.0, 0.25)
                transaction_velocity[i] = rng.choice([0, 1])
                location_deviation[i] = rng.uniform(0.72, 0.96)
            elif p == 2:
                device_novelty[i] = rng.uniform(0.0, 0.25)
                transaction_velocity[i] = rng.choice([3, 4, 5])
                location_deviation[i] = rng.uniform(0.0, 0.25)
            elif p == 3:
                device_novelty[i] = rng.uniform(0.72, 0.95)
                transaction_velocity[i] = rng.choice([0, 1])
                location_deviation[i] = rng.uniform(0.72, 0.95)
            elif p == 4:
                device_novelty[i] = rng.uniform(0.72, 0.95)
                transaction_velocity[i] = rng.choice([3, 4, 5])
                location_deviation[i] = rng.uniform(0.0, 0.25)
            else:
                device_novelty[i] = rng.uniform(0.0, 0.25)
                transaction_velocity[i] = rng.choice([3, 4, 5])
                location_deviation[i] = rng.uniform(0.72, 0.95)

        merchant_history_score = rng.uniform(0.01, 0.20, size=count)
        customer_tenure_days = rng.randint(30, 600, size=count)
        ip_reputation_score = rng.uniform(0.50, 0.90, size=count)
        amounts = np.clip(rng.lognormal(mean=6.8, sigma=1.1, size=count), 100.0, 25000.0)
        labels = rng.choice(["legitimate", "fraud"], size=count, p=[0.50, 0.50])

    elif cat == "fraud":
        device_novelty = rng.uniform(0.75, 1.0, size=count)
        transaction_velocity = rng.choice([4, 5, 6, 7, 8, 9, 10], size=count, p=[0.20, 0.25, 0.20, 0.15, 0.10, 0.05, 0.05])
        location_deviation = rng.uniform(0.75, 1.0, size=count)
        merchant_history_score = rng.uniform(0.25, 0.95, size=count)
        customer_tenure_days = rng.randint(1, 90, size=count)
        ip_reputation_score = rng.uniform(0.05, 0.40, size=count)
        amounts = np.clip(rng.lognormal(mean=8.2, sigma=1.3, size=count), 1000.0, 85000.0)
        labels = np.array(["fraud"] * count)

    elif cat == "legitimate_but_unusual":
        customer_tenure_days = rng.randint(180, 1500, size=count)
        ip_reputation_score = rng.uniform(0.80, 1.0, size=count)
        merchant_history_score = rng.uniform(0.0, 0.10, size=count)
        location_deviation = rng.uniform(0.0, 0.40, size=count)
        is_high_amount = rng.choice([True, False], size=count, p=[0.5, 0.5])
        amounts = np.zeros(count)
        transaction_velocity = np.zeros(count, dtype=int)
        device_novelty = np.zeros(count)

        for i in range(count):
            if is_high_amount[i]:
                amounts[i] = rng.uniform(25000.0, 95000.0)
                transaction_velocity[i] = rng.choice([0, 1, 2])
                device_novelty[i] = rng.uniform(0.0, 0.85)
            else:
                amounts[i] = rng.uniform(300.0, 4500.0)
                transaction_velocity[i] = rng.choice([4, 5, 6])
                device_novelty[i] = rng.uniform(0.0, 0.30)
        labels = np.array(["legitimate"] * count)

    elif cat == "merchant_anomaly":
        device_novelty = rng.uniform(0.0, 0.25, size=count)
        transaction_velocity = rng.choice([0, 1], size=count, p=[0.80, 0.20])
        location_deviation = rng.uniform(0.0, 0.25, size=count)
        merchant_history_score = rng.uniform(0.35, 0.95, size=count)
        customer_tenure_days = rng.randint(60, 1000, size=count)
        ip_reputation_score = rng.uniform(0.75, 1.0, size=count)
        amounts = np.clip(rng.lognormal(mean=7.0, sigma=1.1, size=count), 200.0, 35000.0)
        labels = rng.choice(["legitimate", "fraud"], size=count, p=[0.50, 0.50])
    else:
        raise ValueError(f"Unknown category: {cat}")

    return pd.DataFrame({
        "transaction_id": txn_ids,
        "timestamp": timestamps,
        "amount": np.round(amounts, 2),
        "device_novelty": np.round(device_novelty, 4),
        "transaction_velocity": transaction_velocity,
        "location_deviation": np.round(location_deviation, 4),
        "merchant_history_score": np.round(merchant_history_score, 4),
        "customer_tenure_days": customer_tenure_days,
        "payment_method": payment_methods,
        "ip_reputation_score": np.round(ip_reputation_score, 4),
        "category_tag": [cat] * count,
        "ground_truth_label": labels,
    })


def generate_dataset(n_events, seed):
    """Generates full synthetic dataset given total event count and seed."""
    rng = np.random.RandomState(seed)
    counts = compute_category_counts(n_events)
    df_list = [build_category_dataset(rng, cat, cnt) for cat, cnt in counts.items() if cnt > 0]
    final_df = pd.concat(df_list, ignore_index=True)
    return final_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
