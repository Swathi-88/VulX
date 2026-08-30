"""
Statistical distribution generators for VulX payment event attributes.
"""

from datetime import datetime, timedelta
import numpy as np


def generate_timestamps(rng, count, start_date=None):
    """Generates ISO 8601 timestamps spread over a 30-day window."""
    if start_date is None:
        end_dt = datetime(2026, 8, 23, 14, 0, 0)
        start_dt = end_dt - timedelta(days=30)
    else:
        start_dt = start_date
        end_dt = start_dt + timedelta(days=30)

    start_ts = start_dt.timestamp()
    end_ts = end_dt.timestamp()
    random_ts = rng.uniform(start_ts, end_ts, size=count)
    random_ts.sort()
    return [datetime.fromtimestamp(ts).isoformat() + "Z" for ts in random_ts]


def sample_payment_methods(rng, count):
    """Samples payment channels with realistic UPI dominance."""
    return rng.choice(
        ["UPI", "card", "netbanking", "wallet"],
        size=count,
        p=[0.60, 0.25, 0.10, 0.05],
    )
