"""
VulX Latency Benchmark Script (benchmark.py)

Measures real execution latencies across 100+ transaction evaluation runs:
  - Fast Decision Path (XGBoost + Bootstrap Uncertainty)
  - Policy Engine Evaluation
  - Execution Simulation & Ledger Recording
  - Async SHAP Explanation Path

Saves output metrics to models/benchmark_results.json and benchmark_results.json.
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from vulx.config import DEFAULT_OUTPUT_CSV, MODELS_DIR
from vulx.models.decision_contract import get_fast_decision, get_explanation
from vulx.policy_engine import evaluate
from vulx.execution_engine import execute
from vulx.ledger import record_event


def run_benchmark(n_runs=100, output_json=None):
    if output_json is None:
        output_json = str(MODELS_DIR / "benchmark_results.json")

    print("=" * 80)
    print(f" RUNNING VULX LATENCY BENCHMARK ({n_runs} RUNS)")
    print("=" * 80)

    # Load dataset sample
    if os.path.exists(DEFAULT_OUTPUT_CSV):
        df = pd.read_csv(DEFAULT_OUTPUT_CSV)
        sample_transactions = df.head(n_runs).to_dict(orient="records")
    else:
        sample_transactions = [
            {
                "transaction_id": f"bench_{i}",
                "amount": 1000.0 + (i * 100),
                "device_novelty": 0.5,
                "transaction_velocity": 2,
                "location_deviation": 0.3,
                "merchant_history_score": 0.1,
                "customer_tenure_days": 180,
                "payment_method": "UPI",
                "ip_reputation_score": 0.8,
            }
            for i in range(n_runs)
        ]

    fast_path_latencies = []
    policy_latencies = []
    execution_latencies = []
    shap_latencies = []
    total_latencies = []

    # Warmup
    _ = get_fast_decision(sample_transactions[0])
    _ = get_explanation(sample_transactions[0])

    for tx in sample_transactions:
        t0 = time.perf_counter()
        contract = get_fast_decision(tx)
        t1 = time.perf_counter()

        policy_res = evaluate(contract)
        t2 = time.perf_counter()

        exec_res = execute(policy_res["routing_decision"], contract)
        record_event({**contract, **policy_res, **exec_res, "ground_truth_label": tx.get("ground_truth_label", "legitimate")})
        t3 = time.perf_counter()

        # Decoupled explanation path
        _ = get_explanation(tx)
        t4 = time.perf_counter()

        fast_path_latencies.append((t1 - t0) * 1000)
        policy_latencies.append((t2 - t1) * 1000)
        execution_latencies.append((t3 - t2) * 1000)
        shap_latencies.append((t4 - t3) * 1000)
        total_latencies.append((t4 - t0) * 1000)

    fast_path_latencies = np.array(fast_path_latencies)
    policy_latencies = np.array(policy_latencies)
    execution_latencies = np.array(execution_latencies)
    shap_latencies = np.array(shap_latencies)
    total_latencies = np.array(total_latencies)

    results = {
        "n_runs": len(sample_transactions),
        "fast_decision_path_ms": {
            "p50": round(float(np.percentile(fast_path_latencies, 50)), 3),
            "p95": round(float(np.percentile(fast_path_latencies, 95)), 3),
            "p99": round(float(np.percentile(fast_path_latencies, 99)), 3),
            "mean": round(float(np.mean(fast_path_latencies)), 3),
        },
        "policy_engine_ms": {
            "p50": round(float(np.percentile(policy_latencies, 50)), 3),
            "p95": round(float(np.percentile(policy_latencies, 95)), 3),
            "p99": round(float(np.percentile(policy_latencies, 99)), 3),
            "mean": round(float(np.mean(policy_latencies)), 3),
        },
        "execution_and_ledger_ms": {
            "p50": round(float(np.percentile(execution_latencies, 50)), 3),
            "p95": round(float(np.percentile(execution_latencies, 95)), 3),
            "p99": round(float(np.percentile(execution_latencies, 99)), 3),
            "mean": round(float(np.mean(execution_latencies)), 3),
        },
        "async_shap_explanation_ms": {
            "p50": round(float(np.percentile(shap_latencies, 50)), 3),
            "p95": round(float(np.percentile(shap_latencies, 95)), 3),
            "p99": round(float(np.percentile(shap_latencies, 99)), 3),
            "mean": round(float(np.mean(shap_latencies)), 3),
        },
        "total_pipeline_ms": {
            "p50": round(float(np.percentile(total_latencies, 50)), 3),
            "p95": round(float(np.percentile(total_latencies, 95)), 3),
            "p99": round(float(np.percentile(total_latencies, 99)), 3),
            "mean": round(float(np.mean(total_latencies)), 3),
        },
        "raw_latencies_ms": total_latencies.tolist(),
    }

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(f"\n  [OK] Fast Decision Path p50:   {results['fast_decision_path_ms']['p50']:.3f} ms")
    print(f"  [OK] Policy Engine p50:        {results['policy_engine_ms']['p50']:.3f} ms")
    print(f"  [OK] Execution & Ledger p50:   {results['execution_and_ledger_ms']['p50']:.3f} ms")
    print(f"  [OK] Async SHAP Explanation p50: {results['async_shap_explanation_ms']['p50']:.3f} ms")
    print(f"  [OK] Total Fast Pipeline p50:  {(results['fast_decision_path_ms']['p50'] + results['policy_engine_ms']['p50']):.3f} ms")

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved benchmark metrics to '{output_json}' & 'benchmark_results.json'")
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    run_benchmark()
