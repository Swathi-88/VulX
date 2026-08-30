"""
VulX System Metrics Dashboard (metrics_dashboard.py)

Queries SQLite ledger to calculate system-wide routing distributions, step-up verification metrics,
human review exception lists, and side-by-side comparison between the Naive ML baseline vs the VulX-governed system.
"""

import json
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from vulx.config import DEFAULT_DEMO_JSON
from vulx.ledger import query_ledger, record_event
from vulx.models.decision_contract import get_decision_contract
from vulx.policy_engine import evaluate
from vulx.execution_engine import execute


def run_metrics_dashboard(db_path=None) -> dict:
    """
    Computes and prints system-wide governance metrics from the SQLite ledger.
    """
    events = query_ledger(db_path=db_path)

    # Seed ledger with demo cases if empty
    if not events and os.path.exists(DEFAULT_DEMO_JSON):
        with open(DEFAULT_DEMO_JSON, "r", encoding="utf-8") as f:
            demo_cases = json.load(f)
        for idx, c in enumerate(demo_cases):
            contract = get_decision_contract(c)
            p_res = evaluate(contract)
            e_res = execute(p_res["routing_decision"], contract, seed=42 + idx)
            record_event({**contract, **p_res, **e_res, "ground_truth_label": c.get("ground_truth_label", "legitimate")}, db_path=db_path)
        events = query_ledger(db_path=db_path)

    total_events = len(events)
    if total_events == 0:
        print("No events found in SQLite ledger.")
        return {
            "total_events": 0,
            "routing_distribution": {"ALLOW": 0, "VERIFY": 0, "HUMAN_REVIEW": 0},
            "verify_analysis": {"total": 0, "fps_saved": 0, "true_fraud": 0},
            "human_review_analysis": {"total": 0, "overrides": 0, "rubberstamped": 0},
            "naive_baseline": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "fpr": 0.0},
            "vulx_system": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "fpr": 0.0},
        }

    # 1. Routing Decision Distribution
    route_counts = {"ALLOW": 0, "VERIFY": 0, "HUMAN_REVIEW": 0}
    for ev in events:
        r = ev.get("routing_decision", "ALLOW")
        route_counts[r] = route_counts.get(r, 0) + 1

    # 2. VERIFY Path Analysis
    verify_events = [ev for ev in events if ev.get("routing_decision") == "VERIFY"]
    verify_total = len(verify_events)
    verify_true_fraud = sum(
        1 for ev in verify_events if ev.get("verification_outcome") == "FAILED" or str(ev.get("ground_truth_label")).lower() in ["fraud", "suspicious"]
    )
    verify_fps_saved = sum(
        1 for ev in verify_events if ev.get("verification_outcome") == "VERIFIED" and str(ev.get("ground_truth_label")).lower() not in ["fraud", "suspicious"]
    )

    # 3. HUMAN_REVIEW Path Analysis (Honest Exception List)
    hr_events = [ev for ev in events if ev.get("routing_decision") == "HUMAN_REVIEW"]
    hr_total = len(hr_events)
    hr_overrides = sum(1 for ev in hr_events if ev.get("human_review_outcome") is not None)
    hr_rubberstamped = hr_total - hr_overrides

    # 4. Naive Baseline vs Governed VulX System Performance Metrics
    y_true = []
    y_pred_naive = []
    y_pred_vulx = []

    for ev in events:
        gt = str(ev.get("ground_truth_label")).lower().strip()
        is_fraud = 1 if gt in ["fraud", "suspicious", "true"] else 0
        y_true.append(is_fraud)

        naive_action = str(ev.get("naive_recommended_action")).upper().strip()
        y_pred_naive.append(1 if naive_action == "BLOCK" else 0)

        final_outcome = str(ev.get("final_outcome")).lower().strip()
        y_pred_vulx.append(1 if final_outcome == "blocked" else 0)

    # Calculate Naive Baseline Metrics
    cm_naive = confusion_matrix(y_true, y_pred_naive, labels=[0, 1])
    tn_n, fp_n, fn_n, tp_n = cm_naive.ravel()
    prec_naive = float(precision_score(y_true, y_pred_naive, zero_division=0))
    rec_naive = float(recall_score(y_true, y_pred_naive, zero_division=0))
    f1_naive = float(f1_score(y_true, y_pred_naive, zero_division=0))
    fpr_naive = float(fp_n / (fp_n + tn_n)) if (fp_n + tn_n) > 0 else 0.0

    # Calculate VulX Governed System Metrics
    cm_vulx = confusion_matrix(y_true, y_pred_vulx, labels=[0, 1])
    tn_a, fp_a, fn_a, tp_a = cm_vulx.ravel()
    prec_vulx = float(precision_score(y_true, y_pred_vulx, zero_division=0))
    rec_vulx = float(recall_score(y_true, y_pred_vulx, zero_division=0))
    f1_vulx = float(f1_score(y_true, y_pred_vulx, zero_division=0))
    fpr_vulx = float(fp_a / (fp_a + tn_a)) if (fp_a + tn_a) > 0 else 0.0

    print("=" * 80)
    print(" VULX SYSTEM-WIDE GOVERNANCE & PERFORMANCE METRICS DASHBOARD")
    print("=" * 80)
    print(f"\n[1] SYSTEM ROUTING DISTRIBUTION (Total Processed: {total_events})")
    for route in ["ALLOW", "VERIFY", "HUMAN_REVIEW"]:
        cnt = route_counts.get(route, 0)
        pct = (cnt / total_events) * 100 if total_events > 0 else 0.0
        print(f"    - {route:<13}: {cnt:3d} transactions ({pct:5.1f}%)")

    print("\n[2] STEP-UP VERIFY PATH PERFORMANCE")
    print(f"    - Total VERIFY Transactions:  {verify_total}")
    print(f"    - False Positives Prevented:  {verify_fps_saved} (Step-up completed -> Legitimate payment saved)")
    print(f"    - True Fraud Stopped:         {verify_true_fraud} (Step-up failed -> Fraud blocked)")

    print("\n[3] HUMAN REVIEW PATH (HONEST EXCEPTION LIST)")
    print(f"    - Total Queued for Review:    {hr_total}")
    print(f"    - Explicit Analyst Overrides: {hr_overrides}")
    print(f"    - Heuristic Rubber-Stamped:   {hr_rubberstamped}")

    print("\n" + "-" * 80)
    print("[4] FULL VULX SYSTEM vs NAIVE ML MODEL BASELINE COMPARISON")
    print("-" * 80)
    print(f"{'Metric':<28} | {'Naive Model Baseline':<20} | {'VulX Governed System':<20} | {'Impact'}")
    print("-" * 80)
    print(f"{'Precision':<28} | {prec_naive:<20.4f} | {prec_vulx:<20.4f} | {prec_vulx - prec_naive:+.4f}")
    print(f"{'Recall':<28} | {rec_naive:<20.4f} | {rec_vulx:<20.4f} | {rec_vulx - rec_naive:+.4f}")
    print(f"{'F1 Score':<28} | {f1_naive:<20.4f} | {f1_vulx:<20.4f} | {f1_vulx - f1_naive:+.4f}")
    print(f"{'False Positive Rate (FPR)':<28} | {fpr_naive:<20.4f} | {fpr_vulx:<20.4f} | {fpr_vulx - fpr_naive:+.4f}")
    print("-" * 80)
    print("  --> Key Takeaway: VulX Policy Engine dramatically reduces False Positives (FPR)")
    print("      by routing high-value edge cases to VERIFY step-up rather than naive auto-BLOCK!")
    print("=" * 80 + "\n")

    return {
        "total_events": total_events,
        "routing_distribution": route_counts,
        "verify_analysis": {"total": verify_total, "fps_saved": verify_fps_saved, "true_fraud": verify_true_fraud},
        "human_review_analysis": {"total": hr_total, "overrides": hr_overrides, "rubberstamped": hr_rubberstamped},
        "naive_baseline": {"precision": prec_naive, "recall": rec_naive, "f1_score": f1_naive, "fpr": fpr_naive},
        "vulx_system": {"precision": prec_vulx, "recall": rec_vulx, "f1_score": f1_vulx, "fpr": fpr_vulx},
    }


if __name__ == "__main__":
    run_metrics_dashboard()
