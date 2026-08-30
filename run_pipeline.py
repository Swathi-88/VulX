"""
VulX End-to-End Pipeline Demo Script (run_pipeline.py)

Wires together Phase 1 (Data), Phase 2 (Decision Contract), Phase 3 (Policy Engine),
and Phase 4 (Execution & SQLite Ledger) into a single runnable pipeline.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from vulx.config import DEFAULT_DEMO_JSON
from vulx.execution_engine import execute
from vulx.ledger import query_ledger, record_event
from vulx.models.decision_contract import get_decision_contract
from vulx.policy_engine import evaluate


def process_transaction(transaction: dict, seed: int = 42) -> dict:
    """
    Runs a single transaction event through the complete 4-phase VulX pipeline.
    """
    # Phase 2: Decision Contract & SHAP explainability
    contract = get_decision_contract(transaction)

    # Phase 3: Policy Engine evaluation & rationale trace
    policy_res = evaluate(contract)

    # Phase 4: Execution Engine simulation
    exec_res = execute(
        routing_decision=policy_res["routing_decision"],
        transaction=contract,
        seed=seed,
    )

    # Phase 4: SQLite Ledger Audit Trail
    event_payload = {
        **contract,
        **policy_res,
        **exec_res,
        "ground_truth_label": transaction.get("ground_truth_label", "legitimate"),
        "timestamp": transaction.get("timestamp", "2026-08-26T15:00:00Z"),
    }

    ledger_record = record_event(event_payload)
    return {
        "contract": contract,
        "policy": policy_res,
        "execution": exec_res,
        "ledger": ledger_record,
    }


def run_demo_pipeline():
    print("=" * 85)
    print(" VULX INTEGRATED PIPELINE DEMO — PHASES 1 to 4")
    print("=" * 85)

    if not os.path.exists(DEFAULT_DEMO_JSON):
        print(f"Error: Demo file '{DEFAULT_DEMO_JSON}' not found. Please generate events first.")
        sys.exit(1)

    with open(DEFAULT_DEMO_JSON, "r", encoding="utf-8") as f:
        demo_cases = json.load(f)

    # Step-by-step trace for the showcase false positive case (Case 1)
    kp_case = demo_cases[0]
    res_kp = process_transaction(kp_case, seed=42)

    c = res_kp["contract"]
    p = res_kp["policy"]
    e = res_kp["execution"]
    l = res_kp["ledger"]

    print("\n" + "-" * 85)
    print(" STEP-BY-STEP TRACE: ₹45,000 SHOWCASE FALSE POSITIVE TRANSACTION")
    print("-" * 85)

    top_sigs_str = ", ".join(
        [f"{s['feature']} ({s['contribution']:+.3f})" for s in c["top_contributing_signals"]]
    )
    print(f"  [1] Vulcan ML Model says:  {c['risk_probability']*100:.1f}% risk, Naive Action: {c['naive_recommended_action']}")
    print(f"                             Top signals: [{top_sigs_str}]")

    print(f"  [2] VulX Policy Engine:   Routed to {p['routing_decision']}")
    print(f"                             Rationale: {p['rationale_trace'][-1]}")

    ver_str = f" ({e['verification_outcome']})" if e["verification_outcome"] else ""
    print(f"  [3] Execution Engine:      Action taken: {e['action_taken']}{ver_str}")
    print(f"                             Final Payment Outcome: {e['final_outcome'].upper()}")

    corr_desc = "Prevented False Positive (Saved customer purchase)" if l['correctness'] == "TN" and c['naive_recommended_action'] == "BLOCK" else l['correctness']
    print(f"  [4] SQLite Ledger Record:  Audit Correctness: {l['correctness']} [{corr_desc}]")
    print(f"                             Legal Basis Tag: {l['legal_basis_tag']} | Retention: {l['retention_class']}")
    print("-" * 85)

    # Process all showcase cases and build summary table
    print("\n" + "=" * 85)
    print(" SUMMARY TABLE ACROSS ALL 8 DEMO SHOWCASE CASES")
    print("=" * 85)

    header = f"{'Tx ID':<22} | {'Amount':<10} | {'Naive Act':<9} | {'Policy Route':<12} | {'Outcome':<9} | {'GT Label':<10} | {'Correct'}"
    print(header)
    print("-" * len(header))

    summary_records = []
    for idx, case in enumerate(demo_cases):
        res = process_transaction(case, seed=42 + idx)
        l_rec = res["ledger"]
        summary_records.append(l_rec)

        tx_id_short = case["transaction_id"][:22]
        amt_str = f"INR {case['amount']:,.0f}"
        naive = res["contract"]["naive_recommended_action"]
        route = res["policy"]["routing_decision"]
        final_out = l_rec["final_outcome"]
        gt_lbl = case.get("ground_truth_label", "legitimate")
        corr = l_rec["correctness"]

        print(f"{tx_id_short:<22} | {amt_str:<10} | {naive:<9} | {route:<12} | {final_out:<9} | {gt_lbl:<10} | {corr}")

    print("-" * len(header))

    # Calculate overall metrics
    all_ledger = query_ledger()
    total_events = len(all_ledger)
    fps_prevented = sum(
        1 for r in summary_records if r["naive_recommended_action"] == "BLOCK" and r["correctness"] == "TN"
    )

    print(f"\n  Total Audit Events Recorded in SQLite: {total_events}")
    print(f"  False Positives Prevented (BLOCK -> Completed): {fps_prevented}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_demo_pipeline()
