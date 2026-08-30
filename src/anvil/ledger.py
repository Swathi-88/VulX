"""
Anvil SQLite Ledger Layer (Phase 4)
Provides persistent audit trails for payment risk decisions, execution outcomes, and compliance tags.
"""

import json
import os
import sqlite3
import uuid
from typing import Dict, List, Optional
from anvil.config import DEFAULT_LEDGER_DB_PATH

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ledger_events (
    event_id TEXT PRIMARY KEY,
    transaction_id TEXT,
    timestamp TEXT,
    risk_probability REAL,
    naive_recommended_action TEXT,
    routing_decision TEXT,
    rationale_trace TEXT,
    action_taken TEXT,
    verification_outcome TEXT,
    human_review_outcome TEXT,
    final_outcome TEXT,
    ground_truth_label TEXT,
    correctness TEXT,
    legal_basis_tag TEXT,
    retention_class TEXT
);
"""


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = DEFAULT_LEDGER_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None):
    """Creates the SQLite ledger table if it does not exist."""
    with get_connection(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()


def compute_correctness(final_outcome: str, ground_truth_label: str) -> str:
    """
    Computes TP / FP / TN / FN correctness metric:
      - TP = blocked and truly fraud
      - FP = blocked and truly legitimate
      - TN = completed and truly legitimate
      - FN = completed and truly fraud
    """
    gt_clean = ground_truth_label.lower().strip()
    is_fraud = gt_clean in ["fraud", "suspicious", "true"]
    is_blocked = final_outcome.lower().strip() == "blocked"

    if is_blocked and is_fraud:
        return "TP"
    elif is_blocked and not is_fraud:
        return "FP"
    elif not is_blocked and not is_fraud:
        return "TN"
    else:  # not is_blocked and is_fraud
        return "FN"


def compute_legal_basis(event_data: dict) -> str:
    """
    Tags legal_basis_tag as 'cross_merchant_derived' if any top signal is tagged cross_merchant_derived
    or if purpose constraint was triggered, else 'own_history'.
    """
    top_signals = event_data.get("top_contributing_signals", [])
    for sig in top_signals:
        if isinstance(sig, dict):
            tags = sig.get("tags", [])
            feat = sig.get("feature")
            if "cross_merchant_derived" in tags or feat in [
                "merchant_history_score",
                "ip_reputation_score",
            ]:
                return "cross_merchant_derived"

    rationale = event_data.get("rationale_trace", [])
    if any("purpose_constraint=triggered" in str(r) for r in rationale):
        return "cross_merchant_derived"

    return "own_history"


def compute_retention_class(final_outcome: str) -> str:
    """
    Tags retention_class as 'financial_record_required' if payment completed, else 'short_term_ops'.
    """
    if final_outcome.lower().strip() == "completed":
        return "financial_record_required"
    return "short_term_ops"


def record_event(event: dict, db_path: Optional[str] = None) -> dict:
    """
    Records an audited payment decision event into the SQLite ledger.
    """
    init_db(db_path)

    event_id = event.get("event_id", f"evt_{uuid.uuid4().hex[:12]}")
    tx_id = event.get("transaction_id", "unknown_id")
    ts = event.get("timestamp", "2026-08-26T15:00:00Z")
    risk_prob = float(event.get("risk_probability", 0.0))
    naive_action = event.get("naive_recommended_action", "ALLOW")
    routing_dec = event.get("routing_decision", "ALLOW")

    raw_trace = event.get("rationale_trace", [])
    if isinstance(raw_trace, list):
        trace_json = json.dumps(raw_trace)
    else:
        trace_json = str(raw_trace)

    action_taken = event.get("action_taken", routing_dec)
    ver_outcome = event.get("verification_outcome")
    hr_outcome = event.get("human_review_outcome")
    final_outcome = event.get("final_outcome", "completed")
    gt_label = event.get("ground_truth_label", "legitimate")

    correctness = compute_correctness(final_outcome, gt_label)
    legal_basis = compute_legal_basis(event)
    retention_class = compute_retention_class(final_outcome)

    record_dict = {
        "event_id": event_id,
        "transaction_id": tx_id,
        "timestamp": ts,
        "risk_probability": risk_prob,
        "naive_recommended_action": naive_action,
        "routing_decision": routing_dec,
        "rationale_trace": trace_json,
        "action_taken": action_taken,
        "verification_outcome": ver_outcome,
        "human_review_outcome": hr_outcome,
        "final_outcome": final_outcome,
        "ground_truth_label": gt_label,
        "correctness": correctness,
        "legal_basis_tag": legal_basis,
        "retention_class": retention_class,
    }

    insert_sql = """
    INSERT OR REPLACE INTO ledger_events (
        event_id, transaction_id, timestamp, risk_probability, naive_recommended_action,
        routing_decision, rationale_trace, action_taken, verification_outcome,
        human_review_outcome, final_outcome, ground_truth_label, correctness,
        legal_basis_tag, retention_class
    ) VALUES (
        :event_id, :transaction_id, :timestamp, :risk_probability, :naive_recommended_action,
        :routing_decision, :rationale_trace, :action_taken, :verification_outcome,
        :human_review_outcome, :final_outcome, :ground_truth_label, :correctness,
        :legal_basis_tag, :retention_class
    )
    """

    with get_connection(db_path) as conn:
        conn.execute(insert_sql, record_dict)
        conn.commit()

    return record_dict


def query_ledger(filter_dict: Optional[dict] = None, db_path: Optional[str] = None) -> List[dict]:
    """
    Queries the SQLite ledger with optional filter dictionary parameters.
    """
    init_db(db_path)
    sql = "SELECT * FROM ledger_events"
    params = []

    if filter_dict:
        conditions = []
        for key, val in filter_dict.items():
            conditions.append(f"{key} = ?")
            params.append(val)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

    sql += " ORDER BY rowid ASC"

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        results = []
        for row in rows:
            r_dict = dict(row)
            try:
                r_dict["rationale_trace"] = json.loads(r_dict["rationale_trace"])
            except Exception:
                pass
            results.append(r_dict)
        return results
