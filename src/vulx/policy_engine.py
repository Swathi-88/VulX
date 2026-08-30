"""
VulX Policy Engine (Phase 3)
Policy-as-code routing engine that evaluates standard decision contracts against policies.yaml rules.
"""

import os
from typing import Optional, Dict, Any
import yaml
from vulx.config import DEFAULT_POLICIES_PATH


def load_policies(policy_path: Optional[str] = None) -> dict:
    """
    Loads policies configuration from YAML file.
    Pulls from DEFAULT_POLICIES_PATH if policy_path is None.
    """
    if policy_path is None:
        policy_path = DEFAULT_POLICIES_PATH

    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy configuration file '{policy_path}' not found.")

    with open(policy_path, "r", encoding="utf-8") as f:
        policies = yaml.safe_load(f)

    return policies


def evaluate(
    decision_contract: dict,
    policy_path: Optional[str] = None,
    policies_dict: Optional[dict] = None,
) -> dict:
    """
    Evaluates a Standard Decision Contract against policies.yaml rules.
    Runs 6 checks in order:
      1. Purpose constraint check
      2. Severity tiers evaluation
      3. False positive cost lookup
      4. Reversibility evaluation
      5. Uncertainty & confidence mapping
      6. Routing rules decision table (top-to-bottom)
    """
    if policies_dict is not None:
        policies = policies_dict
    elif isinstance(policy_path, dict):
        policies = policy_path
    elif policy_path is not None:
        policies = load_policies(policy_path)
    else:
        policies = load_policies()

    rationale_trace = []

    # -------------------------------------------------------------
    # 1. Purpose Constraint Check
    # -------------------------------------------------------------
    pc_rules = policies.get("purpose_constraint_rules", {})
    restricted_tags = pc_rules.get("restricted_tags", ["cross_merchant_derived"])
    cross_merchant_features = pc_rules.get(
        "cross_merchant_features", ["merchant_history_score", "ip_reputation_score"]
    )

    top_signals = decision_contract.get("top_contributing_signals", [])
    triggered_features = []

    for signal in top_signals:
        if isinstance(signal, dict):
            feat_name = signal.get("feature")
            sig_tags = signal.get("tags", [])
            has_tag = any(t in restricted_tags for t in sig_tags)
            is_cm_feat = feat_name in cross_merchant_features
            if has_tag or is_cm_feat:
                triggered_features.append(feat_name)

    purpose_constraint_triggered = len(triggered_features) > 0
    if purpose_constraint_triggered:
        feat_str = ", ".join(set(triggered_features))
        rationale_trace.append(
            f"purpose_constraint=triggered because signal '{feat_str}' is tagged as cross_merchant_derived "
            f"(requires min VERIFY, disallows silent auto-BLOCK)"
        )
    else:
        rationale_trace.append(
            "purpose_constraint=passed because no cross_merchant_derived signals in top contributors"
        )

    # -------------------------------------------------------------
    # 2. Severity Tiers Evaluation
    # -------------------------------------------------------------
    sev_rules = policies.get("severity_tiers", {})
    low_max = float(sev_rules.get("low_max", 1000.0))
    medium_max = float(sev_rules.get("medium_max", 10000.0))

    raw_amount = decision_contract.get("amount")
    if raw_amount is None:
        tx_obj = decision_contract.get("transaction")
        raw_amount = tx_obj.get("amount") if isinstance(tx_obj, dict) else 0.0
    amount = float(raw_amount) if raw_amount is not None else 0.0

    if amount < low_max:
        severity = "low"
        rationale_trace.append(f"severity=low because amount={amount} < {low_max}")
    elif amount <= medium_max:
        severity = "medium"
        rationale_trace.append(
            f"severity=medium because amount={amount} is between {low_max} and {medium_max}"
        )
    else:
        severity = "high"
        rationale_trace.append(f"severity=high because amount={amount} > {medium_max}")

    # -------------------------------------------------------------
    # 3. False Positive Cost Evaluation
    # -------------------------------------------------------------
    fp_rules = policies.get("false_positive_cost_table", {})
    tenure_cfg = fp_rules.get("tenure_buckets", {})
    short_max_days = int(tenure_cfg.get("short_max_days", 30))
    medium_max_days = int(tenure_cfg.get("medium_max_days", 365))
    fp_matrix = fp_rules.get("matrix", {})

    raw_tenure = decision_contract.get("customer_tenure_days")
    if raw_tenure is None:
        tx_obj = decision_contract.get("transaction")
        raw_tenure = tx_obj.get("customer_tenure_days") if isinstance(tx_obj, dict) else 0
    customer_tenure_days = int(raw_tenure) if raw_tenure is not None else 0

    if customer_tenure_days <= short_max_days:
        tenure_bucket = "short"
        tenure_label = f"short-tenure, <= {short_max_days}d"
    elif customer_tenure_days <= medium_max_days:
        tenure_bucket = "medium"
        tenure_label = f"medium-tenure, {short_max_days}-{medium_max_days}d"
    else:
        tenure_bucket = "long"
        tenure_label = f"long-tenure, > {medium_max_days}d"

    fp_cost = fp_matrix.get(tenure_bucket, {}).get(severity, "medium")
    rationale_trace.append(
        f"FP_cost={fp_cost} because customer_tenure_days={customer_tenure_days} ({tenure_label}) and severity={severity}"
    )

    # -------------------------------------------------------------
    # 4. Reversibility Check
    # -------------------------------------------------------------
    rev_rules = policies.get("reversibility_rules", {})
    reversible_actions = rev_rules.get("reversible_actions", ["ALLOW", "REVIEW", "BLOCK"])
    irreversible_actions = rev_rules.get("irreversible_actions", ["PERMANENT_SUSPEND", "HARD_BLOCK"])
    step_up_avail = rev_rules.get("step_up_verification_available", True)

    naive_action = decision_contract.get("naive_recommended_action", "ALLOW")

    if naive_action in irreversible_actions:
        is_reversible = False
        rationale_trace.append(
            f"reversibility=irreversible because naive_action={naive_action} is listed as irreversible"
        )
    else:
        is_reversible = True
        step_up_str = (
            " (step-up verification path available)"
            if naive_action == "BLOCK" and step_up_avail
            else ""
        )
        rationale_trace.append(
            f"reversibility=reversible because naive_action={naive_action} is reversible{step_up_str}"
        )

    # -------------------------------------------------------------
    # 5. Uncertainty & Confidence Mapping
    # -------------------------------------------------------------
    unc_rules = policies.get("uncertainty_thresholds", {})
    std_dev_cfg = unc_rules.get("std_dev_thresholds", {})
    std_low_max = float(std_dev_cfg.get("low_max", 0.05))
    std_med_max = float(std_dev_cfg.get("medium_max", 0.15))

    dist_cfg = unc_rules.get("distance_from_0_5_thresholds", {})
    dist_high_min = float(dist_cfg.get("high_min", 0.35))
    dist_med_min = float(dist_cfg.get("medium_min", 0.15))

    conf_matrix = unc_rules.get("confidence_matrix", {})

    unc_data = decision_contract.get("prediction_uncertainty", {})
    std_dev = float(unc_data.get("std_dev", 0.0))
    unc_level_attr = unc_data.get("uncertainty_level")

    if unc_level_attr:
        unc_bucket = unc_level_attr
    elif std_dev < std_low_max:
        unc_bucket = "low"
    elif std_dev <= std_med_max:
        unc_bucket = "medium"
    else:
        unc_bucket = "high"

    risk_prob = float(decision_contract.get("risk_probability", 0.5))
    dist_from_0_5 = round(abs(risk_prob - 0.5), 4)

    if dist_from_0_5 >= dist_high_min:
        dist_bucket = "high"
    elif dist_from_0_5 >= dist_med_min:
        dist_bucket = "medium"
    else:
        dist_bucket = "low"

    confidence = conf_matrix.get(unc_bucket, {}).get(dist_bucket, "medium")
    rationale_trace.append(
        f"confidence={confidence} because uncertainty_level={unc_bucket} (std_dev={std_dev:.4f}) "
        f"and distance_from_0.5={dist_from_0_5:.4f}"
    )

    # -------------------------------------------------------------
    # 6. Routing Rules Decision Table
    # -------------------------------------------------------------
    eval_context = {
        "amount": amount,
        "customer_tenure_days": customer_tenure_days,
        "severity": severity,
        "fp_cost": fp_cost,
        "is_reversible": is_reversible,
        "naive_recommended_action": naive_action,
        "uncertainty_level": unc_bucket,
        "confidence": confidence,
        "purpose_constraint_triggered": purpose_constraint_triggered,
        "risk_probability": risk_prob,
        "std_dev": std_dev,
        "True": True,
        "False": False,
    }

    routing_rules = policies.get("routing_rules", [])
    routing_decision = "HUMAN_REVIEW"
    final_rationale = "final: HUMAN_REVIEW because default fallback reached"

    for rule in routing_rules:
        cond_str = rule.get("condition", "False")
        try:
            matched = eval(cond_str, {"__builtins__": {}}, eval_context)
            if matched:
                routing_decision = rule.get("decision", "HUMAN_REVIEW")
                final_rationale = rule.get(
                    "rationale",
                    f"final: {routing_decision} because {rule.get('name', 'rule matched')}",
                )
                break
        except Exception as e:
            continue

    rationale_trace.append(final_rationale)

    return {
        "transaction_id": decision_contract.get("transaction_id", "unknown_id"),
        "routing_decision": routing_decision,
        "rationale_trace": rationale_trace,
    }
