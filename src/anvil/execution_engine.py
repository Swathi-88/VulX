"""
Anvil Execution Engine (Phase 4)
Simulates verification step-up, human analyst review, and payment completion/blocking.
"""

import random
from typing import Optional


def execute(
    routing_decision: str,
    transaction: dict,
    success_prob: float = 0.85,
    analyst_override: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    Executes a routing decision (ALLOW, VERIFY, HUMAN_REVIEW) on a transaction.

    Args:
        routing_decision: "ALLOW", "VERIFY", or "HUMAN_REVIEW"
        transaction: transaction dictionary or decision contract
        success_prob: step-up verification success probability (default 0.85)
        analyst_override: optional explicit analyst action ("APPROVE" / "REJECT")
        seed: optional random seed for deterministic simulation

    Returns:
        {
            "action_taken": "ALLOW" | "VERIFY" | "HUMAN_REVIEW",
            "verification_outcome": "VERIFIED" | "FAILED" | None,
            "human_review_outcome": "APPROVED" | "REJECTED" | None,
            "final_outcome": "completed" | "blocked"
        }
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    decision = routing_decision.upper().strip()

    if decision == "ALLOW":
        return {
            "action_taken": "ALLOW",
            "verification_outcome": None,
            "human_review_outcome": None,
            "final_outcome": "completed",
        }

    elif decision == "BLOCK":
        return {
            "action_taken": "BLOCK",
            "verification_outcome": None,
            "human_review_outcome": None,
            "final_outcome": "blocked",
        }

    elif decision == "VERIFY":
        # Simulate step-up verification draw against success_prob
        draw = rng.random()
        if draw <= success_prob:
            outcome = "VERIFIED"
            final = "completed"
        else:
            outcome = "FAILED"
            final = "blocked"
        return {
            "action_taken": "VERIFY",
            "verification_outcome": outcome,
            "human_review_outcome": None,
            "final_outcome": final,
        }

    elif decision == "HUMAN_REVIEW":
        if analyst_override is not None:
            override_clean = analyst_override.upper().strip()
            if override_clean in ["APPROVE", "APPROVED", "ALLOW", "PASS"]:
                hr_outcome = "APPROVED"
                final = "completed"
            else:
                hr_outcome = "REJECTED"
                final = "blocked"
        else:
            # Automated fallback analyst heuristic (risk_probability < 0.6 -> approve, else block)
            raw_risk = transaction.get("risk_probability")
            if raw_risk is None:
                raw_risk = transaction.get("risk_score", 0.5)
            risk_prob = float(raw_risk) if raw_risk is not None else 0.5
            if risk_prob < 0.60:
                hr_outcome = "APPROVED"
                final = "completed"
            else:
                hr_outcome = "REJECTED"
                final = "blocked"

        return {
            "action_taken": "HUMAN_REVIEW",
            "verification_outcome": None,
            "human_review_outcome": hr_outcome,
            "final_outcome": final,
        }

    else:
        # Fallback unknown routing decision -> block safely
        return {
            "action_taken": decision,
            "verification_outcome": None,
            "human_review_outcome": None,
            "final_outcome": "blocked",
        }
