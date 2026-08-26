"""
Anvil Policy Engine Root Re-export & CLI helper
"""

import os
from anvil.policy_engine import evaluate, load_policies

__all__ = ["evaluate", "load_policies"]

if __name__ == "__main__":
    import json
    from anvil.config import DEFAULT_DEMO_JSON
    from anvil.models.decision_contract import get_decision_contract

    print("Evaluating demo cases with Anvil Policy Engine...")
    if os.path.exists(DEFAULT_DEMO_JSON):
        with open(DEFAULT_DEMO_JSON, "r", encoding="utf-8") as f:
            cases = json.load(f)
        for case in cases[:2]:
            contract = get_decision_contract(case)
            res = evaluate(contract)
            print(f"\nTransaction ID: {res['transaction_id']}")
            print(f"Routing Decision: {res['routing_decision']}")
            print("Rationale Trace:")
            for trace in res["rationale_trace"]:
                print(f"  - {trace}")
