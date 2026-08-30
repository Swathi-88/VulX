"""
Anvil Fraud Detection & Policy Engine — Interactive Streamlit Dashboard (app.py)

Visual pitch & demo interface for Anvil pipeline stages, policy editing, and SQLite ledger.
"""

import json
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from anvil.config import DEFAULT_DEMO_JSON, DEFAULT_POLICIES_PATH
from anvil.execution_engine import execute
from anvil.ledger import query_ledger, record_event
from anvil.models.decision_contract import get_decision_contract
from anvil.policy_engine import evaluate, load_policies


def run_streamlit_app():
    if not HAS_STREAMLIT:
        print("Streamlit package is not installed. Run 'pip install streamlit' to launch the web dashboard.")
        return

    st.set_page_config(
        page_title="Anvil Fraud Decision & Policy Engine",
        layout="wide",
    )

    st.title("Anvil: Policy-as-Code Fraud Routing System")
    st.caption("Standard Decision Contracts • Explainable SHAP Signals • Policy Routing • SQLite Audit Ledger")

    tab1, tab2, tab3 = st.tabs(["Live Pipeline Demo", "Policy Config (YAML)", "SQLite Audit Ledger"])

    # Load Demo Cases
    demo_cases = []
    if os.path.exists(DEFAULT_DEMO_JSON):
        with open(DEFAULT_DEMO_JSON, "r", encoding="utf-8") as f:
            demo_cases = json.load(f)

    with tab1:
        st.subheader("Select Transaction Vector")
        case_options = {
            f"{c['transaction_id']} (₹{c['amount']:,.0f} - {c['category_tag']})": c
            for c in demo_cases
        }

        selected_key = st.selectbox("Choose demo case:", list(case_options.keys()))
        case = case_options[selected_key]

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### Transaction Event Data")
            st.json(case)

        # Execute Pipeline
        contract = get_decision_contract(case)
        policy_res = evaluate(contract)
        exec_res = execute(policy_res["routing_decision"], contract, seed=42)

        event_payload = {
            **contract,
            **policy_res,
            **exec_res,
            "ground_truth_label": case.get("ground_truth_label", "legitimate"),
            "timestamp": case.get("timestamp", "2026-08-26T15:00:00Z"),
        }
        ledger_rec = record_event(event_payload)

        with col_right:
            st.markdown("### Pipeline Progression & Decision Trace")

            # Stage 1: Vulcan Model
            st.info(
                f"**1. Vulcan ML Model:** Risk Probability = **{contract['risk_probability']*100:.1f}%** | "
                f"Naive Action = **{contract['naive_recommended_action']}**"
            )

            # Stage 2: Policy Routing
            route_color = "green" if policy_res["routing_decision"] == "ALLOW" else ("orange" if policy_res["routing_decision"] == "VERIFY" else "red")
            st.markdown(f"**2. Anvil Policy Engine:** Routing Decision = **:{route_color}[{policy_res['routing_decision']}]**")
            with st.expander("View Policy Rationale Trace", expanded=True):
                for idx, r in enumerate(policy_res["rationale_trace"], 1):
                    st.write(f"**{idx}.** {r}")

            # Stage 3: Execution
            st.success(
                f"**3. Execution Engine:** Action = **{exec_res['action_taken']}** | "
                f"Outcome = **{exec_res['final_outcome'].upper()}**"
            )

            # Stage 4: Ledger
            st.warning(
                f"**4. SQLite Audit Ledger:** Correctness = **{ledger_rec['correctness']}** | "
                f"Legal Basis = **{ledger_rec['legal_basis_tag']}** | "
                f"Retention = **{ledger_rec['retention_class']}**"
            )

    with tab2:
        st.subheader("Live Policy Configuration Editor")
        st.caption("Editing YAML numbers updates decision rules live across all evaluation pipeline runs.")

        if os.path.exists(DEFAULT_POLICIES_PATH):
            with open(DEFAULT_POLICIES_PATH, "r", encoding="utf-8") as f:
                yaml_content = f.read()

            new_yaml = st.text_area("policies.yaml Content", value=yaml_content, height=400)
            if st.button("Save & Apply Policy Rules"):
                try:
                    yaml.safe_load(new_yaml)
                    with open(DEFAULT_POLICIES_PATH, "w", encoding="utf-8") as f:
                        f.write(new_yaml)
                    st.success("Successfully updated policies.yaml!")
                except Exception as e:
                    st.error(f"Invalid YAML format: {e}")

    with tab3:
        st.subheader("SQLite Audit Ledger Database")
        records = query_ledger()
        if records:
            st.dataframe(records, use_container_width=True)
        else:
            st.info("No ledger records recorded yet. Run pipeline to populate ledger.")


if __name__ == "__main__":
    if HAS_STREAMLIT:
        run_streamlit_app()
    else:
        print("Streamlit is not installed. You can still run 'python run_pipeline.py' for terminal output.")
