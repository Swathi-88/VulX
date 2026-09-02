"""
VulX Prototype Live Dashboard (dashboard.py)

Real-time Streamlit application powered by real pipeline files and functions:
  - Tab 1: Live Run (real-time perf_counter deltas, decoupled SHAP explanation bar chart)
  - Tab 2: Compare: Naive vs VulX (batch comparison, diverged rows highlighted)
  - Tab 3: Ledger Explorer (live SQLite query_ledger, FP prevented metric, human review load)
  - Tab 4: Policy Lab (interactive in-memory policy slider evaluation)
  - Tab 5: Metrics & Retrain Impact (real metrics.json, retrain_comparison.json, benchmark_results.json)
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# Ensure src is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from vulx.config import (
    DEFAULT_DEMO_JSON,
    DEFAULT_LEDGER_DB_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_POLICIES_PATH,
    MODEL_ENSEMBLE_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from vulx.execution_engine import execute
from vulx.ledger import query_ledger, record_event
from vulx.models.decision_contract import (
    get_decision_contract,
    get_explanation,
    get_fast_decision,
    load_models,
)
from vulx.policy_engine import evaluate, load_policies

st.set_page_config(
    page_title="VulX Fraud Routing & Governance Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Caching Functions
# -----------------------------------------------------------------------------
@st.cache_resource
def cached_load_models():
    """Loads and caches XGBoost primary and ensemble models once."""
    if os.path.exists(DEFAULT_MODEL_PATH):
        return load_models(DEFAULT_MODEL_PATH, str(MODEL_ENSEMBLE_DIR))
    return None, None, None


@st.cache_data(ttl=3)
def cached_query_ledger():
    """Queries SQLite ledger with a short 3-second TTL for live dashboard updates."""
    if os.path.exists(DEFAULT_LEDGER_DB_PATH):
        return query_ledger(db_path=DEFAULT_LEDGER_DB_PATH)
    return []


# -----------------------------------------------------------------------------
# Helper File Loader
# -----------------------------------------------------------------------------
def load_json_file(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# Load models at startup
cached_load_models()

# -----------------------------------------------------------------------------
# Sidebar Status Checklist
# -----------------------------------------------------------------------------
st.sidebar.title("VulX System State")
st.sidebar.caption("Real-Time Artifact Checklist")

events_csv_path = DEFAULT_OUTPUT_CSV if os.path.exists(DEFAULT_OUTPUT_CSV) else "events.csv"
demo_json_path = DEFAULT_DEMO_JSON if os.path.exists(DEFAULT_DEMO_JSON) else "demo_cases.json"
model_pkl_path = DEFAULT_MODEL_PATH
metrics_json_path = DEFAULT_METRICS_PATH if os.path.exists(DEFAULT_METRICS_PATH) else "metrics.json"
policies_yaml_path = DEFAULT_POLICIES_PATH if os.path.exists(DEFAULT_POLICIES_PATH) else "policies.yaml"
ledger_db_path = DEFAULT_LEDGER_DB_PATH
retrain_path = str(MODELS_DIR / "retrain_comparison.json") if os.path.exists(str(MODELS_DIR / "retrain_comparison.json")) else "retrain_comparison.json"
bench_path = str(MODELS_DIR / "benchmark_results.json") if os.path.exists(str(MODELS_DIR / "benchmark_results.json")) else "benchmark_results.json"

# Check events.csv
if os.path.exists(events_csv_path):
    df_raw_check = pd.read_csv(events_csv_path)
    st.sidebar.success(f"[OK] events.csv ({len(df_raw_check):,} rows)")
else:
    st.sidebar.warning("[MISSING] events.csv not found (Run generate_events.py)")

# Check demo_cases.json
demo_cases_list = load_json_file(demo_json_path)
if demo_cases_list:
    st.sidebar.success(f"[OK] demo_cases.json ({len(demo_cases_list)} scenarios)")
else:
    st.sidebar.warning("[MISSING] demo_cases.json not found")

# Check model.pkl
if os.path.exists(model_pkl_path):
    ens_count = len([f for f in os.listdir(str(MODEL_ENSEMBLE_DIR)) if f.endswith(".pkl")]) if os.path.exists(str(MODEL_ENSEMBLE_DIR)) else 0
    st.sidebar.success(f"[OK] model.pkl & {ens_count} ensemble models")
else:
    st.sidebar.error("[MISSING] model.pkl not found (Run train_model.py)")

# Check metrics.json
if os.path.exists(metrics_json_path):
    st.sidebar.success("[OK] metrics.json")
else:
    st.sidebar.warning("[MISSING] metrics.json not found")

# Check policies.yaml
if os.path.exists(policies_yaml_path):
    st.sidebar.success("[OK] policies.yaml")
else:
    st.sidebar.error("[MISSING] policies.yaml not found")

# Check SQLite ledger
ledger_events_check = cached_query_ledger()
if os.path.exists(ledger_db_path):
    st.sidebar.success(f"[OK] vulx_ledger.db ({len(ledger_events_check)} events)")
else:
    st.sidebar.warning("[EMPTY] vulx_ledger.db empty / not created")

# Check retrain_comparison.json
if os.path.exists(retrain_path):
    st.sidebar.success("[OK] retrain_comparison.json")
else:
    st.sidebar.info("[INFO] retrain_comparison.json (Run retrain_and_compare.py)")

# Check benchmark_results.json
if os.path.exists(bench_path):
    st.sidebar.success("[OK] benchmark_results.json")
else:
    st.sidebar.info("[INFO] benchmark_results.json (Run benchmark.py)")

st.sidebar.markdown("---")
if st.sidebar.button("Refresh Ledger Data"):
    st.cache_data.clear()
    st.rerun()


# -----------------------------------------------------------------------------
# Main Dashboard Header & Navigation
# -----------------------------------------------------------------------------
st.title("VulX: Policy-as-Code Routing & Governance Platform")
st.caption("Standard Decision Contracts • SHAP Explainability • Policy Routing • SQLite Audit Ledger • Feedback Retraining")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Live Run",
    "Compare: Naive vs VulX",
    "Ledger Explorer",
    "Policy Lab",
    "Metrics & Retrain Impact",
])


# =============================================================================
# TAB 1: LIVE RUN
# =============================================================================
with tab1:
    st.subheader("Real-Time Transaction Evaluation & Execution Trace")

    col_sel1, col_sel2 = st.columns([2, 1])

    with col_sel1:
        tx_source = st.radio("Select Transaction Source:", ["Showcase Demo Vectors (demo_cases.json)", "Random Sample (events.csv)"], horizontal=True)

    selected_tx = None

    if tx_source.startswith("Showcase") and demo_cases_list:
        case_map = {f"{c['transaction_id']} (INR {c.get('amount',0):,.0f} - {c.get('category_tag')})": c for c in demo_cases_list}
        chosen_key = st.selectbox("Pick showcase scenario:", list(case_map.keys()))
        selected_tx = case_map[chosen_key]

    elif os.path.exists(events_csv_path):
        df_sample = pd.read_csv(events_csv_path)
        sample_idx = st.number_input("Select row index from events.csv:", min_value=0, max_value=len(df_sample) - 1, value=0)
        selected_tx = df_sample.iloc[sample_idx].to_dict()

    if selected_tx is None:
        st.error("No valid transaction data available. Please generate events or demo cases first.")
    else:
        st.markdown("#### Transaction Event Input")
        tx_info_cols = st.columns(5)
        tx_info_cols[0].metric("Transaction ID", str(selected_tx.get("transaction_id"))[:12])
        tx_info_cols[1].metric("Amount", f"INR {selected_tx.get('amount', 0.0):,.2f}")
        tx_info_cols[2].metric("Customer Tenure", f"{selected_tx.get('customer_tenure_days', 0)} days")
        tx_info_cols[3].metric("Payment Method", str(selected_tx.get("payment_method", "UPI")))
        tx_info_cols[4].metric("Ground Truth", str(selected_tx.get("ground_truth_label", "legitimate")).upper())

        if st.button("Run Transaction", type="primary", use_container_width=True):
            # Time Stage 1: Fast Decision Path
            t0 = time.perf_counter()
            fast_contract = get_fast_decision(selected_tx, DEFAULT_MODEL_PATH, str(MODEL_ENSEMBLE_DIR))
            t1 = time.perf_counter()
            fast_path_ms = (t1 - t0) * 1000

            # Time Stage 2: Policy Engine Evaluation
            policy_res = evaluate(fast_contract, DEFAULT_POLICIES_PATH)
            t2 = time.perf_counter()
            policy_ms = (t2 - t1) * 1000

            # Time Stage 3: Execution Simulation
            exec_res = execute(policy_res["routing_decision"], fast_contract, seed=42)
            t3 = time.perf_counter()
            exec_ms = (t3 - t2) * 1000

            # Time Stage 4: SQLite Ledger Recording
            event_payload = {
                **fast_contract,
                **policy_res,
                **exec_res,
                "ground_truth_label": selected_tx.get("ground_truth_label", "legitimate"),
                "timestamp": selected_tx.get("timestamp", "2026-08-26T15:00:00Z"),
            }
            rec_event = record_event(event_payload, DEFAULT_LEDGER_DB_PATH)
            t4 = time.perf_counter()
            ledger_ms = (t4 - t3) * 1000
            total_ms = (t4 - t0) * 1000

            st.markdown(f"### Total Fast-Path Latency: **{total_ms:.2f} ms**")

            # Step-by-Step Vertical Trace Cards
            st.markdown("#### Step-by-Step Execution Trace")

            st.info(
                f"**1. Vulcan ML Model (Fast Path):** `risk_probability` = **{fast_contract['risk_probability']*100:.1f}%** | "
                f"`uncertainty` = **{fast_contract['prediction_uncertainty']['uncertainty_level']}** (`std_dev`={fast_contract['prediction_uncertainty']['std_dev']:.4f}) | "
                f"Naive Action = **{fast_contract['naive_recommended_action']}**  \n"
                f"*Source: `model.pkl` & 5 Bootstrap Ensembles (`{fast_path_ms:.2f} ms`)*"
            )

            route_dec = policy_res["routing_decision"]
            if route_dec == "ALLOW":
                route_badge = "[ALLOW]"
            elif route_dec == "BLOCK":
                route_badge = "[BLOCK]"
            elif route_dec == "VERIFY":
                route_badge = "[VERIFY]"
            else:
                route_badge = "[HUMAN_REVIEW]"
            st.markdown(
                f"**2. VulX Policy Engine:** Routing Decision = **{route_badge}**  \n"
                f"**Rationale Trace:**  \n"
                + "\n".join([f"&nbsp;&nbsp;&nbsp;&nbsp;• `{r}`" for r in policy_res["rationale_trace"]])
                + f"  \n*Source: `policies.yaml` (`{policy_ms:.2f} ms`)*"
            )

            outcome_str = exec_res["final_outcome"].upper()
            ver_suffix = f" ({exec_res['verification_outcome']})" if exec_res["verification_outcome"] else ""
            st.success(
                f"**3. Execution Engine:** Action Taken = **{exec_res['action_taken']}{ver_suffix}** | "
                f"Final Outcome = **{outcome_str}**  \n"
                f"*Source: `execution_engine.py` (`{exec_ms:.2f} ms`)*"
            )

            corr_badge = f"**{rec_event['correctness']}**"
            if rec_event["correctness"] == "TN" and fast_contract["naive_recommended_action"] == "BLOCK":
                corr_badge += " (Prevented False Positive — Saved customer purchase!)"
            st.warning(
                f"**4. SQLite Audit Ledger:** Correctness = {corr_badge} | "
                f"Legal Basis = **{rec_event['legal_basis_tag']}** | "
                f"Retention = **{rec_event['retention_class']}**  \n"
                f"*Source: `vulx_ledger.db` (`{ledger_ms:.2f} ms`)*"
            )

            # Stage 5: Decoupled Async SHAP Explanation
            st.markdown("---")
            st.markdown("#### Decoupled Explainability (Async SHAP Signals)")
            t_s0 = time.perf_counter()
            top_signals = get_explanation(selected_tx, DEFAULT_MODEL_PATH)
            t_s1 = time.perf_counter()
            shap_ms = (t_s1 - t_s0) * 1000

            st.caption(f"Computed **AFTER** the decision in **{shap_ms:.2f} ms** (proving decoupled latency architecture).")

            if top_signals:
                df_shap = pd.DataFrame(top_signals)
                df_shap["abs_contrib"] = df_shap["contribution"].abs()
                fig_shap = px.bar(
                    df_shap,
                    x="contribution",
                    y="feature",
                    orientation="h",
                    color="contribution",
                    color_continuous_scale="RdYlGn_r",
                    title=f"Top SHAP Risk Signals (Computed in {shap_ms:.2f} ms)",
                )
                fig_shap.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_shap, use_container_width=True)


# =============================================================================
# TAB 2: COMPARE: NAIVE VS VULX
# =============================================================================
with tab2:
    st.subheader("Batch Evaluation Comparison: Raw ML Naive Model vs VulX Policy Engine")

    if not os.path.exists(events_csv_path):
        st.error("File `events.csv` not found. Please run `generate_events.py` first.")
    else:
        df_all_events = pd.read_csv(events_csv_path)

        col_b1, col_b2 = st.columns([2, 1])
        with col_b1:
            batch_size = st.slider("Select Batch Size from events.csv:", min_value=10, max_value=200, value=50, step=10)

        # Stratified sampling across category_tag to include edge cases
        try:
            df_batch = df_all_events.groupby("category_tag", group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(2, int(batch_size * (len(x) / len(df_all_events))))), random_state=42)
            ).head(batch_size)
        except Exception:
            df_batch = df_all_events.head(batch_size)

        batch_records = []
        wrongful_naive_blocks = 0
        vulx_saved_via_verify = 0

        for _, row in df_batch.iterrows():
            tx_dict = row.to_dict()
            contract = get_fast_decision(tx_dict, DEFAULT_MODEL_PATH, str(MODEL_ENSEMBLE_DIR))
            naive_act = contract["naive_recommended_action"]

            policy_res = evaluate(contract, DEFAULT_POLICIES_PATH)
            vulx_act = policy_res["routing_decision"]

            gt = str(tx_dict.get("ground_truth_label")).lower().strip()
            is_legit = gt in ["legitimate", "normal", "legitimate_but_unusual"]

            if naive_act == "BLOCK" and is_legit:
                wrongful_naive_blocks += 1
                if vulx_act == "VERIFY":
                    vulx_saved_via_verify += 1

            diverged = (naive_act != vulx_act)

            batch_records.append({
                "transaction_id": str(tx_dict.get("transaction_id"))[:16],
                "amount": f"INR {tx_dict.get('amount', 0.0):,.0f}",
                "category_tag": tx_dict.get("category_tag"),
                "naive_action": naive_act,
                "vulx_action": vulx_act,
                "ground_truth_label": gt,
                "diverged": diverged,
            })

        df_comparison_result = pd.DataFrame(batch_records)
        df_comparison_result.sort_values(by="diverged", ascending=False, inplace=True)

        # Aggregate Live Count Metric Banner
        st.markdown("### Live Batch Aggregate Governance Impact")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Batch Size", len(df_batch))
        col_m2.metric("Naive Model Wrongful Blocks", wrongful_naive_blocks, delta="- High False Positives", delta_color="inverse")
        col_m3.metric("VulX Prevented via VERIFY", vulx_saved_via_verify, delta="+ Recovered Purchases", delta_color="normal")

        st.markdown(
            f"> **Live Governance Finding**: Naive ML model would have wrongfully blocked **{wrongful_naive_blocks}** legitimate transactions. "
            f"VulX Policy Engine routed **{vulx_saved_via_verify}** of them to **VERIFY** step-up authentication, saving legitimate sales!"
        )

        st.markdown("#### Transaction Decision Comparison Table (Diverged Rows Highlighted)")

        def highlight_diverged(row):
            if row["diverged"]:
                return ["background-color: #2b1f0d; font-weight: bold;"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_comparison_result.style.apply(highlight_diverged, axis=1),
            use_container_width=True,
            height=400,
        )


# =============================================================================
# TAB 3: LEDGER EXPLORER
# =============================================================================
with tab3:
    st.subheader("SQLite Audit Ledger Live Query & Governance Explorer")

    ledger_rows = cached_query_ledger()

    if not ledger_rows:
        st.info("SQLite ledger `vulx_ledger.db` is currently empty on this instance.")
        if st.button("⚡ Pre-Populate Ledger with Demo Cases (One-Click)", type="primary"):
            if demo_cases_list:
                for case in demo_cases_list:
                    contract = get_fast_decision(case, DEFAULT_MODEL_PATH, str(MODEL_ENSEMBLE_DIR))
                    pol_res = evaluate(contract, DEFAULT_POLICIES_PATH)
                    ex_res = execute(pol_res["routing_decision"], contract, seed=42)
                    payload = {
                        **contract,
                        **pol_res,
                        **ex_res,
                        "ground_truth_label": case.get("ground_truth_label", "legitimate"),
                        "timestamp": case.get("timestamp", "2026-08-26T15:00:00Z"),
                    }
                    record_event(payload, DEFAULT_LEDGER_DB_PATH)
                st.cache_data.clear()
                st.rerun()
            else:
                st.warning("No demo cases found to populate.")
    else:
        df_ledger = pd.DataFrame(ledger_rows)

        # Compute Summary Metrics from REAL Ledger Contents
        total_ledger_events = len(df_ledger)

        # False positives prevented by VERIFY (routing_decision == VERIFY and correctness == TN)
        fp_prevented_count = sum(
            1 for _, r in df_ledger.iterrows()
            if r.get("routing_decision") == "VERIFY" and r.get("correctness") == "TN"
        )

        # Human Review Load
        hr_count = sum(1 for _, r in df_ledger.iterrows() if r.get("routing_decision") == "HUMAN_REVIEW")
        hr_pct = (hr_count / total_ledger_events * 100) if total_ledger_events > 0 else 0.0

        st.markdown("#### Real Ledger Summary Panel")
        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
        col_l1.metric("Total Recorded Events", total_ledger_events)
        col_l2.metric("False Positives Prevented (VERIFY)", fp_prevented_count, help="routing_decision=VERIFY & correctness=TN")
        col_l3.metric("Human Review Load", f"{hr_count} ({hr_pct:.1f}%)", help="Honest exception list requiring analyst oversight")
        col_l4.metric("Legal Basis: Cross-Merchant", sum(1 for _, r in df_ledger.iterrows() if r.get("legal_basis_tag") == "cross_merchant_derived"))

        # Visual Charts
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            route_dist = df_ledger["routing_decision"].value_counts().reset_index()
            route_dist.columns = ["Routing Decision", "Count"]
            fig_pie = px.pie(route_dist, names="Routing Decision", values="Count", title="Ledger Routing Decision Breakdown", hole=0.4)
            fig_pie.update_layout(height=280)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_c2:
            corr_dist = df_ledger["correctness"].value_counts().reset_index()
            corr_dist.columns = ["Correctness", "Count"]
            fig_bar = px.bar(corr_dist, x="Correctness", y="Count", color="Correctness", title="Ledger Audit Correctness Breakdown (TP/FP/TN/FN)")
            fig_bar.update_layout(height=280)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Interactive Table Filters
        st.markdown("#### Filter Ledger Audit Records")
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            sel_correctness = st.multiselect("Filter by Correctness:", list(df_ledger["correctness"].unique()), default=list(df_ledger["correctness"].unique()))
        with f_col2:
            sel_routing = st.multiselect("Filter by Routing Decision:", list(df_ledger["routing_decision"].unique()), default=list(df_ledger["routing_decision"].unique()))
        with f_col3:
            sel_legal = st.multiselect("Filter by Legal Basis Tag:", list(df_ledger["legal_basis_tag"].unique()), default=list(df_ledger["legal_basis_tag"].unique()))

        filtered_df = df_ledger[
            (df_ledger["correctness"].isin(sel_correctness)) &
            (df_ledger["routing_decision"].isin(sel_routing)) &
            (df_ledger["legal_basis_tag"].isin(sel_legal))
        ]

        st.dataframe(filtered_df, use_container_width=True, height=350)


# =============================================================================
# TAB 4: POLICY LAB (INTERACTIVE POLICY EDITING)
# =============================================================================
with tab4:
    st.subheader("Interactive Policy Lab & Live Threshold Tuning")
    st.caption("Adjust policy parameters in-memory and watch routing decisions re-evaluate live across batch transactions.")

    if not os.path.exists(policies_yaml_path):
        st.error("File `policies.yaml` not found.")
    else:
        with open(policies_yaml_path, "r", encoding="utf-8") as f:
            disk_policies = yaml.safe_load(f)

        # Create editable copy in session state
        if "in_memory_policies" not in st.session_state:
            st.session_state["in_memory_policies"] = json.loads(json.dumps(disk_policies))

        pol_curr = st.session_state["in_memory_policies"]

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.markdown("#### 1. Severity Tier Thresholds")
            sev_low = st.number_input("Low Severity Max Amount (INR):", value=float(pol_curr["severity_tiers"].get("low_max", 1000.0)), step=500.0)
            sev_med = st.number_input("Medium Severity Max Amount (INR):", value=float(pol_curr["severity_tiers"].get("medium_max", 10000.0)), step=1000.0)

            pol_curr["severity_tiers"]["low_max"] = sev_low
            pol_curr["severity_tiers"]["medium_max"] = sev_med

        with col_p2:
            st.markdown("#### 2. Customer Tenure Buckets")
            ten_short = st.number_input("Short Tenure Max Days:", value=int(pol_curr["false_positive_cost_table"]["tenure_buckets"].get("short_max_days", 30)), step=5)
            ten_med = st.number_input("Medium Tenure Max Days:", value=int(pol_curr["false_positive_cost_table"]["tenure_buckets"].get("medium_max_days", 365)), step=15)

            pol_curr["false_positive_cost_table"]["tenure_buckets"]["short_max_days"] = ten_short
            pol_curr["false_positive_cost_table"]["tenure_buckets"]["medium_max_days"] = ten_med

        if st.button("Reset to disk policies.yaml defaults"):
            st.session_state["in_memory_policies"] = json.loads(json.dumps(disk_policies))
            st.rerun()

        st.markdown("---")
        st.markdown("#### Live Re-evaluated Routing Table (In-Memory Policy Rules)")

        if os.path.exists(events_csv_path):
            df_lab = pd.read_csv(events_csv_path).head(30)
            lab_results = []

            for _, row in df_lab.iterrows():
                tx_d = row.to_dict()
                contract = get_fast_decision(tx_d, DEFAULT_MODEL_PATH, str(MODEL_ENSEMBLE_DIR))

                # Evaluate using IN-MEMORY policy
                p_eval = evaluate(contract, policies_dict=pol_curr)

                lab_results.append({
                    "transaction_id": str(tx_d.get("transaction_id"))[:16],
                    "amount": f"INR {tx_d.get('amount', 0.0):,.0f}",
                    "tenure_days": tx_d.get("customer_tenure_days"),
                    "naive_action": contract["naive_recommended_action"],
                    "re_evaluated_routing": p_eval["routing_decision"],
                    "final_rationale": p_eval["rationale_trace"][-1],
                })

            df_lab_res = pd.DataFrame(lab_results)
            st.dataframe(df_lab_res, use_container_width=True, height=350)


# =============================================================================
# TAB 5: METRICS & RETRAIN IMPACT
# =============================================================================
with tab5:
    st.subheader("Model Evaluation Metrics, Retraining Impact & Latency Benchmarks")

    # Section 1: Base Model Metrics
    st.markdown("### 1. Base XGBoost Model Performance (Held-Out Test Set)")
    base_metrics = load_json_file(metrics_json_path)

    if base_metrics:
        m_cols = st.columns(4)
        m_cols[0].metric("Precision", f"{base_metrics.get('precision', 0.0):.4f}")
        m_cols[1].metric("Recall", f"{base_metrics.get('recall', 0.0):.4f}")
        m_cols[2].metric("F1 Score", f"{base_metrics.get('f1_score', 0.0):.4f}")
        m_cols[3].metric("AUC-ROC", f"{base_metrics.get('auc_roc', 0.0):.4f}")
    else:
        st.warning("File `metrics.json` not found. Please run `train_model.py` first.")

    st.markdown("---")

    # Section 2: Feedback Retraining Comparison
    st.markdown("### 2. Feedback Retraining Impact (retrain_comparison.json)")
    retrain_data = load_json_file(retrain_path)

    if not retrain_data:
        st.info("[NOTE] `retrain_comparison.json` not found. Run `python retrain_and_compare.py` first to generate real retraining metrics.")
    else:
        base_m = retrain_data.get("baseline_model", {})
        ret_m = retrain_data.get("retrained_model", {})

        metrics_names = ["Precision", "Recall", "F1 Score", "False Positive Rate"]
        base_vals = [base_m.get("precision", 0), base_m.get("recall", 0), base_m.get("f1_score", 0), base_m.get("false_positive_rate", 0)]
        ret_vals = [ret_m.get("precision", 0), ret_m.get("recall", 0), ret_m.get("f1_score", 0), ret_m.get("false_positive_rate", 0)]

        fig_retrain = go.Figure(data=[
            go.Bar(name="Original Baseline Model", x=metrics_names, y=base_vals, marker_color="#3366cc"),
            go.Bar(name="Retrained Model (Feedback Added)", x=metrics_names, y=ret_vals, marker_color="#109618"),
        ])
        fig_retrain.update_layout(barmode="group", height=320, title="Real Before/After Retraining Performance Comparison")
        st.plotly_chart(fig_retrain, use_container_width=True)

        st.caption(f"Evaluated on identical held-out test set ({retrain_data.get('test_set_size', 400)} samples) with {retrain_data.get('feedback_samples_added', 0)} feedback correction samples added.")

    st.markdown("---")

    # Section 3: Latency Benchmark Results
    st.markdown("### 3. Execution Latency Distribution (benchmark_results.json)")
    bench_data = load_json_file(bench_path)

    if not bench_data:
        st.info("[NOTE] `benchmark_results.json` not found. Run `python benchmark.py` to measure real latency distributions.")
    else:
        b_cols = st.columns(4)
        b_cols[0].metric("Fast Decision Path p50", f"{bench_data['fast_decision_path_ms']['p50']:.3f} ms")
        b_cols[1].metric("Policy Engine p50", f"{bench_data['policy_engine_ms']['p50']:.3f} ms")
        b_cols[2].metric("Execution & Ledger p50", f"{bench_data['execution_and_ledger_ms']['p50']:.3f} ms")
        b_cols[3].metric("Async SHAP p50", f"{bench_data['async_shap_explanation_ms']['p50']:.3f} ms")

        if "raw_latencies_ms" in bench_data:
            fig_bench = px.histogram(
                x=bench_data["raw_latencies_ms"],
                nbins=30,
                title=f"Total Pipeline Latency Distribution ({bench_data['n_runs']} runs)",
                labels={"x": "Total Pipeline Latency (ms)"},
            )
            fig_bench.update_layout(height=280)
            st.plotly_chart(fig_bench, use_container_width=True)
