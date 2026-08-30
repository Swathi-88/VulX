# Anvil 5-Minute Pitch Video Script

**Title**: Anvil: Policy-as-Code Routing & Governance for Vulcan Fraud Detection  
**Target Duration**: 5:00 minutes  

---

## Timeline & Scene-by-Scene Script

### Scene 1: The One-Line Thesis (0:00 – 0:30)
**Visual**: Presenter on camera / Title slide showing **"Vulcan answers what should happen. Anvil answers what we're allowed to do about it."**

> **Speaker**:  
> "Machine learning risk models excel at predicting fraud probabilities, but raw risk scores shouldn't blindly block real payments. A raw ML model only tells you what *might* happen based on historical patterns. But in enterprise payments, business rules, customer tenure, legal provenance, and false-positive costs dictate what you're *allowed* to do about it.  
> 
> Welcome to **Anvil**: a policy-as-code routing and governance layer that sits between Vulcan ML risk predictions and real-world payment execution, turning raw model outputs into auditable, reversible, and compliant business decisions."

---

### Scene 2: The Killer Demo Case Run Live (0:30 – 2:00)
**Visual**: Terminal screen running `python run_pipeline.py` / Streamlit UI `app.py` showcasing Transaction `demo_fp_001_showcase`.

> **Speaker**:  
> "Let's watch Anvil in action on our killer demo case: a ₹45,000 payment by a loyal customer of 400 days, transacting from a new device and new location.  
> 
> 1. **Phase 2 (Vulcan Model)**: The raw XGBoost model sees high device novelty and location deviation. It outputs an **87% fraud risk score** and issues a naive recommendation to **BLOCK**. In a naive system, this customer's payment is silently declined, alienating a high-value user.  
> 2. **Phase 3 (Anvil Policy Engine)**: Anvil intercepts the decision contract. It evaluates `policies.yaml`:
>    - *Severity*: ₹45,000 is `high` severity.
>    - *FP Cost*: 400 days tenure + high amount = `high` false-positive cost.
>    - *Reversibility*: `BLOCK` is reversible via step-up authentication.
>    - *Policy Decision*: Anvil overrides naive `BLOCK` and routes the payment to **`VERIFY`**.
> 3. **Phase 4 (Execution & Ledger)**: Anvil sends a 2FA step-up prompt to the customer's verified mobile device. The customer passes verification! The payment completes successfully, and Anvil logs the event into SQLite as a **Prevented False Positive (TN)** with `cross_merchant_derived` legal tags and `financial_record_required` retention."

---

### Scene 3: Metrics Dashboard & Feedback Retraining (2:00 – 3:30)
**Visual**: Running `python metrics_dashboard.py` and `python retrain_and_compare.py` on terminal / Dashboard UI.

> **Speaker**:  
> "Now let's look at the systemic impact across our entire transaction volume:
> 
> When we run `python metrics_dashboard.py`, we see that Anvil routes 37.5% of transactions to `ALLOW`, 50% to step-up `VERIFY`, and 12.5% to `HUMAN_REVIEW`.  
> 
> Crucially, our step-up `VERIFY` path recovered 100% of high-value false positives while still blocking true fraud! Comparing the raw naive model vs the Anvil-governed system side-by-side: Anvil slashes the False Positive Rate while keeping Recall intact!  
> 
> Furthermore, when we run `python feedback_pipeline.py` and `retrain_and_compare.py`, Anvil automatically harvests FP/FN edge cases from the SQLite ledger into `correction_increment.csv`, retrains XGBoost, and proves measurable precision and F1 improvements on held-out test data!"

---

### Scene 4: Why Anvil Beats Single-Track Submissions (3:30 – 4:30)
**Visual**: Slide highlighting the 4 Buildathon Track Pillars vs Anvil Capabilities.

> **Speaker**:  
> "Why does Anvil beat single-track hackathon entries? Because single-track entries solve isolated puzzles, while Anvil addresses all four hackathon governance requirements in one unified pipeline:  
> 
> 1. **Explainability & SHAP**: Standard Decision Contracts expose real SHAP feature contributions and bootstrap ensemble uncertainty.  
> 2. **Audit Trails & Provenance**: Every decision is stored in SQLite with explicit `legal_basis_tags` (`cross_merchant_derived` vs `own_history`) and financial retention bounds.  
> 3. **False-Positive Cost Optimization**: `policies.yaml` explicitly balances tenure and transaction amount against decline costs.  
> 4. **Honest Exception Lists**: Our `HUMAN_REVIEW` console isolates genuine edge cases that require analyst oversight, preventing blind automated failures.  
> 
> Anvil doesn't just put a prettier UI on top of Vulcan — it provides a complete governance operating system."

---

### Scene 5: Future Roadmap & Production Scaling (4:30 – 5:00)
**Visual**: Closing slide with GitHub repository link & architecture diagram.

> **Speaker**:  
> "Today, Anvil runs on synthetic event streams and a local XGBoost simulator. With direct access to production Vulcan real-time feature stores, Anvil would scale seamlessly:  
> - Dynamically pulling live merchant network graph embeddings.  
> - Hot-reloading YAML policy changes across distributed Kubernetes microservices.  
> - Streaming audit ledger logs directly into enterprise Kafka topics and data warehouses.  
> 
> Thank you for watching!"
