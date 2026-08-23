"""
Terminal output formatters and reporting utilities for Anvil event summary statistics.
"""

def print_summary_table(df):
    """Prints a clean summary table of category counts, percentages, and label balance."""
    total_events = len(df)
    categories = ["normal", "suspicious", "borderline", "fraud", "legitimate_but_unusual", "merchant_anomaly"]

    print("\n" + "=" * 80)
    print(f" ANVIL SYNTHETIC EVENT GENERATION SUMMARY (Total Events: {total_events})")
    print("=" * 80)

    header = f"{'Category Tag':<25} | {'Count':<7} | {'Pct':<7} | {'Legitimate':<10} | {'Fraud':<7} | {'Fraud %':<8}"
    print(header)
    print("-" * 80)

    for cat in categories:
        sub = df[df["category_tag"] == cat]
        cnt = len(sub)
        pct = (cnt / total_events) * 100
        leg = len(sub[sub["ground_truth_label"] == "legitimate"])
        frd = len(sub[sub["ground_truth_label"] == "fraud"])
        frd_pct = (frd / cnt * 100) if cnt > 0 else 0.0

        print(f"{cat:<25} | {cnt:<7} | {pct:>5.1f}%  | {leg:<10} | {frd:<7} | {frd_pct:>6.1f}%")

    print("-" * 80)
    tot_leg = len(df[df["ground_truth_label"] == "legitimate"])
    tot_frd = len(df[df["ground_truth_label"] == "fraud"])
    tot_frd_pct = (tot_frd / total_events) * 100
    print(f"{'OVERALL TOTAL':<25} | {total_events:<7} | {'100.0%':<7} | {tot_leg:<10} | {tot_frd:<7} | {tot_frd_pct:>6.1f}%")
    print("=" * 80 + "\n")
