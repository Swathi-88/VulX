#!/usr/bin/env python3
"""
Anvil Synthetic Payment Event Generator
----------------------------------------
Generates synthetic payment transactions for evaluating fraud detection prototypes.
Features correlated risk signals, 6 category distributions, log-normal amount distributions,
and reproducible showcase demo vectors.

Usage:
    python generate_events.py [--n_events 2000] [--seed 42] [--output data/raw/events.csv] [--demo_output data/processed/demo_cases.json]
"""

import argparse
import json
import sys
from pathlib import Path

# Add src/ directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anvil.config import DEFAULT_N_EVENTS, DEFAULT_SEED, DEFAULT_OUTPUT_CSV, DEFAULT_DEMO_JSON
from anvil.generator.event_builder import generate_dataset
from anvil.generator.showcase import get_showcase_demo_cases
from anvil.utils.output_formatter import print_summary_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic payment events for Anvil fraud detection prototype."
    )
    parser.add_argument(
        "--n_events",
        type=int,
        default=DEFAULT_N_EVENTS,
        help=f"Number of synthetic events to generate (default: {DEFAULT_N_EVENTS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV file path for dataset (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--demo_output",
        type=str,
        default=DEFAULT_DEMO_JSON,
        help=f"Output JSON file path for hand-picked demo cases (default: {DEFAULT_DEMO_JSON})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Initializing Anvil Payment Event Generator (n_events={args.n_events}, seed={args.seed})...")

    # Ensure parent directories exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.demo_output).parent.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    df = generate_dataset(args.n_events, args.seed)

    # Save CSV
    df.to_csv(args.output, index=False)
    print(f"Saved dataset ({len(df)} records) to '{args.output}'.")

    # Generate and save demo cases
    demo_cases = get_showcase_demo_cases()
    with open(args.demo_output, "w", encoding="utf-8") as f:
        json.dump(demo_cases, f, indent=2)
    print(f"Saved {len(demo_cases)} showcase demo cases to '{args.demo_output}'.")

    # Print summary table
    print_summary_table(df)


if __name__ == "__main__":
    main()
