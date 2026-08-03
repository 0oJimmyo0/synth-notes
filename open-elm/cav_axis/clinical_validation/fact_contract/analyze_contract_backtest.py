#!/usr/bin/env python3
"""Compare contract-audit routes with completed human medication labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract_note_coverage_csv", required=True)
    parser.add_argument("--human_review_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def yes(value: object) -> bool:
    return str(value).strip().lower() == "yes"


def main() -> None:
    args = parse_args()
    coverage = pd.read_csv(Path(args.contract_note_coverage_csv).resolve())
    human = pd.read_csv(Path(args.human_review_csv).resolve())
    if "case_id" not in coverage.columns or "case_id" not in human.columns:
        raise KeyError("coverage and review CSVs must contain case_id")
    if coverage.case_id.duplicated().any() or human.case_id.duplicated().any():
        raise ValueError("This minimal analyzer requires one final note per case; subset repeated-anchor reviews first.")
    required = {"human_medication_error_yes_no", "human_severe_medication_error_yes_no"}
    if missing := required.difference(human.columns):
        raise KeyError(f"human review missing columns: {sorted(missing)}")
    merged = coverage.merge(human, on="case_id", how="inner", validate="one_to_one")
    route = ~merged.contract_pass.astype(bool)
    severe = merged.human_severe_medication_error_yes_no.map(yes)
    any_error = merged.human_medication_error_yes_no.map(yes)
    def sensitivity(label: pd.Series) -> float | None:
        return float((route & label).sum() / label.sum()) if label.any() else None
    summary = {
        "n_linked_notes": int(len(merged)),
        "contract_route_rate": float(route.mean()) if len(merged) else 0.0,
        "severe_error_sensitivity": sensitivity(severe),
        "any_medication_error_sensitivity": sensitivity(any_error),
        "false_rejection_rate_among_human_no_medication_error": float((route & ~any_error).sum() / (~any_error).sum()) if (~any_error).any() else None,
        "decision_boundary": "Backtest only. Do not use as an automatic clinical gate until severe failures and false rejections are manually audited.",
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "contract_backtest_row_table.csv", index=False)
    (output_dir / "contract_backtest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
