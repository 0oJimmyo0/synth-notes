#!/usr/bin/env python3
"""Prepare a prediction-blind atomic-contract addendum for human agreement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent_source_review_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review = pd.read_csv(args.independent_source_review_csv, dtype=str).fillna("")
    required = {
        "case_id", "fact_id", "field", "generation_value", "source_fact_id",
        "source_char_start", "source_char_end", "independent_contract_status",
        "independent_contract_section",
    }
    if missing := required - set(review.columns):
        raise KeyError(f"independent source review is missing columns: {sorted(missing)}")
    parents = review.loc[review.independent_contract_status.str.lower().eq("required")].copy()
    if parents.empty:
        raise ValueError("independent source review has no required rows")
    columns = [
        "case_id", "fact_id", "field", "generation_value", "source_fact_id",
        "source_char_start", "source_char_end", "independent_contract_section",
    ]
    template = parents.loc[:, columns].copy()
    template = template.rename(columns={"fact_id": "parent_fact_id", "generation_value": "source_generation_value"})
    template["independent_parent_resolution"] = "pending"
    template["independent_atom_sequence"] = ""
    template["independent_atomic_status"] = ""
    template["independent_atomic_section"] = ""
    template["independent_atomic_generation_value"] = ""
    template["independent_source_span_start"] = ""
    template["independent_source_span_end"] = ""
    template["independent_atom_reviewer_note"] = ""

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "independent_atomic_contract_addendum_RESTRICTED.csv"
    template.to_csv(output, index=False)
    summary = {
        "scope": "prediction_blind_independent_atomic_contract_addendum",
        "n_cases": int(template.case_id.nunique()),
        "n_required_source_parent_rows": int(len(template)),
        "review_instruction": "Set each parent to atomic_created or route_manual, then create one or more atomic rows per atomic_created parent.",
        "security_note": "The template contains restricted source-derived text and must remain on approved project storage.",
    }
    (output_dir / "independent_atomic_contract_addendum_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
