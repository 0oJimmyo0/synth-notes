#!/usr/bin/env python3
"""Compute text-free inter-reviewer agreement on overlapping blinded outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FIELDS = [
    "principal_diagnosis_supported_yes_no",
    "hospital_course_supported_yes_no",
    "major_procedures_supported_yes_no_not_applicable",
    "discharge_medications_supported_yes_no",
    "disposition_supported_yes_no",
    "follow_up_supported_yes_no",
    "instructions_supported_yes_no",
    "unsupported_major_claim_yes_no",
    "critical_omission_yes_no",
    "internal_clinical_consistency_score_1to5",
    "overall_factual_faithfulness_score_1to5",
    "overall_clinical_usability_pass_fail",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure agreement on a blinded review subset.")
    parser.add_argument("--primary_review_csv", required=True)
    parser.add_argument("--second_review_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def cohen_kappa(left: pd.Series, right: pd.Series) -> float | None:
    if not len(left):
        return None
    observed = float(left.eq(right).mean())
    left_p = left.value_counts(normalize=True)
    right_p = right.value_counts(normalize=True)
    expected = sum(float(left_p.get(value, 0.0) * right_p.get(value, 0.0)) for value in set(left_p.index) | set(right_p.index))
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def main() -> None:
    args = parse_args()
    primary = pd.read_csv(Path(args.primary_review_csv).resolve())
    second = pd.read_csv(Path(args.second_review_csv).resolve())
    if primary.blinded_output_id.duplicated().any() or second.blinded_output_id.duplicated().any():
        raise ValueError("each review file must contain unique blinded_output_id values")
    if missing := set(FIELDS).difference(primary.columns) | set(FIELDS).difference(second.columns):
        raise KeyError(f"review files are missing fields: {sorted(missing)}")
    merged = primary[["blinded_output_id", *FIELDS]].merge(
        second[["blinded_output_id", *FIELDS]], on="blinded_output_id", suffixes=("_primary", "_second"), how="inner", validate="one_to_one"
    )
    if merged.empty:
        raise ValueError("the two review files do not share blinded_output_id values")
    rows = []
    for field in FIELDS:
        left = merged[f"{field}_primary"].fillna("").astype(str).str.strip().str.lower()
        right = merged[f"{field}_second"].fillna("").astype(str).str.strip().str.lower()
        rows.append({
            "field": field,
            "n_overlap": len(merged),
            "exact_agreement_rate": float(left.eq(right).mean()),
            "cohen_kappa": cohen_kappa(left, right),
        })
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "reviewer_agreement_summary.csv", index=False)
    summary = {
        "n_overlapping_blinded_outputs": int(len(merged)),
        "mean_exact_agreement_rate": float(table.exact_agreement_rate.mean()),
        "median_exact_agreement_rate": float(table.exact_agreement_rate.median()),
        "security_note": "Agreement outputs exclude fact-ledger text, synthetic notes, and reviewer comments.",
    }
    (output_dir / "reviewer_agreement_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
