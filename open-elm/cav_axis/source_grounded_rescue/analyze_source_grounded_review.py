#!/usr/bin/env python3
"""Validate and unblind a ledger-grounded rescue review without exporting text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TEXT_COLUMNS = {"verified_fact_ledger", "synthetic_note", "reviewer_notes"}
SUPPORT_COLUMNS = [
    "principal_diagnosis_supported_yes_no",
    "hospital_course_supported_yes_no",
    "major_procedures_supported_yes_no_not_applicable",
    "discharge_medications_supported_yes_no",
    "disposition_supported_yes_no",
    "follow_up_supported_yes_no",
    "instructions_supported_yes_no",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze blinded source-grounded rescue review labels.")
    parser.add_argument("--review_csv", required=True)
    parser.add_argument("--blinded_key_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def normalize_yes_no(series: pd.Series, field: str) -> pd.Series:
    value = series.fillna("").astype(str).str.strip().str.lower()
    invalid = set(value).difference({"yes", "no"})
    if invalid:
        raise ValueError(f"{field} has invalid values: {sorted(invalid)}")
    return value


def main() -> None:
    args = parse_args()
    review = pd.read_csv(Path(args.review_csv).resolve())
    key = pd.read_csv(Path(args.blinded_key_csv).resolve())
    required_review = set(SUPPORT_COLUMNS + [
        "blinded_output_id", "case_id", "unsupported_major_claim_yes_no",
        "critical_omission_yes_no", "internal_clinical_consistency_score_1to5",
        "overall_factual_faithfulness_score_1to5", "overall_clinical_usability_pass_fail",
    ])
    required_key = {"blinded_output_id", "case_id", "condition", "model_condition", "arm"}
    if missing := required_review.difference(review.columns):
        raise KeyError(f"review is missing columns: {sorted(missing)}")
    if missing := required_key.difference(key.columns):
        raise KeyError(f"key is missing columns: {sorted(missing)}")
    if review.blinded_output_id.duplicated().any() or key.blinded_output_id.duplicated().any():
        raise ValueError("review and key must have unique blinded_output_id values")

    merged = review.merge(key, on=["blinded_output_id", "case_id"], how="inner", validate="one_to_one")
    if len(merged) != len(review) or len(merged) != len(key):
        raise ValueError("review/key rows do not match one-to-one")
    for column in ["unsupported_major_claim_yes_no", "critical_omission_yes_no"]:
        merged[column] = normalize_yes_no(merged[column], column)
    for column in SUPPORT_COLUMNS:
        allowed = {"yes", "no"}
        if column == "major_procedures_supported_yes_no_not_applicable":
            allowed.add("not_applicable")
        value = merged[column].fillna("").astype(str).str.strip().str.lower()
        invalid = set(value).difference(allowed)
        if invalid:
            raise ValueError(f"{column} has invalid values: {sorted(invalid)}")
        merged[column] = value
    for column in ["internal_clinical_consistency_score_1to5", "overall_factual_faithfulness_score_1to5"]:
        merged[column] = pd.to_numeric(merged[column], errors="raise").astype(int)
        if not merged[column].between(1, 5).all():
            raise ValueError(f"{column} must be between 1 and 5")
    merged["overall_clinical_usability_pass_fail"] = merged["overall_clinical_usability_pass_fail"].fillna("").astype(str).str.strip().str.lower()
    if invalid := set(merged["overall_clinical_usability_pass_fail"]).difference({"pass", "fail"}):
        raise ValueError(f"invalid pass/fail values: {sorted(invalid)}")

    merged["rule_derived_pass"] = (
        merged["unsupported_major_claim_yes_no"].eq("no")
        & merged["critical_omission_yes_no"].eq("no")
        & merged["internal_clinical_consistency_score_1to5"].ge(4)
        & merged["overall_factual_faithfulness_score_1to5"].ge(4)
    )
    merged["reviewer_rule_match"] = merged["rule_derived_pass"].eq(merged["overall_clinical_usability_pass_fail"].eq("pass"))
    if not merged["reviewer_rule_match"].all():
        bad = merged.loc[~merged.reviewer_rule_match, "blinded_output_id"].tolist()
        raise ValueError(f"pass/fail values violate required rule: {bad}")

    rows = []
    for condition, group in merged.groupby("condition", sort=True):
        row = {
            "condition": condition,
            "model_condition": group["model_condition"].iloc[0],
            "arm": group["arm"].iloc[0],
            "n_notes": len(group),
            "pass_count": int(group.rule_derived_pass.sum()),
            "pass_rate": float(group.rule_derived_pass.mean()),
            "unsupported_major_claim_rate": float(group.unsupported_major_claim_yes_no.eq("yes").mean()),
            "critical_omission_rate": float(group.critical_omission_yes_no.eq("yes").mean()),
            "mean_internal_consistency": float(group.internal_clinical_consistency_score_1to5.mean()),
            "mean_factual_faithfulness": float(group.overall_factual_faithfulness_score_1to5.mean()),
        }
        for column in SUPPORT_COLUMNS:
            row[f"{column}_rate"] = float(group[column].eq("yes").mean())
        rows.append(row)
    condition_summary = pd.DataFrame(rows).sort_values(["pass_rate", "mean_factual_faithfulness"], ascending=False)

    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    safe_columns = [column for column in merged.columns if column not in TEXT_COLUMNS]
    merged[safe_columns].to_csv(output_dir / "source_grounded_review_unblinded_label_matrix.csv", index=False)
    condition_summary.to_csv(output_dir / "source_grounded_review_condition_summary.csv", index=False)
    overall = {
        "n_notes": int(len(merged)),
        "n_cases": int(merged.case_id.nunique()),
        "pass_count": int(merged.rule_derived_pass.sum()),
        "pass_rate": float(merged.rule_derived_pass.mean()),
        "unsupported_major_claim_count": int(merged.unsupported_major_claim_yes_no.eq("yes").sum()),
        "critical_omission_count": int(merged.critical_omission_yes_no.eq("yes").sum()),
        "pass_rule_verified": True,
        "security_note": "Derived outputs exclude verified ledger text, synthetic notes, and reviewer free-text comments.",
    }
    (output_dir / "source_grounded_review_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    table_columns = ["condition", "n_notes", "pass_count", "pass_rate", "unsupported_major_claim_rate", "critical_omission_rate", "mean_factual_faithfulness"]
    header = "| " + " | ".join(table_columns) + " |"
    divider = "|" + "|".join(["---"] * len(table_columns)) + "|"
    table_rows = [
        "| " + " | ".join(str(row[column]) for column in table_columns) + " |"
        for _, row in condition_summary[table_columns].iterrows()
    ]
    lines = ["# Source-Grounded Rescue Review", "", f"- Cases: {overall['n_cases']}", f"- Notes: {overall['n_notes']}", f"- Rule-verified passes: {overall['pass_count']}/{overall['n_notes']} ({overall['pass_rate']:.1%})", f"- Unsupported major claims: {overall['unsupported_major_claim_count']}", f"- Critical omissions: {overall['critical_omission_count']}", "", "## Condition Comparison", "", header, divider, *table_rows, "", "Interpret results using the predeclared study-specific feasibility thresholds; this review does not establish clinical deployment performance."]
    (output_dir / "source_grounded_review_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(overall, indent=2))
    print(condition_summary[["condition", "n_notes", "pass_count", "pass_rate", "unsupported_major_claim_rate", "critical_omission_rate", "mean_factual_faithfulness"]].to_string(index=False))


if __name__ == "__main__":
    main()
