#!/usr/bin/env python3
"""Analyze a medication judge against frozen human review labels.

This produces feasibility metrics. Small held-out samples must be reported with
counts and confidence intervals; this script never labels a judge as deployed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local medication-judge decisions with human review.")
    parser.add_argument("--judge_output_path", required=True)
    parser.add_argument("--human_review_csv", action="append", required=True, help="One or more frozen human review CSVs.")
    parser.add_argument("--reference_csv", help="Optional derived reference with blinded_output_id and reference_material_discrepancy.")
    parser.add_argument(
        "--positive_label",
        choices=("severe_failure", "any_medication_error"),
        default="severe_failure",
        help="Human-positive definition when --reference_csv is not supplied.",
    )
    parser.add_argument("--expected_repeats", type=int, default=3, help="Required outputs per note for a complete stability assessment.")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def yes(value: object) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1", "fail"}


def main() -> None:
    args = parse_args()
    if args.expected_repeats < 1:
        raise ValueError("--expected_repeats must be at least 1")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    human = pd.concat([pd.read_csv(path).fillna("") for path in args.human_review_csv], ignore_index=True, sort=False)
    if args.reference_csv:
        reference = pd.read_csv(args.reference_csv).fillna("")
        if "reference_material_discrepancy" not in reference:
            raise KeyError("reference CSV requires reference_material_discrepancy")
        human = reference
    if "blinded_output_id" not in human:
        raise KeyError("human review requires blinded_output_id")
    if human.blinded_output_id.duplicated().any():
        raise ValueError("Human-review blinded_output_id values must be unique across supplied files.")
    def severe_label(row: pd.Series) -> bool:
        medication_label = str(row.get("human_severe_medication_error_yes_no", "")).strip()
        if medication_label:
            return yes(medication_label)
        return yes(row.get("unsupported_major_claim_yes_no")) or yes(row.get("critical_omission_yes_no"))

    if args.reference_csv:
        human["human_severe_failure"] = human["reference_material_discrepancy"].astype(bool)
        human_label_source = "adjudicated material-discrepancy reference"
        positive_label_name = "reference_material_discrepancy"
    elif args.positive_label == "any_medication_error":
        if "human_medication_error_yes_no" not in human.columns:
            raise KeyError("--positive_label any_medication_error requires human_medication_error_yes_no.")
        human["human_severe_failure"] = human["human_medication_error_yes_no"].map(yes)
        human_label_source = "medication-specific human label"
        positive_label_name = "human_medication_error"
    else:
        human["human_severe_failure"] = human.apply(severe_label, axis=1)
        human_label_source = "medication-specific label when available; otherwise unsupported_major_claim OR critical_omission"
        positive_label_name = "human_severe_failure"
    judge_rows = []
    with Path(args.judge_output_path).resolve().open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            payload = record.get("judge_output", record)
            output_id = str(record.get("blinded_output_id", record.get("task_id", "").split("::")[-1]))
            judge_rows.append({
                "blinded_output_id": output_id,
                "judge_final_reject": bool(payload.get("final_reject", False)) if isinstance(payload, dict) else False,
                "judge_requires_human_review": bool(payload.get("requires_human_review", payload.get("final_reject", False))) if isinstance(payload, dict) else True,
                "judge_schema_valid": isinstance(payload, dict) and bool(record.get("schema_valid", True)) and record.get("parse_error") is None,
            })
    judge_repeats = pd.DataFrame(judge_rows)
    if judge_repeats.empty:
        raise ValueError("No parseable judge outputs were found.")
    judge = judge_repeats.groupby("blinded_output_id", as_index=False).agg(
        judge_repeat_count=("judge_final_reject", "size"),
        judge_any_reject=("judge_final_reject", "any"),
        judge_all_reject=("judge_final_reject", "all"),
        judge_any_requires_human_review=("judge_requires_human_review", "any"),
        judge_label_stable=("judge_final_reject", lambda values: values.nunique() == 1),
        judge_schema_valid=("judge_schema_valid", "all"),
    )
    judge["judge_complete_repeats"] = judge["judge_repeat_count"] == args.expected_repeats
    # Instability and invalid output are review routes, never automatic passes.
    judge["judge_final_reject"] = judge.judge_any_reject
    judge["judge_review_route"] = (
        judge.judge_any_requires_human_review
        | ~judge.judge_label_stable
        | ~judge.judge_schema_valid
        | ~judge.judge_complete_repeats
    )
    merged = human.merge(judge, on="blinded_output_id", how="left", validate="one_to_one")
    if merged.judge_final_reject.isna().any():
        raise ValueError("Some human-reviewed outputs have no judge result.")
    merged["judge_final_reject"] = merged.judge_final_reject.astype(bool)
    merged["judge_review_route"] = merged.judge_review_route.astype(bool)
    tp = int((merged.human_severe_failure & merged.judge_final_reject).sum())
    fn = int((merged.human_severe_failure & ~merged.judge_final_reject).sum())
    fp = int((~merged.human_severe_failure & merged.judge_final_reject).sum())
    tn = int((~merged.human_severe_failure & ~merged.judge_final_reject).sum())
    safe = lambda n, d: float(n / d) if d else None
    summary = {
        "n_notes": len(merged), "positive_label_name": positive_label_name,
        "reference_positive_count": int(merged.human_severe_failure.sum()),
        "human_label_source": human_label_source,
        "true_positive": tp, "false_negative": fn, "false_positive": fp, "true_negative": tn,
        "positive_label_sensitivity": safe(tp, tp + fn), "specificity": safe(tn, tn + fp),
        "false_rejection_rate": safe(fp, fp + tn),
        "human_review_route_rate": float(merged.judge_review_route.mean()),
        "material_discrepancy_route_sensitivity": safe(int((merged.human_severe_failure & merged.judge_review_route).sum()), int(merged.human_severe_failure.sum())),
        "schema_valid_rate": float(merged.judge_schema_valid.mean()),
        "complete_repeat_rate": float(merged.judge_complete_repeats.mean()),
        "repeat_label_stability_rate": float(merged.judge_label_stable.mean()),
        "review_route_rate_from_instability_invalidity_or_incompleteness": float((~merged.judge_label_stable | ~merged.judge_schema_valid | ~merged.judge_complete_repeats).mean()),
        "interpretation": "Feasibility only. Do not use this as an automatic clinical gate without an independent prospective validation region.",
    }
    if positive_label_name == "human_severe_failure":
        summary["severe_error_sensitivity"] = summary["positive_label_sensitivity"]
    merged.drop(columns=[c for c in ("verified_fact_ledger", "synthetic_note", "reviewer_notes") if c in merged], errors="ignore").to_csv(output_dir / "medication_judge_human_comparison.csv", index=False)
    (output_dir / "medication_judge_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
