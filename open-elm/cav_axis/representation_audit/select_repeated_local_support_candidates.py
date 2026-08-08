#!/usr/bin/env python3
"""Select one generated candidate per case using frozen repeated local support.

Selection does not use clinical labels, MedGemma output, or target-region
membership. It ranks each eligible candidate set by worst-case support across
frozen reference splits and explicitly records cases with no eligible candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support_summary_csv", required=True)
    parser.add_argument("--generation_manifest_path", required=True)
    parser.add_argument("--posthoc_constraint_audit_path", default=None)
    parser.add_argument("--course_constraint_audit_path", default=None)
    parser.add_argument("--expected_candidates_per_case", type=int, default=4)
    parser.add_argument(
        "--analysis_scope",
        default="generated_candidate_selection",
        help="Provenance label for this selection run (for example heldout_test).",
    )
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    support = pd.read_csv(Path(args.support_summary_csv).resolve())
    required = {"rescue_id", "case_id", "candidate_index", "mean_support", "min_support", "max_support"}
    if missing := required.difference(support.columns):
        raise KeyError(f"Support summary is missing columns: {sorted(missing)}")
    if support.rescue_id.duplicated().any():
        raise ValueError("Support summary has duplicate rescue IDs.")
    counts = support.groupby("case_id").rescue_id.size()
    if not counts.eq(args.expected_candidates_per_case).all():
        bad = counts.loc[~counts.eq(args.expected_candidates_per_case)].to_dict()
        raise ValueError(f"Cases do not have {args.expected_candidates_per_case} scored candidates: {bad}")
    all_candidate_case_ids = set(support.case_id.astype(str))
    manifest = pd.read_json(Path(args.generation_manifest_path).resolve(), lines=True)
    if "rescue_id" not in manifest or manifest.rescue_id.duplicated().any():
        raise ValueError("Generation manifest must contain unique rescue_id values.")
    if args.course_constraint_audit_path:
        course_audit = pd.read_json(Path(args.course_constraint_audit_path).resolve(), lines=True)
        required_audit = {"rescue_id", "course_constraint_pass", "hit_max_new_tokens"}
        if missing := required_audit.difference(course_audit.columns):
            raise KeyError(f"Course constraint audit is missing columns: {sorted(missing)}")
        if course_audit.rescue_id.duplicated().any():
            raise ValueError("Course constraint audit has duplicate rescue IDs.")
        support = support.merge(
            course_audit[["rescue_id", "course_constraint_pass", "hit_max_new_tokens"]].rename(
                columns={
                    "course_constraint_pass": "current_course_constraint_pass",
                    "hit_max_new_tokens": "current_hit_max_new_tokens",
                }
            ), on="rescue_id", how="left", validate="one_to_one"
        )
        if support[["current_course_constraint_pass", "current_hit_max_new_tokens"]].isna().any().any():
            raise ValueError("Some support rows are absent from the course constraint audit.")
        support = support.loc[
            support.current_course_constraint_pass.astype(bool)
            & ~support.current_hit_max_new_tokens.astype(bool)
        ].copy()
        if support.empty:
            raise ValueError("No eligible non-capped constraint-passing candidates remain.")
    eligible_case_ids = set(support.case_id.astype(str))
    unselected_case_ids = sorted(all_candidate_case_ids.difference(eligible_case_ids))
    selected = support.sort_values(
        ["case_id", "min_support", "mean_support", "candidate_index"],
        ascending=[True, False, False, True], kind="stable",
    ).groupby("case_id", as_index=False, sort=True).head(1).copy()
    selected["selection_rank"] = 1
    selected["selection_rule"] = "max_min_support_then_mean_support_then_lowest_candidate_index"
    output = selected.merge(manifest, on=["rescue_id", "case_id", "candidate_index"], how="left", validate="one_to_one")
    if output.generated_text.isna().any():
        raise ValueError("Selected support rows could not be joined to generated outputs.")
    posthoc_constraint_audit_used = False
    if "current_course_constraint_pass" in output:
        if not output.current_course_constraint_pass.astype(bool).all():
            raise ValueError("Selected outputs include a current failed course constraint.")
        if output.current_hit_max_new_tokens.astype(bool).any():
            raise ValueError("Selected outputs include a max-token cap hit.")
    elif not output.course_constraint_pass.astype(bool).all():
        if not args.posthoc_constraint_audit_path:
            raise ValueError("Selected outputs include a failed course constraint.")
        audit = json.loads(Path(args.posthoc_constraint_audit_path).resolve().read_text())
        if int(audit.get("posthoc_constraint_failure_count", -1)) != 0:
            raise ValueError("Post-hoc constraint audit does not clear all generated outputs.")
        posthoc_constraint_audit_used = True
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_json(output_dir / "selected_generated_manifest.jsonl", orient="records", lines=True)
    output.drop(columns=[column for column in ["generated_text", "hospital_course_text"] if column in output]).to_csv(
        output_dir / "selected_candidate_scores.csv", index=False
    )
    pd.DataFrame({
        "case_id": unselected_case_ids,
        "selection_status": "no_eligible_non_capped_constraint_passing_candidate",
    }).to_csv(output_dir / "unselected_cases.csv", index=False)
    summary = {
        "scope": args.analysis_scope,
        "n_cases_candidate_pool": len(all_candidate_case_ids),
        "n_cases": int(output.case_id.nunique()),
        "n_selected_candidates": int(len(output)),
        "n_unselected_cases": len(unselected_case_ids),
        "selection_rule": "max_min_support_then_mean_support_then_lowest_candidate_index",
        "target_region_rule_used": False,
        "clinical_or_judge_labels_used": False,
        "posthoc_constraint_audit_used": posthoc_constraint_audit_used,
        "course_constraint_audit_used": bool(args.course_constraint_audit_path),
        "selected_by_support_arm_and_stratum": output.groupby(
            ["support_arm", "cohort_stratum"], dropna=False
        ).size().rename("n").reset_index().to_dict(orient="records"),
        "security_note": "The selected manifest contains synthetic text and remains on approved project storage.",
    }
    (output_dir / "selected_candidate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
