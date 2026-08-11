#!/usr/bin/env python3
"""Select a fresh, outcome-blind workload-study cohort and review form."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def stable_score(seed: int, *parts: object) -> str:
    return hashlib.sha256(f"{seed}|".encode() + "|".join(map(str, parts)).encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility_manifest_csv", required=True)
    parser.add_argument("--exclude_manifest_csv", action="append", default=[])
    parser.add_argument("--utility_split", default="train")
    parser.add_argument("--n_cases", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_cases <= 0 or args.n_cases % 10:
        raise ValueError("n_cases must be positive and divisible by 10 for balanced workflow and quintile strata")
    frame = pd.read_csv(args.utility_manifest_csv, dtype=str).fillna("")
    required = {"dataset_row_id", "subject_id", "text_length", "utility_split"}
    if missing := required - set(frame.columns):
        raise KeyError(f"utility manifest is missing columns: {sorted(missing)}")
    eligible = frame.loc[frame.utility_split.eq(args.utility_split)].copy()
    excluded_rows, excluded_subjects = set(), set()
    for path in args.exclude_manifest_csv:
        prior = pd.read_csv(path, dtype=str).fillna("")
        if "dataset_row_id" in prior:
            excluded_rows |= set(prior.dataset_row_id)
        if "subject_id" in prior:
            excluded_subjects |= set(prior.subject_id)
    eligible = eligible.loc[
        ~eligible.dataset_row_id.isin(excluded_rows) & ~eligible.subject_id.isin(excluded_subjects)
    ].copy()
    eligible["note_score"] = [stable_score(args.seed, "note", value) for value in eligible.dataset_row_id]
    # The utility manifest may repeat canonical rows; retain one deterministic
    # representation before enforcing one selected note per patient.
    eligible = eligible.sort_values("note_score", kind="stable").drop_duplicates("dataset_row_id", keep="first")
    one_per_subject = eligible.drop_duplicates("subject_id", keep="first").copy()
    one_per_subject["text_length"] = pd.to_numeric(one_per_subject.text_length, errors="raise")
    one_per_subject["length_quintile"] = pd.qcut(
        one_per_subject.text_length.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    per_quintile = args.n_cases // 5
    pieces = []
    for quintile, group in one_per_subject.groupby("length_quintile", sort=True):
        selected = group.sort_values("note_score", kind="stable").head(per_quintile).copy()
        if len(selected) != per_quintile:
            raise ValueError(f"insufficient eligible cases in length quintile {quintile}")
        selected["workflow_score"] = [stable_score(args.seed, "workflow", value) for value in selected.dataset_row_id]
        selected = selected.sort_values("workflow_score", kind="stable")
        selected["workflow_condition"] = ["full_manual"] * (per_quintile // 2) + ["automation_assisted"] * (per_quintile // 2)
        pieces.append(selected)
    selected = pd.concat(pieces, ignore_index=True).sort_values(["length_quintile", "dataset_row_id"], kind="stable")
    if selected.subject_id.nunique() != len(selected):
        raise ValueError("selected cohort must contain one case per patient")
    if selected.dataset_row_id.nunique() != len(selected):
        raise ValueError("selected cohort must contain one canonical dataset row per case")
    selected["workload_case_rank"] = range(1, len(selected) + 1)
    manifest_columns = [column for column in [
        "workload_case_rank", "dataset_row_id", "subject_id", "note_id", "source_split",
        "utility_split", "text_length", "length_quintile", "workflow_condition",
    ] if column in selected]
    manifest = selected.loc[:, manifest_columns]
    review_form = manifest.copy()
    for column in [
        "review_status", "reviewer_id", "review_start_iso", "review_end_iso", "pause_seconds",
        "active_review_seconds", "final_case_outcome", "candidate_shown_count",
        "candidate_auto_accept_count", "candidate_manual_review_count", "candidate_accepted_unchanged_count",
        "candidate_edited_count", "candidate_rejected_count", "manual_obligation_added_count",
        "unsupported_candidate_accepted_yes_no", "critical_final_omission_yes_no",
        "medication_or_disposition_error_yes_no", "reviewer_note",
    ]:
        review_form[column] = "pending" if column == "review_status" else ""
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "workload_case_manifest_v1.csv", index=False)
    review_form.to_csv(output_dir / "workload_review_form_v1.csv", index=False)
    summary = {
        "scope": "fresh_utility_train_outcome_blind_workload_study_selection",
        "seed": args.seed,
        "n_cases": len(manifest),
        "n_unique_subjects": int(manifest.subject_id.nunique()),
        "workflow_counts": manifest.workflow_condition.value_counts().to_dict(),
        "length_quintile_counts": manifest.length_quintile.value_counts().sort_index().to_dict(),
        "excluded_dataset_rows": len(excluded_rows),
        "excluded_subjects": len(excluded_subjects),
        "security_note": "Outputs contain provenance IDs, assignments, and derived length strata only; no source text or outcomes are accessed.",
    }
    (output_dir / "workload_case_manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
