#!/usr/bin/env python3
"""Select an identity-safe, outcome-blind workload-study cohort.

The utility split pools original train and dev records, where numeric
``dataset_row_id`` values overlap. This selector therefore uses the compound
source identity (source_split, dataset_row_id) throughout.
"""

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
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_cases <= 0 or args.n_cases % 10:
        raise ValueError("n_cases must be positive and divisible by 10")
    frame = pd.read_csv(args.utility_manifest_csv, dtype=str).fillna("")
    required = {"dataset_row_id", "subject_id", "note_id", "source_split", "text_length", "utility_split"}
    if missing := required - set(frame.columns):
        raise KeyError(f"utility manifest is missing columns: {sorted(missing)}")
    frame["dataset_row_id"] = pd.to_numeric(frame["dataset_row_id"], errors="raise").astype(int)
    eligible = frame.loc[frame.utility_split.eq(args.utility_split)].copy()
    identity = ["source_split", "dataset_row_id"]
    if eligible.duplicated(identity).any():
        raise ValueError("utility manifest has duplicate source identities")

    excluded_identities, excluded_subjects = set(), set()
    for item in args.exclude_manifest_csv:
        prior = pd.read_csv(item, dtype=str).fillna("")
        if set(identity).issubset(prior.columns):
            excluded_identities |= set(zip(prior.source_split, pd.to_numeric(prior.dataset_row_id, errors="raise")))
        if "subject_id" in prior:
            excluded_subjects |= set(prior.subject_id)
    eligible = eligible.loc[
        ~eligible.apply(lambda row: (row.source_split, int(row.dataset_row_id)) in excluded_identities, axis=1)
        & ~eligible.subject_id.isin(excluded_subjects)
    ].copy()
    eligible["identity_key"] = eligible.source_split + ":" + eligible.dataset_row_id.astype(str)
    eligible["note_score"] = [stable_score(args.seed, "note", value) for value in eligible.identity_key]
    eligible = eligible.sort_values("note_score", kind="stable").drop_duplicates("subject_id", keep="first").copy()
    eligible["text_length"] = pd.to_numeric(eligible.text_length, errors="raise")
    eligible["length_quintile"] = pd.qcut(
        eligible.text_length.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    per_quintile = args.n_cases // 5
    pieces = []
    for quintile, group in eligible.groupby("length_quintile", sort=True):
        selected = group.sort_values("note_score", kind="stable").head(per_quintile).copy()
        if len(selected) != per_quintile:
            raise ValueError(f"insufficient eligible cases in length quintile {quintile}")
        selected["workflow_score"] = [stable_score(args.seed, "workflow", value) for value in selected.identity_key]
        selected = selected.sort_values("workflow_score", kind="stable")
        selected["workflow_condition"] = ["full_manual"] * (per_quintile // 2) + ["automation_assisted"] * (per_quintile // 2)
        pieces.append(selected)
    selected = pd.concat(pieces, ignore_index=True).sort_values(["length_quintile", "identity_key"], kind="stable")
    if selected.subject_id.nunique() != len(selected) or selected.duplicated(identity).any():
        raise ValueError("selected cohort must have one patient and one source identity per case")
    selected["workload_case_rank"] = range(1, len(selected) + 1)
    columns = [
        "workload_case_rank", "dataset_row_id", "subject_id", "note_id", "filename", "source_split",
        "utility_split", "text_length", "length_quintile", "workflow_condition",
    ]
    manifest = selected[[column for column in columns if column in selected]].copy()
    review_form = manifest.copy()
    # Timing begins when any source or candidate material for a case is first visible.
    for column in [
        "review_status", "reviewer_id", "review_start_iso", "review_end_iso", "pause_seconds",
        "active_review_seconds", "final_case_outcome", "current_case_safety_escalation_yes_no", "reviewer_note",
    ]:
        review_form[column] = "pending" if column == "review_status" else ""
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "workload_case_manifest_v2.csv", index=False)
    review_form.to_csv(output_dir / "workload_review_form_v2.csv", index=False)
    summary = {
        "scope": "identity_safe_utility_train_outcome_blind_workload_study_selection",
        "seed": args.seed,
        "n_cases": len(manifest),
        "n_unique_subjects": int(manifest.subject_id.nunique()),
        "workflow_counts": manifest.workflow_condition.value_counts().to_dict(),
        "length_quintile_counts": manifest.length_quintile.value_counts().sort_index().to_dict(),
        "excluded_source_identities": len(excluded_identities),
        "excluded_subjects": len(excluded_subjects),
        "identity_key": ["source_split", "dataset_row_id"],
        "security_note": "Outputs contain provenance IDs, assignments, and derived length strata only; no source text or outcomes are accessed.",
    }
    (output_dir / "workload_case_manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
