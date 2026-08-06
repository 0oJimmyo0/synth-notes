#!/usr/bin/env python3
"""Safely append reviewed alias-recovery facts to a completed source ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VALID_STATUSES = {"verified", "corrected", "omitted"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed_review_csv", required=True)
    parser.add_argument("--delta_review_csv", required=True)
    parser.add_argument("--cohort", required=True, choices=("initial", "replenishment"))
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    completed = pd.read_csv(Path(args.completed_review_csv).resolve())
    delta = pd.read_csv(Path(args.delta_review_csv).resolve())
    delta = delta.loc[delta.cohort.astype(str).eq(args.cohort)].copy()
    if delta.empty:
        raise ValueError(f"No delta rows found for cohort={args.cohort!r}.")
    required = {"case_id", "dataset_row_id", "fact_id", "field", "manual_verification_status", "manual_verified_value", "generation_value"}
    for label, frame in (("completed review", completed), ("delta review", delta)):
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{label} missing columns: {sorted(missing)}")
    if set(delta.field.astype(str)) != {"principal_diagnosis"}:
        raise ValueError("This utility accepts only principal_diagnosis recovery rows.")
    statuses = set(delta.manual_verification_status.fillna("").astype(str).str.lower())
    if not statuses.issubset(VALID_STATUSES) or "omitted" in statuses:
        raise ValueError("Recovery rows must have final verified/corrected statuses before merge.")
    delta["manual_verified_value"] = delta.manual_verified_value.fillna("").astype(str).str.strip()
    delta["generation_value"] = delta.generation_value.fillna("").astype(str).str.strip()
    if (delta.manual_verified_value.eq("") | delta.generation_value.eq("")).any():
        raise ValueError("Retained recovery rows require nonblank manual_verified_value and generation_value.")
    old_keys = set(zip(completed.dataset_row_id.astype(int), completed.field.astype(str)))
    new_keys = set(zip(delta.dataset_row_id.astype(int), delta.field.astype(str)))
    if old_keys.intersection(new_keys):
        raise ValueError("Recovery data overlaps an existing field; refusing to overwrite a completed review.")
    if completed.fact_id.astype(str).isin(delta.fact_id.astype(str)).any():
        raise ValueError("Recovery fact_id collides with a completed-review fact_id.")

    # Preserve the completed review schema; recovery-only metadata is reported
    # in the summary rather than leaking into the source-fact ledger.
    for column in completed.columns:
        if column not in delta:
            delta[column] = ""
    delta = delta[completed.columns].copy()
    # Case-level reviewer decisions must be repeated on the appended fact row;
    # otherwise a still-blocked case appears internally inconsistent to the
    # downstream validator.
    case_level_columns = ("case_blocked", "blocked_missing_required_fields", "case_blocked_reason", "case_blocking_reasons")
    for column in case_level_columns:
        if column not in completed.columns:
            continue
        values = completed.groupby("case_id", dropna=False)[column].agg(
            lambda series: next((value for value in series if pd.notna(value) and str(value).strip()), "")
        )
        delta[column] = delta.case_id.map(values).fillna("")
    merged = pd.concat([completed, delta], ignore_index=True)

    prior_missing = {
        str(row.case_id): str(row.missing_required_fields)
        for row in pd.read_csv(Path(args.delta_review_csv).resolve()).loc[
            lambda frame: frame.cohort.astype(str).eq(args.cohort)
        ].itertuples(index=False)
    }
    unblocked = [case_id for case_id, missing in prior_missing.items() if missing == "principal_diagnosis"]
    if "case_blocked" in merged.columns:
        merged.loc[merged.case_id.astype(str).isin(unblocked), "case_blocked"] = False
    for column in ("blocked_missing_required_fields", "case_blocked_reason", "case_blocking_reasons"):
        if column in merged.columns:
            merged.loc[merged.case_id.astype(str).isin(unblocked), column] = ""

    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    summary = {
        "cohort": args.cohort,
        "n_original_rows": int(len(completed)),
        "n_recovery_rows_added": int(len(delta)),
        "n_merged_rows": int(len(merged)),
        "n_cases_unblocked_by_principal_diagnosis_recovery": int(len(unblocked)),
        "unblocked_case_ids": sorted(unblocked),
        "security_note": "Output remains source-derived and must remain on approved project storage.",
    }
    Path(args.summary_json).resolve().write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
