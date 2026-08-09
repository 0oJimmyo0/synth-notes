#!/usr/bin/env python3
"""Audit a note-linked, hospital-observed 30-day return outcome without note text.

This is a feasibility audit, not a final downstream-task label definition. It
reports event prevalence, competing death, and right-censoring so the project
can decide whether observed return is a defensible utility endpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_manifest_path", required=True)
    parser.add_argument("--admissions_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--window_days", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_days <= 0:
        raise ValueError("window_days must be positive")

    split = pd.read_csv(Path(args.split_manifest_path).resolve(), low_memory=False)
    required_split = {"dataset_row_id", "subject_id", "hadm_id", "split", "patient_disjoint_from_train"}
    if missing := required_split.difference(split.columns):
        raise KeyError(f"Split manifest missing columns: {sorted(missing)}")
    if split.hadm_id.duplicated().any():
        raise ValueError("The split manifest must contain one row per hadm_id for this audit.")

    admissions = pd.read_csv(
        Path(args.admissions_path).resolve(),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime"],
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    if admissions.hadm_id.duplicated().any():
        raise ValueError("Admissions input has duplicate hadm_id values.")

    index = split[["dataset_row_id", "subject_id", "hadm_id", "split", "patient_disjoint_from_train"]].merge(
        admissions, on="hadm_id", how="left", validate="one_to_one", suffixes=("_split", "_admission")
    )
    subject_mismatch = (
        index.subject_id_split.notna()
        & index.subject_id_admission.notna()
        & index.subject_id_split.ne(index.subject_id_admission)
    )
    if subject_mismatch.any():
        raise ValueError(f"{int(subject_mismatch.sum())} subject_id values disagree across inputs.")
    index["subject_id"] = index.subject_id_admission.fillna(index.subject_id_split)
    index = index.drop(columns=["subject_id_split", "subject_id_admission"])

    admission_index = admissions.set_index("hadm_id")
    index["next_admittime"] = pd.NaT
    # Compute next admission from all observed hospital admissions, not only
    # note-linked rows. This avoids treating missing note coverage as no return.
    by_subject = {
        subject: group.admittime.sort_values().to_numpy(dtype="datetime64[ns]")
        for subject, group in admissions.groupby("subject_id", sort=False)
    }
    next_values = []
    for row in index[["subject_id", "dischtime"]].itertuples(index=False):
        if pd.isna(row.subject_id) or pd.isna(row.dischtime):
            next_values.append(pd.NaT)
            continue
        times = by_subject[row.subject_id]
        position = np.searchsorted(times, np.datetime64(row.dischtime), side="right")
        next_values.append(pd.Timestamp(times[position]) if position < len(times) else pd.NaT)
    index["next_admittime"] = next_values

    window = pd.Timedelta(days=args.window_days)
    data_end = admissions.dischtime.max()
    index["window_end"] = index.dischtime + window
    index["missing_admission_link"] = index.admittime.isna() | index.dischtime.isna()
    index["death_within_window"] = (
        index.deathtime.notna()
        & index.dischtime.notna()
        & index.deathtime.gt(index.dischtime)
        & index.deathtime.le(index.window_end)
    )
    index["administratively_censored"] = index.window_end.gt(data_end)
    index["outcome_eligible"] = ~(
        index.missing_admission_link | index.death_within_window | index.administratively_censored
    )
    index["observed_return_within_window"] = (
        index.outcome_eligible
        & index.next_admittime.notna()
        & index.next_admittime.le(index.window_end)
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    export_columns = [
        "dataset_row_id", "subject_id", "hadm_id", "split", "patient_disjoint_from_train",
        "missing_admission_link", "death_within_window", "administratively_censored",
        "outcome_eligible", "observed_return_within_window",
    ]
    index[export_columns].to_csv(output_dir / "note_level_30day_observed_return_labels.csv", index=False)

    summary = (
        index.groupby(["split", "patient_disjoint_from_train"], dropna=False)
        .agg(
            note_rows=("dataset_row_id", "size"),
            outcome_eligible_count=("outcome_eligible", "sum"),
            return_count=("observed_return_within_window", "sum"),
            competing_death_count=("death_within_window", "sum"),
            administrative_censor_count=("administratively_censored", "sum"),
            missing_admission_link_count=("missing_admission_link", "sum"),
        )
        .reset_index()
    )
    summary["outcome_eligible_rate"] = summary.outcome_eligible_count / summary.note_rows
    summary["return_prevalence_among_eligible"] = np.where(
        summary.outcome_eligible_count > 0,
        summary.return_count / summary.outcome_eligible_count,
        np.nan,
    )
    summary.to_csv(output_dir / "observed_return_feasibility_by_split.csv", index=False)

    overall = {
        "scope": "note_linked_hospital_observed_return_feasibility_only",
        "window_days": args.window_days,
        "data_end_dischtime": str(data_end),
        "n_note_rows": int(len(index)),
        "n_outcome_eligible": int(index.outcome_eligible.sum()),
        "n_observed_returns": int(index.observed_return_within_window.sum()),
        "n_competing_deaths": int(index.death_within_window.sum()),
        "n_administratively_censored": int(index.administratively_censored.sum()),
        "n_missing_admission_link": int(index.missing_admission_link.sum()),
        "security_note": "Outputs contain provenance IDs and derived outcome flags only; no note text.",
        "limitation": "Observed return captures only admissions recorded in this source system and is not all-cause readmission.",
    }
    (output_dir / "observed_return_feasibility_summary.json").write_text(json.dumps(overall, indent=2) + "\n")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
