#!/usr/bin/env python3
"""Characterize train/dev observed-return timing and next-admission types.

This is a definition audit only. It does not select a final label or access an
unopened utility-test cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility_train_dev_manifest", required=True)
    parser.add_argument("--admissions_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--window_days", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    utility = pd.read_csv(Path(args.utility_train_dev_manifest).resolve(), dtype=str).fillna("")
    needed = {"dataset_row_id", "subject_id", "hadm_id", "split"}
    if missing := needed.difference(utility.columns):
        raise KeyError(f"utility manifest missing columns: {sorted(missing)}")
    if utility.hadm_id.duplicated().any():
        raise ValueError("utility manifest must contain one row per admission")
    admissions = pd.read_csv(
        Path(args.admissions_path).resolve(),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type", "admission_location"],
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    admissions["subject_id"] = admissions.subject_id.astype(str)
    admissions["hadm_id"] = pd.to_numeric(admissions.hadm_id, errors="raise").astype(int)
    if admissions.hadm_id.duplicated().any():
        raise ValueError("admissions has duplicate hadm_id values")
    admissions = admissions.sort_values(["subject_id", "admittime", "hadm_id"], kind="stable")
    next_by_hadm: dict[int, tuple[int, pd.Timestamp, str, str]] = {}
    for _, group in admissions.groupby("subject_id", sort=False):
        records = list(group[["hadm_id", "admittime", "dischtime", "admission_type", "admission_location"]].itertuples(index=False, name=None))
        for position, (hadm_id, _, dischtime, _, _) in enumerate(records):
            if pd.isna(dischtime):
                continue
            for candidate in records[position + 1:]:
                if candidate[1] > dischtime:
                    next_by_hadm[int(hadm_id)] = (candidate[0], candidate[1], candidate[3], candidate[4])
                    break
    index = utility[["dataset_row_id", "subject_id", "hadm_id", "split"]].copy()
    index["subject_id"] = index.subject_id.astype(str)
    index["hadm_id"] = pd.to_numeric(index.hadm_id, errors="raise").astype(int)
    index = index.merge(
        admissions[["hadm_id", "subject_id", "dischtime", "deathtime"]],
        on=["hadm_id", "subject_id"], how="left", validate="one_to_one",
    )
    next_records = [next_by_hadm.get(hadm_id) for hadm_id in index.hadm_id]
    index["next_hadm_id"] = [record[0] if record else np.nan for record in next_records]
    index["next_admittime"] = [record[1] if record else pd.NaT for record in next_records]
    index["next_admission_type"] = [record[2] if record else "" for record in next_records]
    index["next_admission_location"] = [record[3] if record else "" for record in next_records]
    index["hours_to_next_admission"] = (index.next_admittime - index.dischtime).dt.total_seconds() / 3600
    index["next_within_window"] = index.hours_to_next_admission.between(0, args.window_days * 24, inclusive="both")
    index["same_day_next_admission"] = index.next_within_window & index.hours_to_next_admission.lt(24)
    index["elective_next_admission"] = index.next_within_window & index.next_admission_type.fillna("").str.lower().eq("elective")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset_row_id", "subject_id", "hadm_id", "split", "next_hadm_id", "hours_to_next_admission",
        "next_within_window", "same_day_next_admission", "elective_next_admission",
        "next_admission_type", "next_admission_location",
    ]
    index[columns].to_csv(output_dir / "train_dev_next_admission_characterization.csv", index=False)
    within = index.loc[index.next_within_window].copy()
    summary = within.groupby(["split", "next_admission_type"], dropna=False).agg(
        returns=("hadm_id", "size"),
        same_day_returns=("same_day_next_admission", "sum"),
        elective_returns=("elective_next_admission", "sum"),
        median_hours_to_next_admission=("hours_to_next_admission", "median"),
    ).reset_index()
    summary.to_csv(output_dir / "train_dev_return_characterization_by_type.csv", index=False)
    overall = {
        "scope": "train_dev_only_observed_return_definition_audit",
        "window_days": args.window_days,
        "n_rows": int(len(index)),
        "n_next_within_window": int(index.next_within_window.sum()),
        "n_same_day_next_admissions": int(index.same_day_next_admission.sum()),
        "n_elective_next_admissions": int(index.elective_next_admission.sum()),
        "security_note": "Outputs contain provenance IDs and derived admission timing/type fields only; no note text.",
        "limitation": "Admission type and timing do not perfectly identify planned returns or transfers.",
    }
    (output_dir / "train_dev_return_characterization_summary.json").write_text(json.dumps(overall, indent=2) + "\n")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
