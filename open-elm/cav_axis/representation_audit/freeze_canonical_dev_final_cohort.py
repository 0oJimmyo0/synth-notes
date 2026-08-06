#!/usr/bin/env python3
"""Freeze a balanced, source-complete development cohort before generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined_readiness_csv", required=True)
    parser.add_argument("--initial_cohort_csv", required=True)
    parser.add_argument("--replenishment_reserve_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_per_stratum", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined = pd.read_csv(Path(args.combined_readiness_csv).resolve())
    initial = pd.read_csv(Path(args.initial_cohort_csv).resolve())
    reserve = pd.read_csv(Path(args.replenishment_reserve_csv).resolve())
    required = {"dataset_row_id", "support_arm", "cohort_stratum", "source_wave", "ledger_ready_for_generation"}
    missing = required.difference(combined.columns)
    if missing:
        raise KeyError(f"Combined readiness map missing columns: {sorted(missing)}")
    provenance_columns = [
        "dataset_row_id", "note_id", "subject_id", "patient_disjoint_from_train", "mean_top_50_support",
    ]
    subject_lookup = pd.concat([
        initial[[column for column in provenance_columns if column in initial.columns]],
        reserve[[column for column in provenance_columns if column in reserve.columns]],
    ], ignore_index=True)
    if subject_lookup.dataset_row_id.duplicated().any():
        raise ValueError("Initial cohort and replenishment reserve share dataset_row_id values.")
    for column in subject_lookup.columns:
        if column == "dataset_row_id":
            continue
        lookup = subject_lookup.set_index("dataset_row_id")[column]
        if column not in combined.columns:
            combined[column] = combined.dataset_row_id.map(lookup)
        else:
            existing = combined[column]
            missing_value = existing.isna() | existing.astype(str).str.strip().eq("")
            combined.loc[missing_value, column] = combined.loc[missing_value, "dataset_row_id"].map(lookup)
    if combined.subject_id.isna().any():
        raise ValueError("A combined readiness row lacks frozen subject provenance.")
    ready = combined.loc[combined.ledger_ready_for_generation.astype(bool)].copy()
    initial_priority = initial[["dataset_row_id", "selection_rank_within_stratum"]].rename(
        columns={"selection_rank_within_stratum": "source_priority"}
    )
    reserve_priority = reserve[["dataset_row_id", "replacement_priority"]].rename(
        columns={"replacement_priority": "source_priority"}
    )
    ready_initial = ready.loc[ready.source_wave.eq("initial")].merge(
        initial_priority, on="dataset_row_id", how="left", validate="one_to_one"
    )
    ready_reserve = ready.loc[ready.source_wave.eq("replenishment")].merge(
        reserve_priority, on="dataset_row_id", how="left", validate="one_to_one"
    )
    chosen = []
    for arm in ("stable_sparse", "stable_dense"):
        for stratum in ("patient_disjoint", "patient_overlap"):
            group = pd.concat([
                ready_initial.loc[(ready_initial.support_arm.eq(arm)) & (ready_initial.cohort_stratum.eq(stratum))],
                ready_reserve.loc[(ready_reserve.support_arm.eq(arm)) & (ready_reserve.cohort_stratum.eq(stratum))],
            ], ignore_index=True)
            group["wave_priority"] = group.source_wave.map({"initial": 0, "replenishment": 1})
            group = group.sort_values(["wave_priority", "source_priority", "dataset_row_id"], kind="stable")
            group = group.drop_duplicates("subject_id", keep="first")
            selected = group.head(args.n_per_stratum).copy()
            if len(selected) != args.n_per_stratum:
                raise ValueError(f"Only {len(selected)} source-complete unique-subject cases available for {arm}/{stratum}.")
            selected["final_selection_rank_within_stratum"] = range(1, len(selected) + 1)
            chosen.append(selected)
    final = pd.concat(chosen, ignore_index=True)
    if final.subject_id.astype(str).duplicated().any():
        raise ValueError("Final cohort repeats a subject across strata.")
    final["final_case_id"] = final.dataset_row_id.map(lambda value: f"dev_{int(value)}")
    final = final.sort_values(["support_arm", "cohort_stratum", "final_selection_rank_within_stratum"])
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    keep = [
        "final_case_id", "dataset_row_id", "note_id", "subject_id", "support_arm", "cohort_stratum",
        "patient_disjoint_from_train", "source_wave", "case_id", "source_priority",
        "final_selection_rank_within_stratum", "mean_top_50_support",
    ]
    final[[column for column in keep if column in final]].to_csv(
        output_dir / "canonical_dev_support_vanilla_final40.csv", index=False
    )
    summary = {
        "scope": "development_only_balanced_source_complete_cohort_before_any_synthetic_generation",
        "n_notes": int(len(final)),
        "n_unique_subjects": int(final.subject_id.nunique()),
        "n_per_stratum": args.n_per_stratum,
        "selection_order": "initial validated cohort first, then deterministic replenishment priority",
        "counts": final.groupby(["support_arm", "cohort_stratum"]).size().rename("n").reset_index().to_dict(orient="records"),
        "security_note": "Output contains provenance IDs and derived support labels only; no source-note text.",
    }
    (output_dir / "canonical_dev_support_vanilla_final40_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
