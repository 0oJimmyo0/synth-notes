#!/usr/bin/env python3
"""Build source-eligibility replenishment reserves for a frozen dev cohort.

Selection is based only on frozen local-support arm, patient-disjoint stratum,
and source-review readiness.  It intentionally does not read synthetic output
or outcome labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def stable_hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic_csv", required=True)
    parser.add_argument("--original_cohort_csv", required=True)
    parser.add_argument("--readiness_map_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_ready_per_stratum", type=int, default=10)
    parser.add_argument("--reserve_multiplier", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_ready_per_stratum <= 0 or args.reserve_multiplier <= 0:
        raise ValueError("Target and reserve multiplier must be positive.")
    diagnostic = pd.read_csv(Path(args.diagnostic_csv).resolve())
    original = pd.read_csv(Path(args.original_cohort_csv).resolve())
    readiness = pd.read_csv(Path(args.readiness_map_csv).resolve())
    original_ids = set(original.dataset_row_id.astype(int))
    original_subjects = set(original.subject_id.astype(str))
    ready_counts = readiness.loc[readiness.ledger_ready_for_generation.astype(bool)].groupby(
        ["support_arm", "cohort_stratum"]
    ).size().to_dict()
    arm_masks = {
        "stable_sparse": diagnostic.stable_sparse_k50_with_adjacent.astype(bool),
        "stable_dense": diagnostic.stable_dense_k50_with_adjacent.astype(bool),
    }
    selected, used_subjects = [], set(original_subjects)
    for arm, arm_mask in arm_masks.items():
        for stratum, is_disjoint in (("patient_disjoint", True), ("patient_overlap", False)):
            observed = int(ready_counts.get((arm, stratum), 0))
            needed = max(0, args.target_ready_per_stratum - observed)
            if needed == 0:
                continue
            reserve_n = needed * args.reserve_multiplier
            candidates = diagnostic.loc[
                arm_mask
                & diagnostic.patient_disjoint_from_train.astype(bool).eq(is_disjoint)
                & ~diagnostic.dataset_row_id.astype(int).isin(original_ids)
                & ~diagnostic.subject_id.astype(str).isin(used_subjects)
            ].copy()
            candidates["selection_hash"] = candidates.apply(
                lambda row: stable_hash(args.seed, f"{arm}|{stratum}|{row.subject_id}|{row.source_index}"), axis=1
            )
            candidates = candidates.sort_values("selection_hash").drop_duplicates("subject_id", keep="first")
            chosen = candidates.head(reserve_n).copy()
            if len(chosen) != reserve_n:
                raise ValueError(f"Only {len(chosen)} candidates available for {arm}/{stratum}; need {reserve_n}.")
            chosen["support_arm"] = arm
            chosen["cohort_stratum"] = stratum
            chosen["ready_cases_before_replenishment"] = observed
            chosen["ready_cases_needed"] = needed
            chosen["replacement_priority"] = range(1, len(chosen) + 1)
            selected.append(chosen)
            used_subjects.update(chosen.subject_id.astype(str))
    reserve = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    if reserve.subject_id.astype(str).duplicated().any():
        raise ValueError("Replenishment reserve repeats a subject.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    keep = [
        "source_index", "dataset_row_id", "note_id", "case_id", "subject_id", "patient_disjoint_from_train",
        "support_arm", "cohort_stratum", "ready_cases_before_replenishment", "ready_cases_needed",
        "replacement_priority", "mean_top_50_support", "sparse_frequency_k25", "sparse_frequency_k50", "sparse_frequency_k100",
    ]
    reserve[[column for column in keep if column in reserve]].to_csv(
        output_dir / "canonical_dev_support_replenishment_reserve.csv", index=False
    )
    summary = {
        "scope": "development_only_source_eligibility_replenishment_before_any_synthetic_generation",
        "selection_seed": args.seed,
        "target_ready_per_stratum": args.target_ready_per_stratum,
        "reserve_multiplier": args.reserve_multiplier,
        "n_reserve_notes": int(len(reserve)),
        "n_unique_subjects": int(reserve.subject_id.nunique()),
        "selection_counts": reserve.groupby(["support_arm", "cohort_stratum"]).size().rename("n").reset_index().to_dict(orient="records"),
        "security_note": "Output contains provenance IDs and derived support labels only; no source-note text.",
    }
    (output_dir / "canonical_dev_support_replenishment_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
