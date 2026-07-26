#!/usr/bin/env python3
"""Freeze a leakage-aware held-out anchor cohort for a target real-note basin."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_cluster_assignments_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_cluster_ids", required=True, help="Comma-separated fixed real-test cluster IDs")
    parser.add_argument(
        "--source_split",
        default="test",
        help="Split represented by dataset_row_id; required when assignments cover multiple splits.",
    )
    parser.add_argument("--eligibility_candidates_path", default=None,
                        help="Optional Tier-1 eligibility CSV; only its dataset_row_id values may be selected.")
    parser.add_argument(
        "--exclude_manifest_path",
        default=None,
        help="Optional prior anchor manifest; its dataset_row_id values are excluded to make this cohort independent.",
    )
    parser.add_argument("--n_anchors", type=int, default=30)
    parser.add_argument("--min_patient_disjoint", type=int, default=0,
                        help="Minimum patient-disjoint anchors when the eligible reserve permits it.")
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def stable_rank(dataset_row_id: int, seed: int) -> int:
    # Deterministic pseudo-random ordering without relying on machine RNG state.
    return int(hashlib.sha256(f"{seed}:{dataset_row_id}".encode("utf-8")).hexdigest(), 16)


def main() -> None:
    args = parse_args()
    target_ids = {int(value) for value in args.target_cluster_ids.split(",") if value.strip()}
    if not target_ids:
        raise ValueError("--target_cluster_ids must contain at least one cluster ID")
    frame = pd.read_csv(Path(args.real_cluster_assignments_path).resolve())
    required = {"dataset_row_id", "note_id", "subject_id", "hadm_id", "cluster_id", "patient_disjoint_from_train"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Cluster assignments missing columns: {sorted(missing)}")
    frame["dataset_row_id"] = pd.to_numeric(frame["dataset_row_id"], errors="raise").astype(int)
    # In the full-real table dataset_row_id is local to each split. Filter before
    # uniqueness checks so train/dev/test row IDs are never mixed.
    if "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str) == str(args.source_split)].copy()
    if frame["dataset_row_id"].duplicated().any():
        raise ValueError(f"Cluster assignments contain duplicate dataset_row_id values within split={args.source_split!r}.")
    candidates = frame.loc[frame["cluster_id"].isin(target_ids)].copy()
    if args.eligibility_candidates_path:
        eligibility = pd.read_csv(Path(args.eligibility_candidates_path).resolve())
        if "dataset_row_id" not in eligibility.columns:
            raise KeyError("Eligibility CSV must contain dataset_row_id.")
        if "eligibility_tier" in eligibility.columns:
            eligibility = eligibility.loc[eligibility["eligibility_tier"] == "tier1_complete_review_candidate"].copy()
        eligible_ids = set(pd.to_numeric(eligibility["dataset_row_id"], errors="raise").astype(int))
        candidates = candidates.loc[candidates["dataset_row_id"].isin(eligible_ids)].copy()
    excluded_ids: set[int] = set()
    if args.exclude_manifest_path:
        previous = pd.read_csv(Path(args.exclude_manifest_path).resolve())
        if "dataset_row_id" not in previous.columns:
            raise KeyError("Exclusion manifest must contain dataset_row_id.")
        excluded_ids = set(pd.to_numeric(previous["dataset_row_id"], errors="raise").astype(int))
        candidates = candidates.loc[~candidates["dataset_row_id"].isin(excluded_ids)].copy()
    if len(candidates) < args.n_anchors:
        raise ValueError(f"Target basin has only {len(candidates)} rows, fewer than requested {args.n_anchors}.")
    candidates["patient_disjoint_from_train"] = candidates["patient_disjoint_from_train"].fillna(False).astype(bool)
    candidates["stable_rank"] = candidates["dataset_row_id"].map(lambda value: stable_rank(int(value), args.seed))
    pd_candidates = candidates.loc[candidates["patient_disjoint_from_train"]].sort_values("stable_rank")
    overlap_candidates = candidates.loc[~candidates["patient_disjoint_from_train"]].sort_values("stable_rank")
    proportional_pd_target = max(1, round(args.n_anchors * len(pd_candidates) / len(candidates)))
    requested_pd_target = max(proportional_pd_target, int(args.min_patient_disjoint))
    pd_target = min(len(pd_candidates), args.n_anchors, requested_pd_target)
    selected = pd.concat([pd_candidates.head(pd_target), overlap_candidates.head(args.n_anchors - pd_target)], ignore_index=True)
    if len(selected) != args.n_anchors:
        selected = candidates.sort_values("stable_rank").head(args.n_anchors).copy()
    selected = selected.drop(columns="stable_rank").sort_values("dataset_row_id").reset_index(drop=True)
    selected.insert(0, "case_id", [f"region_{'_'.join(map(str, sorted(target_ids)))}_{idx:03d}" for idx in range(1, len(selected) + 1)])
    # These stable provenance fields are consumed by prompt-safe ledger
    # serialization and preserve the patient-disjoint review stratum.
    selected["anchor_id"] = selected["dataset_row_id"].map(lambda value: f"anchor_{int(value)}")
    selected["review_stratum"] = np.where(
        selected["patient_disjoint_from_train"].fillna(False).astype(bool),
        "patient_disjoint",
        "patient_overlap",
    )
    selected["target_cluster_ids"] = ",".join(map(str, sorted(target_ids)))
    selected["anchor_selection"] = "deterministic_stratified_target_basin_sample"
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out / "region_anchor_manifest.csv", index=False)
    summary = {
        "target_cluster_ids": sorted(target_ids), "source_split": str(args.source_split), "n_anchors": int(len(selected)),
        "patient_disjoint_count": int(selected["patient_disjoint_from_train"].sum()),
        "cluster_counts": {str(key): int(value) for key, value in selected.cluster_id.value_counts().sort_index().items()},
        "seed": int(args.seed), "selection": "deterministic target-basin sample stratified by patient-disjoint status",
        "min_patient_disjoint_requested": int(args.min_patient_disjoint),
        "eligibility_candidates_path": str(Path(args.eligibility_candidates_path).resolve()) if args.eligibility_candidates_path else None,
        "exclude_manifest_path": str(Path(args.exclude_manifest_path).resolve()) if args.exclude_manifest_path else None,
        "excluded_prior_anchor_count": len(excluded_ids),
    }
    (out / "region_anchor_manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
