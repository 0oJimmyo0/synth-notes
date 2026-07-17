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
    parser.add_argument("--n_anchors", type=int, default=30)
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
    if frame["dataset_row_id"].duplicated().any():
        raise ValueError("Cluster assignments contain duplicate dataset_row_id values.")
    candidates = frame.loc[frame["cluster_id"].isin(target_ids)].copy()
    if len(candidates) < args.n_anchors:
        raise ValueError(f"Target basin has only {len(candidates)} rows, fewer than requested {args.n_anchors}.")
    candidates["patient_disjoint_from_train"] = candidates["patient_disjoint_from_train"].fillna(False).astype(bool)
    candidates["stable_rank"] = candidates["dataset_row_id"].map(lambda value: stable_rank(int(value), args.seed))
    pd_candidates = candidates.loc[candidates["patient_disjoint_from_train"]].sort_values("stable_rank")
    overlap_candidates = candidates.loc[~candidates["patient_disjoint_from_train"]].sort_values("stable_rank")
    pd_target = min(len(pd_candidates), max(1, round(args.n_anchors * len(pd_candidates) / len(candidates))))
    selected = pd.concat([pd_candidates.head(pd_target), overlap_candidates.head(args.n_anchors - pd_target)], ignore_index=True)
    if len(selected) != args.n_anchors:
        selected = candidates.sort_values("stable_rank").head(args.n_anchors).copy()
    selected = selected.drop(columns="stable_rank").sort_values("dataset_row_id").reset_index(drop=True)
    selected.insert(0, "case_id", [f"region_{'_'.join(map(str, sorted(target_ids)))}_{idx:03d}" for idx in range(1, len(selected) + 1)])
    selected["target_cluster_ids"] = ",".join(map(str, sorted(target_ids)))
    selected["anchor_selection"] = "deterministic_stratified_target_basin_sample"
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out / "region_anchor_manifest.csv", index=False)
    summary = {
        "target_cluster_ids": sorted(target_ids), "n_anchors": int(len(selected)),
        "patient_disjoint_count": int(selected["patient_disjoint_from_train"].sum()),
        "cluster_counts": {str(key): int(value) for key, value in selected.cluster_id.value_counts().sort_index().items()},
        "seed": int(args.seed), "selection": "deterministic target-basin sample stratified by patient-disjoint status",
    }
    (out / "region_anchor_manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
