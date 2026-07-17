#!/usr/bin/env python3
"""Freeze a stratified nested backbone-control subset before generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic nested fact-only backbone-control subset.")
    parser.add_argument("--case_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_cases", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output_stem", default="untouched_backbone_fact_only_nested_control")
    return parser.parse_args()


def stable_rank(case_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(Path(args.case_manifest_path).resolve())
    required = {"case_id", "review_stratum", "patient_disjoint_from_train"}
    if missing := required.difference(frame.columns):
        raise KeyError(f"case manifest missing columns: {sorted(missing)}")
    if frame.case_id.duplicated().any():
        raise ValueError("case manifest must have one row per case_id")
    if args.n_cases <= 0 or args.n_cases > len(frame):
        raise ValueError("--n_cases must be between 1 and the number of available cases")

    frame = frame.copy()
    frame["patient_disjoint_from_train"] = frame.patient_disjoint_from_train.fillna(False).astype(bool)
    frame["rank"] = frame.case_id.astype(str).map(lambda value: stable_rank(value, args.seed))
    exact_target = round(args.n_cases * (frame.review_stratum.astype(str).eq("exact_pooled").mean()))
    exact_target = min(max(exact_target, 1), args.n_cases - 1)
    centroid_target = args.n_cases - exact_target
    pd_target = round(args.n_cases * frame.patient_disjoint_from_train.mean())

    exact_pd = frame.loc[frame.review_stratum.astype(str).eq("exact_pooled") & frame.patient_disjoint_from_train].sort_values("rank")
    selected_pd = exact_pd.head(min(pd_target, exact_target)).copy()
    exact_remaining = frame.loc[
        frame.review_stratum.astype(str).eq("exact_pooled") & ~frame.case_id.isin(selected_pd.case_id)
    ].sort_values("rank")
    selected_exact = pd.concat([selected_pd, exact_remaining.head(exact_target - len(selected_pd))], ignore_index=True)
    centroid = frame.loc[frame.review_stratum.astype(str).eq("centroid_only")].sort_values("rank")
    selected_centroid = centroid.head(centroid_target)
    selected = pd.concat([selected_exact, selected_centroid], ignore_index=True)
    if len(selected) != args.n_cases:
        raise ValueError("insufficient cases for requested deterministic strata")
    selected = selected.drop(columns="rank").sort_values("case_id").reset_index(drop=True)

    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = str(args.output_stem).strip()
    if not output_stem:
        raise ValueError("--output_stem must not be empty")
    selected.to_csv(output_dir / f"{output_stem}_manifest.csv", index=False)
    summary = {
        "n_cases": int(len(selected)),
        "seed": int(args.seed),
        "selection_rule": "deterministic hash rank with proportional exact-pooled/centroid-only allocation and patient-disjoint representation where available",
        "review_stratum_counts": {str(key): int(value) for key, value in selected.review_stratum.value_counts().items()},
        "patient_disjoint_count": int(selected.patient_disjoint_from_train.sum()),
        "output_stem": output_stem,
    }
    (output_dir / "untouched_backbone_fact_only_nested_control_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
