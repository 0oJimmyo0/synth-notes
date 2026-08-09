#!/usr/bin/env python3
"""Create a fresh patient-grouped utility split without reading outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_split_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source_splits", default="train,dev")
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--dev_fraction", type=float, default=0.15)
    return parser.parse_args()


def group_for_subject(subject_id: str, seed: int, train_fraction: float, dev_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{subject_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train_fraction:
        return "train"
    if value < train_fraction + dev_fraction:
        return "dev"
    return "test_unopened"


def main() -> None:
    args = parse_args()
    if not 0 < args.train_fraction < 1 or not 0 < args.dev_fraction < 1 or args.train_fraction + args.dev_fraction >= 1:
        raise ValueError("fractions must be positive and leave a nonzero unopened test fraction")
    source_splits = {item.strip() for item in args.source_splits.split(",") if item.strip()}
    frame = pd.read_csv(Path(args.base_split_manifest_path).resolve(), low_memory=False)
    needed = {"dataset_row_id", "subject_id", "hadm_id", "split"}
    if missing := needed.difference(frame.columns):
        raise KeyError(f"base split manifest missing columns: {sorted(missing)}")
    frame = frame.loc[frame.split.astype(str).isin(source_splits)].copy()
    if frame.empty:
        raise ValueError("no rows remain after source split filtering")
    if frame.hadm_id.duplicated().any():
        raise ValueError("utility split requires one note row per hadm_id")
    frame["subject_id"] = frame.subject_id.astype(str)
    frame["utility_split"] = frame.subject_id.map(
        lambda subject_id: group_for_subject(subject_id, args.seed, args.train_fraction, args.dev_fraction)
    )
    # Preserve the original note-level split for provenance while exposing the
    # new patient-grouped assignment through the conventional `split` column
    # expected by outcome-feasibility tooling.
    frame["source_split"] = frame["split"]
    frame["split"] = frame["utility_split"]
    frame["patient_disjoint_from_train"] = frame.utility_split.ne("train")
    groups = frame.groupby("subject_id").utility_split.nunique()
    if groups.ne(1).any():
        raise ValueError("a subject was assigned to more than one utility split")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "utility_patient_grouped_manifest.csv", index=False)
    frame.loc[frame.utility_split.isin({"train", "dev"})].to_csv(output_dir / "utility_train_dev_manifest.csv", index=False)
    frame.loc[frame.utility_split.eq("test_unopened")].to_csv(output_dir / "utility_test_manifest_UNOPENED.csv", index=False)
    counts = frame.groupby("utility_split").agg(notes=("dataset_row_id", "size"), subjects=("subject_id", "nunique")).reset_index()
    counts.to_csv(output_dir / "utility_patient_grouped_split_counts.csv", index=False)
    summary = {
        "scope": "fresh_patient_grouped_utility_split_without_outcome_access",
        "seed": args.seed,
        "source_splits": sorted(source_splits),
        "fractions": {"train": args.train_fraction, "dev": args.dev_fraction, "test_unopened": 1 - args.train_fraction - args.dev_fraction},
        "counts": counts.to_dict(orient="records"),
        "security_note": "Outputs contain provenance IDs and split labels only; no note text or outcomes are accessed.",
    }
    (output_dir / "utility_patient_grouped_split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
