#!/usr/bin/env python3
"""Derive frozen sparse/dense cohort labels from repeated local-support runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--source_split", choices=("dev", "test"), required=True)
    parser.add_argument("--seeds", default="20260811,20260812,20260813,20260814,20260815")
    parser.add_argument("--sparse_frequency_min", type=float, default=0.80)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    records = []
    for seed in seeds:
        path = Path(args.run_root).resolve() / f"seed_{seed}" / f"canonical_{args.source_split}_local_support.jsonl"
        frame = pd.read_json(path, lines=True)
        if frame.source_index.duplicated().any():
            raise ValueError(f"Duplicate source indices in {path}.")
        frame["split_seed"] = seed
        records.append(frame)
    combined = pd.concat(records, ignore_index=True)
    required = {"source_index", "dataset_row_id", "note_id", "case_id", "subject_id", "patient_disjoint_from_train"}
    if missing := required - set(combined):
        raise KeyError(f"Support records missing columns: {sorted(missing)}")
    base_columns = sorted(required)
    base = combined.sort_values("split_seed").drop_duplicates("source_index")[base_columns]
    output = base.copy()
    for k in (25, 50, 100):
        column = f"mean_top_{k}_support"
        if column not in combined:
            raise KeyError(f"Support records are missing {column}.")
        output = output.merge(
            combined.groupby("source_index", as_index=False)[column].mean(),
            on="source_index", how="left", validate="one_to_one",
        )
        sparse_rows = []
        for _, frame in combined.groupby("split_seed"):
            n_sparse = max(1, int(np.ceil(len(frame) * 0.10)))
            sparse_rows.append(frame.nsmallest(n_sparse, column)[["source_index"]])
        frequency = pd.concat(sparse_rows).groupby("source_index").size().div(len(seeds))
        output[f"sparse_frequency_k{k}"] = output.source_index.map(frequency).fillna(0.0)
    output["stable_sparse_k50_with_adjacent"] = (
        output.sparse_frequency_k25.ge(args.sparse_frequency_min)
        & output.sparse_frequency_k50.ge(args.sparse_frequency_min)
    )
    output["stable_dense_k50_with_adjacent"] = (
        output.sparse_frequency_k25.eq(0.0)
        & output.sparse_frequency_k50.eq(0.0)
        & output.sparse_frequency_k100.eq(0.0)
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / f"canonical_{args.source_split}_support_cohort_diagnostic.csv", index=False)
    summary = {
        "source_split": args.source_split,
        "seeds": seeds,
        "n_rows": int(len(output)),
        "stable_sparse_count": int(output.stable_sparse_k50_with_adjacent.sum()),
        "stable_dense_count": int(output.stable_dense_k50_with_adjacent.sum()),
        "sparse_frequency_min": args.sparse_frequency_min,
        "security_note": "Output contains provenance IDs and derived support labels only; no source-note text.",
    }
    (output_dir / f"canonical_{args.source_split}_support_cohort_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
