#!/usr/bin/env python3
"""
Build a deterministic anchor-manifest subset for small completion diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small deterministic subset of an anchor manifest.")
    parser.add_argument("--input_path", required=True, help="Input CSV or JSONL manifest")
    parser.add_argument("--output_path", required=True, help="Output CSV path")
    parser.add_argument("--n_rows", type=int, default=24, help="Number of anchors to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--stratify_col",
        default="patient_disjoint_from_train",
        help="Optional column for simple proportional stratified sampling; leave empty to disable",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_table(input_path)
    n_rows = min(int(args.n_rows), len(df))
    stratify_col = str(args.stratify_col).strip()

    if stratify_col and stratify_col in df.columns and df[stratify_col].notna().any():
        parts = []
        group_sizes = df[stratify_col].fillna("unknown").value_counts()
        allocated = 0
        labels = list(group_sizes.index)
        for idx, label in enumerate(labels):
            group = df.loc[df[stratify_col].fillna("unknown") == label].copy()
            if idx == len(labels) - 1:
                take = n_rows - allocated
            else:
                frac = len(group) / len(df)
                take = min(len(group), max(1, round(n_rows * frac)))
            allocated += take
            parts.append(group.sample(n=take, random_state=int(args.seed), replace=False))
        out_df = pd.concat(parts, ignore_index=True).head(n_rows).copy()
    else:
        out_df = df.sample(n=n_rows, random_state=int(args.seed), replace=False).copy()

    out_df = out_df.reset_index(drop=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved subset manifest to: {output_path}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
