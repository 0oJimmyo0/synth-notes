#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    load_and_merge_tables,
    normalize_rows,
    parse_csv_list,
    parse_float_list,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit how held-out embeddings move under saved CAV axes."
    )
    parser.add_argument("--bank_dir", required=True, help="Directory created by fit_axis_bank.py")
    parser.add_argument("--embeddings_path", required=True, help="Path to sentence_embeddings.npy")
    parser.add_argument("--metadata_path", required=True, help="Path to sentence_embeddings_metadata.csv")
    parser.add_argument("--factors_path", required=True, help="Same factor CSV used for fitting")
    parser.add_argument("--join_cols", required=True, help="Same join columns used during fitting")
    parser.add_argument(
        "--alphas",
        default="-2,-1,-0.5,0.5,1,2",
        help="Comma-separated steering magnitudes",
    )
    parser.add_argument(
        "--axis_indices",
        default="",
        help="Optional comma-separated axis indices to audit. Defaults to [0, recommended_k).",
    )
    parser.add_argument(
        "--top_targets",
        type=int,
        default=5,
        help="Number of positive and negative target outputs to report per axis",
    )
    parser.add_argument(
        "--output_name",
        default="steering_audit.csv",
        help="CSV filename for detailed audit rows",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    bank_dir = Path(args.bank_dir)
    bank = np.load(bank_dir / "axis_bank.npz")
    target_manifest = pd.read_csv(bank_dir / "target_manifest.csv")
    split_manifest = pd.read_csv(bank_dir / "split_manifest.csv")
    with open(bank_dir / "axis_bank_summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    join_cols = parse_csv_list(args.join_cols)
    alpha_values = parse_float_list(args.alphas)

    embeddings, merged_df = load_and_merge_tables(
        embeddings_path=args.embeddings_path,
        metadata_path=args.metadata_path,
        factors_path=args.factors_path,
        join_cols=join_cols,
    )

    merged_df = merged_df.merge(split_manifest, on=["embedding_row_id", *join_cols], how="inner", validate="one_to_one")
    heldout_rows = merged_df.loc[merged_df["split"] == "test", "embedding_row_id"].to_numpy(dtype=int)
    x_test = embeddings[heldout_rows]

    axes = bank["axes"]
    coefficient_matrix = bank["coefficient_matrix"]
    intercept = bank["intercept"]
    axis_loadings = bank["axis_loadings"]

    if args.axis_indices:
        axis_indices = [int(value) for value in parse_csv_list(args.axis_indices)]
    else:
        recommended_k = int(summary["recommended_k"])
        axis_indices = list(range(recommended_k))

    rows = []
    summaries = []
    base_scores = x_test @ coefficient_matrix.T + intercept

    for axis_idx in axis_indices:
        axis = axes[:, axis_idx]
        loading_vector = axis_loadings[:, axis_idx]

        positive_rank = np.argsort(loading_vector)[::-1][: args.top_targets]
        negative_rank = np.argsort(loading_vector)[: args.top_targets]
        tracked_indices = list(dict.fromkeys([*positive_rank.tolist(), *negative_rank.tolist()]))

        axis_summary = {
            "axis": axis_idx,
            "positive_targets": [],
            "negative_targets": [],
        }

        for target_idx in positive_rank:
            row = target_manifest.iloc[target_idx]
            axis_summary["positive_targets"].append(
                {
                    "target_column": row["target_column"],
                    "factor": row["factor"],
                    "level": row["level"],
                    "loading": float(loading_vector[target_idx]),
                }
            )
        for target_idx in negative_rank:
            row = target_manifest.iloc[target_idx]
            axis_summary["negative_targets"].append(
                {
                    "target_column": row["target_column"],
                    "factor": row["factor"],
                    "level": row["level"],
                    "loading": float(loading_vector[target_idx]),
                }
            )
        summaries.append(axis_summary)

        for alpha in alpha_values:
            shifted = normalize_rows(x_test + (alpha * axis))
            shifted_scores = shifted @ coefficient_matrix.T + intercept

            for target_idx in tracked_indices:
                target_row = target_manifest.iloc[target_idx]
                delta = shifted_scores[:, target_idx] - base_scores[:, target_idx]
                rows.append(
                    {
                        "axis": axis_idx,
                        "alpha": alpha,
                        "factor": target_row["factor"],
                        "target_column": target_row["target_column"],
                        "level": target_row["level"],
                        "target_kind": target_row["target_kind"],
                        "loading": float(loading_vector[target_idx]),
                        "mean_base_score": float(base_scores[:, target_idx].mean()),
                        "mean_shifted_score": float(shifted_scores[:, target_idx].mean()),
                        "mean_delta": float(delta.mean()),
                        "std_delta": float(delta.std(ddof=0)),
                    }
                )

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(bank_dir / args.output_name, index=False)

    trend_rows = []
    for (axis_idx, target_column), group in audit_df.groupby(["axis", "target_column"]):
        ordered = group.sort_values("alpha")
        if len(ordered) >= 2:
            corr = float(np.corrcoef(ordered["alpha"].to_numpy(), ordered["mean_shifted_score"].to_numpy())[0, 1])
        else:
            corr = np.nan
        trend_rows.append(
            {
                "axis": axis_idx,
                "target_column": target_column,
                "trend_corr": corr,
            }
        )

    trend_df = pd.DataFrame(trend_rows)
    trend_df.to_csv(bank_dir / "steering_trends.csv", index=False)
    save_json(bank_dir / "steering_summary.json", {"axis_summaries": summaries, "alphas": alpha_values})

    print(f"Saved steering audit to: {bank_dir / args.output_name}")
    print(f"Saved steering trend summary to: {bank_dir / 'steering_trends.csv'}")


if __name__ == "__main__":
    main()
