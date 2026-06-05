#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    compute_axis_bank,
    encode_targets,
    filter_complete_cases,
    fit_multioutput_ridge,
    load_and_merge_tables,
    parse_csv_list,
    parse_float_list,
    reconstruct_coefficients,
    save_json,
    score_predictions,
    split_rows,
    summarize_metric_table,
    top_axis_targets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a multi-factor linear axis bank from note embeddings, then compress it with SVD."
    )
    parser.add_argument("--embeddings_path", required=True, help="Path to sentence_embeddings.npy")
    parser.add_argument("--metadata_path", required=True, help="Path to sentence_embeddings_metadata.csv")
    parser.add_argument("--factors_path", required=True, help="CSV with demographic / cohort factor columns")
    parser.add_argument(
        "--join_cols",
        required=True,
        help="Comma-separated keys used to join metadata to factor table, e.g. subject_id,hadm_id",
    )
    parser.add_argument(
        "--factor_cols",
        required=True,
        help="Comma-separated factor columns to model, e.g. age,gender,race,insurance",
    )
    parser.add_argument(
        "--continuous_factors",
        default="",
        help="Optional comma-separated factor names to force as continuous",
    )
    parser.add_argument(
        "--categorical_factors",
        default="",
        help="Optional comma-separated factor names to force as categorical",
    )
    parser.add_argument(
        "--group_col",
        default="subject_id",
        help="Optional group column for held-out split, usually subject_id",
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Held-out split size")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--alphas",
        default="0.01,0.1,1,10,100",
        help="Comma-separated ridge alpha grid",
    )
    parser.add_argument("--max_axes", type=int, default=16, help="Maximum number of SVD axes to keep")
    parser.add_argument(
        "--energy_threshold",
        type=float,
        default=0.95,
        help="Recommended axis count should explain at least this fraction of probe energy",
    )
    parser.add_argument(
        "--score_ratio_threshold",
        type=float,
        default=0.99,
        help="Recommended axis count should recover at least this fraction of the full held-out score",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for axis bank outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    join_cols = parse_csv_list(args.join_cols)
    factor_cols = parse_csv_list(args.factor_cols)
    continuous_factors = parse_csv_list(args.continuous_factors)
    categorical_factors = parse_csv_list(args.categorical_factors)
    alphas = parse_float_list(args.alphas)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, merged_df = load_and_merge_tables(
        embeddings_path=args.embeddings_path,
        metadata_path=args.metadata_path,
        factors_path=args.factors_path,
        join_cols=join_cols,
    )
    embeddings, merged_df = filter_complete_cases(embeddings, merged_df, factor_cols)
    encoded = encode_targets(
        merged_df,
        factor_cols=factor_cols,
        continuous_overrides=continuous_factors,
        categorical_overrides=categorical_factors,
    )

    train_idx, test_idx = split_rows(
        merged_df,
        test_size=args.test_size,
        random_state=args.random_state,
        group_col=args.group_col,
    )

    x_train = embeddings[train_idx]
    x_test = embeddings[test_idx]
    y_train = encoded.matrix[train_idx]
    y_test = encoded.matrix[test_idx]

    scaler, ridge, coef_raw, intercept_raw = fit_multioutput_ridge(x_train, y_train, alphas=alphas)
    full_pred = x_test @ coef_raw.T + intercept_raw
    full_metrics = score_predictions(y_test, full_pred, encoded.manifest)
    full_summary = summarize_metric_table(full_metrics)

    axes, singular_values, energy_ratio, left_vectors = compute_axis_bank(
        coef_raw, max_axes=args.max_axes
    )
    cumulative_energy = np.cumsum(energy_ratio)
    axis_loadings = coef_raw @ axes

    truncated_rows = []
    full_clipped_score = full_summary.get("mean_clipped_score", np.nan)
    for k in range(1, axes.shape[1] + 1):
        coef_k = reconstruct_coefficients(axes, singular_values, left_vectors, k=k)
        pred_k = x_test @ coef_k.T + intercept_raw
        metrics_k = score_predictions(y_test, pred_k, encoded.manifest)
        summary_k = summarize_metric_table(metrics_k)
        truncated_clipped_score = summary_k.get("mean_clipped_score", np.nan)
        summary_k.update(
            {
                "k": k,
                "cumulative_energy": float(cumulative_energy[k - 1]),
                "full_score_ratio": (
                    float(truncated_clipped_score / full_clipped_score)
                    if not np.isnan(truncated_clipped_score)
                    and not np.isnan(full_clipped_score)
                    and full_clipped_score != 0.0
                    else np.nan
                ),
            }
        )
        truncated_rows.append(summary_k)

    truncated_df = pd.DataFrame(truncated_rows)
    valid_k = truncated_df[
        (truncated_df["cumulative_energy"] >= args.energy_threshold)
        & (truncated_df["full_score_ratio"] >= args.score_ratio_threshold)
    ]
    recommended_k = int(valid_k["k"].iloc[0]) if not valid_k.empty else int(truncated_df["k"].iloc[-1])

    split_manifest = merged_df.loc[:, ["embedding_row_id", *join_cols]].copy()
    split_manifest["split"] = "train"
    split_manifest.loc[test_idx, "split"] = "test"

    encoded.manifest.to_csv(output_dir / "target_manifest.csv", index=False)
    full_metrics.to_csv(output_dir / "heldout_target_metrics_full.csv", index=False)
    truncated_df.to_csv(output_dir / "heldout_axis_count_curve.csv", index=False)
    split_manifest.to_csv(output_dir / "split_manifest.csv", index=False)

    np.savez(
        output_dir / "axis_bank.npz",
        axes=axes,
        singular_values=singular_values,
        energy_ratio=energy_ratio,
        cumulative_energy=cumulative_energy,
        coefficient_matrix=coef_raw,
        intercept=intercept_raw,
        axis_loadings=axis_loadings,
        feature_mean=scaler.mean_,
        feature_scale=scaler.scale_,
        train_indices=train_idx,
        test_indices=test_idx,
    )

    summary = {
        "n_rows_after_complete_case_filter": int(len(merged_df)),
        "embedding_dim": int(embeddings.shape[1]),
        "n_factors": int(len(factor_cols)),
        "n_target_columns": int(encoded.matrix.shape[1]),
        "factor_types": encoded.factor_types,
        "ridge_alpha": float(ridge.alpha_),
        "max_axes": int(axes.shape[1]),
        "recommended_k": recommended_k,
        "energy_threshold": float(args.energy_threshold),
        "score_ratio_threshold": float(args.score_ratio_threshold),
        "full_model_summary": full_summary,
        "top_targets_by_axis": top_axis_targets(axis_loadings, encoded.manifest, top_n=5),
    }
    save_json(output_dir / "axis_bank_summary.json", summary)

    print(f"Saved axis bank to: {output_dir}")
    print(f"Rows after filtering: {len(merged_df)}")
    print(f"Target columns in coefficient matrix: {encoded.matrix.shape[1]}")
    print(f"Ridge alpha: {ridge.alpha_}")
    print(f"Recommended axis count: {recommended_k}")
    if full_summary:
        print(f"Full held-out summary: {full_summary}")


if __name__ == "__main__":
    main()
