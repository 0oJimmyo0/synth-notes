#!/usr/bin/env python3
"""
Evaluate whether synthetic notes enrich a pooled local basin rather than a single exact cluster.

This script is designed for the current Phase 2b setting where decoded notes
stay near a target cluster but may fall into adjacent official clusters after
decode -> re-embed. It supports:

- pooled-basin win rate
- pooled-basin margin against external comparison clusters
- leakage-aware stratification
- basic clinical/subgroup coherence summaries for the real pooled basin
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset


DEFAULT_PROFILE_FIELDS = [
    "service",
    "admission_type",
    "insurance",
    "age_bin",
    "sex_gender",
    "icu_flag",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pooled-basin enrichment for a synthetic run.")
    parser.add_argument("--real_dataset_path", required=True, help="Real HF dataset path, e.g. encoded_testing_filtered")
    parser.add_argument("--factors_path", required=True, help="Factor table with cluster_id and subgroup fields")
    parser.add_argument("--synthetic_manifest_path", required=True, help="Synthetic generation manifest JSONL")
    parser.add_argument("--synthetic_embeddings_path", required=True, help="Re-embedded synthetic notes (.npy)")
    parser.add_argument("--output_dir", required=True, help="Output directory for basin evaluation artifacts")
    parser.add_argument(
        "--basin_cluster_ids",
        required=True,
        help="Comma-separated official cluster IDs defining the pooled target basin, e.g. 29,9,17,45",
    )
    parser.add_argument(
        "--comparison_cluster_ids",
        required=True,
        help="Comma-separated official cluster IDs used in the local-basin competition analysis",
    )
    parser.add_argument(
        "--separate_cluster_ids",
        default="",
        help="Comma-separated cluster IDs to report separately in the summary, e.g. 7",
    )
    parser.add_argument(
        "--profile_fields",
        default=",".join(DEFAULT_PROFILE_FIELDS),
        help="Comma-separated subgroup fields to profile for the real basin",
    )
    parser.add_argument(
        "--top_levels_per_field",
        type=int,
        default=5,
        help="Number of top subgroup levels to keep per field in the profile table",
    )
    return parser


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_commit(script_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(script_dir.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def save_json(path: Path, payload: dict[str, Any]) -> None:
    def _default(v: Any) -> Any:
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        raise TypeError(f"Object of type {v.__class__.__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def load_real_embeddings(dataset_path: Path) -> np.ndarray:
    dataset = Dataset.load_from_disk(str(dataset_path))
    rows = []
    for row in dataset["domain_embeddings"]:
        arr = np.asarray(row[0], dtype=np.float32)
        rows.append(arr)
    return normalize_rows(np.vstack(rows))


def load_factor_table(path: Path, n_real_rows: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "dataset_row_id" not in df.columns:
        raise KeyError("Factor table must contain dataset_row_id")
    if "cluster_id" not in df.columns:
        raise KeyError("Factor table must contain cluster_id")
    df["dataset_row_id"] = pd.to_numeric(df["dataset_row_id"], errors="coerce")
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
    df = df.dropna(subset=["dataset_row_id", "cluster_id"]).copy()
    df["dataset_row_id"] = df["dataset_row_id"].astype(int)
    df["cluster_id"] = df["cluster_id"].astype(int)
    df = df[(df["dataset_row_id"] >= 0) & (df["dataset_row_id"] < n_real_rows)].copy()
    df = df.drop_duplicates(subset=["dataset_row_id"], keep="first").reset_index(drop=True)
    return df


def build_centroids(real_embeddings: np.ndarray, factors_df: pd.DataFrame, cluster_ids: list[int]) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for cluster_id in cluster_ids:
        ids = factors_df.loc[factors_df["cluster_id"] == cluster_id, "dataset_row_id"].to_numpy()
        if len(ids) == 0:
            raise ValueError(f"No real rows found for cluster_id={cluster_id}")
        centroid = real_embeddings[ids].mean(axis=0, keepdims=True)
        centroids[cluster_id] = normalize_rows(centroid)[0]
    return centroids


def leakage_group(values: pd.Series) -> pd.Series:
    normed = values.astype(str).str.lower()
    out = pd.Series(index=values.index, dtype=object)
    out.loc[normed == "true"] = "patient_disjoint"
    out.loc[normed == "false"] = "patient_overlap"
    out = out.fillna("unknown")
    return out


def profile_rows(df: pd.DataFrame, label: str, profile_fields: list[str], top_k: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in profile_fields:
        if field not in df.columns:
            continue
        vc = df[field].fillna("NA").astype(str).value_counts(normalize=True).head(top_k)
        for level, fraction in vc.items():
            rows.append(
                {
                    "profile_label": label,
                    "field": field,
                    "label": level,
                    "fraction": float(fraction),
                    "n_rows": int(len(df)),
                }
            )
    return pd.DataFrame(rows)


def summarize_group(
    df: pd.DataFrame,
    basin_cluster_ids: set[int],
    separate_cluster_ids: set[int],
    target_cluster_id: int,
) -> dict[str, Any]:
    best_counts = df["best_cluster_among_candidates"].value_counts().sort_index().to_dict()
    basin_mask = df["best_cluster_among_candidates"].isin(basin_cluster_ids)
    summary = {
        "n_rows": int(len(df)),
        "pooled_basin_win_rate": float(basin_mask.mean()) if len(df) else float("nan"),
        "target_cluster_id": int(target_cluster_id),
        "exact_target_cluster_win_rate": float((df["best_cluster_among_candidates"] == target_cluster_id).mean())
        if len(df)
        else float("nan"),
        "mean_basin_margin_vs_best_external": float(df["pooled_basin_margin_vs_best_external"].mean()) if len(df) else float("nan"),
        "median_basin_margin_vs_best_external": float(df["pooled_basin_margin_vs_best_external"].median()) if len(df) else float("nan"),
        "mean_target_cluster_margin_vs_best_other": float(df["target_cluster_margin_vs_best_other"].mean()) if len(df) else float("nan"),
        "median_target_cluster_margin_vs_best_other": float(df["target_cluster_margin_vs_best_other"].median()) if len(df) else float("nan"),
        "best_cluster_counts": {str(k): int(v) for k, v in best_counts.items()},
    }
    if separate_cluster_ids:
        for cluster_id in sorted(separate_cluster_ids):
            summary[f"win_rate_cluster_{cluster_id}"] = float((df["best_cluster_among_candidates"] == cluster_id).mean()) if len(df) else float("nan")
    return summary


def main() -> None:
    args = build_parser().parse_args()

    real_dataset_path = Path(args.real_dataset_path).resolve()
    factors_path = Path(args.factors_path).resolve()
    synthetic_manifest_path = Path(args.synthetic_manifest_path).resolve()
    synthetic_embeddings_path = Path(args.synthetic_embeddings_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    basin_cluster_ids = parse_int_list(args.basin_cluster_ids)
    comparison_cluster_ids = parse_int_list(args.comparison_cluster_ids)
    separate_cluster_ids = set(parse_int_list(args.separate_cluster_ids))
    profile_fields = parse_csv_list(args.profile_fields)

    basin_cluster_set = set(basin_cluster_ids)
    comparison_cluster_set = set(comparison_cluster_ids)
    if not basin_cluster_set:
        raise ValueError("--basin_cluster_ids must not be empty")
    if not comparison_cluster_set:
        raise ValueError("--comparison_cluster_ids must not be empty")
    if not basin_cluster_set.issubset(comparison_cluster_set):
        raise ValueError("All basin_cluster_ids must also be included in comparison_cluster_ids")

    real_embeddings = load_real_embeddings(real_dataset_path)
    factors_df = load_factor_table(factors_path, n_real_rows=len(real_embeddings))
    comparison_cluster_ids_sorted = sorted(comparison_cluster_set)
    centroids = build_centroids(real_embeddings, factors_df, comparison_cluster_ids_sorted)

    manifest_df = pd.read_json(synthetic_manifest_path, lines=True)
    synthetic_embeddings = normalize_rows(np.load(synthetic_embeddings_path))
    if len(manifest_df) != len(synthetic_embeddings):
        raise ValueError(
            f"Manifest rows ({len(manifest_df)}) do not match synthetic embeddings rows ({len(synthetic_embeddings)})"
        )

    row_df = manifest_df[["generation_id", "dataset_row_id", "note_id", "subject_id", "hadm_id"]].copy()
    if "patient_disjoint_from_train" in manifest_df.columns:
        row_df["patient_disjoint_from_train"] = manifest_df["patient_disjoint_from_train"]
    else:
        row_df["patient_disjoint_from_train"] = pd.NA
    row_df["analysis_group"] = leakage_group(row_df["patient_disjoint_from_train"])

    cosine_cols: list[str] = []
    for cluster_id in comparison_cluster_ids_sorted:
        col = f"cos_to_cluster_{cluster_id}"
        row_df[col] = synthetic_embeddings @ centroids[cluster_id]
        cosine_cols.append(col)

    row_df["best_cluster_among_candidates"] = (
        row_df[cosine_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)
    )

    basin_cols = [f"cos_to_cluster_{cluster_id}" for cluster_id in basin_cluster_ids]
    external_cluster_ids = [cluster_id for cluster_id in comparison_cluster_ids_sorted if cluster_id not in basin_cluster_set]
    external_cols = [f"cos_to_cluster_{cluster_id}" for cluster_id in external_cluster_ids]

    row_df["best_basin_cosine"] = row_df[basin_cols].max(axis=1)
    row_df["best_basin_cluster"] = row_df[basin_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)

    if external_cols:
        row_df["best_external_cosine"] = row_df[external_cols].max(axis=1)
        row_df["best_external_cluster"] = row_df[external_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)
    else:
        row_df["best_external_cosine"] = np.nan
        row_df["best_external_cluster"] = pd.NA

    target_cluster_id = basin_cluster_ids[0]
    target_col = f"cos_to_cluster_{target_cluster_id}"
    other_cols = [col for col in cosine_cols if col != target_col]
    row_df["target_cluster_margin_vs_best_other"] = row_df[target_col] - row_df[other_cols].max(axis=1)
    row_df["pooled_basin_margin_vs_best_external"] = row_df["best_basin_cosine"] - row_df["best_external_cosine"]
    row_df["wins_pooled_basin"] = row_df["best_cluster_among_candidates"].isin(basin_cluster_set)

    real_basin_df = factors_df.loc[factors_df["cluster_id"].isin(basin_cluster_set)].copy()
    real_separate_df = factors_df.loc[factors_df["cluster_id"].isin(separate_cluster_ids)].copy() if separate_cluster_ids else pd.DataFrame()

    group_rows = []
    full_df = row_df.copy()
    group_rows.append(
        {
            "analysis_group": "full",
            **summarize_group(full_df, basin_cluster_set, separate_cluster_ids, target_cluster_id),
        }
    )
    for group_name in ["patient_disjoint", "patient_overlap"]:
        subset = row_df.loc[row_df["analysis_group"] == group_name].copy()
        group_rows.append(
            {
                "analysis_group": group_name,
                **summarize_group(subset, basin_cluster_set, separate_cluster_ids, target_cluster_id),
            }
        )
    group_df = pd.DataFrame(group_rows)

    profile_frames = [profile_rows(real_basin_df, "real_pooled_basin", profile_fields, args.top_levels_per_field)]
    if not real_separate_df.empty:
        profile_frames.append(profile_rows(real_separate_df, "real_separate_clusters", profile_fields, args.top_levels_per_field))
    for cluster_id in basin_cluster_ids:
        cluster_df = factors_df.loc[factors_df["cluster_id"] == cluster_id].copy()
        profile_frames.append(profile_rows(cluster_df, f"real_cluster_{cluster_id}", profile_fields, args.top_levels_per_field))
    if separate_cluster_ids:
        for cluster_id in sorted(separate_cluster_ids):
            cluster_df = factors_df.loc[factors_df["cluster_id"] == cluster_id].copy()
            profile_frames.append(profile_rows(cluster_df, f"real_cluster_{cluster_id}", profile_fields, args.top_levels_per_field))
    profile_df = pd.concat([df for df in profile_frames if not df.empty], ignore_index=True)

    summary = {
        "created_at": now_iso(),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "script_path": str(Path(__file__).resolve()),
        "real_dataset_path": str(real_dataset_path),
        "factors_path": str(factors_path),
        "synthetic_manifest_path": str(synthetic_manifest_path),
        "synthetic_embeddings_path": str(synthetic_embeddings_path),
        "output_dir": str(output_dir),
        "basin_cluster_ids": basin_cluster_ids,
        "comparison_cluster_ids": comparison_cluster_ids_sorted,
        "separate_cluster_ids": sorted(separate_cluster_ids),
        "target_cluster_id": int(target_cluster_id),
        "n_real_rows": int(len(real_embeddings)),
        "n_synthetic_rows": int(len(row_df)),
        "full_metrics": group_rows[0],
        "patient_disjoint_metrics": group_rows[1],
        "patient_overlap_metrics": group_rows[2],
    }

    row_table_path = output_dir / "pooled_basin_row_table.csv"
    group_metrics_path = output_dir / "pooled_basin_group_metrics.csv"
    profile_path = output_dir / "pooled_basin_profile.csv"
    summary_path = output_dir / "pooled_basin_summary.json"
    report_path = output_dir / "pooled_basin_summary.md"

    row_df.to_csv(row_table_path, index=False)
    group_df.to_csv(group_metrics_path, index=False)
    profile_df.to_csv(profile_path, index=False)
    save_json(summary_path, summary)

    md_lines = [
        "# Pooled Basin Evaluation",
        "",
        f"- Basin clusters: `{','.join(map(str, basin_cluster_ids))}`",
        f"- Comparison clusters: `{','.join(map(str, comparison_cluster_ids_sorted))}`",
        f"- Separate clusters: `{','.join(map(str, sorted(separate_cluster_ids))) if separate_cluster_ids else '<none>'}`",
        f"- Synthetic rows: `{len(row_df)}`",
        "",
        "## Group Metrics",
        "",
    ]
    for _, row in group_df.iterrows():
        md_lines.append(
            f"- `{row['analysis_group']}`: "
            f"pooled_basin_win_rate={row['pooled_basin_win_rate']:.4f}, "
            f"exact_target_cluster_win_rate={row['exact_target_cluster_win_rate']:.4f}, "
            f"mean_basin_margin_vs_best_external={row['mean_basin_margin_vs_best_external']:.4f}"
        )
    md_lines += [
        "",
        "## Files",
        "",
        "- `pooled_basin_summary.json`",
        "- `pooled_basin_group_metrics.csv`",
        "- `pooled_basin_row_table.csv`",
        "- `pooled_basin_profile.csv`",
    ]
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Saved pooled basin summary to: {summary_path}")
    print(f"Saved pooled basin group metrics to: {group_metrics_path}")
    print(f"Saved pooled basin row table to: {row_table_path}")
    print(f"Saved pooled basin profile to: {profile_path}")
    print(json.dumps(summary["full_metrics"], indent=2))


if __name__ == "__main__":
    main()
