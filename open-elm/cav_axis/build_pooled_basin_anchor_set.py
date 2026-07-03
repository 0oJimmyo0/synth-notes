#!/usr/bin/env python3
"""
Build a pooled-basin anchor set from the official real-manifold mapping outputs.

This script is used when exact single-cluster crossing is too strict for the
actual project target. It selects anchors that are hard for pooled-basin entry
or low-density basin enrichment, rather than merely hard for reassignment to
one exact KMeans bucket.
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build pooled-basin anchor rows from real-manifold outputs.")
    parser.add_argument("--dataset_path", required=True, help="HF dataset path, e.g. encoded_testing_filtered")
    parser.add_argument("--cluster_assignments_path", required=True, help="CSV from real_*_cluster_assignments.csv")
    parser.add_argument("--cluster_summary_path", required=True, help="CSV from real_*_cluster_summary.csv")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--basin_cluster_ids", required=True, help="Comma-separated basin clusters, e.g. 29,9,17,45")
    parser.add_argument(
        "--source_split",
        default="test",
        help="Optional split restriction, usually test for held-out pilots",
    )
    parser.add_argument(
        "--candidate_pool",
        default="outside_basin",
        choices=["outside_basin", "all"],
        help="Which rows are eligible as anchors",
    )
    parser.add_argument(
        "--ranking_mode",
        default="closest_outside_basin",
        choices=["closest_outside_basin", "closest_low_density_basin"],
        help="How to rank candidate anchors",
    )
    parser.add_argument(
        "--low_density_quantile",
        type=float,
        default=0.2,
        help="Quantile used to define low-density clusters from density_proxy",
    )
    parser.add_argument("--n_anchors", type=int, default=256, help="Number of anchors to keep")
    parser.add_argument(
        "--patient_disjoint_only",
        action="store_true",
        help="Restrict anchors to patient-disjoint rows only",
    )
    parser.add_argument(
        "--comparison_cluster_ids",
        default="",
        help="Optional explicit cluster list for local-basin scoring. Defaults to basin + top nearby external clusters by centroid similarity.",
    )
    parser.add_argument(
        "--max_external_clusters",
        type=int,
        default=4,
        help="When comparison_cluster_ids is omitted, how many external clusters to include automatically",
    )
    return parser


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    def _default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def load_real_embeddings(dataset_path: Path) -> np.ndarray:
    dataset = Dataset.load_from_disk(str(dataset_path))
    rows = []
    for row in dataset["domain_embeddings"]:
        arr = np.asarray(row[0], dtype=np.float32)
        rows.append(arr)
    return normalize_rows(np.vstack(rows))


def build_centroids(real_embeddings: np.ndarray, assignments_df: pd.DataFrame, cluster_ids: list[int]) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for cluster_id in cluster_ids:
        ids = assignments_df.loc[assignments_df["cluster_id"] == cluster_id, "dataset_row_id"].astype(int).to_numpy()
        if len(ids) == 0:
            continue
        centroids[cluster_id] = normalize_rows(real_embeddings[ids].mean(axis=0, keepdims=True))[0]
    return centroids


def main() -> None:
    args = build_parser().parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    cluster_assignments_path = Path(args.cluster_assignments_path).resolve()
    cluster_summary_path = Path(args.cluster_summary_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    basin_cluster_ids = parse_int_list(args.basin_cluster_ids)
    if not basin_cluster_ids:
        raise ValueError("--basin_cluster_ids must not be empty")
    basin_cluster_set = set(basin_cluster_ids)

    assignments_df = pd.read_csv(cluster_assignments_path)
    summary_df = pd.read_csv(cluster_summary_path)
    real_embeddings = load_real_embeddings(dataset_path)

    assignments_df["dataset_row_id"] = pd.to_numeric(assignments_df["dataset_row_id"], errors="coerce")
    assignments_df["cluster_id"] = pd.to_numeric(assignments_df["cluster_id"], errors="coerce")
    assignments_df = assignments_df.dropna(subset=["dataset_row_id", "cluster_id"]).copy()
    assignments_df["dataset_row_id"] = assignments_df["dataset_row_id"].astype(int)
    assignments_df["cluster_id"] = assignments_df["cluster_id"].astype(int)
    assignments_df = assignments_df[(assignments_df["dataset_row_id"] >= 0) & (assignments_df["dataset_row_id"] < len(real_embeddings))].copy()
    assignments_df = assignments_df.drop_duplicates(subset=["dataset_row_id"], keep="first").reset_index(drop=True)

    if args.source_split and "split" in assignments_df.columns:
        assignments_df = assignments_df.loc[assignments_df["split"].astype(str) == str(args.source_split)].copy()

    if args.patient_disjoint_only and "patient_disjoint_from_train" in assignments_df.columns:
        flag = assignments_df["patient_disjoint_from_train"].astype(str).str.lower()
        assignments_df = assignments_df.loc[flag == "true"].copy()

    low_density_threshold = float(summary_df["density_proxy"].quantile(float(args.low_density_quantile)))
    low_density_clusters = sorted(
        set(
            pd.to_numeric(
                summary_df.loc[summary_df["density_proxy"] <= low_density_threshold, "cluster_id"],
                errors="coerce",
            ).dropna().astype(int).tolist()
        )
    )
    low_density_basin_clusters = sorted(set(low_density_clusters) & basin_cluster_set)
    if not low_density_basin_clusters:
        raise ValueError("No low-density basin clusters found; adjust basin_cluster_ids or low_density_quantile.")

    comparison_cluster_ids = parse_int_list(args.comparison_cluster_ids)
    if not comparison_cluster_ids:
        candidate_cluster_ids = sorted(assignments_df["cluster_id"].unique().tolist())
        centroids = build_centroids(real_embeddings, assignments_df, candidate_cluster_ids)
        basin_centroid = normalize_rows(
            np.mean(np.stack([centroids[cid] for cid in basin_cluster_ids if cid in centroids], axis=0), axis=0, keepdims=True)
        )[0]
        external_scores = []
        for cid in candidate_cluster_ids:
            if cid in basin_cluster_set or cid not in centroids:
                continue
            external_scores.append((cid, float(np.dot(centroids[cid], basin_centroid))))
        external_scores.sort(key=lambda item: item[1], reverse=True)
        comparison_cluster_ids = basin_cluster_ids + [cid for cid, _ in external_scores[: int(args.max_external_clusters)]]

    comparison_cluster_ids = sorted(dict.fromkeys(comparison_cluster_ids))
    centroids = build_centroids(real_embeddings, assignments_df, comparison_cluster_ids)
    missing = [cid for cid in comparison_cluster_ids if cid not in centroids]
    if missing:
        raise ValueError(f"Missing centroids for cluster IDs: {missing}")

    row_df = assignments_df.copy()
    row_embeddings = real_embeddings[row_df["dataset_row_id"].astype(int).to_numpy()]

    for cid in comparison_cluster_ids:
        row_df[f"cos_to_cluster_{cid}"] = row_embeddings @ centroids[cid]

    basin_cols = [f"cos_to_cluster_{cid}" for cid in basin_cluster_ids]
    external_cluster_ids = [cid for cid in comparison_cluster_ids if cid not in basin_cluster_set]
    external_cols = [f"cos_to_cluster_{cid}" for cid in external_cluster_ids]
    low_density_cols = [f"cos_to_cluster_{cid}" for cid in low_density_basin_clusters]
    non_low_density_cols = [f"cos_to_cluster_{cid}" for cid in comparison_cluster_ids if cid not in set(low_density_basin_clusters)]

    row_df["best_basin_cosine"] = row_df[basin_cols].max(axis=1)
    row_df["best_basin_cluster"] = row_df[basin_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)
    row_df["best_external_cosine"] = row_df[external_cols].max(axis=1) if external_cols else np.nan
    row_df["best_external_cluster"] = (
        row_df[external_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int) if external_cols else pd.NA
    )
    row_df["pooled_basin_margin_vs_external"] = row_df["best_basin_cosine"] - row_df["best_external_cosine"]
    row_df["wins_pooled_basin"] = row_df["cluster_id"].isin(basin_cluster_set)

    row_df["best_low_density_basin_cosine"] = row_df[low_density_cols].max(axis=1)
    row_df["best_low_density_basin_cluster"] = row_df[low_density_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)
    if non_low_density_cols:
        row_df["best_non_low_density_cosine"] = row_df[non_low_density_cols].max(axis=1)
        row_df["best_non_low_density_cluster"] = row_df[non_low_density_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)
    else:
        row_df["best_non_low_density_cosine"] = np.nan
        row_df["best_non_low_density_cluster"] = pd.NA
    row_df["low_density_basin_margin"] = row_df["best_low_density_basin_cosine"] - row_df["best_non_low_density_cosine"]
    row_df["in_low_density_basin"] = row_df["cluster_id"].isin(set(low_density_basin_clusters))

    if args.candidate_pool == "outside_basin":
        row_df = row_df.loc[~row_df["cluster_id"].isin(basin_cluster_set)].copy()

    if args.ranking_mode == "closest_outside_basin":
        # Hard pooled-basin anchors are rows with the smallest basin-vs-external margin.
        sort_cols = ["pooled_basin_margin_vs_external", "best_basin_cosine"]
        ascending = [True, False]
    else:
        # Hard low-density anchors are rows with the smallest low-density margin.
        sort_cols = ["low_density_basin_margin", "best_low_density_basin_cosine"]
        ascending = [True, False]

    anchors_df = (
        row_df.sort_values(sort_cols, ascending=ascending)
        .head(int(args.n_anchors))
        .copy()
        .reset_index(drop=True)
    )
    anchors_df["anchor_rank"] = np.arange(1, len(anchors_df) + 1)
    anchors_df["anchor_set_name"] = "pooled_basin_anchor_set"
    anchors_df["ranking_mode"] = args.ranking_mode
    anchors_df["candidate_pool"] = args.candidate_pool
    anchors_df["basin_cluster_ids"] = ",".join(str(x) for x in basin_cluster_ids)
    anchors_df["comparison_cluster_ids"] = ",".join(str(x) for x in comparison_cluster_ids)
    anchors_df["low_density_basin_cluster_ids"] = ",".join(str(x) for x in low_density_basin_clusters)
    anchors_df["pooled_basin_anchor_flag"] = 1

    out_csv = output_dir / "pooled_basin_anchor_manifest.csv"
    summary_json = output_dir / "pooled_basin_anchor_summary.json"
    out_md = output_dir / "pooled_basin_anchor_summary.md"

    anchors_df.to_csv(out_csv, index=False)

    summary = {
        "created_at": now_iso(),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "script_path": str(Path(__file__).resolve()),
        "dataset_path": str(dataset_path),
        "cluster_assignments_path": str(cluster_assignments_path),
        "cluster_summary_path": str(cluster_summary_path),
        "output_dir": str(output_dir),
        "basin_cluster_ids": basin_cluster_ids,
        "comparison_cluster_ids": comparison_cluster_ids,
        "low_density_basin_cluster_ids": low_density_basin_clusters,
        "source_split": args.source_split,
        "candidate_pool": args.candidate_pool,
        "ranking_mode": args.ranking_mode,
        "low_density_quantile": float(args.low_density_quantile),
        "low_density_threshold": low_density_threshold,
        "n_candidate_rows": int(len(row_df)),
        "n_anchor_rows": int(len(anchors_df)),
        "anchor_cluster_counts": {str(k): int(v) for k, v in anchors_df["cluster_id"].value_counts().sort_index().to_dict().items()},
        "mean_anchor_basin_margin": float(anchors_df["pooled_basin_margin_vs_external"].mean()) if len(anchors_df) else float("nan"),
        "mean_anchor_low_density_margin": float(anchors_df["low_density_basin_margin"].mean()) if len(anchors_df) else float("nan"),
    }
    save_json(summary_json, summary)

    md_lines = [
        "# Pooled Basin Anchor Summary",
        "",
        f"- Basin clusters: `{','.join(str(x) for x in basin_cluster_ids)}`",
        f"- Comparison clusters: `{','.join(str(x) for x in comparison_cluster_ids)}`",
        f"- Low-density basin clusters: `{','.join(str(x) for x in low_density_basin_clusters)}`",
        f"- Candidate pool: `{args.candidate_pool}`",
        f"- Ranking mode: `{args.ranking_mode}`",
        f"- Split: `{args.source_split}`",
        f"- Anchor rows: `{len(anchors_df)}`",
        "",
        "## Summary",
        "",
        f"- mean pooled_basin_margin_vs_external: `{summary['mean_anchor_basin_margin']:.6f}`",
        f"- mean low_density_basin_margin: `{summary['mean_anchor_low_density_margin']:.6f}`",
        f"- anchor cluster counts: `{summary['anchor_cluster_counts']}`",
        "",
        "## Files",
        "",
        "- `pooled_basin_anchor_manifest.csv`",
        "- `pooled_basin_anchor_summary.json`",
    ]
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Saved pooled basin anchors to: {out_csv}")
    print(f"Saved pooled basin summary to: {summary_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
