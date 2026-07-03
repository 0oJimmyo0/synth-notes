#!/usr/bin/env python3
"""
Build a two-stage basin transport dataset for Phase 2b.

Stage 1:
- move source anchors toward a pooled target basin
- evaluate pooled-basin entry pre-decode

Stage 2:
- only for anchors whose best Stage-1 row enters the basin,
  refine toward a low-density sub-basin
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset

from build_barycentric_transport_dataset import (
    barycentric_target,
    extract_embedding_vector,
    get_git_commit,
    load_dataset_rows,
    merge_optional_metadata,
    normalize_rows,
    normalize_scalar,
    maybe_int,
    parse_csv_list,
    parse_float_list,
    save_json,
    top1_per_cluster,
)


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a two-stage pooled-basin transport HF dataset.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--factors_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_selection_query", required=True)
    parser.add_argument("--split_manifest_path", default=None)
    parser.add_argument(
        "--join_cols",
        default="source_row_id,embedding_row_id,dataset_row_id,note_id,subject_id,hadm_id",
    )
    parser.add_argument("--source_split", default=None)

    parser.add_argument("--stage1_basin_cluster_ids", required=True)
    parser.add_argument("--stage1_external_cluster_ids", required=True)
    parser.add_argument("--stage1_target_selection_query", default=None)
    parser.add_argument("--stage1_alphas", required=True)
    parser.add_argument("--stage1_competitor_gammas", required=True)
    parser.add_argument("--stage1_k_target_neighbors", type=int, default=16)
    parser.add_argument("--stage1_k_competitor_neighbors", type=int, default=16)

    parser.add_argument("--stage2_low_density_cluster_ids", required=True)
    parser.add_argument("--stage2_target_selection_query", default=None)
    parser.add_argument("--stage2_competitor_cluster_ids", default="")
    parser.add_argument("--stage2_alphas", required=True)
    parser.add_argument("--stage2_competitor_gammas", required=True)
    parser.add_argument("--stage2_k_target_neighbors", type=int, default=16)
    parser.add_argument("--stage2_k_competitor_neighbors", type=int, default=16)

    parser.add_argument(
        "--neighbor_weight_mode",
        default="softmax_cosine",
        choices=["softmax_cosine", "inverse_distance", "uniform"],
    )
    parser.add_argument("--neighbor_temperature", type=float, default=0.02)
    parser.add_argument("--max_source_rows", type=int, default=None)
    parser.add_argument("--max_stage1_target_rows", type=int, default=None)
    parser.add_argument("--max_stage2_target_rows", type=int, default=None)
    parser.add_argument("--normalize_after_steering", action="store_true")
    parser.add_argument("--output_stem", default="two_stage_basin_transport")
    return parser


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_cluster_centroids(
    merged_df: pd.DataFrame,
    dataset: Dataset,
    cluster_ids: list[int],
) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for cid in cluster_ids:
        rows = merged_df.loc[pd.to_numeric(merged_df["cluster_id"], errors="coerce") == cid, "dataset_row_id"].astype(int).to_numpy()
        if len(rows) == 0:
            continue
        embs = normalize_rows(np.stack([extract_embedding_vector(dataset[int(i)]) for i in rows], axis=0))
        centroids[cid] = normalize_rows(embs.mean(axis=0, keepdims=True))[0]
    return centroids


def build_local_pool(
    dataset: Dataset,
    merged_df: pd.DataFrame,
    query: str,
    max_rows: int | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    df = merged_df.query(query, engine="python").copy()
    if max_rows is not None:
        df = df.head(max_rows).copy()
    if df.empty:
        raise ValueError(f"No rows remained after query: {query}")
    row_ids = df["dataset_row_id"].astype(int).to_numpy()
    embs = normalize_rows(np.stack([extract_embedding_vector(dataset[int(i)]) for i in row_ids], axis=0))
    return df.reset_index(drop=True), embs


def build_competitor_pools(
    dataset: Dataset,
    merged_df: pd.DataFrame,
    competitor_cluster_ids: list[int],
    source_embeddings: np.ndarray,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, tuple[np.ndarray, np.ndarray]]]:
    cluster_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    cluster_top1: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cid in competitor_cluster_ids:
        subset = merged_df.loc[pd.to_numeric(merged_df["cluster_id"], errors="coerce") == cid].copy()
        if subset.empty:
            continue
        row_ids = subset["dataset_row_id"].astype(int).to_numpy()
        embs = normalize_rows(np.stack([extract_embedding_vector(dataset[int(i)]) for i in row_ids], axis=0))
        cluster_arrays[cid] = (row_ids, embs)
        cluster_top1[cid] = top1_per_cluster(source_embeddings, embs)
    return cluster_arrays, cluster_top1


def compute_best_competitor_mix(
    anchor_embedding: np.ndarray,
    source_order: int,
    competitor_cluster_ids: list[int],
    competitor_cluster_arrays: dict[int, tuple[np.ndarray, np.ndarray]],
    competitor_cluster_top1: dict[int, tuple[np.ndarray, np.ndarray]],
    k_neighbors: int,
    neighbor_weight_mode: str,
    neighbor_temperature: float,
) -> dict[str, Any]:
    if not competitor_cluster_ids:
        return {
            "best_competitor_cluster": None,
            "best_competitor_dataset_row_id": None,
            "best_competitor_anchor_cosine": None,
            "competitor_neighbor_count": 0,
            "competitor_neighbor_mean_cosine": None,
            "competitor_weight_entropy": None,
            "competitor_mix": None,
        }

    candidates: list[tuple[int, float]] = []
    for cid in competitor_cluster_ids:
        if cid not in competitor_cluster_top1:
            continue
        scores, _ = competitor_cluster_top1[cid]
        candidates.append((cid, float(scores[source_order])))
    if not candidates:
        return {
            "best_competitor_cluster": None,
            "best_competitor_dataset_row_id": None,
            "best_competitor_anchor_cosine": None,
            "competitor_neighbor_count": 0,
            "competitor_neighbor_mean_cosine": None,
            "competitor_weight_entropy": None,
            "competitor_mix": None,
        }

    candidates.sort(key=lambda item: item[1], reverse=True)
    best_competitor_cluster = int(candidates[0][0])
    best_competitor_anchor_cosine = float(candidates[0][1])
    comp_row_ids, comp_embeddings = competitor_cluster_arrays[best_competitor_cluster]
    k_comp = min(int(k_neighbors), len(comp_row_ids))
    comp_sims = comp_embeddings @ anchor_embedding
    top_idx = np.argpartition(-comp_sims, kth=k_comp - 1)[:k_comp]
    top_sims = comp_sims[top_idx]
    order = np.argsort(-top_sims)
    top_idx = top_idx[order]
    top_sims = top_sims[order]
    comp_neighbor_ids = comp_row_ids[top_idx]
    comp_neighbor_embs = comp_embeddings[top_idx]
    comp_neighbor_dist = (1.0 - top_sims).astype(np.float32)
    competitor_mix, competitor_weights = barycentric_target(
        comp_neighbor_embs,
        cosine_distances=comp_neighbor_dist,
        mode=neighbor_weight_mode,
        temperature=neighbor_temperature,
    )
    competitor_entropy = float(-np.sum(competitor_weights * np.log(np.clip(competitor_weights, 1e-12, None))))
    return {
        "best_competitor_cluster": best_competitor_cluster,
        "best_competitor_dataset_row_id": int(comp_neighbor_ids[0]),
        "best_competitor_anchor_cosine": best_competitor_anchor_cosine,
        "competitor_neighbor_count": int(k_comp),
        "competitor_neighbor_mean_cosine": float(np.mean(top_sims)),
        "competitor_weight_entropy": competitor_entropy,
        "competitor_mix": competitor_mix,
    }


def score_stage1_basin(
    shifted_embedding: np.ndarray,
    basin_centroids: dict[int, np.ndarray],
    external_centroids: dict[int, np.ndarray],
) -> tuple[float, int | None, float, int | None, float]:
    basin_scores = {cid: float(np.dot(shifted_embedding, centroid)) for cid, centroid in basin_centroids.items()}
    external_scores = {cid: float(np.dot(shifted_embedding, centroid)) for cid, centroid in external_centroids.items()}
    best_basin_cluster, best_basin_cos = max(basin_scores.items(), key=lambda item: item[1])
    if external_scores:
        best_external_cluster, best_external_cos = max(external_scores.items(), key=lambda item: item[1])
    else:
        best_external_cluster, best_external_cos = (None, float("nan"))
    return best_basin_cos, best_basin_cluster, best_external_cos, best_external_cluster, float(best_basin_cos - best_external_cos)


def score_stage2_low_density(
    shifted_embedding: np.ndarray,
    low_density_centroids: dict[int, np.ndarray],
    non_low_density_centroids: dict[int, np.ndarray],
) -> tuple[float, int | None, float, int | None, float]:
    low_scores = {cid: float(np.dot(shifted_embedding, centroid)) for cid, centroid in low_density_centroids.items()}
    non_scores = {cid: float(np.dot(shifted_embedding, centroid)) for cid, centroid in non_low_density_centroids.items()}
    best_low_cluster, best_low_cos = max(low_scores.items(), key=lambda item: item[1])
    if non_scores:
        best_non_cluster, best_non_cos = max(non_scores.items(), key=lambda item: item[1])
    else:
        best_non_cluster, best_non_cos = (None, float("nan"))
    return best_low_cos, best_low_cluster, best_non_cos, best_non_cluster, float(best_low_cos - best_non_cos)


def make_dataset_and_manifest(rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]], output_dir: Path, output_stem: str) -> None:
    ds = Dataset.from_list(rows)
    ds.save_to_disk(str(output_dir))
    pd.DataFrame(manifest_rows).to_csv(output_dir / f"{output_stem}_dataset_manifest.csv", index=False)


def main() -> None:
    args = build_parser().parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    factors_path = Path(args.factors_path).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    preferred_join_cols = parse_csv_list(args.join_cols)
    stage1_basin_cluster_ids = parse_int_list(args.stage1_basin_cluster_ids)
    stage1_external_cluster_ids = parse_int_list(args.stage1_external_cluster_ids)
    stage2_low_density_cluster_ids = parse_int_list(args.stage2_low_density_cluster_ids)
    stage1_alphas = parse_float_list(args.stage1_alphas)
    stage1_competitor_gammas = parse_float_list(args.stage1_competitor_gammas)
    stage2_alphas = parse_float_list(args.stage2_alphas)
    stage2_competitor_gammas = parse_float_list(args.stage2_competitor_gammas)
    stage2_competitor_cluster_ids = parse_int_list(args.stage2_competitor_cluster_ids)
    if not stage2_competitor_cluster_ids:
        stage2_competitor_cluster_ids = sorted(set(stage1_basin_cluster_ids + stage1_external_cluster_ids) - set(stage2_low_density_cluster_ids))

    dataset, dataset_df = load_dataset_rows(dataset_path)
    if args.source_split and "split" not in dataset_df.columns:
        dataset_df["split"] = args.source_split

    merged_df, active_join_cols, join_report = merge_optional_metadata(
        dataset_df,
        factors_path=factors_path,
        split_manifest_path=args.split_manifest_path,
        preferred_join_cols=preferred_join_cols,
        source_split=args.source_split,
    )
    if args.source_split and "split" in merged_df.columns:
        merged_df = merged_df.loc[merged_df["split"] == args.source_split].copy()

    source_df = merged_df.query(args.source_selection_query, engine="python").copy()
    if args.max_source_rows is not None:
        source_df = source_df.head(args.max_source_rows).copy()
    if source_df.empty:
        raise ValueError("No source rows remained after applying --source_selection_query.")

    stage1_target_query = args.stage1_target_selection_query or f"cluster_id in [{','.join(str(x) for x in stage1_basin_cluster_ids)}]"
    stage2_target_query = args.stage2_target_selection_query or f"cluster_id in [{','.join(str(x) for x in stage2_low_density_cluster_ids)}]"

    stage1_target_df, stage1_target_embeddings = build_local_pool(dataset, merged_df, stage1_target_query, args.max_stage1_target_rows)
    stage2_target_df, stage2_target_embeddings = build_local_pool(dataset, merged_df, stage2_target_query, args.max_stage2_target_rows)

    source_row_ids = source_df["dataset_row_id"].astype(int).to_numpy()
    source_embeddings = normalize_rows(np.stack([extract_embedding_vector(dataset[int(i)]) for i in source_row_ids], axis=0))

    stage1_comp_arrays, stage1_comp_top1 = build_competitor_pools(dataset, merged_df, stage1_external_cluster_ids, source_embeddings)
    stage2_comp_arrays, _ = build_competitor_pools(dataset, merged_df, stage2_competitor_cluster_ids, source_embeddings)

    stage1_basin_centroids = build_cluster_centroids(merged_df, dataset, stage1_basin_cluster_ids)
    stage1_external_centroids = build_cluster_centroids(merged_df, dataset, stage1_external_cluster_ids)
    stage2_low_density_centroids = build_cluster_centroids(merged_df, dataset, stage2_low_density_cluster_ids)
    stage2_non_low_density_centroids = build_cluster_centroids(merged_df, dataset, stage2_competitor_cluster_ids)

    stage1_k_target = min(int(args.stage1_k_target_neighbors), len(stage1_target_df))
    stage2_k_target = min(int(args.stage2_k_target_neighbors), len(stage2_target_df))

    stage1_rows: list[dict[str, Any]] = []
    stage1_manifest_rows: list[dict[str, Any]] = []
    stage1_best_per_source: dict[int, dict[str, Any]] = {}

    stage1_target_scores = source_embeddings @ stage1_target_embeddings.T

    for source_order, (_, source_row) in enumerate(source_df.reset_index(drop=True).iterrows()):
        dataset_row_id = int(source_row["dataset_row_id"])
        example = dataset[dataset_row_id]
        source_embedding = extract_embedding_vector(example)
        source_unit = normalize_rows(source_embedding.reshape(1, -1))[0]

        sims = stage1_target_scores[source_order]
        top_idx = np.argpartition(-sims, kth=stage1_k_target - 1)[:stage1_k_target]
        top_sims = sims[top_idx]
        order = np.argsort(-top_sims)
        top_idx = top_idx[order]
        top_sims = top_sims[order]
        target_neighbor_ids = stage1_target_df["dataset_row_id"].astype(int).to_numpy()[top_idx]
        target_neighbor_embs = stage1_target_embeddings[top_idx]
        target_neighbor_dist = (1.0 - top_sims).astype(np.float32)
        target_mix, target_weights = barycentric_target(
            target_neighbor_embs,
            cosine_distances=target_neighbor_dist,
            mode=args.neighbor_weight_mode,
            temperature=float(args.neighbor_temperature),
        )
        target_weight_entropy = float(-np.sum(target_weights * np.log(np.clip(target_weights, 1e-12, None))))

        comp_info = compute_best_competitor_mix(
            anchor_embedding=source_unit,
            source_order=source_order,
            competitor_cluster_ids=stage1_external_cluster_ids,
            competitor_cluster_arrays=stage1_comp_arrays,
            competitor_cluster_top1=stage1_comp_top1,
            k_neighbors=int(args.stage1_k_competitor_neighbors),
            neighbor_weight_mode=args.neighbor_weight_mode,
            neighbor_temperature=float(args.neighbor_temperature),
        )

        for alpha in stage1_alphas:
            for gamma in stage1_competitor_gammas:
                if comp_info["competitor_mix"] is not None:
                    competitor_aware_target = ((1.0 + gamma) * target_mix) - (gamma * comp_info["competitor_mix"])
                    competitor_aware_target = normalize_rows(competitor_aware_target.reshape(1, -1))[0]
                else:
                    competitor_aware_target = target_mix

                shifted = ((1.0 - alpha) * source_unit) + (alpha * competitor_aware_target)
                if args.normalize_after_steering:
                    shifted = normalize_rows(shifted.reshape(1, -1))[0]
                shifted = shifted.astype(np.float32)

                best_basin_cos, best_basin_cluster, best_external_cos, best_external_cluster, basin_margin = score_stage1_basin(
                    shifted,
                    basin_centroids=stage1_basin_centroids,
                    external_centroids=stage1_external_centroids,
                )
                row = {
                    "input_ids": example["input_ids"],
                    "domain_embeddings": [shifted.tolist()],
                    "source_row_id": maybe_int(source_row.get("source_row_id", source_row.get("dataset_row_id"))),
                    "source_row_id_resolved": int(dataset_row_id),
                    "dataset_row_id": maybe_int(source_row.get("dataset_row_id")),
                    "embedding_row_id": maybe_int(source_row.get("embedding_row_id")),
                    "note_id": normalize_scalar(source_row.get("note_id")),
                    "subject_id": maybe_int(source_row.get("subject_id")),
                    "hadm_id": maybe_int(source_row.get("hadm_id")),
                    "split": normalize_scalar(source_row.get("split", args.source_split)),
                    "source_embedding_id": str(maybe_int(source_row.get("embedding_row_id", source_row.get("dataset_row_id")))),
                    "patient_disjoint_from_train": normalize_scalar(source_row.get("patient_disjoint_from_train")),
                    "hadm_disjoint_from_train": normalize_scalar(source_row.get("hadm_disjoint_from_train")),
                    "note_disjoint_from_train": normalize_scalar(source_row.get("note_disjoint_from_train")),
                    "patient_overlap_with_train": normalize_scalar(source_row.get("patient_overlap_with_train")),
                    "hadm_overlap_with_train": normalize_scalar(source_row.get("hadm_overlap_with_train")),
                    "note_overlap_with_train": normalize_scalar(source_row.get("note_overlap_with_train")),
                    "axis_id": 0,
                    "axis_label": f"{args.output_stem}__stage1",
                    "alpha": float(alpha),
                    "competitor_gamma": float(gamma),
                    "normalized_after_steering": bool(args.normalize_after_steering),
                    "random_shift_norm": float(np.linalg.norm(shifted - source_unit)),
                    "editor_model": None,
                    "edited_text": None,
                    "post_edit_source_cosine": float(np.dot(source_unit, shifted)),
                    "source_dataset_path": str(dataset_path),
                    "source_split": normalize_scalar(source_row.get("split", args.source_split)),
                    "selection_query": args.source_selection_query,
                    "transport_mode": "two_stage_stage1",
                    "stage1_best_basin_cosine": best_basin_cos,
                    "stage1_best_basin_cluster": best_basin_cluster,
                    "stage1_best_external_cosine": best_external_cos,
                    "stage1_best_external_cluster": best_external_cluster,
                    "stage1_basin_margin_vs_external": basin_margin,
                    "stage1_enters_basin": int(basin_margin > 0),
                    "stage1_nearest_target_dataset_row_id": int(target_neighbor_ids[0]),
                    "stage1_target_neighbor_entropy": target_weight_entropy,
                    "stage1_best_competitor_cluster": comp_info["best_competitor_cluster"],
                    "stage1_best_competitor_dataset_row_id": comp_info["best_competitor_dataset_row_id"],
                    "stage1_local_competitor_cosine": comp_info["best_competitor_anchor_cosine"],
                }
                stage1_rows.append(row)
                stage1_manifest_rows.append(
                    {
                        "shifted_dataset_row_id": len(stage1_manifest_rows),
                        "dataset_row_id": row["dataset_row_id"],
                        "source_row_id_resolved": row["source_row_id_resolved"],
                        "alpha": row["alpha"],
                        "competitor_gamma": row["competitor_gamma"],
                        "stage1_basin_margin_vs_external": row["stage1_basin_margin_vs_external"],
                        "stage1_enters_basin": row["stage1_enters_basin"],
                    }
                )
                if row["stage1_enters_basin"] == 1:
                    prev = stage1_best_per_source.get(int(dataset_row_id))
                    prev_margin = (
                        None
                        if prev is None
                        else float(prev["stage1_row"]["stage1_basin_margin_vs_external"])
                    )
                    if prev is None or float(row["stage1_basin_margin_vs_external"]) > prev_margin:
                        stage1_best_per_source[int(dataset_row_id)] = {
                            "source_row": source_row.to_dict(),
                            "source_unit": source_unit.copy(),
                            "stage1_shifted": shifted.copy(),
                            "stage1_row": row.copy(),
                        }

    stage2_rows: list[dict[str, Any]] = []
    stage2_manifest_rows: list[dict[str, Any]] = []

    stage2_target_ids = stage2_target_df["dataset_row_id"].astype(int).to_numpy()
    for dataset_row_id, payload in stage1_best_per_source.items():
        source_row = payload["source_row"]
        stage1_shifted = payload["stage1_shifted"]
        example = dataset[int(dataset_row_id)]

        sims = stage2_target_embeddings @ stage1_shifted
        top_idx = np.argpartition(-sims, kth=stage2_k_target - 1)[:stage2_k_target]
        top_sims = sims[top_idx]
        order = np.argsort(-top_sims)
        top_idx = top_idx[order]
        top_sims = top_sims[order]
        target_neighbor_ids = stage2_target_ids[top_idx]
        target_neighbor_embs = stage2_target_embeddings[top_idx]
        target_neighbor_dist = (1.0 - top_sims).astype(np.float32)
        target_mix, target_weights = barycentric_target(
            target_neighbor_embs,
            cosine_distances=target_neighbor_dist,
            mode=args.neighbor_weight_mode,
            temperature=float(args.neighbor_temperature),
        )
        target_weight_entropy = float(-np.sum(target_weights * np.log(np.clip(target_weights, 1e-12, None))))

        # Recompute competitor choice against the Stage-1 shifted anchor.
        comp_candidates = []
        for cid, (row_ids, embs) in stage2_comp_arrays.items():
            sims_comp = embs @ stage1_shifted
            comp_candidates.append((cid, float(sims_comp.max())))
        comp_candidates.sort(key=lambda item: item[1], reverse=True)
        if comp_candidates:
            best_comp_cluster = int(comp_candidates[0][0])
            row_ids, embs = stage2_comp_arrays[best_comp_cluster]
            k_comp = min(int(args.stage2_k_competitor_neighbors), len(row_ids))
            sims_comp = embs @ stage1_shifted
            top_idx = np.argpartition(-sims_comp, kth=k_comp - 1)[:k_comp]
            top_sims = sims_comp[top_idx]
            order = np.argsort(-top_sims)
            top_idx = top_idx[order]
            top_sims = top_sims[order]
            comp_neighbor_ids = row_ids[top_idx]
            comp_neighbor_embs = embs[top_idx]
            comp_neighbor_dist = (1.0 - top_sims).astype(np.float32)
            competitor_mix, competitor_weights = barycentric_target(
                comp_neighbor_embs,
                cosine_distances=comp_neighbor_dist,
                mode=args.neighbor_weight_mode,
                temperature=float(args.neighbor_temperature),
            )
            competitor_entropy = float(-np.sum(competitor_weights * np.log(np.clip(competitor_weights, 1e-12, None))))
            best_comp_dataset_row_id = int(comp_neighbor_ids[0])
            best_comp_cos = float(top_sims[0])
        else:
            best_comp_cluster = None
            competitor_mix = None
            competitor_entropy = None
            best_comp_dataset_row_id = None
            best_comp_cos = None

        for alpha in stage2_alphas:
            for gamma in stage2_competitor_gammas:
                if competitor_mix is not None:
                    competitor_aware_target = ((1.0 + gamma) * target_mix) - (gamma * competitor_mix)
                    competitor_aware_target = normalize_rows(competitor_aware_target.reshape(1, -1))[0]
                else:
                    competitor_aware_target = target_mix

                shifted = ((1.0 - alpha) * stage1_shifted) + (alpha * competitor_aware_target)
                if args.normalize_after_steering:
                    shifted = normalize_rows(shifted.reshape(1, -1))[0]
                shifted = shifted.astype(np.float32)

                best_basin_cos, best_basin_cluster, best_external_cos, best_external_cluster, basin_margin = score_stage1_basin(
                    shifted,
                    basin_centroids=stage1_basin_centroids,
                    external_centroids=stage1_external_centroids,
                )
                best_low_cos, best_low_cluster, best_non_cos, best_non_cluster, low_margin = score_stage2_low_density(
                    shifted,
                    low_density_centroids=stage2_low_density_centroids,
                    non_low_density_centroids=stage2_non_low_density_centroids,
                )
                row = {
                    "input_ids": example["input_ids"],
                    "domain_embeddings": [shifted.tolist()],
                    "source_row_id": maybe_int(source_row.get("source_row_id", source_row.get("dataset_row_id"))),
                    "source_row_id_resolved": int(dataset_row_id),
                    "dataset_row_id": maybe_int(source_row.get("dataset_row_id")),
                    "embedding_row_id": maybe_int(source_row.get("embedding_row_id")),
                    "note_id": normalize_scalar(source_row.get("note_id")),
                    "subject_id": maybe_int(source_row.get("subject_id")),
                    "hadm_id": maybe_int(source_row.get("hadm_id")),
                    "split": normalize_scalar(source_row.get("split", args.source_split)),
                    "source_embedding_id": str(maybe_int(source_row.get("embedding_row_id", source_row.get("dataset_row_id")))),
                    "patient_disjoint_from_train": normalize_scalar(source_row.get("patient_disjoint_from_train")),
                    "hadm_disjoint_from_train": normalize_scalar(source_row.get("hadm_disjoint_from_train")),
                    "note_disjoint_from_train": normalize_scalar(source_row.get("note_disjoint_from_train")),
                    "patient_overlap_with_train": normalize_scalar(source_row.get("patient_overlap_with_train")),
                    "hadm_overlap_with_train": normalize_scalar(source_row.get("hadm_overlap_with_train")),
                    "note_overlap_with_train": normalize_scalar(source_row.get("note_overlap_with_train")),
                    "axis_id": 0,
                    "axis_label": f"{args.output_stem}__stage2",
                    "alpha": float(alpha),
                    "competitor_gamma": float(gamma),
                    "normalized_after_steering": bool(args.normalize_after_steering),
                    "random_shift_norm": float(np.linalg.norm(shifted - payload["source_unit"])),
                    "editor_model": None,
                    "edited_text": None,
                    "post_edit_source_cosine": float(np.dot(payload["source_unit"], shifted)),
                    "source_dataset_path": str(dataset_path),
                    "source_split": normalize_scalar(source_row.get("split", args.source_split)),
                    "selection_query": args.source_selection_query,
                    "transport_mode": "two_stage_stage2",
                    "stage1_selected_alpha": payload["stage1_row"]["alpha"],
                    "stage1_selected_competitor_gamma": payload["stage1_row"]["competitor_gamma"],
                    "stage1_selected_basin_margin": payload["stage1_row"]["stage1_basin_margin_vs_external"],
                    "stage2_best_basin_cosine": best_basin_cos,
                    "stage2_best_basin_cluster": best_basin_cluster,
                    "stage2_best_external_cosine": best_external_cos,
                    "stage2_best_external_cluster": best_external_cluster,
                    "stage2_basin_margin_vs_external": basin_margin,
                    "stage2_enters_basin": int(basin_margin > 0),
                    "stage2_best_low_density_cosine": best_low_cos,
                    "stage2_best_low_density_cluster": best_low_cluster,
                    "stage2_best_non_low_density_cosine": best_non_cos,
                    "stage2_best_non_low_density_cluster": best_non_cluster,
                    "stage2_low_density_margin": low_margin,
                    "stage2_enters_low_density": int(low_margin > 0),
                    "stage2_nearest_target_dataset_row_id": int(target_neighbor_ids[0]),
                    "stage2_target_neighbor_entropy": target_weight_entropy,
                    "stage2_best_competitor_cluster": best_comp_cluster,
                    "stage2_best_competitor_dataset_row_id": best_comp_dataset_row_id,
                    "stage2_local_competitor_cosine": best_comp_cos,
                    "stage2_competitor_entropy": competitor_entropy,
                }
                stage2_rows.append(row)
                stage2_manifest_rows.append(
                    {
                        "shifted_dataset_row_id": len(stage2_manifest_rows),
                        "dataset_row_id": row["dataset_row_id"],
                        "source_row_id_resolved": row["source_row_id_resolved"],
                        "alpha": row["alpha"],
                        "competitor_gamma": row["competitor_gamma"],
                        "stage1_selected_alpha": row["stage1_selected_alpha"],
                        "stage1_selected_competitor_gamma": row["stage1_selected_competitor_gamma"],
                        "stage1_selected_basin_margin": row["stage1_selected_basin_margin"],
                        "stage2_basin_margin_vs_external": row["stage2_basin_margin_vs_external"],
                        "stage2_low_density_margin": row["stage2_low_density_margin"],
                        "stage2_enters_basin": row["stage2_enters_basin"],
                        "stage2_enters_low_density": row["stage2_enters_low_density"],
                    }
                )

    stage1_dir = output_root / f"{args.output_stem}_stage1"
    stage2_dir = output_root / f"{args.output_stem}_stage2"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)
    make_dataset_and_manifest(stage1_rows, stage1_manifest_rows, stage1_dir, f"{args.output_stem}_stage1")
    make_dataset_and_manifest(stage2_rows, stage2_manifest_rows, stage2_dir, f"{args.output_stem}_stage2")

    summary_payload = {
        "created_at": now_iso(),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "script_path": str(Path(__file__).resolve()),
        "dataset_path": str(dataset_path),
        "factors_path": str(factors_path),
        "output_dir": str(output_root),
        "output_stage1_dir": str(stage1_dir),
        "output_stage2_dir": str(stage2_dir),
        "split_manifest_path": str(Path(args.split_manifest_path).resolve()) if args.split_manifest_path else None,
        "join_report": join_report,
        "active_join_cols": active_join_cols,
        "source_selection_query": args.source_selection_query,
        "source_split": args.source_split,
        "stage1_target_selection_query": stage1_target_query,
        "stage2_target_selection_query": stage2_target_query,
        "stage1_basin_cluster_ids": stage1_basin_cluster_ids,
        "stage1_external_cluster_ids": stage1_external_cluster_ids,
        "stage2_low_density_cluster_ids": stage2_low_density_cluster_ids,
        "stage2_competitor_cluster_ids": stage2_competitor_cluster_ids,
        "stage1_alphas": stage1_alphas,
        "stage1_competitor_gammas": stage1_competitor_gammas,
        "stage2_alphas": stage2_alphas,
        "stage2_competitor_gammas": stage2_competitor_gammas,
        "n_source_rows": int(len(source_df)),
        "n_stage1_rows": int(len(stage1_rows)),
        "n_stage1_entered_basin_sources": int(len(stage1_best_per_source)),
        "n_stage2_rows": int(len(stage2_rows)),
        "cli_args": vars(args),
    }
    save_json(output_root / f"{args.output_stem}_run_metadata.json", summary_payload)

    print(f"Saved Stage-1 dataset to: {stage1_dir}")
    print(f"Saved Stage-2 dataset to: {stage2_dir}")
    print(f"Selected source rows: {len(source_df)}")
    print(f"Stage-1 shifted rows: {len(stage1_rows)}")
    print(f"Stage-1 basin-entry source rows: {len(stage1_best_per_source)}")
    print(f"Stage-2 shifted rows: {len(stage2_rows)}")


if __name__ == "__main__":
    main()
