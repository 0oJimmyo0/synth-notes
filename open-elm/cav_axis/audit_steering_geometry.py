#!/usr/bin/env python3
"""
Audit steering geometry before scaling generation.

This script is designed for Phase 2b diagnostics:
1. cluster separability audit for a target region such as cluster_target_29
2. pre-decode proximity audit for shifted embeddings across alpha values
3. alpha-sensitivity summary
4. optional note-level comparison using generated-note embeddings

The goal is to answer a technical question before launching more generation:
does additive steering actually move embeddings toward the intended target
region, or does it only create broad manifold perturbations?
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from common import normalize_rows, parse_csv_list, parse_float_list, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit steering geometry before decode and optionally after decode.")
    parser.add_argument("--fit_dataset_path", required=True, help="HF dataset used to fit target geometry, usually train")
    parser.add_argument("--eval_dataset_path", required=True, help="HF dataset used as steering source/eval set, usually test")
    parser.add_argument("--factors_path", required=True, help="Factor table containing target columns such as cluster_target_29")
    parser.add_argument("--output_dir", required=True, help="Directory for diagnostic outputs")
    parser.add_argument("--target_column", required=True, help="Binary target column, e.g. cluster_target_29")
    parser.add_argument("--direction_bank_path", required=True, help="Local direction bank .npz")
    parser.add_argument(
        "--direction_labels",
        required=True,
        help="Comma-separated direction labels from the bank, e.g. local_centroid_difference__cluster_target_29,local_one_vs_rest_linear__cluster_target_29",
    )
    parser.add_argument("--alphas", required=True, help="Comma-separated steering strengths, e.g. 0,0.25,0.5,1.0")
    parser.add_argument("--split_manifest_path", default=None, help="Optional filtered-aligned split manifest")
    parser.add_argument(
        "--join_cols",
        default="source_row_id,embedding_row_id,dataset_row_id,note_id,subject_id,hadm_id",
        help="Preferred join columns for metadata merges",
    )
    parser.add_argument("--fit_split", default="train", help="Split label for fit dataset if not embedded")
    parser.add_argument("--eval_split", default="test", help="Split label for eval dataset if not embedded")
    parser.add_argument(
        "--source_selection_query",
        default=None,
        help="Optional pandas query over eval metadata; default is target_column == 1",
    )
    parser.add_argument("--max_source_rows", type=int, default=None, help="Optional cap on eval source anchors")
    parser.add_argument(
        "--normalize_after_steering",
        action="store_true",
        help="L2-normalize shifted embeddings after additive steering",
    )
    parser.add_argument(
        "--neighborhood_k",
        type=int,
        default=5,
        help="k used to estimate target-neighborhood radius from fit positives",
    )
    parser.add_argument(
        "--generated_manifest_paths",
        default=None,
        help="Optional comma-separated generated manifest JSONL paths for note-level comparison",
    )
    parser.add_argument(
        "--generated_embeddings_paths",
        default=None,
        help="Optional comma-separated .npy paths aligned to generated manifests",
    )
    parser.add_argument(
        "--generated_run_labels",
        default=None,
        help="Optional comma-separated display labels for generated runs",
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


def normalize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_dataset_rows(dataset_path: Path) -> tuple[Dataset, pd.DataFrame, np.ndarray]:
    dataset = Dataset.load_from_disk(str(dataset_path))
    base_df = pd.DataFrame({"dataset_row_id": np.arange(len(dataset), dtype=int)})
    metadata_cols = [col for col in dataset.column_names if col not in {"input_ids", "domain_embeddings"}]
    if metadata_cols:
        metadata_df = dataset.select_columns(metadata_cols).to_pandas()
        base_df = pd.concat([base_df, metadata_df.reset_index(drop=True)], axis=1)

    embeddings = []
    for example in dataset:
        emb = example["domain_embeddings"]
        if not isinstance(emb, list) or not emb:
            raise ValueError("Expected each dataset row to carry a non-empty domain_embeddings list.")
        embeddings.append(np.asarray(emb[0], dtype=np.float32))
    embedding_matrix = normalize_rows(np.vstack(embeddings).astype(np.float32))
    return dataset, base_df, embedding_matrix


def choose_join_keys(frames: list[pd.DataFrame], preferred_keys: list[str]) -> list[str]:
    common_cols = set(frames[0].columns)
    for frame in frames[1:]:
        common_cols &= set(frame.columns)

    preferred_groups = [
        ["split", "dataset_row_id"],
        ["source_row_id"],
        ["embedding_row_id"],
        ["dataset_row_id"],
        ["note_id", "subject_id", "hadm_id"],
        ["note_id"],
        ["subject_id", "hadm_id"],
    ]
    for key in preferred_keys:
        if key in {"note_id", "subject_id", "hadm_id"}:
            continue
        if [key] not in preferred_groups:
            preferred_groups.append([key])
    for keys in preferred_groups:
        if all(key in common_cols for key in keys):
            return keys
    raise ValueError(f"Could not detect stable join keys. Shared columns were: {sorted(common_cols)}")


def normalize_join_cols(df: pd.DataFrame, join_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in join_cols:
        if col not in out.columns:
            raise KeyError(f"Join column missing: {col}")
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().all():
            out[col] = numeric.astype(int)
        else:
            out[col] = out[col].astype(str).str.strip()
    return out


def merge_metadata(
    base_df: pd.DataFrame,
    factors_path: Path,
    split_manifest_path: str | None,
    preferred_join_cols: list[str],
    source_split: str | None,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    merged = base_df.copy()
    join_report: dict[str, list[str]] = {}
    active_join_cols = ["dataset_row_id"]

    if split_manifest_path:
        split_df = pd.read_csv(split_manifest_path)
        join_cols = choose_join_keys([merged, split_df], preferred_join_cols)
        if join_cols == ["dataset_row_id"] and "split" in split_df.columns:
            if source_split:
                split_df = split_df.loc[split_df["split"].astype(str) == str(source_split)].copy()
                join_report["split_manifest_filtered_to_split"] = [str(source_split)]
        merged = normalize_join_cols(merged, join_cols)
        split_df = normalize_join_cols(split_df, join_cols)
        duplicate_mask = split_df.duplicated(subset=join_cols, keep=False)
        if duplicate_mask.any():
            raise ValueError(f"Split manifest has duplicate rows for join keys {join_cols}; deduplicate it first.")
        merged = merged.merge(split_df, on=join_cols, how="left", validate="one_to_one", suffixes=("", "_split"))
        active_join_cols = join_cols
        join_report["split_manifest_join_cols"] = join_cols

    factors_df = pd.read_csv(factors_path)
    join_cols = choose_join_keys([merged, factors_df], preferred_join_cols)
    merged = normalize_join_cols(merged, join_cols)
    factors_df = normalize_join_cols(factors_df, join_cols)
    duplicate_mask = factors_df.duplicated(subset=join_cols, keep=False)
    if duplicate_mask.any():
        raise ValueError(f"Factor table has duplicate rows for join keys {join_cols}; deduplicate it first.")
    merged = merged.merge(factors_df, on=join_cols, how="left", validate="one_to_one", suffixes=("", "_factor"))
    active_join_cols = join_cols
    join_report["factors_join_cols"] = join_cols
    return merged, active_join_cols, join_report


def load_directions(direction_bank_path: Path, direction_labels: list[str]) -> dict[str, np.ndarray]:
    bank = np.load(direction_bank_path, allow_pickle=True)
    directions = np.asarray(bank["directions"], dtype=np.float32)
    bank_labels = [str(item) for item in bank["direction_labels"].tolist()]
    label_to_idx = {label: idx for idx, label in enumerate(bank_labels)}
    missing = [label for label in direction_labels if label not in label_to_idx]
    if missing:
        raise KeyError(f"Direction labels not found in bank: {missing}")
    return {label: directions[:, label_to_idx[label]].astype(np.float32) for label in direction_labels}


def prepare_binary_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if target_column not in df.columns:
        raise KeyError(f"Target column not found: {target_column}")
    numeric = pd.to_numeric(df[target_column], errors="coerce")
    out = df.loc[numeric.notna()].copy()
    out[target_column] = numeric.loc[out.index].astype(int)
    out = out.loc[out[target_column].isin([0, 1])].copy()
    return out


def fit_target_geometry(
    train_embeddings: np.ndarray,
    train_df: pd.DataFrame,
    eval_embeddings: np.ndarray,
    eval_df: pd.DataFrame,
    target_column: str,
    neighborhood_k: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, LogisticRegression, NearestNeighbors, float]:
    y_train = train_df[target_column].to_numpy(dtype=int)
    y_eval = eval_df[target_column].to_numpy(dtype=int)

    pos_train = train_embeddings[y_train == 1]
    neg_train = train_embeddings[y_train == 0]
    if len(pos_train) == 0 or len(neg_train) == 0:
        raise ValueError(f"Target column {target_column} needs both positive and negative fit rows.")

    pos_centroid = normalize_rows(pos_train.mean(axis=0, keepdims=True))[0].astype(np.float32)
    neg_centroid = normalize_rows(neg_train.mean(axis=0, keepdims=True))[0].astype(np.float32)

    clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=4000,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(train_embeddings, y_train)

    train_scores = clf.predict_proba(train_embeddings)[:, 1]
    eval_scores = clf.predict_proba(eval_embeddings)[:, 1]

    pos_neighbors = min(max(2, neighborhood_k + 1), len(pos_train))
    target_nn = NearestNeighbors(metric="cosine", n_neighbors=1)
    target_nn.fit(pos_train)

    intra_nn = NearestNeighbors(metric="cosine", n_neighbors=pos_neighbors)
    intra_nn.fit(pos_train)
    intra_distances, _ = intra_nn.kneighbors(pos_train)
    kth_distances = intra_distances[:, -1]
    target_radius = float(np.median(kth_distances))

    summary = {
        "target_column": target_column,
        "n_fit_rows": int(len(train_df)),
        "n_fit_positive": int((y_train == 1).sum()),
        "n_fit_negative": int((y_train == 0).sum()),
        "n_eval_rows": int(len(eval_df)),
        "n_eval_positive": int((y_eval == 1).sum()),
        "n_eval_negative": int((y_eval == 0).sum()),
        "train_auroc": float(roc_auc_score(y_train, train_scores)),
        "train_average_precision": float(average_precision_score(y_train, train_scores)),
        "eval_auroc": float(roc_auc_score(y_eval, eval_scores)),
        "eval_average_precision": float(average_precision_score(y_eval, eval_scores)),
        "positive_centroid_norm": float(np.linalg.norm(pos_centroid)),
        "negative_centroid_norm": float(np.linalg.norm(neg_centroid)),
        "target_neighborhood_radius_cosine_distance": target_radius,
        "positive_within_class_dispersion": float(np.mean(1.0 - (pos_train @ pos_centroid))),
        "negative_within_class_dispersion": float(np.mean(1.0 - (neg_train @ neg_centroid))),
        "centroid_separation_cosine_distance": float(1.0 - np.dot(pos_centroid, neg_centroid)),
    }
    return summary, pos_centroid, neg_centroid, clf, target_nn, target_radius


def score_embeddings(
    embeddings: np.ndarray,
    source_embeddings: np.ndarray,
    pos_centroid: np.ndarray,
    neg_centroid: np.ndarray,
    clf: LogisticRegression,
    target_nn: NearestNeighbors,
    target_radius: float,
) -> pd.DataFrame:
    source_shift_cosine = np.sum(source_embeddings * embeddings, axis=1)
    target_centroid_similarity = embeddings @ pos_centroid
    negative_centroid_similarity = embeddings @ neg_centroid
    classifier_score = clf.predict_proba(embeddings)[:, 1]
    nn_distances, _ = target_nn.kneighbors(embeddings)
    nearest_target_distance = nn_distances[:, 0]
    return pd.DataFrame(
        {
            "source_shift_cosine": source_shift_cosine,
            "target_centroid_similarity": target_centroid_similarity,
            "negative_centroid_similarity": negative_centroid_similarity,
            "target_margin": target_centroid_similarity - negative_centroid_similarity,
            "classifier_score": classifier_score,
            "classifier_predicted_target": (classifier_score >= 0.5).astype(int),
            "nearest_target_distance": nearest_target_distance,
            "nearest_target_cosine": 1.0 - nearest_target_distance,
            "in_target_neighborhood": (nearest_target_distance <= target_radius).astype(int),
        }
    )


def summarize_vs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        df.loc[df["alpha"] == 0]
        .set_index("dataset_row_id")[
            [
                "target_centroid_similarity",
                "target_margin",
                "classifier_score",
                "nearest_target_distance",
                "in_target_neighborhood",
            ]
        ]
        .rename(
            columns={
                "target_centroid_similarity": "baseline_target_centroid_similarity",
                "target_margin": "baseline_target_margin",
                "classifier_score": "baseline_classifier_score",
                "nearest_target_distance": "baseline_nearest_target_distance",
                "in_target_neighborhood": "baseline_in_target_neighborhood",
            }
        )
    )
    merged = df.merge(baseline, left_on="dataset_row_id", right_index=True, how="left")
    merged["delta_target_centroid_similarity"] = (
        merged["target_centroid_similarity"] - merged["baseline_target_centroid_similarity"]
    )
    merged["delta_target_margin"] = merged["target_margin"] - merged["baseline_target_margin"]
    merged["delta_classifier_score"] = merged["classifier_score"] - merged["baseline_classifier_score"]
    merged["delta_nearest_target_distance"] = (
        merged["nearest_target_distance"] - merged["baseline_nearest_target_distance"]
    )
    return merged


def alpha_sensitivity_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction_label, direction_df in df.groupby("direction_label"):
        alpha_sorted = sorted(direction_df["alpha"].unique().tolist())
        per_row = []
        for _, row_df in direction_df.groupby("dataset_row_id"):
            row_df = row_df.sort_values("alpha")
            score_values = row_df["classifier_score"].to_numpy()
            dist_values = row_df["nearest_target_distance"].to_numpy()
            per_row.append(
                {
                    "monotonic_score_increase": bool(np.all(np.diff(score_values) >= -1e-8)),
                    "monotonic_distance_decrease": bool(np.all(np.diff(dist_values) <= 1e-8)),
                }
            )
        monotonic_df = pd.DataFrame(per_row)

        best_alpha_row = (
            direction_df.groupby("alpha", as_index=False)["classifier_score"].mean().sort_values("classifier_score", ascending=False).iloc[0]
        )
        rows.append(
            {
                "direction_label": direction_label,
                "alphas_evaluated": ",".join(str(alpha) for alpha in alpha_sorted),
                "fraction_monotonic_score_increase": float(monotonic_df["monotonic_score_increase"].mean()),
                "fraction_monotonic_distance_decrease": float(monotonic_df["monotonic_distance_decrease"].mean()),
                "best_alpha_by_mean_classifier_score": float(best_alpha_row["alpha"]),
                "best_mean_classifier_score": float(best_alpha_row["classifier_score"]),
            }
        )
    return pd.DataFrame(rows)


def choose_best_entry_regime(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(
            columns=[
                "selection_rank",
                "direction_label",
                "alpha",
                "target_neighborhood_entry_rate",
                "fraction_distance_improved",
                "mean_nearest_target_distance",
                "mean_source_shift_cosine",
                "mean_classifier_score",
            ]
        )
    ranked = summary_df.sort_values(
        [
            "target_neighborhood_entry_rate",
            "fraction_distance_improved",
            "mean_nearest_target_distance",
            "mean_source_shift_cosine",
            "mean_classifier_score",
        ],
        ascending=[False, False, True, False, False],
    ).reset_index(drop=True)
    out = ranked[
        [
            "direction_label",
            "alpha",
            "target_neighborhood_entry_rate",
            "fraction_distance_improved",
            "mean_nearest_target_distance",
            "mean_source_shift_cosine",
            "mean_classifier_score",
        ]
    ].copy()
    out.insert(0, "selection_rank", np.arange(1, len(out) + 1, dtype=int))
    return out


def run_predecode_audit(
    source_df: pd.DataFrame,
    source_embeddings: np.ndarray,
    directions: dict[str, np.ndarray],
    alphas: list[float],
    normalize_after_steering: bool,
    pos_centroid: np.ndarray,
    neg_centroid: np.ndarray,
    clf: LogisticRegression,
    target_nn: NearestNeighbors,
    target_radius: float,
) -> pd.DataFrame:
    rows = []
    for direction_label, direction_vector in directions.items():
        for alpha in alphas:
            shifted = source_embeddings + (alpha * direction_vector.reshape(1, -1))
            if normalize_after_steering:
                shifted = normalize_rows(shifted).astype(np.float32)
            metrics_df = score_embeddings(
                embeddings=shifted,
                source_embeddings=source_embeddings,
                pos_centroid=pos_centroid,
                neg_centroid=neg_centroid,
                clf=clf,
                target_nn=target_nn,
                target_radius=target_radius,
            )
            block = pd.concat([source_df.reset_index(drop=True), metrics_df], axis=1)
            block["direction_label"] = direction_label
            block["alpha"] = float(alpha)
            rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    return summarize_vs_baseline(out)


def summarize_predecode(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (direction_label, alpha), group_df in df.groupby(["direction_label", "alpha"]):
        rows.append(
            {
                "direction_label": direction_label,
                "alpha": float(alpha),
                "n_source_rows": int(len(group_df)),
                "mean_source_shift_cosine": float(group_df["source_shift_cosine"].mean()),
                "mean_target_centroid_similarity": float(group_df["target_centroid_similarity"].mean()),
                "mean_delta_target_centroid_similarity": float(group_df["delta_target_centroid_similarity"].mean()),
                "mean_target_margin": float(group_df["target_margin"].mean()),
                "mean_delta_target_margin": float(group_df["delta_target_margin"].mean()),
                "mean_classifier_score": float(group_df["classifier_score"].mean()),
                "mean_delta_classifier_score": float(group_df["delta_classifier_score"].mean()),
                "predicted_target_rate": float(group_df["classifier_predicted_target"].mean()),
                "mean_nearest_target_distance": float(group_df["nearest_target_distance"].mean()),
                "mean_delta_nearest_target_distance": float(group_df["delta_nearest_target_distance"].mean()),
                "fraction_distance_improved": float((group_df["delta_nearest_target_distance"] < 0).mean()),
                "target_neighborhood_entry_rate": float(group_df["in_target_neighborhood"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["direction_label", "alpha"]).reset_index(drop=True)


def infer_run_label(path: Path, fallback_idx: int) -> str:
    return path.parent.name or f"generated_run_{fallback_idx}"


def run_note_level_audit(
    generated_manifest_paths: list[Path],
    generated_embeddings_paths: list[Path],
    generated_run_labels: list[str],
    selected_source_ids: set[int],
    pos_centroid: np.ndarray,
    neg_centroid: np.ndarray,
    clf: LogisticRegression,
    target_nn: NearestNeighbors,
    target_radius: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_frames = []
    summary_rows = []
    for idx, (manifest_path, embeddings_path) in enumerate(zip(generated_manifest_paths, generated_embeddings_paths)):
        run_label = generated_run_labels[idx] if idx < len(generated_run_labels) else infer_run_label(manifest_path, idx)
        manifest_df = pd.read_json(manifest_path, lines=True)
        embeddings = np.load(embeddings_path)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        embeddings = normalize_rows(np.asarray(embeddings, dtype=np.float32))
        if len(manifest_df) != embeddings.shape[0]:
            raise ValueError(f"Manifest/embedding mismatch for {manifest_path}: {len(manifest_df)} vs {embeddings.shape[0]}")

        manifest_df["dataset_row_id"] = pd.to_numeric(manifest_df["dataset_row_id"], errors="raise").astype(int)
        manifest_df = manifest_df.loc[manifest_df["dataset_row_id"].isin(selected_source_ids)].reset_index(drop=True)
        if manifest_df.empty:
            continue

        embeddings = embeddings[manifest_df.index.to_numpy(dtype=int)]
        dummy_source = embeddings.copy()
        metrics_df = score_embeddings(
            embeddings=embeddings,
            source_embeddings=dummy_source,
            pos_centroid=pos_centroid,
            neg_centroid=neg_centroid,
            clf=clf,
            target_nn=target_nn,
            target_radius=target_radius,
        )
        block = pd.concat([manifest_df.reset_index(drop=True), metrics_df], axis=1)
        block["run_label"] = run_label
        row_frames.append(block)

        alpha_values = sorted(pd.to_numeric(block.get("alpha"), errors="coerce").dropna().unique().tolist()) if "alpha" in block.columns else []
        summary_rows.append(
            {
                "run_label": run_label,
                "generation_condition": str(block.get("generation_condition").dropna().iloc[0]) if "generation_condition" in block.columns and block["generation_condition"].notna().any() else run_label,
                "axis_label": str(block.get("axis_label").dropna().iloc[0]) if "axis_label" in block.columns and block["axis_label"].notna().any() else None,
                "n_rows": int(len(block)),
                "unique_source_rows": int(block["dataset_row_id"].nunique()),
                "alphas_present": ",".join(str(alpha) for alpha in alpha_values),
                "mean_target_centroid_similarity": float(block["target_centroid_similarity"].mean()),
                "mean_target_margin": float(block["target_margin"].mean()),
                "mean_classifier_score": float(block["classifier_score"].mean()),
                "predicted_target_rate": float(block["classifier_predicted_target"].mean()),
                "mean_nearest_target_distance": float(block["nearest_target_distance"].mean()),
                "target_neighborhood_entry_rate": float(block["in_target_neighborhood"].mean()),
            }
        )

    if not row_frames:
        return pd.DataFrame(), pd.DataFrame()
    row_df = pd.concat(row_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("run_label").reset_index(drop=True)
    return row_df, summary_df


def markdown_report(
    separability: dict[str, Any],
    predecode_summary: pd.DataFrame,
    alpha_summary: pd.DataFrame,
    best_regime_df: pd.DataFrame,
    note_summary: pd.DataFrame,
    output_dir: Path,
) -> str:
    lines = []
    lines.append("# Steering Geometry Audit")
    lines.append("")
    lines.append("## Target Separability")
    lines.append(f"- target_column: `{separability['target_column']}`")
    lines.append(f"- fit positives / negatives: {separability['n_fit_positive']} / {separability['n_fit_negative']}")
    lines.append(f"- eval positives / negatives: {separability['n_eval_positive']} / {separability['n_eval_negative']}")
    lines.append(f"- eval AUROC: {separability['eval_auroc']:.4f}")
    lines.append(f"- eval average precision: {separability['eval_average_precision']:.4f}")
    lines.append(f"- centroid separation cosine distance: {separability['centroid_separation_cosine_distance']:.4f}")
    lines.append("")
    lines.append("## Pre-decode Summary")
    best_rows = predecode_summary.sort_values("mean_classifier_score", ascending=False).head(6)
    for _, row in best_rows.iterrows():
        lines.append(
            f"- `{row['direction_label']}` alpha={row['alpha']}: "
            f"classifier_score={row['mean_classifier_score']:.4f}, "
            f"delta_score={row['mean_delta_classifier_score']:.4f}, "
            f"target_rate={row['predicted_target_rate']:.4f}, "
            f"distance={row['mean_nearest_target_distance']:.4f}"
        )
    lines.append("")
    lines.append("## Alpha Sensitivity")
    for _, row in alpha_summary.iterrows():
        lines.append(
            f"- `{row['direction_label']}`: "
            f"monotonic score increase={row['fraction_monotonic_score_increase']:.3f}, "
            f"monotonic distance decrease={row['fraction_monotonic_distance_decrease']:.3f}, "
            f"best alpha={row['best_alpha_by_mean_classifier_score']}"
        )
    if not best_regime_df.empty:
        lines.append("")
        lines.append("## Best Pre-decode Entry Regimes")
        for _, row in best_regime_df.head(5).iterrows():
            lines.append(
                f"- rank {int(row['selection_rank'])}: `{row['direction_label']}` alpha={row['alpha']}: "
                f"entry_rate={row['target_neighborhood_entry_rate']:.4f}, "
                f"distance_improved={row['fraction_distance_improved']:.4f}, "
                f"mean_distance={row['mean_nearest_target_distance']:.4f}, "
                f"source_cosine={row['mean_source_shift_cosine']:.4f}"
            )
    if not note_summary.empty:
        lines.append("")
        lines.append("## Note-level Summary")
        for _, row in note_summary.iterrows():
            lines.append(
                f"- `{row['run_label']}`: "
                f"classifier_score={row['mean_classifier_score']:.4f}, "
                f"target_rate={row['predicted_target_rate']:.4f}, "
                f"distance={row['mean_nearest_target_distance']:.4f}"
            )
    lines.append("")
    lines.append("## Outputs")
    lines.append("- `steering_separability_summary.json`")
    lines.append("- `predecode_row_level_metrics.csv`")
    lines.append("- `predecode_direction_alpha_summary.csv`")
    lines.append("- `alpha_sensitivity_summary.csv`")
    if not note_summary.empty:
        lines.append("- `note_level_row_metrics.csv`")
        lines.append("- `note_level_summary.csv`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()

    fit_dataset_path = Path(args.fit_dataset_path).resolve()
    eval_dataset_path = Path(args.eval_dataset_path).resolve()
    factors_path = Path(args.factors_path).resolve()
    direction_bank_path = Path(args.direction_bank_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    direction_labels = parse_csv_list(args.direction_labels)
    alphas = parse_float_list(args.alphas)
    preferred_join_cols = parse_csv_list(args.join_cols)
    if not direction_labels:
        raise ValueError("--direction_labels must specify at least one direction")
    if not alphas:
        raise ValueError("--alphas must specify at least one value")

    _, fit_base_df, fit_embeddings = load_dataset_rows(fit_dataset_path)
    _, eval_base_df, eval_embeddings = load_dataset_rows(eval_dataset_path)

    if args.fit_split and "split" not in fit_base_df.columns:
        fit_base_df["split"] = args.fit_split
    if args.eval_split and "split" not in eval_base_df.columns:
        eval_base_df["split"] = args.eval_split

    fit_df, fit_join_cols, fit_join_report = merge_metadata(
        fit_base_df,
        factors_path=factors_path,
        split_manifest_path=args.split_manifest_path,
        preferred_join_cols=preferred_join_cols,
        source_split=args.fit_split,
    )
    eval_df, eval_join_cols, eval_join_report = merge_metadata(
        eval_base_df,
        factors_path=factors_path,
        split_manifest_path=args.split_manifest_path,
        preferred_join_cols=preferred_join_cols,
        source_split=args.eval_split,
    )

    fit_df = prepare_binary_target(fit_df, args.target_column).sort_values("dataset_row_id").reset_index(drop=True)
    eval_df = prepare_binary_target(eval_df, args.target_column).sort_values("dataset_row_id").reset_index(drop=True)
    fit_embeddings = fit_embeddings[fit_df["dataset_row_id"].to_numpy(dtype=int)]
    eval_embeddings = eval_embeddings[eval_df["dataset_row_id"].to_numpy(dtype=int)]

    if args.source_selection_query:
        source_df = eval_df.query(args.source_selection_query, engine="python").copy()
    else:
        source_df = eval_df.loc[eval_df[args.target_column] == 1].copy()
    if args.max_source_rows is not None:
        source_df = source_df.head(args.max_source_rows).copy()
    if source_df.empty:
        raise ValueError("No eval source rows remained after selection.")

    source_df = source_df.sort_values("dataset_row_id").reset_index(drop=True)
    source_embeddings = eval_embeddings[source_df["dataset_row_id"].to_numpy(dtype=int)]

    directions = load_directions(direction_bank_path, direction_labels)
    separability, pos_centroid, neg_centroid, clf, target_nn, target_radius = fit_target_geometry(
        train_embeddings=fit_embeddings,
        train_df=fit_df,
        eval_embeddings=eval_embeddings,
        eval_df=eval_df,
        target_column=args.target_column,
        neighborhood_k=args.neighborhood_k,
    )

    predecode_row_df = run_predecode_audit(
        source_df=source_df,
        source_embeddings=source_embeddings,
        directions=directions,
        alphas=alphas,
        normalize_after_steering=bool(args.normalize_after_steering),
        pos_centroid=pos_centroid,
        neg_centroid=neg_centroid,
        clf=clf,
        target_nn=target_nn,
        target_radius=target_radius,
    )
    predecode_summary_df = summarize_predecode(predecode_row_df)
    alpha_summary_df = alpha_sensitivity_summary(predecode_row_df)
    best_regime_df = choose_best_entry_regime(predecode_summary_df)

    note_row_df = pd.DataFrame()
    note_summary_df = pd.DataFrame()
    manifest_paths = [Path(p).resolve() for p in parse_csv_list(args.generated_manifest_paths)]
    embedding_paths = [Path(p).resolve() for p in parse_csv_list(args.generated_embeddings_paths)]
    run_labels = parse_csv_list(args.generated_run_labels)
    if manifest_paths or embedding_paths:
        if len(manifest_paths) != len(embedding_paths):
            raise ValueError("generated_manifest_paths and generated_embeddings_paths must have the same length.")
        note_row_df, note_summary_df = run_note_level_audit(
            generated_manifest_paths=manifest_paths,
            generated_embeddings_paths=embedding_paths,
            generated_run_labels=run_labels,
            selected_source_ids=set(source_df["dataset_row_id"].astype(int).tolist()),
            pos_centroid=pos_centroid,
            neg_centroid=neg_centroid,
            clf=clf,
            target_nn=target_nn,
            target_radius=target_radius,
        )

    predecode_row_df.to_csv(output_dir / "predecode_row_level_metrics.csv", index=False)
    predecode_summary_df.to_csv(output_dir / "predecode_direction_alpha_summary.csv", index=False)
    alpha_summary_df.to_csv(output_dir / "alpha_sensitivity_summary.csv", index=False)
    best_regime_df.to_csv(output_dir / "best_predecode_entry_regimes.csv", index=False)
    if not note_row_df.empty:
        note_row_df.to_csv(output_dir / "note_level_row_metrics.csv", index=False)
        note_summary_df.to_csv(output_dir / "note_level_summary.csv", index=False)

    payload = {
        "created_at": now_iso(),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "script_path": str(Path(__file__).resolve()),
        "fit_dataset_path": str(fit_dataset_path),
        "eval_dataset_path": str(eval_dataset_path),
        "factors_path": str(factors_path),
        "direction_bank_path": str(direction_bank_path),
        "direction_labels": direction_labels,
        "alphas": alphas,
        "target_column": args.target_column,
        "source_selection_query": args.source_selection_query or f"{args.target_column} == 1",
        "normalize_after_steering": bool(args.normalize_after_steering),
        "neighborhood_k": int(args.neighborhood_k),
        "n_selected_source_rows": int(len(source_df)),
        "fit_join_cols": fit_join_cols,
        "eval_join_cols": eval_join_cols,
        "fit_join_report": fit_join_report,
        "eval_join_report": eval_join_report,
        "separability_summary": separability,
        "predecode_best_by_classifier_score": predecode_summary_df.sort_values(
            "mean_classifier_score", ascending=False
        ).head(10).to_dict(orient="records"),
        "best_predecode_entry_regimes": best_regime_df.head(10).to_dict(orient="records"),
        "alpha_sensitivity_summary": alpha_summary_df.to_dict(orient="records"),
        "note_level_summary": note_summary_df.to_dict(orient="records") if not note_summary_df.empty else [],
        "cli_args": vars(args),
    }
    save_json(output_dir / "steering_separability_summary.json", payload)
    (output_dir / "steering_separability_summary.md").write_text(
        markdown_report(separability, predecode_summary_df, alpha_summary_df, best_regime_df, note_summary_df, output_dir),
        encoding="utf-8",
    )

    print(f"Saved steering diagnostics to: {output_dir}")
    print(f"Selected eval source rows: {len(source_df)}")
    print(f"Directions audited: {', '.join(direction_labels)}")
    print(f"Alphas audited: {', '.join(str(alpha) for alpha in alphas)}")


if __name__ == "__main__":
    main()
