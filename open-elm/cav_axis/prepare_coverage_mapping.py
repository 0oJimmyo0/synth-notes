#!/usr/bin/env python3
"""
Phase 1 coverage-mapping preparation.

Mode 1: real_only_precompute
  - runs now on held-out test embeddings only

Mode 1b: real_all_filtered_precompute
  - runs on the full filtered real cohort across train/dev/test

Mode 2: real_vs_synthetic
  - guarded until vanilla generation, audit, leakage flags, and synthetic embeddings exist
  - computes coverage in the full normalized 1024-d BGE embedding space

Quantitative claims in this script are based on the full high-dimensional space.
PCA / UMAP outputs are visualization only.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


DEFAULT_DATASET_PATH = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/encoded_testing_filtered"
)
DEFAULT_TRAIN_DATASET_PATH = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/encoded_training_filtered"
)
DEFAULT_DEV_DATASET_PATH = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/encoded_dev_filtered"
)
DEFAULT_SPLIT_MANIFEST_PATH = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/leakage_audit/split_manifest_note_level.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/coverage/real_only_precompute"
)
DEFAULT_WHOLE_COHORT_MANIFEST = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/leakage_audit/split_manifest_note_level_full.csv"
)
DEFAULT_EMBEDDING_METADATA_PATH = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "embeddings-BAAI-bge-large-en-v1.5/sentence_embeddings_metadata.csv"
)
DEFAULT_SYNTHETIC_MANIFEST_PATH = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/synthetic_notes/"
    "synthetic_notes_test_vanilla_seed42_manifest.jsonl"
)
DEFAULT_SYNTHETIC_EMBEDDINGS_PATH = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/generation_audit/vanilla_test_seed42/"
    "generated_note_embeddings_bge_large.npy"
)
DEFAULT_AUDIT_JSON_PATH = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/generation_audit/vanilla_test_seed42/"
    "generation_audit_baseline.json"
)

LEAKAGE_COLS = [
    "patient_overlap_with_train",
    "hadm_overlap_with_train",
    "note_overlap_with_train",
    "patient_disjoint_from_train",
    "hadm_disjoint_from_train",
    "note_disjoint_from_train",
]
SUBGROUP_CANDIDATE_MAP = {
    "age_bin": ["age_bin", "age_group"],
    "sex_gender": ["sex_gender", "gender", "sex"],
    "race_ethnicity": ["race_ethnicity", "race", "ethnicity"],
    "insurance": ["insurance"],
    "admission_type": ["admission_type"],
    "service": ["service", "curr_service"],
    "los_bin": ["los_bin", "length_of_stay_bin"],
    "icu_flag": ["icu_flag", "is_icu", "icu_stay_flag"],
}
GROUP_ORDER = ["full_test", "patient_disjoint", "patient_overlap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase 1 coverage-mapping artifacts.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["real_only_precompute", "real_all_filtered_precompute", "real_vs_synthetic"],
        help="Coverage-prep mode",
    )
    parser.add_argument("--real_dataset_path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--train_dataset_path", default=DEFAULT_TRAIN_DATASET_PATH)
    parser.add_argument("--dev_dataset_path", default=DEFAULT_DEV_DATASET_PATH)
    parser.add_argument("--split_manifest_path", default=DEFAULT_SPLIT_MANIFEST_PATH)
    parser.add_argument("--whole_cohort_manifest_path", default=DEFAULT_WHOLE_COHORT_MANIFEST)
    parser.add_argument("--embedding_metadata_path", default=DEFAULT_EMBEDDING_METADATA_PATH)
    parser.add_argument("--extra_metadata_path", default=None)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--prdc_k", type=int, default=5)
    parser.add_argument("--low_density_quantile", type=float, default=0.2)
    parser.add_argument("--distribution_sample_size", type=int, default=2000)
    parser.add_argument("--max_points_for_plot", type=int, default=3000)
    parser.add_argument(
        "--synthetic_manifest_path",
        default=DEFAULT_SYNTHETIC_MANIFEST_PATH,
    )
    parser.add_argument(
        "--synthetic_embeddings_path",
        default=DEFAULT_SYNTHETIC_EMBEDDINGS_PATH,
    )
    parser.add_argument(
        "--audit_json_path",
        default=DEFAULT_AUDIT_JSON_PATH,
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def load_real_embeddings(dataset_path: Path) -> np.ndarray:
    dataset = Dataset.load_from_disk(str(dataset_path))
    rows = []
    for example in dataset:
        emb = example["domain_embeddings"][0]
        rows.append(np.asarray(emb, dtype=np.float32))
    return normalize_matrix(np.vstack(rows))


def load_split_named_embeddings(dataset_path: Path, split_name: str) -> pd.DataFrame:
    dataset = Dataset.load_from_disk(str(dataset_path))
    rows = []
    for idx, example in enumerate(dataset):
        emb = np.asarray(example["domain_embeddings"][0], dtype=np.float32)
        rows.append({"split": split_name, "dataset_row_id": idx, "embedding": emb})
    df = pd.DataFrame(rows)
    df["embedding"] = df["embedding"].map(lambda arr: arr / max(np.linalg.norm(arr), 1e-12))
    return df


def load_split_manifest(split_manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(split_manifest_path)
    df["dataset_row_id"] = pd.to_numeric(df["dataset_row_id"], errors="raise").astype(int)
    return df


def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def load_synthetic_embeddings(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return normalize_matrix(np.asarray(arr, dtype=np.float32))


def boolish(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def leakage_group(series: pd.Series) -> pd.Series:
    return series.map(
        lambda x: "patient_disjoint"
        if boolish(x) is True
        else "patient_overlap"
        if boolish(x) is False
        else "unknown"
    )


def infer_density(cluster_size: int, mean_distance_to_centroid: float) -> float:
    return float(cluster_size / max(mean_distance_to_centroid, 1e-8))


def fit_real_clusters(embeddings: np.ndarray, n_clusters: int, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_seed, batch_size=2048, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    centers = normalize_matrix(kmeans.cluster_centers_.astype(np.float32))
    return labels, centers


def assign_to_centers(embeddings: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    similarities = embeddings @ centers.T
    labels = similarities.argmax(axis=1)
    distances = 1.0 - similarities[np.arange(len(embeddings)), labels]
    return labels.astype(int), distances.astype(np.float32)


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def detect_join_keys(df: pd.DataFrame) -> list[str]:
    preferred_sets = [
        ["source_row_id"],
        ["embedding_row_id"],
        ["dataset_row_id"],
        ["note_id", "subject_id", "hadm_id"],
        ["note_id"],
        ["subject_id", "hadm_id"],
        ["hadm_id"],
        ["subject_id"],
    ]
    available = set(df.columns)
    for keys in preferred_sets:
        if all(key in available for key in keys):
            return keys
    return []


def load_extra_metadata(extra_metadata_path: str | None) -> pd.DataFrame | None:
    if not extra_metadata_path:
        return None
    path = Path(extra_metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"extra_metadata_path does not exist: {path}")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    raise ValueError(f"Unsupported extra metadata file type: {path.suffix}")


def enrich_with_extra_metadata(df: pd.DataFrame, extra_df: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if extra_df is None:
        return df, warnings

    left_keys = detect_join_keys(df)
    right_keys = [key for key in left_keys if key in extra_df.columns]
    if not left_keys or not right_keys or left_keys != right_keys:
        warnings.append("Extra metadata path provided, but no stable join key overlap was found; skipping extra metadata join.")
        return df, warnings

    keep_cols = []
    for canonical, aliases in SUBGROUP_CANDIDATE_MAP.items():
        for alias in aliases:
            if alias in extra_df.columns:
                keep_cols.append(alias)
                break
    if not keep_cols:
        warnings.append("Extra metadata file was loaded, but it does not contain subgroup candidate columns.")
        return df, warnings

    join_df = extra_df[right_keys + keep_cols].drop_duplicates(subset=right_keys)
    merged = df.merge(join_df, how="left", on=right_keys, suffixes=("", "_extra"))
    return merged, warnings


def canonicalize_subgroup_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    out = df.copy()
    for canonical, aliases in SUBGROUP_CANDIDATE_MAP.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            if alias in out.columns:
                out[canonical] = out[alias]
                break
        else:
            warnings.append(f"Missing subgroup metadata field '{canonical}' in current manifest/joined metadata.")
    return out, warnings


def build_test_manifest(
    split_manifest: pd.DataFrame,
    n_real_rows: int,
    extra_metadata: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    test_manifest = (
        split_manifest.loc[split_manifest["split"] == "test"]
        .sort_values("dataset_row_id")
        .reset_index(drop=True)
    )
    if len(test_manifest) != n_real_rows:
        raise ValueError(
            f"Filtered split manifest test rows ({len(test_manifest)}) do not match encoded_testing_filtered rows ({n_real_rows})."
        )
    if not np.array_equal(test_manifest["dataset_row_id"].to_numpy(), np.arange(n_real_rows)):
        raise ValueError("Filtered split manifest dataset_row_id is not aligned 0..N-1 with encoded_testing_filtered.")

    test_manifest, join_warnings = enrich_with_extra_metadata(test_manifest, extra_metadata)
    warnings.extend(join_warnings)
    test_manifest, subgroup_warnings = canonicalize_subgroup_columns(test_manifest)
    warnings.extend(subgroup_warnings)
    test_manifest["leakage_group"] = leakage_group(test_manifest["patient_disjoint_from_train"])
    return test_manifest, warnings


def build_all_filtered_manifest(
    split_manifest: pd.DataFrame,
    train_dataset_path: Path,
    dev_dataset_path: Path,
    test_dataset_path: Path,
    extra_metadata: pd.DataFrame | None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    warnings: list[str] = []
    split_frames = []
    embedding_frames = []

    for split_name, dataset_path in [
        ("train", train_dataset_path),
        ("dev", dev_dataset_path),
        ("test", test_dataset_path),
    ]:
        split_df = (
            split_manifest.loc[split_manifest["split"] == split_name]
            .sort_values("dataset_row_id")
            .reset_index(drop=True)
        )
        emb_df = load_split_named_embeddings(dataset_path, split_name)

        if len(split_df) != len(emb_df):
            raise ValueError(
                f"Filtered split manifest rows for {split_name} ({len(split_df)}) do not match {dataset_path.name} rows ({len(emb_df)})."
            )
        if not np.array_equal(split_df["dataset_row_id"].to_numpy(), np.arange(len(split_df))):
            raise ValueError(f"Filtered split manifest dataset_row_id for {split_name} is not aligned 0..N-1.")

        split_frames.append(split_df)
        embedding_frames.append(emb_df)

    manifest_df = pd.concat(split_frames, axis=0, ignore_index=True)
    embedding_df = pd.concat(embedding_frames, axis=0, ignore_index=True)

    if not np.array_equal(manifest_df["split"].astype(str).to_numpy(), embedding_df["split"].astype(str).to_numpy()):
        raise ValueError("Combined split order mismatch between manifest and embedding stack.")
    if not np.array_equal(manifest_df["dataset_row_id"].to_numpy(), embedding_df["dataset_row_id"].to_numpy()):
        raise ValueError("Combined dataset_row_id mismatch between manifest and embedding stack.")

    manifest_df, join_warnings = enrich_with_extra_metadata(manifest_df, extra_metadata)
    warnings.extend(join_warnings)
    manifest_df, subgroup_warnings = canonicalize_subgroup_columns(manifest_df)
    warnings.extend(subgroup_warnings)
    manifest_df["leakage_group"] = leakage_group(manifest_df["patient_disjoint_from_train"])

    embeddings = normalize_matrix(np.vstack(embedding_df["embedding"].to_list()))
    return manifest_df.reset_index(drop=True), embeddings, warnings


def compute_real_cluster_outputs(
    embeddings: np.ndarray,
    test_manifest: pd.DataFrame,
    n_clusters: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    labels, centers = fit_real_clusters(embeddings, n_clusters=n_clusters, random_seed=random_seed)
    centroid_dist = 1.0 - np.sum(embeddings * centers[labels], axis=1)

    assign_df = test_manifest.copy().reset_index(drop=True)
    assign_df["cluster_id"] = labels
    assign_df["distance_to_centroid"] = centroid_dist

    rows: list[dict[str, Any]] = []
    for cluster_id, cluster_df in assign_df.groupby("cluster_id", sort=True):
        group_counts = cluster_df["leakage_group"].value_counts().to_dict()
        size = int(len(cluster_df))
        mean_dist = float(cluster_df["distance_to_centroid"].mean())
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": size,
                "cluster_fraction": float(size / len(assign_df)),
                "mean_distance_to_centroid": mean_dist,
                "median_distance_to_centroid": float(cluster_df["distance_to_centroid"].median()),
                "density_proxy": infer_density(size, mean_dist),
                "patient_disjoint_count": int(group_counts.get("patient_disjoint", 0)),
                "patient_overlap_count": int(group_counts.get("patient_overlap", 0)),
                "unknown_group_count": int(group_counts.get("unknown", 0)),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    return assign_df, summary_df, centers


def compute_group_cluster_summary(assign_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, group_df in group_iter(assign_df):
        if group_df.empty:
            rows.append({"group_name": group_name, "n_rows": 0, "n_clusters_present": 0})
            continue
        counts = group_df["cluster_id"].value_counts()
        rows.append(
            {
                "group_name": group_name,
                "n_rows": int(len(group_df)),
                "n_clusters_present": int(counts.shape[0]),
                "largest_cluster_fraction": float(counts.max() / len(group_df)),
                "smallest_nonzero_cluster_fraction": float(counts.min() / len(group_df)),
                "mean_distance_to_centroid": float(group_df["distance_to_centroid"].mean()),
                "median_distance_to_centroid": float(group_df["distance_to_centroid"].median()),
            }
        )
    return pd.DataFrame(rows)


def compute_split_cluster_summary(assign_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, split_df in assign_df.groupby("split", sort=False):
        counts = split_df["cluster_id"].value_counts()
        rows.append(
            {
                "split": split_name,
                "n_rows": int(len(split_df)),
                "n_clusters_present": int(counts.shape[0]),
                "largest_cluster_fraction": float(counts.max() / len(split_df)),
                "smallest_nonzero_cluster_fraction": float(counts.min() / len(split_df)),
                "mean_distance_to_centroid": float(split_df["distance_to_centroid"].mean()),
                "median_distance_to_centroid": float(split_df["distance_to_centroid"].median()),
                "patient_disjoint_rate": float((split_df["leakage_group"] == "patient_disjoint").mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_subgroup_cluster_summary(assign_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    available = [name for name in SUBGROUP_CANDIDATE_MAP if name in assign_df.columns and assign_df[name].notna().any()]
    if not available:
        warnings.append(
            "No subgroup metadata fields are present in the filtered split manifest or joined metadata; subgroup summaries remain empty."
        )
        return pd.DataFrame(columns=["subgroup_name", "subgroup_value", "cluster_id", "count", "fraction_within_subgroup"]), warnings

    rows: list[dict[str, Any]] = []
    for subgroup_name in available:
        for subgroup_value, subgroup_df in assign_df.groupby(subgroup_name, dropna=False):
            counts = subgroup_df["cluster_id"].value_counts()
            denom = len(subgroup_df)
            for cluster_id, count in counts.items():
                rows.append(
                    {
                        "subgroup_name": subgroup_name,
                        "subgroup_value": subgroup_value,
                        "cluster_id": int(cluster_id),
                        "count": int(count),
                        "fraction_within_subgroup": float(count / denom),
                    }
                )
    return pd.DataFrame(rows), warnings


def group_iter(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("full_test", df),
        ("patient_disjoint", df.loc[df["leakage_group"] == "patient_disjoint"]),
        ("patient_overlap", df.loc[df["leakage_group"] == "patient_overlap"]),
    ]


def nearest_neighbor_distances(query: np.ndarray, reference: np.ndarray, n_neighbors: int = 1) -> tuple[np.ndarray, np.ndarray]:
    model = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors, algorithm="brute")
    model.fit(reference)
    distances, indices = model.kneighbors(query, return_distance=True)
    return distances, indices


def kth_real_neighbor_radius(real_embeddings: np.ndarray, k: int) -> np.ndarray:
    effective_neighbors = min(len(real_embeddings), k + 1)
    model = NearestNeighbors(metric="cosine", n_neighbors=effective_neighbors, algorithm="brute")
    model.fit(real_embeddings)
    distances, _ = model.kneighbors(real_embeddings, return_distance=True)
    if effective_neighbors <= 1:
        return np.zeros(len(real_embeddings), dtype=np.float32)
    return distances[:, -1].astype(np.float32)


def compute_distribution_metrics(
    real_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    prdc_k: int,
) -> dict[str, float | int | None]:
    if len(real_embeddings) == 0 or len(synthetic_embeddings) == 0:
        return {
            "n_real": int(len(real_embeddings)),
            "n_synthetic": int(len(synthetic_embeddings)),
            "real_to_synth_coverage": None,
            "synthetic_to_real_precision": None,
            "synthetic_density": None,
            "real_to_synth_mean_nn_distance": None,
            "synthetic_to_real_mean_nn_distance": None,
            "real_neighbor_radius_median": None,
            "synthetic_neighbor_radius_median": None,
        }

    real_radii = kth_real_neighbor_radius(real_embeddings, prdc_k)
    syn_radii = kth_real_neighbor_radius(synthetic_embeddings, prdc_k) if len(synthetic_embeddings) > 1 else np.zeros(len(synthetic_embeddings), dtype=np.float32)

    real_to_syn_dist, _ = nearest_neighbor_distances(real_embeddings, synthetic_embeddings, n_neighbors=1)
    syn_to_real_dist, syn_to_real_idx = nearest_neighbor_distances(synthetic_embeddings, real_embeddings, n_neighbors=1)

    real_to_syn = real_to_syn_dist[:, 0]
    syn_to_real = syn_to_real_dist[:, 0]
    nearest_real_idx = syn_to_real_idx[:, 0]

    real_coverage = float(np.mean(real_to_syn <= real_radii))
    synthetic_precision = float(np.mean(syn_to_real <= real_radii[nearest_real_idx]))

    density_neighbors = nearest_neighbor_distances(synthetic_embeddings, real_embeddings, n_neighbors=min(len(real_embeddings), prdc_k))[0]
    real_radii_for_neighbors = real_radii[np.argsort(np.zeros_like(real_radii))]  # placeholder to keep dtype stable
    del real_radii_for_neighbors
    counts = []
    if len(real_embeddings) > 0:
        nn_model = NearestNeighbors(metric="cosine", n_neighbors=min(len(real_embeddings), max(prdc_k * 4, 10)), algorithm="brute")
        nn_model.fit(real_embeddings)
        syn_distances, syn_indices = nn_model.kneighbors(synthetic_embeddings, return_distance=True)
        for d_row, i_row in zip(syn_distances, syn_indices, strict=False):
            counts.append(int(np.sum(d_row <= real_radii[i_row])))
    synthetic_density = float(np.mean(np.asarray(counts, dtype=np.float32) / max(prdc_k, 1))) if counts else None

    return {
        "n_real": int(len(real_embeddings)),
        "n_synthetic": int(len(synthetic_embeddings)),
        "real_to_synth_coverage": real_coverage,
        "synthetic_to_real_precision": synthetic_precision,
        "synthetic_density": synthetic_density,
        "real_to_synth_mean_nn_distance": float(real_to_syn.mean()),
        "synthetic_to_real_mean_nn_distance": float(syn_to_real.mean()),
        "real_neighbor_radius_median": float(np.median(real_radii)),
        "synthetic_neighbor_radius_median": float(np.median(syn_radii)) if len(syn_radii) else None,
    }


def sample_indices(n_rows: int, sample_size: int, seed: int) -> np.ndarray:
    if n_rows <= sample_size:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=sample_size, replace=False))


def approx_mmd_and_energy(
    real_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    sample_size: int,
    seed: int,
) -> dict[str, float | int | None]:
    if len(real_embeddings) == 0 or len(synthetic_embeddings) == 0:
        return {"distribution_sample_size": 0, "approx_mmd_rbf": None, "approx_energy_distance": None}

    real_idx = sample_indices(len(real_embeddings), sample_size, seed)
    syn_idx = sample_indices(len(synthetic_embeddings), sample_size, seed + 1)
    real = real_embeddings[real_idx]
    syn = synthetic_embeddings[syn_idx]

    rr = np.clip(1.0 - real @ real.T, 0.0, None)
    ss = np.clip(1.0 - syn @ syn.T, 0.0, None)
    rs = np.clip(1.0 - real @ syn.T, 0.0, None)

    median_dist = np.median(rs)
    gamma = 1.0 / max(2.0 * (median_dist**2), 1e-8)
    k_rr = np.exp(-gamma * rr**2)
    k_ss = np.exp(-gamma * ss**2)
    k_rs = np.exp(-gamma * rs**2)
    approx_mmd = float(k_rr.mean() + k_ss.mean() - 2.0 * k_rs.mean())

    eu_rr = np.sqrt(np.clip(2.0 * rr, 0.0, None))
    eu_ss = np.sqrt(np.clip(2.0 * ss, 0.0, None))
    eu_rs = np.sqrt(np.clip(2.0 * rs, 0.0, None))
    energy = float(2.0 * eu_rs.mean() - eu_rr.mean() - eu_ss.mean())

    return {
        "distribution_sample_size": int(min(sample_size, len(real_embeddings), len(synthetic_embeddings))),
        "approx_mmd_rbf": approx_mmd,
        "approx_energy_distance": energy,
    }


def compute_cluster_occupancy_table(
    real_assign_df: pd.DataFrame,
    synthetic_assign_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    density_lookup = cluster_summary_df.set_index("cluster_id").to_dict(orient="index")

    for group_name in GROUP_ORDER:
        real_group = real_assign_df if group_name == "full_test" else real_assign_df.loc[real_assign_df["leakage_group"] == group_name]
        syn_group = synthetic_assign_df if group_name == "full_test" else synthetic_assign_df.loc[synthetic_assign_df["leakage_group"] == group_name]
        real_counts = real_group["cluster_id"].value_counts().to_dict()
        syn_counts = syn_group["cluster_id"].value_counts().to_dict()
        real_total = max(len(real_group), 1)
        syn_total = max(len(syn_group), 1)

        for cluster_id in sorted(density_lookup):
            info = density_lookup[cluster_id]
            real_count = int(real_counts.get(cluster_id, 0))
            syn_count = int(syn_counts.get(cluster_id, 0))
            rows.append(
                {
                    "analysis_group": group_name,
                    "cluster_id": int(cluster_id),
                    "cluster_size_full_real": int(info["cluster_size"]),
                    "density_proxy_full_real": float(info["density_proxy"]),
                    "real_count": real_count,
                    "real_fraction": float(real_count / real_total),
                    "synthetic_count": syn_count,
                    "synthetic_fraction": float(syn_count / syn_total),
                    "occupancy_gap": float((syn_count / syn_total) - (real_count / real_total)),
                    "synthetic_present": bool(syn_count > 0),
                }
            )
    return pd.DataFrame(rows)


def compute_low_density_cluster_coverage(
    cluster_summary_df: pd.DataFrame,
    occupancy_df: pd.DataFrame,
    low_density_quantile: float,
) -> pd.DataFrame:
    threshold = float(cluster_summary_df["density_proxy"].quantile(low_density_quantile))
    low_density_clusters = set(
        cluster_summary_df.loc[cluster_summary_df["density_proxy"] <= threshold, "cluster_id"].astype(int).tolist()
    )
    rows: list[dict[str, Any]] = []

    for group_name in GROUP_ORDER:
        group_df = occupancy_df.loc[occupancy_df["analysis_group"] == group_name].copy()
        low_df = group_df.loc[group_df["cluster_id"].isin(low_density_clusters)].copy()
        covered = low_df.loc[low_df["synthetic_present"]]
        uncovered = low_df.loc[~low_df["synthetic_present"], "cluster_id"].astype(int).tolist()

        rows.append(
            {
                "row_type": "summary",
                "analysis_group": group_name,
                "low_density_quantile": low_density_quantile,
                "density_threshold": threshold,
                "n_low_density_clusters": int(len(low_df)),
                "n_low_density_clusters_covered": int(len(covered)),
                "low_density_cluster_coverage_fraction": float(len(covered) / max(len(low_df), 1)),
                "uncovered_cluster_ids": ",".join(map(str, uncovered)),
                "cluster_id": None,
                "real_fraction": None,
                "synthetic_fraction": None,
                "density_proxy_full_real": None,
            }
        )
        for _, row in low_df.iterrows():
            rows.append(
                {
                    "row_type": "cluster_detail",
                    "analysis_group": group_name,
                    "low_density_quantile": low_density_quantile,
                    "density_threshold": threshold,
                    "n_low_density_clusters": None,
                    "n_low_density_clusters_covered": None,
                    "low_density_cluster_coverage_fraction": None,
                    "uncovered_cluster_ids": None,
                    "cluster_id": int(row["cluster_id"]),
                    "real_fraction": float(row["real_fraction"]),
                    "synthetic_fraction": float(row["synthetic_fraction"]),
                    "density_proxy_full_real": float(row["density_proxy_full_real"]),
                }
            )
    return pd.DataFrame(rows)


def compute_nearest_real_distance_summary(
    real_assign_df: pd.DataFrame,
    real_embeddings: np.ndarray,
    synthetic_assign_df: pd.DataFrame,
    synthetic_embeddings: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name in GROUP_ORDER:
        if group_name == "full_test":
            real_group_df = real_assign_df
            syn_group_df = synthetic_assign_df
        else:
            real_group_df = real_assign_df.loc[real_assign_df["leakage_group"] == group_name]
            syn_group_df = synthetic_assign_df.loc[synthetic_assign_df["leakage_group"] == group_name]

        real_idx = real_group_df.index.to_numpy()
        syn_idx = syn_group_df.index.to_numpy()
        if len(real_idx) == 0 or len(syn_idx) == 0:
            rows.append({"analysis_group": group_name, "n_real_rows": int(len(real_idx)), "n_synthetic_rows": int(len(syn_idx))})
            continue

        distances, _ = nearest_neighbor_distances(synthetic_embeddings[syn_idx], real_embeddings[real_idx], n_neighbors=1)
        d = distances[:, 0]
        rows.append(
            {
                "analysis_group": group_name,
                "n_real_rows": int(len(real_idx)),
                "n_synthetic_rows": int(len(syn_idx)),
                "mean_nearest_real_distance": float(d.mean()),
                "median_nearest_real_distance": float(np.median(d)),
                "p05_nearest_real_distance": float(np.quantile(d, 0.05)),
                "p25_nearest_real_distance": float(np.quantile(d, 0.25)),
                "p75_nearest_real_distance": float(np.quantile(d, 0.75)),
                "p95_nearest_real_distance": float(np.quantile(d, 0.95)),
                "max_nearest_real_distance": float(d.max()),
            }
        )
    return pd.DataFrame(rows)


def compute_stratified_coverage_metrics(
    real_assign_df: pd.DataFrame,
    real_embeddings: np.ndarray,
    synthetic_assign_df: pd.DataFrame,
    synthetic_embeddings: np.ndarray,
    prdc_k: int,
    distribution_sample_size: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name in GROUP_ORDER:
        if group_name == "full_test":
            real_group_df = real_assign_df
            syn_group_df = synthetic_assign_df
        else:
            real_group_df = real_assign_df.loc[real_assign_df["leakage_group"] == group_name]
            syn_group_df = synthetic_assign_df.loc[synthetic_assign_df["leakage_group"] == group_name]

        real_idx = real_group_df.index.to_numpy()
        syn_idx = syn_group_df.index.to_numpy()
        base = compute_distribution_metrics(real_embeddings[real_idx], synthetic_embeddings[syn_idx], prdc_k=prdc_k)
        approx = approx_mmd_and_energy(
            real_embeddings[real_idx],
            synthetic_embeddings[syn_idx],
            sample_size=distribution_sample_size,
            seed=seed,
        )
        rows.append({"analysis_group": group_name, **base, **approx})
    return pd.DataFrame(rows)


def compute_subgroup_coverage_summary(
    real_assign_df: pd.DataFrame,
    synthetic_assign_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    available = [name for name in SUBGROUP_CANDIDATE_MAP if name in real_assign_df.columns and real_assign_df[name].notna().any()]
    if not available:
        warnings.append("Subgroup coverage summary skipped because canonical subgroup metadata fields are unavailable.")
        return pd.DataFrame(), warnings

    density_lookup = cluster_summary_df.set_index("cluster_id")["density_proxy"]
    rows: list[dict[str, Any]] = []
    for subgroup_name in available:
        real_nonnull = real_assign_df.loc[real_assign_df[subgroup_name].notna()]
        syn_nonnull = synthetic_assign_df.loc[synthetic_assign_df[subgroup_name].notna()]
        for subgroup_value in sorted(real_nonnull[subgroup_name].astype(str).unique()):
            real_subset = real_nonnull.loc[real_nonnull[subgroup_name].astype(str) == subgroup_value]
            syn_subset = syn_nonnull.loc[syn_nonnull[subgroup_name].astype(str) == subgroup_value]
            if real_subset.empty:
                continue
            real_clusters = set(real_subset["cluster_id"].astype(int))
            syn_clusters = set(syn_subset["cluster_id"].astype(int))
            covered_clusters = real_clusters & syn_clusters
            rows.append(
                {
                    "subgroup_name": subgroup_name,
                    "subgroup_value": subgroup_value,
                    "n_real_rows": int(len(real_subset)),
                    "n_synthetic_rows": int(len(syn_subset)),
                    "n_real_clusters": int(len(real_clusters)),
                    "n_covered_clusters": int(len(covered_clusters)),
                    "cluster_coverage_fraction": float(len(covered_clusters) / max(len(real_clusters), 1)),
                    "mean_real_cluster_density": float(density_lookup.loc[list(real_clusters)].mean()) if real_clusters else None,
                }
            )
    return pd.DataFrame(rows), warnings


def create_visualization(
    output_dir: Path,
    real_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    real_assign_df: pd.DataFrame,
    synthetic_assign_df: pd.DataFrame,
    max_points_for_plot: int,
    seed: int,
) -> list[str]:
    warnings: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ["matplotlib is unavailable; skipping PCA/UMAP visualization."]

    rng = np.random.default_rng(seed)
    real_idx = sample_indices(len(real_embeddings), max_points_for_plot, seed)
    syn_idx = sample_indices(len(synthetic_embeddings), max_points_for_plot, seed + 1)
    combined = np.vstack([real_embeddings[real_idx], synthetic_embeddings[syn_idx]])
    labels = np.array(["real"] * len(real_idx) + ["synthetic"] * len(syn_idx))
    cluster_ids = np.concatenate([
        real_assign_df.iloc[real_idx]["cluster_id"].to_numpy(),
        synthetic_assign_df.iloc[syn_idx]["cluster_id"].to_numpy(),
    ])

    pca = PCA(n_components=2, random_state=seed)
    pca_coords = pca.fit_transform(combined)

    umap_coords = None
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=seed)
        umap_coords = reducer.fit_transform(combined)
    except Exception:
        warnings.append("umap-learn is unavailable; visualization contains PCA only.")

    ncols = 2 if umap_coords is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    def scatter_panel(ax: Any, coords: np.ndarray, title: str) -> None:
        for kind, color in [("real", "#1f77b4"), ("synthetic", "#ff7f0e")]:
            mask = labels == kind
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.45, c=color, label=kind)
        # Overlay cluster centroids from the plotted sample to help locate sparse regions.
        for cluster_id in np.unique(cluster_ids):
            cluster_mask = cluster_ids == cluster_id
            center = coords[cluster_mask].mean(axis=0)
            ax.scatter(center[0], center[1], s=20, c="black", alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(frameon=False)
        ax.text(
            0.01,
            0.01,
            "Visualization only; quantitative coverage uses full 1024-d normalized BGE space.",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
        )

    scatter_panel(axes[0], pca_coords, "PCA: real vs synthetic")
    if umap_coords is not None:
        scatter_panel(axes[1], umap_coords, "UMAP: real vs synthetic")

    fig.tight_layout()
    fig.savefig(output_dir / "coverage_umap_real_vs_synthetic.png", dpi=200)
    plt.close(fig)
    return warnings


def write_report_md(path: Path, title: str, sections: list[tuple[str, list[str]]]) -> None:
    lines = [f"# {title}", ""]
    for section_title, section_lines in sections:
        lines.append(f"## {section_title}")
        lines.append("")
        lines.extend(section_lines)
        lines.append("")
    path.write_text("\n".join(lines))


def check_real_vs_synthetic_guards(
    args: argparse.Namespace,
    real_manifest: pd.DataFrame,
) -> tuple[list[str], dict[str, Any], pd.DataFrame | None, np.ndarray | None]:
    errors: list[str] = []
    guard_info: dict[str, Any] = {}

    synthetic_manifest_path = Path(args.synthetic_manifest_path)
    audit_json_path = Path(args.audit_json_path)
    synthetic_embeddings_path = Path(args.synthetic_embeddings_path)

    guard_info["synthetic_manifest_exists"] = synthetic_manifest_path.exists()
    guard_info["audit_json_exists"] = audit_json_path.exists()
    guard_info["synthetic_embeddings_exist"] = synthetic_embeddings_path.exists()

    synthetic_manifest_df: pd.DataFrame | None = None
    synthetic_embeddings: np.ndarray | None = None

    if not synthetic_manifest_path.exists():
        errors.append("Synthetic manifest does not exist.")
    if not audit_json_path.exists():
        errors.append("Vanilla audit JSON does not exist.")
    if not synthetic_embeddings_path.exists():
        errors.append("Synthetic embedding file does not exist.")

    if audit_json_path.exists():
        audit_data = json.loads(audit_json_path.read_text())
        audit_status = audit_data.get("readiness_status")
        guard_info["audit_status"] = audit_status
        if audit_status not in {"PASS", "CAUTION"}:
            errors.append(f"Vanilla audit status must be PASS or CAUTION, got {audit_status}.")

    if synthetic_manifest_path.exists():
        synthetic_manifest_df = load_jsonl(synthetic_manifest_path)
        guard_info["synthetic_manifest_rows"] = int(len(synthetic_manifest_df))
        for col in ["generation_condition", "split", "dataset_row_id", *LEAKAGE_COLS]:
            if col not in synthetic_manifest_df.columns:
                errors.append(f"Synthetic manifest is missing required column: {col}")
        if "generated_text" in synthetic_manifest_df.columns and synthetic_manifest_df["generated_text"].isna().any():
            errors.append("Synthetic manifest contains null generated_text rows.")
        if "split" in synthetic_manifest_df.columns:
            split_values = sorted(set(synthetic_manifest_df["split"].dropna().astype(str)))
            guard_info["synthetic_manifest_split_values"] = split_values
            if split_values != ["test"]:
                errors.append(f"Synthetic manifest split values must be ['test'], got {split_values}.")
        if "generation_condition" in synthetic_manifest_df.columns:
            conditions = sorted(set(synthetic_manifest_df["generation_condition"].dropna().astype(str)))
            guard_info["synthetic_generation_conditions"] = conditions
            if conditions != ["vanilla"]:
                errors.append(f"Synthetic manifest generation_condition must be ['vanilla'], got {conditions}.")
        if "dataset_row_id" in synthetic_manifest_df.columns and len(synthetic_manifest_df) == len(real_manifest):
            syn_ids = pd.to_numeric(synthetic_manifest_df["dataset_row_id"], errors="coerce")
            if syn_ids.isna().any():
                errors.append("Synthetic manifest contains non-numeric dataset_row_id values.")
            else:
                syn_ids = syn_ids.astype(int).to_numpy()
                guard_info["synthetic_dataset_row_id_aligned"] = bool(
                    np.array_equal(syn_ids, real_manifest["dataset_row_id"].to_numpy())
                )
                if not guard_info["synthetic_dataset_row_id_aligned"]:
                    errors.append("Synthetic manifest dataset_row_id order does not exactly match filtered test manifest.")

    if synthetic_embeddings_path.exists():
        synthetic_embeddings = load_synthetic_embeddings(synthetic_embeddings_path)
        guard_info["synthetic_embedding_rows"] = int(synthetic_embeddings.shape[0])

    if synthetic_manifest_df is not None and synthetic_embeddings is not None:
        if len(synthetic_manifest_df) != synthetic_embeddings.shape[0]:
            errors.append(
                f"Synthetic embedding row count ({synthetic_embeddings.shape[0]}) does not match synthetic manifest row count ({len(synthetic_manifest_df)})."
            )
        if len(synthetic_manifest_df) != len(real_manifest):
            errors.append(
                f"Synthetic manifest row count ({len(synthetic_manifest_df)}) does not match real held-out row count ({len(real_manifest)})."
            )

    return errors, guard_info, synthetic_manifest_df, synthetic_embeddings


def build_synthetic_assignments(
    synthetic_manifest_df: pd.DataFrame,
    synthetic_embeddings: np.ndarray,
    cluster_centers: np.ndarray,
    extra_metadata: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    synthetic_df = synthetic_manifest_df.copy().reset_index(drop=True)
    labels, distances = assign_to_centers(synthetic_embeddings, cluster_centers)
    synthetic_df["cluster_id"] = labels
    synthetic_df["distance_to_centroid"] = distances
    synthetic_df["leakage_group"] = leakage_group(synthetic_df["patient_disjoint_from_train"])
    synthetic_df, join_warnings = enrich_with_extra_metadata(synthetic_df, extra_metadata)
    warnings.extend(join_warnings)
    synthetic_df, subgroup_warnings = canonicalize_subgroup_columns(synthetic_df)
    warnings.extend(subgroup_warnings)
    return synthetic_df, warnings


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_dataset_path = Path(args.real_dataset_path)
    train_dataset_path = Path(args.train_dataset_path)
    dev_dataset_path = Path(args.dev_dataset_path)
    split_manifest_path = Path(args.split_manifest_path)
    extra_metadata = load_extra_metadata(args.extra_metadata_path)
    split_manifest = load_split_manifest(split_manifest_path)

    run_config = {
        "mode": args.mode,
        "created_at": now_iso(),
        "git_commit": get_git_commit(),
        "script_path": str(Path(__file__).resolve()),
        "real_dataset_path": str(real_dataset_path),
        "train_dataset_path": str(train_dataset_path),
        "dev_dataset_path": str(dev_dataset_path),
        "split_manifest_path": str(split_manifest_path),
        "whole_cohort_manifest_path": args.whole_cohort_manifest_path,
        "embedding_metadata_path": args.embedding_metadata_path,
        "extra_metadata_path": args.extra_metadata_path,
        "output_dir": str(output_dir),
        "n_clusters": int(args.n_clusters),
        "random_seed": int(args.random_seed),
        "prdc_k": int(args.prdc_k),
        "low_density_quantile": float(args.low_density_quantile),
        "distribution_sample_size": int(args.distribution_sample_size),
        "max_points_for_plot": int(args.max_points_for_plot),
        "synthetic_manifest_path": args.synthetic_manifest_path,
        "synthetic_embeddings_path": args.synthetic_embeddings_path,
        "audit_json_path": args.audit_json_path,
    }
    (output_dir / "coverage_run_config.json").write_text(json.dumps(run_config, indent=2))

    if args.mode == "real_only_precompute":
        real_embeddings = load_real_embeddings(real_dataset_path)
        test_manifest, manifest_warnings = build_test_manifest(split_manifest, len(real_embeddings), extra_metadata)
        warnings: list[str] = list(manifest_warnings)
        assign_df, cluster_df, _ = compute_real_cluster_outputs(
            real_embeddings,
            test_manifest,
            n_clusters=args.n_clusters,
            random_seed=args.random_seed,
        )
        group_df = compute_group_cluster_summary(assign_df)
        subgroup_df, subgroup_warnings = compute_subgroup_cluster_summary(assign_df)
        warnings.extend(subgroup_warnings)

        assign_df.to_csv(output_dir / "real_cluster_assignments.csv", index=False)
        cluster_df.to_csv(output_dir / "real_cluster_summary.csv", index=False)
        subgroup_df.to_csv(output_dir / "real_subgroup_cluster_summary.csv", index=False)
        group_df.to_csv(output_dir / "real_group_cluster_summary.csv", index=False)

        summary = {
            **run_config,
            "n_real_rows": int(len(real_embeddings)),
            "row_count_match": True,
            "warnings": warnings,
        }
        (output_dir / "coverage_real_only_precompute.json").write_text(json.dumps(summary, indent=2))
        write_report_md(
            output_dir / "coverage_real_only_precompute.md",
            "Coverage Real-Only Precompute",
            [
                ("Status", ["- Real-only precompute completed successfully.", "- Quantitative coverage claims remain deferred until synthetic embeddings are available."]),
                ("Summary", [f"- n_real_rows: `{len(real_embeddings)}`", f"- n_clusters: `{args.n_clusters}`"]),
                ("Warnings", [f"- {warning}" for warning in warnings] or ["- None"]),
            ],
        )
        print("Saved real-only coverage precompute outputs to:", output_dir)
        return

    if args.mode == "real_all_filtered_precompute":
        all_manifest, real_embeddings, manifest_warnings = build_all_filtered_manifest(
            split_manifest=split_manifest,
            train_dataset_path=train_dataset_path,
            dev_dataset_path=dev_dataset_path,
            test_dataset_path=real_dataset_path,
            extra_metadata=extra_metadata,
        )
        warnings: list[str] = list(manifest_warnings)
        assign_df, cluster_df, _ = compute_real_cluster_outputs(
            real_embeddings,
            all_manifest,
            n_clusters=args.n_clusters,
            random_seed=args.random_seed,
        )
        group_df = compute_group_cluster_summary(assign_df)
        split_df = compute_split_cluster_summary(assign_df)
        subgroup_df, subgroup_warnings = compute_subgroup_cluster_summary(assign_df)
        warnings.extend(subgroup_warnings)

        assign_df.to_csv(output_dir / "real_all_filtered_cluster_assignments.csv", index=False)
        cluster_df.to_csv(output_dir / "real_all_filtered_cluster_summary.csv", index=False)
        group_df.to_csv(output_dir / "real_all_filtered_group_cluster_summary.csv", index=False)
        split_df.to_csv(output_dir / "real_all_filtered_split_cluster_summary.csv", index=False)
        subgroup_df.to_csv(output_dir / "real_all_filtered_subgroup_cluster_summary.csv", index=False)

        summary = {
            **run_config,
            "n_real_rows": int(len(real_embeddings)),
            "split_counts": assign_df["split"].value_counts().to_dict(),
            "row_count_match": True,
            "warnings": warnings,
        }
        (output_dir / "coverage_real_all_filtered_precompute.json").write_text(json.dumps(summary, indent=2))
        write_report_md(
            output_dir / "coverage_real_all_filtered_precompute.md",
            "Coverage Real-All-Filtered Precompute",
            [
                ("Status", ["- All-filtered real precompute completed successfully.", "- This mode is for manifold discovery across train/dev/test, not official held-out synthetic evaluation."]),
                ("Summary", [f"- n_real_rows: `{len(real_embeddings)}`", f"- n_clusters: `{args.n_clusters}`", f"- split_counts: `{summary['split_counts']}`"]),
                ("Warnings", [f"- {warning}" for warning in warnings] or ["- None"]),
            ],
        )
        print("Saved all-filtered real coverage precompute outputs to:", output_dir)
        return

    real_embeddings = load_real_embeddings(real_dataset_path)
    test_manifest, manifest_warnings = build_test_manifest(split_manifest, len(real_embeddings), extra_metadata)
    warnings: list[str] = list(manifest_warnings)
    assign_df, cluster_df, cluster_centers = compute_real_cluster_outputs(
        real_embeddings,
        test_manifest,
        n_clusters=args.n_clusters,
        random_seed=args.random_seed,
    )

    guard_errors, guard_info, synthetic_manifest_df, synthetic_embeddings = check_real_vs_synthetic_guards(args, test_manifest)
    if synthetic_manifest_df is None or synthetic_embeddings is None:
        guard_errors.append("Synthetic manifest or synthetic embeddings could not be loaded.")

    guard_summary = {
        **run_config,
        "guard_info": guard_info,
        "warnings": warnings,
        "errors": guard_errors,
    }
    (output_dir / "coverage_real_vs_synthetic_guard_check.json").write_text(json.dumps(guard_summary, indent=2))
    write_report_md(
        output_dir / "coverage_real_vs_synthetic_guard_check.md",
        "Coverage Real-vs-Synthetic Guard Check",
        [
            ("Status", ["- Guard check completed."]),
            ("Errors", [f"- {err}" for err in guard_errors] or ["- None"]),
            ("Warnings", [f"- {warning}" for warning in warnings] or ["- None"]),
        ],
    )
    if guard_errors:
        raise SystemExit(
            "real_vs_synthetic mode is blocked until all guards pass. See coverage_real_vs_synthetic_guard_check.json"
        )

    synthetic_assign_df, synthetic_warnings = build_synthetic_assignments(
        synthetic_manifest_df=synthetic_manifest_df,
        synthetic_embeddings=synthetic_embeddings,
        cluster_centers=cluster_centers,
        extra_metadata=extra_metadata,
    )
    warnings.extend(synthetic_warnings)

    occupancy_df = compute_cluster_occupancy_table(assign_df, synthetic_assign_df, cluster_df)
    low_density_df = compute_low_density_cluster_coverage(cluster_df, occupancy_df, args.low_density_quantile)
    nearest_distance_df = compute_nearest_real_distance_summary(assign_df, real_embeddings, synthetic_assign_df, synthetic_embeddings)
    coverage_df = compute_stratified_coverage_metrics(
        assign_df,
        real_embeddings,
        synthetic_assign_df,
        synthetic_embeddings,
        prdc_k=args.prdc_k,
        distribution_sample_size=args.distribution_sample_size,
        seed=args.random_seed,
    )
    subgroup_df, subgroup_warnings = compute_subgroup_coverage_summary(assign_df, synthetic_assign_df, cluster_df)
    warnings.extend(subgroup_warnings)
    plot_warnings = create_visualization(
        output_dir=output_dir,
        real_embeddings=real_embeddings,
        synthetic_embeddings=synthetic_embeddings,
        real_assign_df=assign_df,
        synthetic_assign_df=synthetic_assign_df,
        max_points_for_plot=args.max_points_for_plot,
        seed=args.random_seed,
    )
    warnings.extend(plot_warnings)

    occupancy_df.to_csv(output_dir / "cluster_occupancy_real_vs_synthetic.csv", index=False)
    low_density_df.to_csv(output_dir / "low_density_cluster_coverage.csv", index=False)
    nearest_distance_df.to_csv(output_dir / "nearest_real_distance_summary.csv", index=False)
    coverage_df.to_csv(output_dir / "coverage_full_vs_patient_disjoint.csv", index=False)
    if not subgroup_df.empty:
        subgroup_df.to_csv(output_dir / "subgroup_coverage_summary.csv", index=False)

    low_density_summary = low_density_df.loc[low_density_df["row_type"] == "summary"].copy()
    summary = {
        **run_config,
        "n_real_rows": int(len(real_embeddings)),
        "n_synthetic_rows": int(len(synthetic_embeddings)),
        "guard_info": guard_info,
        "warnings": warnings,
        "full_test_metrics": coverage_df.loc[coverage_df["analysis_group"] == "full_test"].to_dict(orient="records")[0],
        "patient_disjoint_metrics": coverage_df.loc[coverage_df["analysis_group"] == "patient_disjoint"].to_dict(orient="records")[0],
        "patient_overlap_metrics": coverage_df.loc[coverage_df["analysis_group"] == "patient_overlap"].to_dict(orient="records")[0],
        "low_density_cluster_summary": low_density_summary.to_dict(orient="records"),
    }
    (output_dir / "coverage_real_vs_synthetic.json").write_text(json.dumps(summary, indent=2))

    readiness_lines = [
        "- Coverage metrics were computed in the full normalized 1024-d BGE space.",
        "- PCA / UMAP outputs, if present, are visualization only and should not be treated as the scientific coverage metric.",
    ]
    warnings_lines = [f"- {warning}" for warning in warnings] or ["- None"]

    metric_lines = []
    for _, row in coverage_df.iterrows():
        metric_lines.append(
            "- "
            + f"{row['analysis_group']}: real_to_synth_coverage={row.get('real_to_synth_coverage')}, "
            + f"synthetic_to_real_precision={row.get('synthetic_to_real_precision')}, "
            + f"synthetic_density={row.get('synthetic_density')}, "
            + f"approx_mmd_rbf={row.get('approx_mmd_rbf')}, "
            + f"approx_energy_distance={row.get('approx_energy_distance')}"
        )

    low_density_lines = []
    for _, row in low_density_summary.iterrows():
        low_density_lines.append(
            "- "
            + f"{row['analysis_group']}: covered {int(row['n_low_density_clusters_covered'])}/{int(row['n_low_density_clusters'])} "
            + f"low-density clusters ({row['low_density_cluster_coverage_fraction']})"
        )

    write_report_md(
        output_dir / "coverage_real_vs_synthetic.md",
        "Coverage Real-vs-Synthetic",
        [
            ("Status", readiness_lines),
            ("Coverage Metrics", metric_lines),
            ("Low-Density Cluster Coverage", low_density_lines or ["- None"]),
            (
                "Outputs",
                [
                    "- `cluster_occupancy_real_vs_synthetic.csv`",
                    "- `low_density_cluster_coverage.csv`",
                    "- `nearest_real_distance_summary.csv`",
                    "- `coverage_full_vs_patient_disjoint.csv`",
                    "- `coverage_umap_real_vs_synthetic.png` if plotting dependencies were available",
                ],
            ),
            ("Warnings", warnings_lines),
        ],
    )

    print("Saved real-vs-synthetic coverage outputs to:", output_dir)


if __name__ == "__main__":
    main()
