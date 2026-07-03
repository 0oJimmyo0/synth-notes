#!/usr/bin/env python3
"""
Build a target-conditioned shifted Hugging Face embedding dataset using
source-specific local target neighborhoods.

This script is intended for Phase 2b local steering redesign:
- choose a binary target region such as `cluster_target_29 == 1`
- choose a source pool such as nearby non-target rows
- for each source row, find k nearest target neighbors
- compute a local target centroid
- move the source embedding partway toward that centroid

The output dataset preserves the same `input_ids` / `domain_embeddings`
structure expected by `generate_synthetic_notes.py`, with manifest-ready
row-level provenance attached.
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

from common import normalize_rows, parse_csv_list, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a projected/local target-shift HF dataset.")
    parser.add_argument("--dataset_path", required=True, help="Source HF dataset path, e.g. encoded_testing_filtered")
    parser.add_argument("--factors_path", required=True, help="Factor/metadata CSV carrying the target column")
    parser.add_argument("--output_dir", required=True, help="Output directory for shifted dataset + metadata")
    parser.add_argument(
        "--target_column",
        required=True,
        help="Binary target column, e.g. cluster_target_29",
    )
    parser.add_argument(
        "--source_selection_query",
        required=True,
        help="Pandas query selecting source anchors, e.g. 'cluster_target_29 == 0'",
    )
    parser.add_argument(
        "--target_selection_query",
        default=None,
        help="Optional pandas query for target pool. Defaults to `<target_column> == 1`.",
    )
    parser.add_argument(
        "--split_manifest_path",
        default=None,
        help="Optional filtered-aligned split manifest to join leakage flags/provenance",
    )
    parser.add_argument(
        "--join_cols",
        default="source_row_id,embedding_row_id,dataset_row_id,note_id,subject_id,hadm_id",
        help="Preferred join columns for metadata merges",
    )
    parser.add_argument(
        "--source_split",
        default=None,
        help="Optional split filter (test/dev/train) applied before source/target queries",
    )
    parser.add_argument(
        "--alphas",
        required=True,
        help="Comma-separated interpolation weights toward local target centroid, e.g. 0.1,0.2,0.25",
    )
    parser.add_argument(
        "--k_target_neighbors",
        type=int,
        default=16,
        help="Number of nearest target neighbors per source row",
    )
    parser.add_argument(
        "--max_source_rows",
        type=int,
        default=None,
        help="Optional cap on selected source rows before alpha expansion",
    )
    parser.add_argument(
        "--max_target_rows",
        type=int,
        default=None,
        help="Optional cap on target pool rows (useful for smoke tests)",
    )
    parser.add_argument(
        "--max_shift_norm",
        type=float,
        default=None,
        help="Optional cap on the raw shift norm before optional renormalization",
    )
    parser.add_argument(
        "--normalize_after_steering",
        action="store_true",
        help="L2-normalize shifted embeddings after the local move",
    )
    parser.add_argument(
        "--output_stem",
        default="projected_target_shift",
        help="Stem for metadata sidecar files written next to the saved HF dataset",
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


def maybe_int(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        return normalize_scalar(value)


def load_dataset_rows(dataset_path: Path) -> tuple[Dataset, pd.DataFrame]:
    dataset = Dataset.load_from_disk(str(dataset_path))
    base_df = pd.DataFrame({"dataset_row_id": np.arange(len(dataset), dtype=int)})
    metadata_cols = [col for col in dataset.column_names if col not in {"input_ids", "domain_embeddings"}]
    if metadata_cols:
        metadata_df = dataset.select_columns(metadata_cols).to_pandas()
        base_df = pd.concat([base_df, metadata_df.reset_index(drop=True)], axis=1)
    return dataset, base_df


def choose_join_keys(frames: list[pd.DataFrame], preferred_keys: list[str]) -> list[str]:
    if not frames:
        raise ValueError("No frames were provided for join-key detection.")
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
    if all(key in preferred_keys for key in ["note_id", "subject_id", "hadm_id"]):
        if ["note_id", "subject_id", "hadm_id"] not in preferred_groups:
            preferred_groups.append(["note_id", "subject_id", "hadm_id"])
    if all(key in preferred_keys for key in ["subject_id", "hadm_id"]):
        if ["subject_id", "hadm_id"] not in preferred_groups:
            preferred_groups.append(["subject_id", "hadm_id"])
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


def merge_optional_metadata(
    base_df: pd.DataFrame,
    factors_path: Path,
    split_manifest_path: str | None,
    preferred_join_cols: list[str],
    source_split: str | None = None,
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
            elif split_df.duplicated(subset=join_cols, keep=False).any():
                raise ValueError(
                    "Split manifest reuses dataset_row_id across multiple splits. "
                    "Pass --source_split so the join can be filtered first."
                )
        merged = normalize_join_cols(merged, join_cols)
        split_df = normalize_join_cols(split_df, join_cols)
        duplicate_mask = split_df.duplicated(subset=join_cols, keep=False)
        if duplicate_mask.any():
            dup_count = int(duplicate_mask.sum())
            raise ValueError(
                f"Split manifest has {dup_count} duplicate rows for join keys {join_cols}; deduplicate it first."
            )
        merged = merged.merge(
            split_df,
            on=join_cols,
            how="left",
            validate="one_to_one",
            suffixes=("", "_split"),
        )
        active_join_cols = join_cols
        join_report["split_manifest_join_cols"] = join_cols

    factors_df = pd.read_csv(factors_path)
    join_cols = choose_join_keys([merged, factors_df], preferred_join_cols)
    merged = normalize_join_cols(merged, join_cols)
    factors_df = normalize_join_cols(factors_df, join_cols)
    duplicate_mask = factors_df.duplicated(subset=join_cols, keep=False)
    if duplicate_mask.any():
        dup_count = int(duplicate_mask.sum())
        raise ValueError(f"Factor table has {dup_count} duplicate rows for join keys {join_cols}; deduplicate it first.")
    merged = merged.merge(
        factors_df,
        on=join_cols,
        how="left",
        validate="one_to_one",
        suffixes=("", "_factor"),
    )
    active_join_cols = join_cols
    join_report["factors_join_cols"] = join_cols

    return merged, active_join_cols, join_report


def extract_embedding_vector(example: dict[str, Any]) -> np.ndarray:
    emb = example["domain_embeddings"]
    if not isinstance(emb, list) or not emb:
        raise ValueError("Expected each dataset row to carry a non-empty domain_embeddings list.")
    return np.asarray(emb[0], dtype=np.float32)


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def cap_shift_norm(shift: np.ndarray, max_shift_norm: float | None) -> tuple[np.ndarray, float]:
    raw_norm = float(np.linalg.norm(shift))
    if max_shift_norm is None or raw_norm <= max_shift_norm or raw_norm <= 0.0:
        return shift, raw_norm
    scaled = shift * (max_shift_norm / raw_norm)
    return scaled.astype(np.float32), float(np.linalg.norm(scaled))


def find_topk_target_neighbors(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    k: int,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return cosine distances and target-row indices for top-k nearest target neighbors.

    Both source_embeddings and target_embeddings are assumed to be L2-normalized.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    if target_embeddings.shape[0] < k:
        raise ValueError("target_embeddings has fewer rows than k")

    all_distances: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []

    for start in range(0, source_embeddings.shape[0], batch_size):
        stop = min(start + batch_size, source_embeddings.shape[0])
        sims = source_embeddings[start:stop] @ target_embeddings.T
        # Larger cosine similarity is better.
        top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        top_sims = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(-top_sims, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_sims = np.take_along_axis(top_sims, order, axis=1)
        all_indices.append(top_idx.astype(np.int32))
        all_distances.append((1.0 - top_sims).astype(np.float32))

    return np.vstack(all_distances), np.vstack(all_indices)


def main() -> None:
    args = build_parser().parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    factors_path = Path(args.factors_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preferred_join_cols = parse_csv_list(args.join_cols)
    alphas = parse_float_list(args.alphas)
    if not alphas:
        raise ValueError("--alphas must specify at least one steering strength")

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

    if args.source_split:
        if "split" not in merged_df.columns:
            raise ValueError("--source_split was provided, but merged metadata has no 'split' column.")
        merged_df = merged_df.loc[merged_df["split"] == args.source_split].copy()

    if args.target_column not in merged_df.columns:
        raise KeyError(f"Target column not found after merge: {args.target_column}")

    numeric = pd.to_numeric(merged_df[args.target_column], errors="coerce")
    merged_df = merged_df.loc[numeric.notna()].copy()
    merged_df[args.target_column] = numeric.loc[merged_df.index].astype(int)
    merged_df = merged_df.loc[merged_df[args.target_column].isin([0, 1])].copy()

    target_query = args.target_selection_query or f"{args.target_column} == 1"
    source_df = merged_df.query(args.source_selection_query, engine="python").copy()
    target_df = merged_df.query(target_query, engine="python").copy()

    if args.max_source_rows is not None:
        source_df = source_df.head(args.max_source_rows).copy()
    if args.max_target_rows is not None:
        target_df = target_df.head(args.max_target_rows).copy()

    if source_df.empty:
        raise ValueError("No source rows remained after applying --source_selection_query.")
    if target_df.empty:
        raise ValueError("No target rows remained after applying target selection.")

    source_row_ids = source_df["dataset_row_id"].astype(int).to_numpy()
    target_row_ids = target_df["dataset_row_id"].astype(int).to_numpy()

    source_embeddings = normalize_rows(np.stack([extract_embedding_vector(dataset[int(i)]) for i in source_row_ids], axis=0))
    target_embeddings = normalize_rows(np.stack([extract_embedding_vector(dataset[int(i)]) for i in target_row_ids], axis=0))

    k_neighbors = min(int(args.k_target_neighbors), len(target_row_ids))
    if k_neighbors <= 0:
        raise ValueError("--k_target_neighbors must be >= 1 after target filtering.")

    neighbor_distances, neighbor_indices = find_topk_target_neighbors(
        source_embeddings=source_embeddings,
        target_embeddings=target_embeddings,
        k=k_neighbors,
    )

    shifted_rows: list[dict[str, Any]] = []
    dataset_manifest_rows: list[dict[str, Any]] = []
    run_metadata_path = output_dir / f"{args.output_stem}_run_metadata.json"
    axis_label = f"projected_local_centroid__{args.target_column}"

    for source_order, (_, source_row) in enumerate(source_df.reset_index(drop=True).iterrows()):
        dataset_row_id = int(source_row["dataset_row_id"])
        example = dataset[dataset_row_id]
        source_embedding = extract_embedding_vector(example)
        source_norm = float(np.linalg.norm(source_embedding))
        if source_norm <= 0:
            raise ValueError("Encountered zero-norm source embedding; cannot steer it safely.")

        source_unit = (source_embedding / source_norm).astype(np.float32)
        nbr_ids = neighbor_indices[source_order]
        nbr_dist = neighbor_distances[source_order]
        local_target_rows = target_row_ids[nbr_ids]
        local_target_embs = target_embeddings[nbr_ids]
        local_centroid = normalize_rows(local_target_embs.mean(axis=0, keepdims=True))[0].astype(np.float32)
        base_residual = (local_centroid - source_unit).astype(np.float32)

        nearest_target_distance = float(nbr_dist[0])
        nearest_target_cosine = float(1.0 - nbr_dist[0])
        mean_neighbor_distance = float(np.mean(nbr_dist))
        mean_neighbor_cosine = float(np.mean(1.0 - nbr_dist))

        for alpha in alphas:
            raw_shift = (float(alpha) * base_residual).astype(np.float32)
            shift_vector, effective_shift_norm = cap_shift_norm(raw_shift, args.max_shift_norm)
            shifted_embedding = source_unit + shift_vector

            if args.normalize_after_steering:
                shifted_embedding = normalize_rows(shifted_embedding.reshape(1, -1))[0].astype(np.float32)
            else:
                shifted_embedding = shifted_embedding.astype(np.float32)

            source_cosine = float(
                np.dot(source_unit, shifted_embedding)
                / max(np.linalg.norm(shifted_embedding), 1e-12)
            )
            target_centroid_cosine = float(
                np.dot(local_centroid, shifted_embedding)
                / max(np.linalg.norm(shifted_embedding), 1e-12)
            )

            shifted_row = {
                "input_ids": example["input_ids"],
                "domain_embeddings": [shifted_embedding.tolist()],
                "source_row_id": maybe_int(source_row.get("source_row_id", source_row.get("dataset_row_id"))),
                "dataset_row_id": maybe_int(source_row.get("dataset_row_id")),
                "embedding_row_id": maybe_int(source_row.get("embedding_row_id")),
                "note_id": normalize_scalar(source_row.get("note_id")),
                "subject_id": maybe_int(source_row.get("subject_id")),
                "hadm_id": maybe_int(source_row.get("hadm_id")),
                "split": normalize_scalar(source_row.get("split", args.source_split)),
                "source_embedding_id": str(
                    maybe_int(source_row.get("embedding_row_id", source_row.get("dataset_row_id")))
                ),
                "patient_disjoint_from_train": normalize_scalar(source_row.get("patient_disjoint_from_train")),
                "hadm_disjoint_from_train": normalize_scalar(source_row.get("hadm_disjoint_from_train")),
                "note_disjoint_from_train": normalize_scalar(source_row.get("note_disjoint_from_train")),
                "patient_overlap_with_train": normalize_scalar(source_row.get("patient_overlap_with_train")),
                "hadm_overlap_with_train": normalize_scalar(source_row.get("hadm_overlap_with_train")),
                "note_overlap_with_train": normalize_scalar(source_row.get("note_overlap_with_train")),
                "axis_id": 0,
                "axis_label": axis_label,
                "alpha": float(alpha),
                "normalized_after_steering": bool(args.normalize_after_steering),
                "random_shift_norm": effective_shift_norm,
                "editor_model": None,
                "edited_text": None,
                "post_edit_source_cosine": source_cosine,
                "source_dataset_path": str(dataset_path),
                "source_split": normalize_scalar(source_row.get("split", args.source_split)),
                "selection_query": args.source_selection_query,
                "target_selection_query": target_query,
                "target_column": args.target_column,
                "nearest_target_dataset_row_id": int(local_target_rows[0]),
                "nearest_target_cosine": nearest_target_cosine,
                "mean_target_neighbor_cosine": mean_neighbor_cosine,
                "target_neighbor_count": int(k_neighbors),
                "local_target_centroid_cosine": target_centroid_cosine,
                "steering_run_metadata_path": str(run_metadata_path),
            }
            shifted_rows.append(shifted_row)
            dataset_manifest_rows.append(
                {
                    "shifted_dataset_row_id": len(dataset_manifest_rows),
                    "source_row_id": shifted_row["source_row_id"],
                    "dataset_row_id": shifted_row["dataset_row_id"],
                    "embedding_row_id": shifted_row["embedding_row_id"],
                    "note_id": shifted_row["note_id"],
                    "subject_id": shifted_row["subject_id"],
                    "hadm_id": shifted_row["hadm_id"],
                    "split": shifted_row["split"],
                    "axis_id": shifted_row["axis_id"],
                    "axis_label": shifted_row["axis_label"],
                    "alpha": shifted_row["alpha"],
                    "normalized_after_steering": shifted_row["normalized_after_steering"],
                    "random_shift_norm": shifted_row["random_shift_norm"],
                    "post_edit_source_cosine": shifted_row["post_edit_source_cosine"],
                    "nearest_target_dataset_row_id": shifted_row["nearest_target_dataset_row_id"],
                    "nearest_target_cosine": shifted_row["nearest_target_cosine"],
                    "mean_target_neighbor_cosine": shifted_row["mean_target_neighbor_cosine"],
                    "local_target_centroid_cosine": shifted_row["local_target_centroid_cosine"],
                }
            )

    shifted_dataset = Dataset.from_list(shifted_rows)
    shifted_dataset.save_to_disk(str(output_dir))

    manifest_csv = output_dir / f"{args.output_stem}_dataset_manifest.csv"
    pd.DataFrame(dataset_manifest_rows).to_csv(manifest_csv, index=False)

    summary_payload = {
        "created_at": now_iso(),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "script_path": str(Path(__file__).resolve()),
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "output_dataset_path": str(output_dir),
        "output_manifest_csv": str(manifest_csv),
        "factors_path": str(factors_path),
        "split_manifest_path": str(Path(args.split_manifest_path).resolve()) if args.split_manifest_path else None,
        "join_report": join_report,
        "active_join_cols": active_join_cols,
        "target_column": args.target_column,
        "source_split": args.source_split,
        "source_selection_query": args.source_selection_query,
        "target_selection_query": target_query,
        "alphas": alphas,
        "k_target_neighbors": int(k_neighbors),
        "max_source_rows": args.max_source_rows,
        "max_target_rows": args.max_target_rows,
        "max_shift_norm": args.max_shift_norm,
        "normalize_after_steering": bool(args.normalize_after_steering),
        "n_source_rows": int(len(source_df)),
        "n_target_rows": int(len(target_df)),
        "n_shifted_rows": int(len(shifted_rows)),
        "embedding_dim": int(source_embeddings.shape[1]),
        "cli_args": vars(args),
    }
    save_json(run_metadata_path, summary_payload)

    print(f"Saved shifted dataset to: {output_dir}")
    print(f"Selected source rows: {len(source_df)}")
    print(f"Target pool rows: {len(target_df)}")
    print(f"Shifted dataset rows: {len(shifted_rows)}")
    print(f"Run metadata: {run_metadata_path}")
    print(f"Dataset manifest CSV: {manifest_csv}")


if __name__ == "__main__":
    main()
