#!/usr/bin/env python3
"""
Build a lightweight adapter-calibration dataset from a real HF note dataset and a
precomputed shifted HF embedding dataset.

The output keeps the standard ELM fields (`input_ids`, `domain_embeddings`) while
adding per-row calibration side information:
- source_domain_embeddings
- target_prototype_embedding
- source/target provenance metadata

This lets us train only the ELM adapter on the same text targets already used by
the repo, while conditioning on edited embeddings that we want the decoder to
handle more faithfully.
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a lightweight adapter-calibration HF dataset.")
    parser.add_argument("--source_dataset_path", required=True, help="Real/source HF dataset path, e.g. encoded_training_filtered")
    parser.add_argument("--shifted_dataset_path", required=True, help="Shifted HF dataset path created by a steering operator")
    parser.add_argument("--factors_path", required=True, help="Factor/metadata CSV carrying the target column")
    parser.add_argument("--output_dir", required=True, help="Directory where the calibration dataset and sidecars will be written")
    parser.add_argument("--target_column", required=True, help="Binary target column, e.g. cluster_target_29")
    parser.add_argument(
        "--target_selection_query",
        default=None,
        help="Optional target-pool query. Defaults to `<target_column> == 1` within the source split.",
    )
    parser.add_argument(
        "--split_manifest_path",
        default=None,
        help="Optional filtered-aligned split manifest for leakage/provenance joins",
    )
    parser.add_argument(
        "--join_cols",
        default="source_row_id,embedding_row_id,dataset_row_id,note_id,subject_id,hadm_id",
        help="Preferred join columns for metadata merges",
    )
    parser.add_argument(
        "--source_split",
        default=None,
        help="Optional split label to filter the source target pool, e.g. train/dev/test",
    )
    parser.add_argument(
        "--k_target_neighbors",
        type=int,
        default=16,
        help="Number of nearest real target neighbors used to build each target prototype",
    )
    parser.add_argument(
        "--neighbor_weight_mode",
        default="softmax_cosine",
        choices=["softmax_cosine", "inverse_distance", "uniform"],
        help="How to weight the local real target neighbors",
    )
    parser.add_argument(
        "--neighbor_temperature",
        type=float,
        default=0.02,
        help="Softmax temperature when using softmax_cosine weighting",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap on shifted rows for a smaller calibration build",
    )
    parser.add_argument(
        "--output_stem",
        default="adapter_calibration",
        help="Stem for sidecar files written next to the saved HF dataset",
    )
    return parser


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return arr / norms


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    def _json_default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    import json

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


def get_git_commit(script_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(script_dir.parent.parent), "rev-parse", "HEAD"],
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


def load_dataset_rows(dataset_path: Path) -> tuple[Dataset, pd.DataFrame, np.ndarray]:
    dataset = Dataset.load_from_disk(str(dataset_path))
    base_df = pd.DataFrame({"dataset_row_id": np.arange(len(dataset), dtype=int)})
    metadata_cols = [col for col in dataset.column_names if col not in {"input_ids", "domain_embeddings"}]
    if metadata_cols:
        metadata_df = dataset.select_columns(metadata_cols).to_pandas()
        base_df = pd.concat([base_df, metadata_df.reset_index(drop=True)], axis=1)

    embeddings = []
    for emb in dataset["domain_embeddings"]:
        if not isinstance(emb, list) or not emb:
            raise ValueError(f"Dataset {dataset_path} has a row with missing domain_embeddings")
        embeddings.append(np.asarray(emb[0], dtype=np.float32))
    embedding_matrix = normalize_rows(np.stack(embeddings, axis=0))
    return dataset, base_df, embedding_matrix


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
        if join_cols == ["dataset_row_id"] and "split" in split_df.columns and source_split:
            split_df = split_df.loc[split_df["split"].astype(str) == str(source_split)].copy()
            join_report["split_manifest_filtered_to_split"] = [str(source_split)]
        merged = normalize_join_cols(merged, join_cols)
        split_df = normalize_join_cols(split_df, join_cols)
        duplicate_mask = split_df.duplicated(subset=join_cols, keep=False)
        if duplicate_mask.any():
            raise ValueError(
                f"Split manifest has {int(duplicate_mask.sum())} duplicate rows for join keys {join_cols}; deduplicate it first."
            )
        merged = merged.merge(split_df, on=join_cols, how="left", validate="one_to_one", suffixes=("", "_split"))
        active_join_cols = join_cols
        join_report["split_manifest_join_cols"] = join_cols

    factors_df = pd.read_csv(factors_path)
    join_cols = choose_join_keys([merged, factors_df], preferred_join_cols)
    merged = normalize_join_cols(merged, join_cols)
    factors_df = normalize_join_cols(factors_df, join_cols)
    duplicate_mask = factors_df.duplicated(subset=join_cols, keep=False)
    if duplicate_mask.any():
        raise ValueError(
            f"Factor table has {int(duplicate_mask.sum())} duplicate rows for join keys {join_cols}; deduplicate it first."
        )
    merged = merged.merge(factors_df, on=join_cols, how="left", validate="one_to_one", suffixes=("", "_factor"))
    active_join_cols = join_cols
    join_report["factors_join_cols"] = join_cols

    return merged, active_join_cols, join_report


def build_source_row_lookup(source_meta: pd.DataFrame) -> dict[str, dict[Any, int]]:
    lookup: dict[str, dict[Any, int]] = {}

    if "source_row_id" in source_meta.columns:
        tmp = source_meta.loc[source_meta["source_row_id"].notna(), ["source_row_id", "dataset_row_id"]].copy()
        if not tmp.empty:
            tmp["source_row_id"] = pd.to_numeric(tmp["source_row_id"], errors="coerce")
            tmp = tmp.dropna(subset=["source_row_id"]).copy()
            tmp["source_row_id"] = tmp["source_row_id"].astype(int)
            tmp["dataset_row_id"] = pd.to_numeric(tmp["dataset_row_id"], errors="coerce").astype(int)
            tmp = tmp.drop_duplicates(subset=["source_row_id"], keep="first")
            lookup["source_row_id"] = dict(zip(tmp["source_row_id"], tmp["dataset_row_id"]))

    if "embedding_row_id" in source_meta.columns:
        tmp = source_meta.loc[source_meta["embedding_row_id"].notna(), ["embedding_row_id", "dataset_row_id"]].copy()
        if not tmp.empty:
            tmp["embedding_row_id"] = pd.to_numeric(tmp["embedding_row_id"], errors="coerce")
            tmp = tmp.dropna(subset=["embedding_row_id"]).copy()
            tmp["embedding_row_id"] = tmp["embedding_row_id"].astype(int)
            tmp["dataset_row_id"] = pd.to_numeric(tmp["dataset_row_id"], errors="coerce").astype(int)
            tmp = tmp.drop_duplicates(subset=["embedding_row_id"], keep="first")
            lookup["embedding_row_id"] = dict(zip(tmp["embedding_row_id"], tmp["dataset_row_id"]))

    if all(col in source_meta.columns for col in ["note_id", "dataset_row_id"]):
        tmp = source_meta.loc[source_meta["note_id"].notna(), ["note_id", "dataset_row_id"]].copy()
        if not tmp.empty:
            tmp["note_id"] = tmp["note_id"].astype(str).str.strip()
            tmp["dataset_row_id"] = pd.to_numeric(tmp["dataset_row_id"], errors="coerce").astype(int)
            tmp = tmp.drop_duplicates(subset=["note_id"], keep="first")
            lookup["note_id"] = dict(zip(tmp["note_id"], tmp["dataset_row_id"]))

    return lookup


def resolve_local_source_dataset_row_id(
    shifted_row: dict[str, Any],
    source_lookup: dict[str, dict[Any, int]],
    source_dataset_size: int,
) -> tuple[int, str]:
    candidates: list[tuple[str, Any]] = []
    if shifted_row.get("source_row_id") is not None:
        candidates.append(("source_row_id", shifted_row.get("source_row_id")))
    if shifted_row.get("embedding_row_id") is not None:
        candidates.append(("embedding_row_id", shifted_row.get("embedding_row_id")))
    if shifted_row.get("note_id") is not None:
        candidates.append(("note_id", str(shifted_row.get("note_id")).strip()))

    for key, raw_value in candidates:
        mapping = source_lookup.get(key)
        if not mapping:
            continue
        try:
            lookup_value = int(raw_value) if key != "note_id" else raw_value
        except Exception:
            lookup_value = raw_value
        local_row_id = mapping.get(lookup_value)
        if local_row_id is None:
            continue
        if 0 <= int(local_row_id) < int(source_dataset_size):
            return int(local_row_id), key

    raise KeyError(
        "Could not resolve a split-local source dataset row id from shifted-row provenance. "
        f"Available shifted keys: source_row_id={shifted_row.get('source_row_id')}, "
        f"embedding_row_id={shifted_row.get('embedding_row_id')}, note_id={shifted_row.get('note_id')}"
    )


def pairwise_topk_cosine(query_embeddings: np.ndarray, target_embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores = query_embeddings @ target_embeddings.T
    if k >= scores.shape[1]:
        idx = np.argsort(-scores, axis=1)
        top_scores = np.take_along_axis(scores, idx, axis=1)
        return top_scores.astype(np.float32), idx.astype(np.int32)

    partition_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    partition_scores = np.take_along_axis(scores, partition_idx, axis=1)
    order = np.argsort(-partition_scores, axis=1)
    top_idx = np.take_along_axis(partition_idx, order, axis=1)
    top_scores = np.take_along_axis(scores, top_idx, axis=1)
    return top_scores.astype(np.float32), top_idx.astype(np.int32)


def compute_neighbor_weights(cosine_similarities: np.ndarray, mode: str, temperature: float) -> np.ndarray:
    if mode == "uniform":
        weights = np.full_like(cosine_similarities, fill_value=1.0 / cosine_similarities.shape[0], dtype=np.float32)
    elif mode == "inverse_distance":
        distances = 1.0 - cosine_similarities
        inv = 1.0 / np.clip(distances, 1e-6, None)
        weights = (inv / np.clip(inv.sum(), 1e-12, None)).astype(np.float32)
    elif mode == "softmax_cosine":
        temp = max(float(temperature), 1e-6)
        shifted = cosine_similarities - float(cosine_similarities.max())
        logits = np.clip(shifted / temp, -80.0, 80.0)
        exps = np.exp(logits)
        weights = (exps / np.clip(exps.sum(), 1e-12, None)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported neighbor weight mode: {mode}")
    return weights


def main() -> None:
    args = build_parser().parse_args()

    source_dataset_path = Path(args.source_dataset_path).resolve()
    shifted_dataset_path = Path(args.shifted_dataset_path).resolve()
    factors_path = Path(args.factors_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preferred_join_cols = parse_csv_list(args.join_cols)

    source_dataset, source_df, source_embeddings = load_dataset_rows(source_dataset_path)
    shifted_dataset, shifted_df, shifted_embeddings = load_dataset_rows(shifted_dataset_path)

    if args.max_rows is not None:
        keep_n = min(int(args.max_rows), len(shifted_dataset))
        shifted_dataset = shifted_dataset.select(range(keep_n))
        shifted_df = shifted_df.iloc[:keep_n].reset_index(drop=True)
        shifted_embeddings = shifted_embeddings[:keep_n]

    merged_source_df, active_join_cols, join_report = merge_metadata(
        source_df,
        factors_path=factors_path,
        split_manifest_path=args.split_manifest_path,
        preferred_join_cols=preferred_join_cols,
        source_split=args.source_split,
    )
    if args.source_split and "split" in merged_source_df.columns:
        merged_source_df = merged_source_df.loc[merged_source_df["split"].astype(str) == str(args.source_split)].copy()

    if args.target_column not in merged_source_df.columns:
        raise KeyError(f"Target column not found after merge: {args.target_column}")
    numeric_target = pd.to_numeric(merged_source_df[args.target_column], errors="coerce")
    merged_source_df = merged_source_df.loc[numeric_target.notna()].copy()
    merged_source_df[args.target_column] = numeric_target.loc[merged_source_df.index].astype(int)
    merged_source_df = merged_source_df.loc[merged_source_df[args.target_column].isin([0, 1])].copy()

    target_query = args.target_selection_query or f"{args.target_column} == 1"
    target_meta = merged_source_df.query(target_query, engine="python").copy()
    if target_meta.empty:
        raise ValueError("No target rows found for the requested target selection.")

    source_lookup = build_source_row_lookup(merged_source_df)

    source_row_col = "source_row_id" if "source_row_id" in shifted_df.columns else "dataset_row_id"
    if source_row_col not in shifted_df.columns:
        raise ValueError("Shifted dataset must carry either source_row_id or dataset_row_id for source alignment.")

    target_row_ids = target_meta["dataset_row_id"].astype(int).to_numpy()
    target_embeddings = source_embeddings[target_row_ids]
    k = min(max(int(args.k_target_neighbors), 1), len(target_embeddings))
    top_scores, top_idx = pairwise_topk_cosine(shifted_embeddings, target_embeddings, k=k)

    calibration_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for row_idx in range(len(shifted_dataset)):
        shifted_example = shifted_dataset[row_idx]
        shifted_embedding = shifted_embeddings[row_idx]
        shifted_row = shifted_df.iloc[row_idx].to_dict()
        source_row_id = int(shifted_row[source_row_col])
        local_source_dataset_row_id, resolution_key = resolve_local_source_dataset_row_id(
            shifted_row=shifted_row,
            source_lookup=source_lookup,
            source_dataset_size=len(source_dataset),
        )
        source_embedding = source_embeddings[local_source_dataset_row_id]

        neighbor_scores = top_scores[row_idx]
        neighbor_idx = top_idx[row_idx]
        neighbor_row_ids = target_row_ids[neighbor_idx]
        weights = compute_neighbor_weights(
            neighbor_scores,
            mode=args.neighbor_weight_mode,
            temperature=args.neighbor_temperature,
        )
        target_prototype = np.sum(target_embeddings[neighbor_idx] * weights[:, None], axis=0)
        target_prototype = normalize_rows(target_prototype.reshape(1, -1))[0]

        row = dict(shifted_example)
        row["source_row_id"] = source_row_id
        row["source_dataset_row_id"] = int(local_source_dataset_row_id)
        row["source_row_resolution_key"] = resolution_key
        row["source_domain_embeddings"] = source_embedding.astype(np.float32).tolist()
        row["target_prototype_embedding"] = target_prototype.astype(np.float32).tolist()
        row["calibration_target_column"] = args.target_column
        row["calibration_target_selection_query"] = target_query
        row["calibration_target_neighbor_count"] = int(k)
        row["calibration_neighbor_weight_mode"] = args.neighbor_weight_mode
        row["calibration_neighbor_temperature"] = float(args.neighbor_temperature)
        row["calibration_nearest_target_dataset_row_id"] = int(neighbor_row_ids[0])
        row["shifted_to_target_prototype_cosine"] = float(np.dot(shifted_embedding, target_prototype))
        row["shifted_to_source_cosine"] = float(np.dot(shifted_embedding, source_embedding))
        calibration_rows.append(row)

        manifest_rows.append(
            {
                "shifted_dataset_row_id": row_idx,
                "source_row_id": source_row_id,
                "source_dataset_row_id": int(local_source_dataset_row_id),
                "source_row_resolution_key": resolution_key,
                "dataset_row_id": maybe_int(row.get("dataset_row_id")),
                "note_id": normalize_scalar(row.get("note_id")),
                "subject_id": maybe_int(row.get("subject_id")),
                "hadm_id": maybe_int(row.get("hadm_id")),
                "split": normalize_scalar(row.get("split")),
                "alpha": normalize_scalar(row.get("alpha")),
                "axis_label": normalize_scalar(row.get("axis_label")),
                "calibration_target_column": args.target_column,
                "calibration_nearest_target_dataset_row_id": int(neighbor_row_ids[0]),
                "shifted_to_target_prototype_cosine": float(np.dot(shifted_embedding, target_prototype)),
                "shifted_to_source_cosine": float(np.dot(shifted_embedding, source_embedding)),
            }
        )

    calibration_dataset = Dataset.from_list(calibration_rows)
    calibration_dataset.save_to_disk(str(output_dir))

    manifest_csv = output_dir / f"{args.output_stem}_dataset_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)

    summary_df = pd.DataFrame(manifest_rows)
    summary_group_cols = ["alpha"] if "alpha" in summary_df.columns and summary_df["alpha"].notna().any() else ["axis_label"]
    summary_csv = output_dir / f"{args.output_stem}_summary.csv"
    (
        summary_df.groupby(summary_group_cols, dropna=False)
        .agg(
            n_rows=("shifted_dataset_row_id", "size"),
            mean_shifted_to_target_prototype_cosine=("shifted_to_target_prototype_cosine", "mean"),
            median_shifted_to_target_prototype_cosine=("shifted_to_target_prototype_cosine", "median"),
            mean_shifted_to_source_cosine=("shifted_to_source_cosine", "mean"),
            median_shifted_to_source_cosine=("shifted_to_source_cosine", "median"),
        )
        .reset_index()
        .to_csv(summary_csv, index=False)
    )

    run_metadata_path = output_dir / f"{args.output_stem}_run_metadata.json"
    save_json(
        run_metadata_path,
        {
            "created_at": now_iso(),
            "git_commit": get_git_commit(Path(__file__).resolve().parent),
            "script_path": str(Path(__file__).resolve()),
            "source_dataset_path": str(source_dataset_path),
            "shifted_dataset_path": str(shifted_dataset_path),
            "factors_path": str(factors_path),
            "split_manifest_path": str(Path(args.split_manifest_path).resolve()) if args.split_manifest_path else None,
            "output_dir": str(output_dir),
            "output_manifest_csv": str(manifest_csv),
            "output_summary_csv": str(summary_csv),
            "target_column": args.target_column,
            "target_selection_query": target_query,
            "source_split": args.source_split,
            "active_join_cols": active_join_cols,
            "join_report": join_report,
            "k_target_neighbors": int(k),
            "neighbor_weight_mode": args.neighbor_weight_mode,
            "neighbor_temperature": float(args.neighbor_temperature),
            "n_rows": int(len(calibration_dataset)),
        },
    )

    print(f"Saved adapter-calibration dataset to: {output_dir}")
    print(f"Saved dataset manifest CSV to: {manifest_csv}")
    print(f"Saved summary CSV to: {summary_csv}")
    print(f"Saved run metadata to: {run_metadata_path}")


if __name__ == "__main__":
    main()
