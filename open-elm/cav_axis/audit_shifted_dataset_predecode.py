#!/usr/bin/env python3
"""
Audit a shifted HF dataset before note generation.

This script is intentionally lightweight: it operates directly on the saved
shifted dataset and compares its embeddings against the real target region in
embedding space, so we can reject weak steering candidates before running ELM
generation.
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
    parser = argparse.ArgumentParser(description="Audit shifted dataset geometry before decode.")
    parser.add_argument("--source_dataset_path", required=True, help="Original HF source dataset, e.g. encoded_testing_filtered")
    parser.add_argument("--shifted_dataset_path", required=True, help="Shifted HF dataset to audit")
    parser.add_argument("--factors_path", required=True, help="Factor table containing the target column")
    parser.add_argument("--target_column", required=True, help="Binary target column, e.g. cluster_target_29")
    parser.add_argument("--output_dir", required=True, help="Directory for audit outputs")
    parser.add_argument("--split_manifest_path", default=None, help="Optional filtered-aligned split manifest")
    parser.add_argument(
        "--join_cols",
        default="source_row_id,embedding_row_id,dataset_row_id,note_id,subject_id,hadm_id",
        help="Preferred join columns for metadata merges",
    )
    parser.add_argument("--source_split", default=None, help="Optional split filter such as test/dev/train")
    parser.add_argument(
        "--target_selection_query",
        default=None,
        help="Optional query for the real target pool. Defaults to `<target_column> == 1`.",
    )
    parser.add_argument(
        "--neighborhood_k",
        type=int,
        default=5,
        help="k used to estimate target-neighborhood radius from the real target pool",
    )
    return parser


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    def _json_default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


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


def load_dataset_rows(dataset_path: Path) -> tuple[Dataset, pd.DataFrame, np.ndarray]:
    dataset = Dataset.load_from_disk(str(dataset_path))
    metadata_cols = [col for col in dataset.column_names if col not in {"input_ids", "domain_embeddings"}]
    if metadata_cols:
        base_df = dataset.select_columns(metadata_cols).to_pandas().reset_index(drop=True)
    else:
        base_df = pd.DataFrame(index=np.arange(len(dataset), dtype=int))
    if "dataset_row_id" not in base_df.columns:
        base_df.insert(0, "dataset_row_id", np.arange(len(dataset), dtype=int))

    embeddings = []
    for emb in dataset["domain_embeddings"]:
        if not isinstance(emb, list) or not emb:
            raise ValueError("Expected each dataset row to carry a non-empty domain_embeddings list.")
        first = emb[0]
        arr = np.asarray(first, dtype=np.float32)
        while arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise ValueError(f"Expected each domain embedding to resolve to 1D, got shape {arr.shape}")
        embeddings.append(arr)
    return dataset, base_df, normalize_rows(np.vstack(embeddings))


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
) -> pd.DataFrame:
    merged = base_df.copy()
    if split_manifest_path:
        split_df = pd.read_csv(split_manifest_path)
        join_cols = choose_join_keys([merged, split_df], preferred_join_cols)
        if join_cols == ["dataset_row_id"] and "split" in split_df.columns and source_split:
            split_df = split_df.loc[split_df["split"].astype(str) == str(source_split)].copy()
        merged = normalize_join_cols(merged, join_cols)
        split_df = normalize_join_cols(split_df, join_cols)
        split_df = split_df.drop_duplicates(subset=join_cols)
        merged = merged.merge(split_df, on=join_cols, how="left", validate="one_to_one", suffixes=("", "_split"))

    factors_df = pd.read_csv(factors_path)
    join_cols = choose_join_keys([merged, factors_df], preferred_join_cols)
    merged = normalize_join_cols(merged, join_cols)
    factors_df = normalize_join_cols(factors_df, join_cols)
    factors_df = factors_df.drop_duplicates(subset=join_cols)
    merged = merged.merge(factors_df, on=join_cols, how="left", validate="one_to_one", suffixes=("", "_factor"))
    return merged


def pairwise_top1_cosine(queries: np.ndarray, targets: np.ndarray, batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    all_scores = []
    all_indices = []
    for start in range(0, queries.shape[0], batch_size):
        stop = min(start + batch_size, queries.shape[0])
        sims = queries[start:stop] @ targets.T
        idx = np.argmax(sims, axis=1)
        scores = sims[np.arange(stop - start), idx]
        all_scores.append(scores.astype(np.float32))
        all_indices.append(idx.astype(np.int32))
    return np.concatenate(all_scores), np.concatenate(all_indices)


def target_radius_from_pool(target_embeddings: np.ndarray, k: int) -> float:
    if len(target_embeddings) <= 1:
        return 0.0
    k = min(max(k, 1), len(target_embeddings) - 1)
    sims = target_embeddings @ target_embeddings.T
    np.fill_diagonal(sims, -np.inf)
    topk = np.partition(-sims, kth=k - 1, axis=1)[:, :k]
    kth_best = -topk.min(axis=1)
    distances = 1.0 - kth_best
    return float(np.median(distances))


def main() -> None:
    args = build_parser().parse_args()

    source_dataset_path = Path(args.source_dataset_path).resolve()
    shifted_dataset_path = Path(args.shifted_dataset_path).resolve()
    factors_path = Path(args.factors_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preferred_join_cols = parse_csv_list(args.join_cols)
    source_ds, source_df, source_embeddings = load_dataset_rows(source_dataset_path)
    _, shifted_df, shifted_embeddings = load_dataset_rows(shifted_dataset_path)

    if args.source_split and "split" not in source_df.columns:
        source_df["split"] = args.source_split

    source_meta = merge_metadata(source_df, factors_path, args.split_manifest_path, preferred_join_cols, args.source_split)
    if args.source_split and "split" in source_meta.columns:
        source_meta = source_meta.loc[source_meta["split"] == args.source_split].copy()

    if args.target_column not in source_meta.columns:
        raise KeyError(f"Target column not found after merge: {args.target_column}")
    numeric = pd.to_numeric(source_meta[args.target_column], errors="coerce")
    source_meta = source_meta.loc[numeric.notna()].copy()
    source_meta[args.target_column] = numeric.loc[source_meta.index].astype(int)
    source_meta = source_meta.loc[source_meta[args.target_column].isin([0, 1])].copy()

    target_query = args.target_selection_query or f"{args.target_column} == 1"
    target_meta = source_meta.query(target_query, engine="python").copy()
    if target_meta.empty:
        raise ValueError("No real target rows found for the requested target selection.")

    target_row_ids = target_meta["dataset_row_id"].astype(int).to_numpy()
    target_embeddings = source_embeddings[target_row_ids]
    target_centroid = normalize_rows(target_embeddings.mean(axis=0, keepdims=True))[0]
    radius = target_radius_from_pool(target_embeddings, args.neighborhood_k)

    nearest_target_cosine, nearest_target_idx = pairwise_top1_cosine(shifted_embeddings, target_embeddings)
    nearest_target_distance = 1.0 - nearest_target_cosine
    source_row_ids = shifted_df["dataset_row_id"].astype(int).to_numpy()
    source_cosine = np.sum(shifted_embeddings * source_embeddings[source_row_ids], axis=1)
    centroid_cosine = shifted_embeddings @ target_centroid

    out_df = shifted_df.copy()
    out_df["nearest_target_dataset_row_id"] = target_row_ids[nearest_target_idx]
    out_df["nearest_target_cosine"] = nearest_target_cosine.astype(np.float32)
    out_df["nearest_target_distance"] = nearest_target_distance.astype(np.float32)
    out_df["target_centroid_cosine"] = centroid_cosine.astype(np.float32)
    out_df["source_cosine_recomputed"] = source_cosine.astype(np.float32)
    out_df["in_target_neighborhood"] = (nearest_target_distance <= radius).astype(int)

    group_cols = ["alpha", "margin_gamma"] if {"alpha", "margin_gamma"}.issubset(out_df.columns) else (["alpha"] if "alpha" in out_df.columns else ["axis_label"])
    summary_df = (
        out_df.groupby(group_cols, dropna=False)
        .agg(
            n_rows=("dataset_row_id", "size"),
            mean_source_cosine=("source_cosine_recomputed", "mean"),
            median_source_cosine=("source_cosine_recomputed", "median"),
            mean_target_centroid_cosine=("target_centroid_cosine", "mean"),
            median_target_centroid_cosine=("target_centroid_cosine", "median"),
            mean_nearest_target_cosine=("nearest_target_cosine", "mean"),
            median_nearest_target_cosine=("nearest_target_cosine", "median"),
            target_neighborhood_entry_rate=("in_target_neighborhood", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )

    best_idx = summary_df["target_neighborhood_entry_rate"].astype(float).idxmax()
    best_row = summary_df.loc[best_idx].to_dict()

    out_df.to_csv(output_dir / "shifted_predecode_row_table.csv", index=False)
    summary_df.to_csv(output_dir / "shifted_predecode_summary.csv", index=False)
    payload = {
        "created_at": now_iso(),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "script_path": str(Path(__file__).resolve()),
        "source_dataset_path": str(source_dataset_path),
        "shifted_dataset_path": str(shifted_dataset_path),
        "factors_path": str(factors_path),
        "split_manifest_path": str(Path(args.split_manifest_path).resolve()) if args.split_manifest_path else None,
        "target_column": args.target_column,
        "target_selection_query": target_query,
        "source_split": args.source_split,
        "target_pool_size": int(len(target_meta)),
        "target_neighborhood_radius": float(radius),
        "group_cols": group_cols,
        "best_group_by_entry_rate": best_row,
        "n_shifted_rows": int(len(out_df)),
    }
    save_json(output_dir / "shifted_predecode_summary.json", payload)

    md_lines = [
        "# Shifted Pre-decode Audit",
        "",
        f"- target_column: `{args.target_column}`",
        f"- target_selection_query: `{target_query}`",
        f"- target_pool_size: `{len(target_meta)}`",
        f"- target_neighborhood_radius: `{radius:.6f}`",
        "",
        "## Best Group",
        "",
        f"- group_cols: `{', '.join(group_cols)}`",
        f"- target_neighborhood_entry_rate: `{best_row['target_neighborhood_entry_rate']:.6f}`",
        f"- mean_nearest_target_cosine: `{best_row['mean_nearest_target_cosine']:.6f}`",
        f"- mean_source_cosine: `{best_row['mean_source_cosine']:.6f}`",
        "",
        "## Files",
        "",
        "- `shifted_predecode_summary.csv`",
        "- `shifted_predecode_row_table.csv`",
    ]
    for col in reversed(group_cols):
        md_lines.insert(9, f"- {col}: `{best_row[col]}`")
    (output_dir / "shifted_predecode_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Saved pre-decode summary to: {output_dir / 'shifted_predecode_summary.csv'}")
    print(f"Saved row-level table to: {output_dir / 'shifted_predecode_row_table.csv'}")
    print(
        "Best group: "
        + ", ".join(f"{col}={best_row[col]}" for col in group_cols)
        + f" with entry rate {best_row['target_neighborhood_entry_rate']:.6f}"
    )


if __name__ == "__main__":
    main()
