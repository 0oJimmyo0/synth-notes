#!/usr/bin/env python3
"""
Build local steering directions for target-specific embedding shifts.

This Phase 2b utility fits target-aware direction vectors directly from the
real embedding space, starting from cluster-local targets such as
`cluster_target_29`. The saved direction bank can then be consumed by
`build_shifted_embedding_dataset.py` without introducing a separate
generation path.
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

from common import normalize_rows, parse_csv_list, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build local steering directions from a real embedding cohort.")
    parser.add_argument("--dataset_path", required=True, help="Source HF dataset path, e.g. encoded_training_filtered")
    parser.add_argument("--factors_path", required=True, help="Factor/metadata CSV carrying the target column")
    parser.add_argument("--output_dir", required=True, help="Output directory for the saved direction bank")
    parser.add_argument(
        "--target_columns",
        required=True,
        help="Comma-separated binary target columns, e.g. cluster_target_29",
    )
    parser.add_argument(
        "--methods",
        default="centroid_difference,one_vs_rest_linear",
        help="Comma-separated methods: centroid_difference, one_vs_rest_linear",
    )
    parser.add_argument(
        "--split_manifest_path",
        default=None,
        help="Optional split manifest to attach leakage/split metadata before filtering",
    )
    parser.add_argument(
        "--join_cols",
        default="source_row_id,embedding_row_id,dataset_row_id,note_id,subject_id,hadm_id",
        help="Preferred join columns for metadata merges",
    )
    parser.add_argument("--source_split", default=None, help="Optional split filter such as train/dev/test")
    parser.add_argument(
        "--selection_query",
        default=None,
        help="Optional pandas query applied after split filtering, before fitting directions",
    )
    parser.add_argument(
        "--max_rows_per_class",
        type=int,
        default=None,
        help="Optional cap per class after balancing the fit frame",
    )
    parser.add_argument(
        "--normalize_directions",
        action="store_true",
        help="L2-normalize learned directions before saving",
    )
    parser.add_argument(
        "--output_stem",
        default="local_direction_bank",
        help="Stem for output files inside output_dir",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for subsampling and linear fitting")
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


def extract_embedding_matrix(dataset: Dataset, row_ids: np.ndarray) -> np.ndarray:
    vectors = []
    for row_id in row_ids:
        emb = dataset[int(row_id)]["domain_embeddings"]
        if not isinstance(emb, list) or not emb:
            raise ValueError("Expected each dataset row to carry a non-empty domain_embeddings list.")
        vectors.append(np.asarray(emb[0], dtype=np.float32))
    return np.stack(vectors, axis=0)


def balanced_frame(df: pd.DataFrame, target_col: str, max_rows_per_class: int | None, random_state: int) -> pd.DataFrame:
    positive = df.loc[df[target_col] == 1].copy()
    negative = df.loc[df[target_col] == 0].copy()
    if positive.empty or negative.empty:
        raise ValueError(f"Target column '{target_col}' must contain both positive and negative rows.")

    target_n = min(len(positive), len(negative))
    if max_rows_per_class is not None:
        target_n = min(target_n, max_rows_per_class)

    positive = positive.sample(n=target_n, random_state=random_state, replace=False)
    negative = negative.sample(n=target_n, random_state=random_state, replace=False)
    return pd.concat([positive, negative], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def fit_centroid_difference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pos = x[y == 1]
    neg = x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Centroid difference requires both classes.")
    return pos.mean(axis=0) - neg.mean(axis=0)


def fit_one_vs_rest_linear(x: np.ndarray, y: np.ndarray, random_state: int) -> np.ndarray:
    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(x, y)
    return model.coef_.reshape(-1).astype(np.float32)


def main() -> None:
    args = build_parser().parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    factors_path = Path(args.factors_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_columns = parse_csv_list(args.target_columns)
    methods = parse_csv_list(args.methods)
    preferred_join_cols = parse_csv_list(args.join_cols)
    allowed_methods = {"centroid_difference", "one_vs_rest_linear"}

    if not target_columns:
        raise ValueError("--target_columns must specify at least one target")
    if not methods:
        raise ValueError("--methods must specify at least one method")
    unknown_methods = [method for method in methods if method not in allowed_methods]
    if unknown_methods:
        raise ValueError(f"Unsupported methods: {unknown_methods}. Allowed: {sorted(allowed_methods)}")

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

    if args.selection_query:
        merged_df = merged_df.query(args.selection_query, engine="python").copy()

    if merged_df.empty:
        raise ValueError("No source rows remained after filtering.")

    directions: list[np.ndarray] = []
    direction_manifest_rows: list[dict[str, Any]] = []
    fit_summaries: list[dict[str, Any]] = []

    for target_col in target_columns:
        if target_col not in merged_df.columns:
            raise KeyError(f"Target column not found after merge: {target_col}")
        numeric = pd.to_numeric(merged_df[target_col], errors="coerce")
        target_df = merged_df.loc[numeric.notna()].copy()
        target_df[target_col] = numeric.loc[target_df.index].astype(int)
        target_df = target_df.loc[target_df[target_col].isin([0, 1])].copy()
        target_df = balanced_frame(
            target_df,
            target_col=target_col,
            max_rows_per_class=args.max_rows_per_class,
            random_state=args.random_state,
        )
        row_ids = target_df["dataset_row_id"].astype(int).to_numpy()
        x = extract_embedding_matrix(dataset, row_ids)
        y = target_df[target_col].to_numpy(dtype=int)

        for method in methods:
            if method == "centroid_difference":
                vector = fit_centroid_difference(x, y)
            elif method == "one_vs_rest_linear":
                vector = fit_one_vs_rest_linear(x, y, random_state=args.random_state)
            else:
                raise AssertionError(f"Unhandled method: {method}")

            if args.normalize_directions:
                vector = normalize_rows(vector.reshape(1, -1))[0].astype(np.float32)
            else:
                vector = vector.astype(np.float32)

            direction_id = len(directions)
            label = f"local_{method}__{target_col}"
            direction_norm = float(np.linalg.norm(vector))
            directions.append(vector)
            direction_manifest_rows.append(
                {
                    "direction_id": direction_id,
                    "direction_label": label,
                    "target_column": target_col,
                    "method": method,
                    "n_rows_fit": int(len(target_df)),
                    "n_positive": int((y == 1).sum()),
                    "n_negative": int((y == 0).sum()),
                    "normalized_direction": bool(args.normalize_directions),
                    "direction_norm": direction_norm,
                }
            )
            fit_summaries.append(
                {
                    "direction_id": direction_id,
                    "direction_label": label,
                    "target_column": target_col,
                    "method": method,
                    "fit_split": args.source_split,
                    "selection_query": args.selection_query,
                    "n_rows_fit": int(len(target_df)),
                    "n_positive": int((y == 1).sum()),
                    "n_negative": int((y == 0).sum()),
                    "direction_norm": direction_norm,
                }
            )

    if not directions:
        raise ValueError("No directions were fit.")

    direction_matrix = np.stack(directions, axis=1).astype(np.float32)
    bank_path = output_dir / f"{args.output_stem}.npz"
    np.savez_compressed(
        bank_path,
        directions=direction_matrix,
        direction_ids=np.asarray([row["direction_id"] for row in direction_manifest_rows], dtype=np.int32),
        direction_labels=np.asarray([row["direction_label"] for row in direction_manifest_rows], dtype=object),
        target_columns=np.asarray([row["target_column"] for row in direction_manifest_rows], dtype=object),
        methods=np.asarray([row["method"] for row in direction_manifest_rows], dtype=object),
    )

    manifest_csv = output_dir / f"{args.output_stem}_manifest.csv"
    pd.DataFrame(direction_manifest_rows).to_csv(manifest_csv, index=False)

    summary_path = output_dir / f"{args.output_stem}_summary.json"
    save_json(
        summary_path,
        {
            "created_at": now_iso(),
            "git_commit": get_git_commit(Path(__file__).resolve().parent),
            "script_path": str(Path(__file__).resolve()),
            "dataset_path": str(dataset_path),
            "factors_path": str(factors_path),
            "output_dir": str(output_dir),
            "direction_bank_path": str(bank_path),
            "direction_manifest_csv": str(manifest_csv),
            "join_report": join_report,
            "active_join_cols": active_join_cols,
            "source_split": args.source_split,
            "selection_query": args.selection_query,
            "target_columns": target_columns,
            "methods": methods,
            "normalize_directions": bool(args.normalize_directions),
            "max_rows_per_class": args.max_rows_per_class,
            "random_state": args.random_state,
            "n_source_rows_after_filtering": int(len(merged_df)),
            "embedding_dim": int(direction_matrix.shape[0]),
            "n_directions": int(direction_matrix.shape[1]),
            "fit_summaries": fit_summaries,
            "cli_args": vars(args),
        },
    )

    print(f"Saved local direction bank to: {bank_path}")
    print(f"Saved direction manifest to: {manifest_csv}")
    print(f"Saved summary to: {summary_path}")
    print(f"Built {direction_matrix.shape[1]} directions with dimension {direction_matrix.shape[0]}")


if __name__ == "__main__":
    main()
