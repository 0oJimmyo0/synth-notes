#!/usr/bin/env python3
"""
Build a decoder-adaptation dataset from real filtered split datasets.

Goal:
- create basin-balanced real-note train/dev subsets
- oversample sparse target-basin clusters if requested
- save split-local HF datasets that can be reused by p1_train.py

This is intentionally simpler than the earlier shifted-embedding calibration path:
it does not require steering metadata and trains on real notes with their original
embeddings, so we can test whether decoder adaptation improves round-trip basin
retention for real target-basin embeddings.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


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


def maybe_int(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        return value


def load_dataset_with_metadata(dataset_path: Path) -> tuple[Dataset, pd.DataFrame]:
    dataset = Dataset.load_from_disk(str(dataset_path))
    df = pd.DataFrame({"dataset_row_id": np.arange(len(dataset), dtype=int)})
    metadata_cols = [c for c in dataset.column_names if c not in {"input_ids", "domain_embeddings"}]
    if metadata_cols:
        df = pd.concat([df, dataset.select_columns(metadata_cols).to_pandas().reset_index(drop=True)], axis=1)
    return dataset, df


def reduce_assignment_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "source_row_id",
        "embedding_row_id",
        "split",
        "dataset_row_id",
        "note_id",
        "subject_id",
        "hadm_id",
        "patient_disjoint_from_train",
        "hadm_disjoint_from_train",
        "note_disjoint_from_train",
        "patient_overlap_with_train",
        "hadm_overlap_with_train",
        "note_overlap_with_train",
        "cluster_id",
        "distance_to_centroid",
    ]
    present = [c for c in keep_cols if c in df.columns]
    return df[present].copy()


def reduce_split_manifest_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "source_row_id",
        "embedding_row_id",
        "split",
        "dataset_row_id",
        "note_id",
        "subject_id",
        "hadm_id",
        "patient_disjoint_from_train",
        "hadm_disjoint_from_train",
        "note_disjoint_from_train",
        "patient_overlap_with_train",
        "hadm_overlap_with_train",
        "note_overlap_with_train",
    ]
    present = [c for c in keep_cols if c in df.columns]
    return df[present].copy()


def choose_common_clusters_auto(
    assignments_df: pd.DataFrame,
    cluster_summary_path: Path,
    target_cluster_ids: list[int],
    n_common_clusters: int,
    split_name: str,
) -> list[int]:
    summary_df = pd.read_csv(cluster_summary_path)
    summary_df["cluster_id"] = pd.to_numeric(summary_df["cluster_id"], errors="coerce")
    if "split" in summary_df.columns:
        summary_df = summary_df.loc[summary_df["split"].astype(str) == str(split_name)].copy()
    size_col = "cluster_size"
    if size_col not in summary_df.columns:
        alt_cols = ["cluster_size_split", "real_count"]
        for alt in alt_cols:
            if alt in summary_df.columns:
                size_col = alt
                break
    summary_df[size_col] = pd.to_numeric(summary_df[size_col], errors="coerce")
    summary_df = summary_df.dropna(subset=["cluster_id", size_col]).copy()
    summary_df["cluster_id"] = summary_df["cluster_id"].astype(int)
    summary_df = summary_df.loc[~summary_df["cluster_id"].isin(set(target_cluster_ids))].copy()
    summary_df = summary_df.sort_values(size_col, ascending=False).head(int(n_common_clusters)).copy()
    common_ids = summary_df["cluster_id"].tolist()
    if not common_ids:
        fallback = (
            assignments_df.loc[~assignments_df["cluster_id"].isin(set(target_cluster_ids)), "cluster_id"]
            .value_counts()
            .head(int(n_common_clusters))
            .index.tolist()
        )
        common_ids = [int(x) for x in fallback]
    if not common_ids:
        raise ValueError("Could not infer common clusters automatically.")
    return common_ids


def sample_rows(df: pd.DataFrame, n_rows: int, rng: random.Random) -> pd.DataFrame:
    if len(df) <= n_rows:
        return df.copy()
    idx = list(df.index)
    rng.shuffle(idx)
    return df.loc[idx[:n_rows]].copy()


def build_split(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset_path).resolve()
    cluster_assignments_path = Path(args.cluster_assignments_path).resolve()
    cluster_summary_path = Path(args.cluster_summary_path).resolve()
    split_manifest_path = Path(args.split_manifest_path).resolve() if args.split_manifest_path else None
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    split_name = str(args.source_split)
    output_dataset_stem = "encoded_training" if split_name == "train" else f"encoded_{split_name}"
    output_dataset_path = output_root / output_dataset_stem
    output_manifest_path = output_root / f"decoder_adaptation_{split_name}_manifest.csv"
    output_summary_path = output_root / f"decoder_adaptation_{split_name}_summary.json"

    target_cluster_ids = parse_int_list(args.target_cluster_ids)
    if not target_cluster_ids:
        raise ValueError("--target_cluster_ids must not be empty")
    preferred_join_cols = parse_csv_list(args.join_cols)

    dataset, base_df = load_dataset_with_metadata(dataset_path)
    base_df["split"] = split_name

    assignments_df = reduce_assignment_columns(pd.read_csv(cluster_assignments_path))
    if "split" in assignments_df.columns:
        assignments_df = assignments_df.loc[assignments_df["split"].astype(str) == split_name].copy()

    join_cols = choose_join_keys([base_df, assignments_df], preferred_join_cols)
    base_df = normalize_join_cols(base_df, join_cols)
    assignments_df = normalize_join_cols(assignments_df, join_cols).drop_duplicates(subset=join_cols)
    merged = base_df.merge(assignments_df, on=join_cols, how="left", validate="one_to_one", suffixes=("", "_assign"))

    if split_manifest_path:
        split_df = reduce_split_manifest_columns(pd.read_csv(split_manifest_path))
        if "split" in split_df.columns:
            split_df = split_df.loc[split_df["split"].astype(str) == split_name].copy()
        split_join_cols = choose_join_keys([merged, split_df], preferred_join_cols)
        merged = normalize_join_cols(merged, split_join_cols)
        split_df = normalize_join_cols(split_df, split_join_cols).drop_duplicates(subset=split_join_cols)
        merged = merged.merge(split_df, on=split_join_cols, how="left", validate="one_to_one", suffixes=("", "_split"))

    if "cluster_id" not in merged.columns:
        raise KeyError("Merged metadata must contain cluster_id")
    merged["cluster_id"] = pd.to_numeric(merged["cluster_id"], errors="coerce")
    merged = merged.dropna(subset=["cluster_id"]).copy()
    merged["cluster_id"] = merged["cluster_id"].astype(int)

    common_cluster_ids = parse_int_list(args.common_cluster_ids)
    if not common_cluster_ids:
        common_cluster_ids = choose_common_clusters_auto(
            assignments_df=merged,
            cluster_summary_path=cluster_summary_path,
            target_cluster_ids=target_cluster_ids,
            n_common_clusters=int(args.n_common_clusters),
            split_name=split_name,
        )

    rng = random.Random(int(args.seed))
    selected_frames: list[pd.DataFrame] = []

    for cid in target_cluster_ids:
        cluster_df = merged.loc[merged["cluster_id"] == cid].copy()
        cluster_df = sample_rows(cluster_df, int(args.rows_per_target_cluster), rng)
        cluster_df["decoder_group_family"] = "target_basin"
        cluster_df["decoder_group_label"] = f"target_cluster_{cid}"
        cluster_df["replication_factor"] = int(args.target_repeat_factor)
        selected_frames.append(cluster_df)

    for cid in common_cluster_ids:
        cluster_df = merged.loc[merged["cluster_id"] == cid].copy()
        cluster_df = sample_rows(cluster_df, int(args.rows_per_common_cluster), rng)
        cluster_df["decoder_group_family"] = "common_cluster"
        cluster_df["decoder_group_label"] = f"common_cluster_{cid}"
        cluster_df["replication_factor"] = int(args.common_repeat_factor)
        selected_frames.append(cluster_df)

    selection_df = pd.concat(selected_frames, ignore_index=True)
    selection_df["replication_factor"] = pd.to_numeric(selection_df["replication_factor"], errors="coerce").fillna(1).astype(int)
    selection_df["replication_factor"] = selection_df["replication_factor"].clip(lower=1)
    selection_df["target_basin_cluster_ids"] = ",".join(str(x) for x in target_cluster_ids)
    selection_df["common_cluster_ids"] = ",".join(str(x) for x in common_cluster_ids)

    output_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    selected_ids = selection_df["dataset_row_id"].astype(int).tolist()
    selected_examples = dataset.select(selected_ids)

    for row, example in zip(selection_df.to_dict(orient="records"), selected_examples):
        rep_factor = int(row["replication_factor"])
        for rep_idx in range(rep_factor):
            out_row = dict(example)
            out_row["dataset_row_id"] = maybe_int(row.get("dataset_row_id"))
            out_row["note_id"] = row.get("note_id")
            out_row["subject_id"] = maybe_int(row.get("subject_id"))
            out_row["hadm_id"] = maybe_int(row.get("hadm_id"))
            out_row["split"] = split_name
            out_row["cluster_id"] = int(row["cluster_id"])
            out_row["decoder_group_family"] = row["decoder_group_family"]
            out_row["decoder_group_label"] = row["decoder_group_label"]
            out_row["patient_disjoint_from_train"] = row.get("patient_disjoint_from_train")
            out_row["source_repeat_index"] = rep_idx
            output_rows.append(out_row)
            manifest_rows.append(
                {
                    "output_row_id": len(manifest_rows),
                    "dataset_row_id": out_row["dataset_row_id"],
                    "note_id": out_row["note_id"],
                    "subject_id": out_row["subject_id"],
                    "hadm_id": out_row["hadm_id"],
                    "split": split_name,
                    "cluster_id": out_row["cluster_id"],
                    "decoder_group_family": out_row["decoder_group_family"],
                    "decoder_group_label": out_row["decoder_group_label"],
                    "patient_disjoint_from_train": out_row["patient_disjoint_from_train"],
                    "source_repeat_index": rep_idx,
                }
            )

    out_ds = Dataset.from_list(output_rows)
    out_ds.save_to_disk(str(output_dataset_path))

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output_manifest_path, index=False)

    summary = {
        "created_at": now_iso(),
        "script_path": str(Path(__file__).resolve()),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "dataset_path": str(dataset_path),
        "cluster_assignments_path": str(cluster_assignments_path),
        "cluster_summary_path": str(cluster_summary_path),
        "split_manifest_path": str(split_manifest_path) if split_manifest_path else None,
        "output_root": str(output_root),
        "output_dataset_path": str(output_dataset_path),
        "output_manifest_path": str(output_manifest_path),
        "source_split": split_name,
        "target_cluster_ids": target_cluster_ids,
        "common_cluster_ids": common_cluster_ids,
        "n_unique_source_rows": int(selection_df["dataset_row_id"].nunique()),
        "n_output_rows": int(len(output_rows)),
        "group_counts_unique": {
            str(k): int(v)
            for k, v in selection_df["decoder_group_label"].value_counts().sort_index().to_dict().items()
        },
        "group_counts_output_rows": {
            str(k): int(v)
            for k, v in manifest_df["decoder_group_label"].value_counts().sort_index().to_dict().items()
        },
        "cli_args": vars(args),
    }
    save_json(output_summary_path, summary)

    print(f"Saved decoder-adaptation dataset to: {output_dataset_path}")
    print(f"Saved manifest to: {output_manifest_path}")
    print(json.dumps(summary["group_counts_output_rows"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build basin-balanced real-note datasets for decoder adaptation.")
    parser.add_argument("--dataset_path", required=True, help="Filtered split HF dataset path, e.g. encoded_training_filtered")
    parser.add_argument("--cluster_assignments_path", required=True, help="Whole filtered cluster assignment CSV")
    parser.add_argument("--cluster_summary_path", required=True, help="Whole filtered cluster summary CSV")
    parser.add_argument("--split_manifest_path", default=None, help="Optional filtered-aligned split manifest")
    parser.add_argument("--output_root", required=True, help="Output root that will contain encoded_<split>")
    parser.add_argument("--source_split", required=True, choices=["train", "dev", "test"])
    parser.add_argument("--target_cluster_ids", required=True, help="Comma-separated target basin cluster IDs")
    parser.add_argument("--common_cluster_ids", default="", help="Optional explicit common cluster IDs")
    parser.add_argument("--n_common_clusters", type=int, default=3)
    parser.add_argument("--rows_per_target_cluster", type=int, default=1024)
    parser.add_argument("--rows_per_common_cluster", type=int, default=1024)
    parser.add_argument("--target_repeat_factor", type=int, default=2)
    parser.add_argument("--common_repeat_factor", type=int, default=1)
    parser.add_argument(
        "--join_cols",
        default="source_row_id,embedding_row_id,dataset_row_id,note_id,subject_id,hadm_id",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    build_split(args)


if __name__ == "__main__":
    main()
