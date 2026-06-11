#!/usr/bin/env python3
"""
Audit split leakage for the MIMIC-IV note/HADM pipeline.

This script now produces two related provenance artifacts:

1. A whole-cohort master manifest covering every embedding row and its assigned split.
2. A filtered-aligned split manifest matching the actual HF datasets used by ELM
   training and generation after long-sequence filtering.

The filtered-aligned manifest is the canonical downstream artifact for vanilla
generation, manifest writing, and audit. The master manifest remains stable as the
full source-of-truth for the embedding cohort and future generation conditions
(vanilla, CAV, editor-assisted variants, etc.).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split


DEFAULT_METADATA = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "embeddings-BAAI-bge-large-en-v1.5/sentence_embeddings_metadata.csv"
)
DEFAULT_DATASET_BASE = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task"
)
DEFAULT_OUTPUT_DIR = f"{DEFAULT_DATASET_BASE}/leakage_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit split leakage and build full + filtered manifests for the note/HADM-aligned MIMIC-IV cohort."
    )
    parser.add_argument(
        "--metadata-path",
        default=DEFAULT_METADATA,
        help="Path to sentence_embeddings_metadata.csv",
    )
    parser.add_argument(
        "--dataset-base-dir",
        default=DEFAULT_DATASET_BASE,
        help="Base directory containing encoded_*_full and encoded_*_filtered datasets",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save manifests and overlap summaries",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Dev split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed used for splitting")
    parser.add_argument(
        "--max-length",
        type=int,
        default=7148,
        help="Sequence length threshold previously used by filter_long_sequences.py for the saved filtered datasets",
    )
    return parser.parse_args()


def clean_ids(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    )


def assign_full_splits(
    metadata_df: pd.DataFrame,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    metadata_df = metadata_df.reset_index(drop=True).copy()
    metadata_df["embedding_row_id"] = metadata_df.index.astype(int)

    train_df, temp_df = train_test_split(
        metadata_df,
        test_size=(dev_ratio + test_ratio),
        random_state=random_seed,
        shuffle=True,
    )

    dev_fraction_of_temp = dev_ratio / (dev_ratio + test_ratio)
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - dev_fraction_of_temp),
        random_state=random_seed,
        shuffle=True,
    )

    split_frames: dict[str, pd.DataFrame] = {}
    for split_name, split_df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        split_df = split_df.copy().reset_index(drop=True)
        split_df["split"] = split_name
        split_df["dataset_row_id_full"] = range(len(split_df))
        split_df["source_row_id"] = split_df["embedding_row_id"]
        split_frames[split_name] = split_df

    master_df = pd.concat([split_frames["train"], split_frames["dev"], split_frames["test"]], ignore_index=True)
    return master_df, split_frames


def actual_training_length(example: dict[str, Any], gen_token_id: int = 128003) -> int | None:
    input_ids = example.get("input_ids")
    if input_ids is None:
        return None
    if not isinstance(input_ids, list):
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        else:
            return None
    try:
        input_ids.index(gen_token_id)
    except ValueError:
        return None
    return len(input_ids) - 1


def infer_kept_indices_from_length(
    full_dataset_path: Path,
    filtered_dataset_path: Path,
    max_length: int,
) -> list[int]:
    full_ds = Dataset.load_from_disk(str(full_dataset_path))
    filtered_ds = Dataset.load_from_disk(str(filtered_dataset_path))

    kept_indices: list[int] = []
    for idx, example in enumerate(full_ds):
        actual_len = actual_training_length(example)
        if actual_len is not None and actual_len <= max_length:
            kept_indices.append(idx)

    if len(kept_indices) != len(filtered_ds):
        raise ValueError(
            f"Inferred {len(kept_indices)} kept rows from '{full_dataset_path}', but "
            f"filtered dataset '{filtered_dataset_path}' has {len(filtered_ds)} rows."
        )

    if kept_indices:
        check_positions = sorted(set([0, len(kept_indices) // 2, len(kept_indices) - 1]))
        for filtered_pos in check_positions:
            full_example = full_ds[kept_indices[filtered_pos]]
            filtered_example = filtered_ds[filtered_pos]
            if full_example["input_ids"] != filtered_example["input_ids"]:
                raise ValueError(
                    f"Filtered dataset order mismatch at position {filtered_pos} between "
                    f"'{full_dataset_path}' and '{filtered_dataset_path}'."
                )

    return kept_indices


def build_filtered_split_frames(
    split_frames_full: dict[str, pd.DataFrame],
    dataset_base_dir: Path,
    max_length: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    filtered_frames: dict[str, pd.DataFrame] = {}
    removed_rows: list[pd.DataFrame] = []
    split_to_dataset_stem = {"train": "training", "dev": "dev", "test": "testing"}

    for split_name, full_df in split_frames_full.items():
        dataset_stem = split_to_dataset_stem[split_name]
        full_path = dataset_base_dir / f"encoded_{dataset_stem}_full"
        filtered_path = dataset_base_dir / f"encoded_{dataset_stem}_filtered"

        if not full_path.exists():
            raise FileNotFoundError(f"Full dataset not found for split '{split_name}': {full_path}")
        if not filtered_path.exists():
            raise FileNotFoundError(f"Filtered dataset not found for split '{split_name}': {filtered_path}")

        kept_full_indices = infer_kept_indices_from_length(full_path, filtered_path, max_length=max_length)
        kept_mask = pd.Series(False, index=full_df.index)
        kept_mask.iloc[kept_full_indices] = True

        filtered_df = full_df.iloc[kept_full_indices].copy().reset_index(drop=True)
        filtered_df["dataset_row_id"] = range(len(filtered_df))
        filtered_df["kept_after_filter"] = True
        filtered_frames[split_name] = filtered_df

        dropped_df = full_df.loc[~kept_mask].copy().reset_index(drop=True)
        if not dropped_df.empty:
            dropped_df["dataset_row_id"] = pd.NA
            dropped_df["kept_after_filter"] = False
            removed_rows.append(dropped_df)

    removed_df = pd.concat(removed_rows, ignore_index=True) if removed_rows else pd.DataFrame()
    return filtered_frames, removed_df


def overlap_record(
    split_a: str,
    split_b: str,
    key_name: str,
    key_series_a: pd.Series,
    key_series_b: pd.Series,
) -> dict[str, object]:
    set_a = set(key_series_a.dropna().tolist())
    set_b = set(key_series_b.dropna().tolist())
    overlap = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a
    return {
        "split_a": split_a,
        "split_b": split_b,
        "key": key_name,
        "unique_a": len(set_a),
        "unique_b": len(set_b),
        "overlap_count": len(overlap),
        "overlap_pct_of_a": (len(overlap) / len(set_a)) if set_a else 0.0,
        "overlap_pct_of_b": (len(overlap) / len(set_b)) if set_b else 0.0,
        "a_only_count": len(only_a),
        "b_only_count": len(only_b),
    }


def compute_overlap_summary(split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    overlap_rows: list[dict[str, object]] = []
    for split_a, split_b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        df_a = split_frames[split_a]
        df_b = split_frames[split_b]
        for key in ["note_id", "hadm_id", "subject_id"]:
            overlap_rows.append(
                overlap_record(
                    split_a=split_a,
                    split_b=split_b,
                    key_name=key,
                    key_series_a=df_a[key],
                    key_series_b=df_b[key],
                )
            )
    return pd.DataFrame(overlap_rows)


def add_overlap_flags(manifest_df: pd.DataFrame, train_reference_df: pd.DataFrame) -> pd.DataFrame:
    df = manifest_df.copy()
    train_subjects = set(df.loc[df["split"] == "train", "subject_id"].dropna().tolist())
    train_hadms = set(df.loc[df["split"] == "train", "hadm_id"].dropna().tolist())
    train_notes = set(df.loc[df["split"] == "train", "note_id"].dropna().tolist())

    if not train_reference_df.empty:
        train_subjects = set(train_reference_df["subject_id"].dropna().tolist())
        train_hadms = set(train_reference_df["hadm_id"].dropna().tolist())
        train_notes = set(train_reference_df["note_id"].dropna().tolist())

    df["patient_overlap_with_train"] = df["subject_id"].isin(train_subjects)
    df["hadm_overlap_with_train"] = df["hadm_id"].isin(train_hadms)
    df["note_overlap_with_train"] = df["note_id"].isin(train_notes)
    df["patient_disjoint_from_train"] = ~df["patient_overlap_with_train"]
    df["hadm_disjoint_from_train"] = ~df["hadm_overlap_with_train"]
    df["note_disjoint_from_train"] = ~df["note_overlap_with_train"]
    return df


def summarize_split_rows(split_df: pd.DataFrame, train_subjects: set[str], train_hadms: set[str]) -> dict[str, object]:
    subject_ids = clean_ids(split_df["subject_id"])
    hadm_ids = clean_ids(split_df["hadm_id"])
    note_ids = clean_ids(split_df["note_id"])
    row_subject_overlap = subject_ids.isin(train_subjects)
    row_hadm_overlap = hadm_ids.isin(train_hadms)
    return {
        "rows": int(len(split_df)),
        "unique_note_ids": int(note_ids.dropna().nunique()),
        "unique_hadm_ids": int(hadm_ids.dropna().nunique()),
        "unique_subject_ids": int(subject_ids.dropna().nunique()),
        "rows_with_subject_overlap_to_train": int(row_subject_overlap.sum()),
        "rows_with_hadm_overlap_to_train": int(row_hadm_overlap.sum()),
        "rows_patient_disjoint_from_train": int((~row_subject_overlap).sum()),
        "rows_hadm_disjoint_from_train": int((~row_hadm_overlap).sum()),
        "pct_patient_disjoint_from_train": float((~row_subject_overlap).mean()),
        "pct_hadm_disjoint_from_train": float((~row_hadm_overlap).mean()),
    }


def canonical_columns(include_filtered_ids: bool = True) -> list[str]:
    cols = [
        "source_row_id",
        "embedding_row_id",
        "split",
        "dataset_row_id_full",
    ]
    if include_filtered_ids:
        cols.extend(["dataset_row_id", "kept_after_filter"])
    cols.extend(
        [
            "filename",
            "note_id",
            "subject_id",
            "hadm_id",
            "note_type",
            "charttime",
            "text_length",
            "text_preview",
            "patient_overlap_with_train",
            "hadm_overlap_with_train",
            "note_overlap_with_train",
            "patient_disjoint_from_train",
            "hadm_disjoint_from_train",
            "note_disjoint_from_train",
        ]
    )
    return cols


def main() -> None:
    args = parse_args()

    metadata_path = Path(args.metadata_path)
    dataset_base_dir = Path(args.dataset_base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not dataset_base_dir.exists():
        raise FileNotFoundError(f"Dataset base directory not found: {dataset_base_dir}")

    metadata_df = pd.read_csv(metadata_path, dtype=str)
    required_cols = ["note_id", "subject_id", "hadm_id", "filename"]
    missing = [col for col in required_cols if col not in metadata_df.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")

    master_full_df, split_frames_full = assign_full_splits(
        metadata_df=metadata_df,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
    )

    for col in ["note_id", "subject_id", "hadm_id"]:
        master_full_df[col] = clean_ids(master_full_df[col])
        for split_name in split_frames_full:
            split_frames_full[split_name][col] = clean_ids(split_frames_full[split_name][col])

    filtered_frames, removed_df = build_filtered_split_frames(
        split_frames_full,
        dataset_base_dir,
        max_length=args.max_length,
    )
    filtered_manifest = pd.concat(
        [filtered_frames["train"], filtered_frames["dev"], filtered_frames["test"]],
        ignore_index=True,
    )
    filtered_manifest = add_overlap_flags(filtered_manifest, filtered_frames["train"])

    train_subjects_filtered = set(filtered_frames["train"]["subject_id"].dropna().tolist())
    train_hadms_filtered = set(filtered_frames["train"]["hadm_id"].dropna().tolist())
    filtered_stats = {
        split_name: summarize_split_rows(split_df, train_subjects_filtered, train_hadms_filtered)
        for split_name, split_df in filtered_frames.items()
    }

    master_full_df["dataset_row_id"] = pd.NA
    master_full_df["kept_after_filter"] = False
    filtered_lookup = filtered_manifest.loc[
        :,
        ["source_row_id", "dataset_row_id", "kept_after_filter"],
    ].copy()
    master_full_df = master_full_df.merge(filtered_lookup, on="source_row_id", how="left", suffixes=("", "_filtered"))
    master_full_df["dataset_row_id"] = master_full_df["dataset_row_id_filtered"].combine_first(master_full_df["dataset_row_id"])
    master_full_df["kept_after_filter"] = master_full_df["kept_after_filter_filtered"].fillna(master_full_df["kept_after_filter"])
    master_full_df = master_full_df.drop(columns=["dataset_row_id_filtered", "kept_after_filter_filtered"])
    master_full_df = add_overlap_flags(master_full_df, filtered_frames["train"])

    filtered_overlap_df = compute_overlap_summary(filtered_frames)
    full_overlap_df = compute_overlap_summary(split_frames_full)

    filtered_manifest = filtered_manifest.loc[:, canonical_columns(include_filtered_ids=True)]
    master_full_df = master_full_df.loc[:, canonical_columns(include_filtered_ids=True)]
    if not removed_df.empty:
        removed_df = add_overlap_flags(removed_df, filtered_frames["train"])
        removed_df = removed_df.loc[:, canonical_columns(include_filtered_ids=True)]

    split_manifest_path = output_dir / "split_manifest_note_level.csv"
    split_manifest_full_path = output_dir / "split_manifest_note_level_full.csv"
    removed_manifest_path = output_dir / "split_manifest_removed_by_filter.csv"
    filtered_overlap_path = output_dir / "split_overlap_summary.csv"
    full_overlap_path = output_dir / "split_overlap_summary_full.csv"
    summary_path = output_dir / "split_leakage_audit_summary.json"

    filtered_manifest.to_csv(split_manifest_path, index=False)
    master_full_df.to_csv(split_manifest_full_path, index=False)
    filtered_overlap_df.to_csv(filtered_overlap_path, index=False)
    full_overlap_df.to_csv(full_overlap_path, index=False)
    if not removed_df.empty:
        removed_df.to_csv(removed_manifest_path, index=False)

    full_sizes = {name: int(len(split_frames_full[name])) for name in ["train", "dev", "test"]}
    filtered_sizes = {name: int(len(filtered_frames[name])) for name in ["train", "dev", "test"]}
    removed_sizes = {name: full_sizes[name] - filtered_sizes[name] for name in ["train", "dev", "test"]}

    summary = {
        "metadata_path": str(metadata_path),
        "dataset_base_dir": str(dataset_base_dir),
        "max_length": int(args.max_length),
        "n_rows_total_full": int(len(master_full_df)),
        "n_rows_total_filtered": int(len(filtered_manifest)),
        "split_sizes_full": full_sizes,
        "split_sizes_filtered": filtered_sizes,
        "split_rows_removed_by_filter": removed_sizes,
        "split_manifest_columns_filtered": filtered_manifest.columns.tolist(),
        "split_manifest_columns_full": master_full_df.columns.tolist(),
        "split_stats_filtered": filtered_stats,
        "filtered_overlap_records": filtered_overlap_df.to_dict(orient="records"),
        "full_overlap_records": full_overlap_df.to_dict(orient="records"),
        "interpretation": {
            "master_manifest_role": "Whole-cohort source provenance across all embedding rows and split assignments.",
            "filtered_manifest_role": "Canonical downstream split manifest aligned to encoded_*_filtered datasets used by ELM training and generation.",
            "leakage_reference": "Overlap flags are computed relative to the filtered training split, because that is what the model actually sees.",
        },
        "output_files": {
            "split_manifest_note_level": str(split_manifest_path),
            "split_manifest_note_level_full": str(split_manifest_full_path),
            "split_manifest_removed_by_filter": str(removed_manifest_path) if removed_df is not None and not removed_df.empty else None,
            "split_overlap_summary": str(filtered_overlap_path),
            "split_overlap_summary_full": str(full_overlap_path),
            "split_leakage_audit_summary": str(summary_path),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved filtered-aligned split manifest to: {split_manifest_path}")
    print(f"Saved whole-cohort split manifest to: {split_manifest_full_path}")
    if not removed_df.empty:
        print(f"Saved removed-by-filter manifest to: {removed_manifest_path}")
    print(f"Saved filtered overlap summary to: {filtered_overlap_path}")
    print(f"Saved full overlap summary to: {full_overlap_path}")
    print(f"Saved audit summary to: {summary_path}")
    print()
    print("Filtered split sizes:")
    for split_name in ["train", "dev", "test"]:
        print(f"  {split_name}: {filtered_sizes[split_name]:,} (removed {removed_sizes[split_name]:,} from full split)")
    print()
    print("Filtered train-vs-held-out overlap:")
    for split_name in ["dev", "test"]:
        stats = filtered_stats[split_name]
        print(
            f"  {split_name}: subject_id overlap="
            f"{stats['rows_with_subject_overlap_to_train']:,}, "
            f"hadm_id overlap={stats['rows_with_hadm_overlap_to_train']:,}"
        )
    print()
    print("Filtered patient-disjoint held-out rows relative to filtered train:")
    for split_name in ["dev", "test"]:
        stats = filtered_stats[split_name]
        print(
            f"  {split_name}: {stats['rows_patient_disjoint_from_train']:,}/{stats['rows']:,} "
            f"({100.0 * stats['pct_patient_disjoint_from_train']:.2f}%)"
        )


if __name__ == "__main__":
    main()
