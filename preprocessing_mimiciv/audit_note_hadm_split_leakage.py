#!/usr/bin/env python3
"""
Audit note-level train/dev/test overlap for the MIMIC-IV note/HADM pipeline.

This reconstructs the note-level split used during HF dataset preparation from the
embedding metadata and reports overlap by note_id, hadm_id, and subject_id.

It also writes a split manifest that downstream evaluation can join against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_METADATA = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "embeddings-BAAI-bge-large-en-v1.5/sentence_embeddings_metadata.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/leakage_audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit note-level split leakage for the note/HADM-aligned MIMIC-IV cohort."
    )
    parser.add_argument(
        "--metadata-path",
        default=DEFAULT_METADATA,
        help="Path to sentence_embeddings_metadata.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save split manifest and overlap summaries",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Dev split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed used for splitting")
    return parser.parse_args()


def assign_splits(
    metadata_df: pd.DataFrame,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> pd.DataFrame:
    if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    metadata_df = metadata_df.reset_index(drop=True).copy()
    metadata_df["embedding_row_id"] = metadata_df.index

    train_df, temp_df = train_test_split(
        metadata_df,
        test_size=(dev_ratio + test_ratio),
        random_state=random_seed,
        shuffle=True,
    )

    dev_size = dev_ratio / (dev_ratio + test_ratio)
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - dev_size),
        random_state=random_seed,
        shuffle=True,
    )

    train_df = train_df.copy().reset_index(drop=True)
    dev_df = dev_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    train_df["split"] = "train"
    dev_df["split"] = "dev"
    test_df["split"] = "test"

    train_df["dataset_row_id"] = range(len(train_df))
    dev_df["dataset_row_id"] = range(len(dev_df))
    test_df["dataset_row_id"] = range(len(test_df))

    for df in (train_df, dev_df, test_df):
        df["source_row_id"] = df["embedding_row_id"]

    combined = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    return combined


def clean_ids(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    )


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


def main() -> None:
    args = parse_args()

    metadata_path = Path(args.metadata_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata_df = pd.read_csv(metadata_path, dtype=str)
    required_cols = ["note_id", "subject_id", "hadm_id", "filename"]
    missing = [col for col in required_cols if col not in metadata_df.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")

    split_manifest = assign_splits(
        metadata_df=metadata_df,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
    )

    split_manifest["note_id"] = clean_ids(split_manifest["note_id"])
    split_manifest["subject_id"] = clean_ids(split_manifest["subject_id"])
    split_manifest["hadm_id"] = clean_ids(split_manifest["hadm_id"])

    split_manifest_path = output_dir / "split_manifest_note_level.csv"
    split_manifest.to_csv(split_manifest_path, index=False)

    splits = {
        name: split_manifest.loc[split_manifest["split"] == name].copy()
        for name in ["train", "dev", "test"]
    }

    overlap_rows: list[dict[str, object]] = []
    split_pairs = [("train", "dev"), ("train", "test"), ("dev", "test")]
    for split_a, split_b in split_pairs:
        df_a = splits[split_a]
        df_b = splits[split_b]
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

    overlap_df = pd.DataFrame(overlap_rows)
    overlap_csv_path = output_dir / "split_overlap_summary.csv"
    overlap_df.to_csv(overlap_csv_path, index=False)

    train_subjects = set(splits["train"]["subject_id"].dropna().tolist())
    train_hadms = set(splits["train"]["hadm_id"].dropna().tolist())
    train_notes = set(splits["train"]["note_id"].dropna().tolist())

    split_manifest["patient_overlap_with_train"] = split_manifest["subject_id"].isin(train_subjects)
    split_manifest["hadm_overlap_with_train"] = split_manifest["hadm_id"].isin(train_hadms)
    split_manifest["note_overlap_with_train"] = split_manifest["note_id"].isin(train_notes)
    split_manifest["patient_disjoint_from_train"] = ~split_manifest["patient_overlap_with_train"]
    split_manifest["hadm_disjoint_from_train"] = ~split_manifest["hadm_overlap_with_train"]
    split_manifest["note_disjoint_from_train"] = ~split_manifest["note_overlap_with_train"]

    split_manifest = split_manifest.loc[
        :,
        [
            "source_row_id",
            "dataset_row_id",
            "embedding_row_id",
            "split",
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
        ],
    ]

    split_stats = {
        split_name: summarize_split_rows(split_df, train_subjects, train_hadms)
        for split_name, split_df in splits.items()
    }

    summary = {
        "metadata_path": str(metadata_path),
        "n_rows_total": int(len(split_manifest)),
        "split_sizes": {name: int(len(df)) for name, df in splits.items()},
        "split_manifest_columns": split_manifest.columns.tolist(),
        "split_stats": split_stats,
        "overlap_records": overlap_rows,
        "interpretation": {
            "split_type": "note_level_random_split",
            "note_id_overlap_expected": "zero if each note appears once",
            "subject_id_overlap_implication": "non-zero overlap means the held-out split is not patient-disjoint",
            "hadm_id_overlap_implication": "non-zero overlap means the held-out split is not admission-disjoint",
        },
    }
    summary_path = output_dir / "split_leakage_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Saved split manifest to:", split_manifest_path)
    print("Saved overlap summary to:", overlap_csv_path)
    print("Saved audit summary to:", summary_path)
    print("")
    print("Split sizes:")
    for split_name, split_df in splits.items():
        print(f"  {split_name}: {len(split_df):,}")
    print("")
    print("Train-vs-held-out overlap:")
    for heldout in ["dev", "test"]:
        subj_overlap = overlap_df[
            (overlap_df["split_a"] == "train")
            & (overlap_df["split_b"] == heldout)
            & (overlap_df["key"] == "subject_id")
        ]["overlap_count"].iloc[0]
        hadm_overlap = overlap_df[
            (overlap_df["split_a"] == "train")
            & (overlap_df["split_b"] == heldout)
            & (overlap_df["key"] == "hadm_id")
        ]["overlap_count"].iloc[0]
        print(f"  {heldout}: subject_id overlap={subj_overlap:,}, hadm_id overlap={hadm_overlap:,}")
    print("")
    print("Patient-disjoint held-out rows relative to train:")
    for heldout in ["dev", "test"]:
        stat = split_stats[heldout]
        print(
            f"  {heldout}: {stat['rows_patient_disjoint_from_train']:,}/{stat['rows']:,} "
            f"({stat['pct_patient_disjoint_from_train']:.2%})"
        )


if __name__ == "__main__":
    main()
