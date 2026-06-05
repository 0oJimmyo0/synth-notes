#!/usr/bin/env python3
"""
Count sizes for the note/HADM-based MIMIC-IV pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count note/HADM pipeline stage sizes")
    parser.add_argument(
        "--raw_discharge_csv",
        default="/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimic-iv-note/2.2/note/discharge.csv",
    )
    parser.add_argument(
        "--pickle_summary_csv",
        default="/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/pickle_ds_note_hadm_all/export_summary_mimiciv_note_hadm.csv",
    )
    parser.add_argument(
        "--embedding_metadata_csv",
        default="/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/embeddings_note_hadm_all/BAAI-bge-large-en-v1.5/sentence_embeddings_metadata.csv",
    )
    parser.add_argument(
        "--dataset_base_dir",
        default="/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task",
    )
    return parser.parse_args()


def load_num_examples(dataset_dir: Path) -> Optional[int]:
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.exists():
        return None
    obj = json.loads(info_path.read_text())
    return obj.get("splits", {}).get("train", {}).get("num_examples")


def main() -> None:
    args = parse_args()

    raw_csv = Path(args.raw_discharge_csv)
    summary_csv = Path(args.pickle_summary_csv)
    meta_csv = Path(args.embedding_metadata_csv)
    dataset_base = Path(args.dataset_base_dir)

    raw = pd.read_csv(raw_csv, usecols=["note_id"], dtype="object")
    summary = pd.read_csv(summary_csv, dtype="object")
    meta = pd.read_csv(meta_csv, usecols=["filename"], dtype="object")

    raw_rows = len(raw)
    wrote_rows = int((summary["status"] == "written").sum()) if "status" in summary.columns else len(summary)
    survive_rows = int(pd.to_numeric(summary["n_dsnotes"], errors="coerce").fillna(0).sum())
    embedded_rows = len(meta)

    train_full = load_num_examples(dataset_base / "encoded_training_full")
    dev_full = load_num_examples(dataset_base / "encoded_dev_full")
    test_full = load_num_examples(dataset_base / "encoded_testing_full")
    train_filtered = load_num_examples(dataset_base / "encoded_training_filtered")
    dev_filtered = load_num_examples(dataset_base / "encoded_dev_filtered")
    test_filtered = load_num_examples(dataset_base / "encoded_testing_filtered")

    full_total = None
    if train_full is not None and dev_full is not None and test_full is not None:
        full_total = int(train_full) + int(dev_full) + int(test_full)

    filtered_total = None
    if train_filtered is not None and dev_filtered is not None and test_filtered is not None:
        filtered_total = int(train_filtered) + int(dev_filtered) + int(test_filtered)

    print("=== Note/HADM MIMIC-IV Pipeline Counts ===")
    print(f"raw_discharge_notes: {raw_rows}")
    print(f"pickle_files_written: {wrote_rows}")
    print(f"notes_surviving_pickle_prep: {survive_rows}")
    print(f"notes_embedded: {embedded_rows}")
    print(f"elm_train_full: {train_full}")
    print(f"elm_dev_full: {dev_full}")
    print(f"elm_test_full: {test_full}")
    print(f"elm_total_full: {full_total}")
    print(f"elm_train_filtered: {train_filtered}")
    print(f"elm_dev_filtered: {dev_filtered}")
    print(f"elm_test_filtered: {test_filtered}")
    print(f"elm_total_filtered: {filtered_total}")


if __name__ == "__main__":
    main()
