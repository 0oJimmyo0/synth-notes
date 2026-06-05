#!/usr/bin/env python3
"""
Count MIMIC-IV note volumes across the ELM pipeline.

Stages counted:
1) Raw discharge notes in mimic-iv-note/2.2/note/discharge.csv
2) Notes surviving current pickle_ds preprocessing logic
   (subject_id+hadm_id join to ICU key table, clipped to admission window,
    and duplicated per ICU stay key as in current pickle generation workflow)
3) Notes embedded (sentence_embeddings_metadata.csv rows)
4) Final examples used by ELM training (encoded_training_filtered)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count MIMIC-IV note sizes across ELM pipeline stages")
    parser.add_argument(
        "--mimiciv_note_dir",
        default="/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimic-iv-note/2.2",
        help="Path to MIMIC-IV-Note root (contains note/discharge.csv)",
    )
    parser.add_argument(
        "--mimiciv_core_dir",
        default="/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1",
        help="Path to MIMIC-IV core root (contains hosp/, haim_mimiciv_key_ids.csv, embeddings/, data/)",
    )
    parser.add_argument(
        "--embedding_subdir",
        default="embeddings/BAAI-bge-large-en-v1.5",
        help="Embedding subdirectory under mimiciv_core_dir",
    )
    parser.add_argument(
        "--task_subdir",
        default="data/clinic_notes/1_task",
        help="Task dataset subdirectory under mimiciv_core_dir",
    )
    return parser.parse_args()


def load_num_examples(dataset_dir: Path) -> Optional[int]:
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text())
    return info.get("splits", {}).get("train", {}).get("num_examples")


def main() -> None:
    args = parse_args()

    mimiciv_note_dir = Path(args.mimiciv_note_dir)
    mimiciv_core_dir = Path(args.mimiciv_core_dir)
    embedding_dir = mimiciv_core_dir / args.embedding_subdir
    task_dir = mimiciv_core_dir / args.task_subdir

    discharge_csv = mimiciv_note_dir / "note" / "discharge.csv"
    keys_csv = mimiciv_core_dir / "haim_mimiciv_key_ids.csv"
    admissions_csv = mimiciv_core_dir / "hosp" / "admissions.csv"
    embedding_metadata_csv = embedding_dir / "sentence_embeddings_metadata.csv"

    required = [discharge_csv, keys_csv, admissions_csv, embedding_metadata_csv, task_dir]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))

    # Raw discharge notes.
    notes = pd.read_csv(
        discharge_csv,
        usecols=["note_id", "subject_id", "hadm_id", "charttime", "text"],
        dtype={"note_id": "object", "subject_id": "Int64", "hadm_id": "Int64", "charttime": "object", "text": "object"},
    )
    raw_discharge_count = len(notes)

    # Current pickle_ds preprocessing logic (from current MIMIC-IV notebook):
    # 1) one patient object per (subject_id, hadm_id, stay_id) key row
    # 2) dsnotes selected by subject_id + hadm_id
    # 3) dsnotes clipped to [admittime, dischtime]
    keys = pd.read_csv(
        keys_csv,
        usecols=["subject_id", "hadm_id", "stay_id"],
        dtype={"subject_id": "Int64", "hadm_id": "Int64", "stay_id": "Int64"},
    )
    admissions = pd.read_csv(
        admissions_csv,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        dtype={"subject_id": "Int64", "hadm_id": "Int64", "admittime": "object", "dischtime": "object"},
    )

    notes["charttime"] = pd.to_datetime(notes["charttime"], errors="coerce")
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")

    key_hadm = keys[["subject_id", "hadm_id"]].drop_duplicates()
    notes_with_key_hadm = notes.merge(
        key_hadm.assign(_in_key_hadm=1),
        on=["subject_id", "hadm_id"],
        how="left",
    )
    notes_in_key_hadm = notes_with_key_hadm["_in_key_hadm"].eq(1)
    notes_in_key_hadm_rows = int(notes_in_key_hadm.sum())
    notes_in_key_hadm_unique = int(notes_with_key_hadm.loc[notes_in_key_hadm, "note_id"].nunique())

    key_adm = keys.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
    joined = key_adm.merge(
        notes[["note_id", "subject_id", "hadm_id", "charttime", "text"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )
    joined_rows = len(joined)
    joined_rows_with_note = int(joined["note_id"].notna().sum())
    joined_rows_without_note = int(joined["note_id"].isna().sum())

    mask = (joined["charttime"] >= joined["admittime"]) & (joined["charttime"] <= joined["dischtime"])
    kept = joined[mask].copy()
    outside_window = joined["note_id"].notna() & (~mask)

    pickle_survive_rows = len(kept)
    pickle_survive_unique_note_ids = int(kept["note_id"].nunique())
    kept_text_len = kept["text"].fillna("").astype(str).str.strip().str.len()
    pickle_survive_text_gt50 = int((kept_text_len > 50).sum())
    notes_outside_window_rows = int(outside_window.sum())
    notes_outside_window_unique = int(joined.loc[outside_window, "note_id"].nunique())

    # Embedded notes.
    embedding_metadata = pd.read_csv(embedding_metadata_csv, usecols=["filename"], dtype={"filename": "object"})
    embedded_count = len(embedding_metadata)

    # Final ELM datasets.
    dataset_counts: Dict[str, Optional[int]] = {}
    for name in [
        "encoded_training_full",
        "encoded_dev_full",
        "encoded_testing_full",
        "encoded_training_filtered",
        "encoded_dev_filtered",
        "encoded_testing_filtered",
    ]:
        dataset_counts[name] = load_num_examples(task_dir / name)

    full_total = None
    if all(dataset_counts[n] is not None for n in ["encoded_training_full", "encoded_dev_full", "encoded_testing_full"]):
        full_total = (
            int(dataset_counts["encoded_training_full"])
            + int(dataset_counts["encoded_dev_full"])
            + int(dataset_counts["encoded_testing_full"])
        )

    filtered_total = None
    if all(
        dataset_counts[n] is not None
        for n in ["encoded_training_filtered", "encoded_dev_filtered", "encoded_testing_filtered"]
    ):
        filtered_total = (
            int(dataset_counts["encoded_training_filtered"])
            + int(dataset_counts["encoded_dev_filtered"])
            + int(dataset_counts["encoded_testing_filtered"])
        )

    print("=== MIMIC-IV ELM Pipeline Counts ===")
    print(f"raw_discharge_notes: {raw_discharge_count}")
    print(f"raw_discharge_notes_with_key_hadm: {notes_in_key_hadm_rows}")
    print(f"raw_unique_note_ids_with_key_hadm: {notes_in_key_hadm_unique}")
    print(f"stay_level_rows_before_time_clip: {joined_rows}")
    print(f"stay_level_rows_with_note_before_time_clip: {joined_rows_with_note}")
    print(f"stay_level_rows_without_note_before_time_clip: {joined_rows_without_note}")
    print(f"rows_dropped_by_time_clip: {notes_outside_window_rows}")
    print(f"unique_note_ids_dropped_by_time_clip: {notes_outside_window_unique}")
    print(f"pickle_ds_surviving_rows: {pickle_survive_rows}")
    print(f"pickle_ds_surviving_unique_note_ids: {pickle_survive_unique_note_ids}")
    print(f"pickle_ds_surviving_rows_text_gt_50: {pickle_survive_text_gt50}")
    print(f"embedded_notes: {embedded_count}")
    print(f"elm_train_examples_filtered: {dataset_counts['encoded_training_filtered']}")
    print(f"elm_dev_examples_filtered: {dataset_counts['encoded_dev_filtered']}")
    print(f"elm_test_examples_filtered: {dataset_counts['encoded_testing_filtered']}")
    print(f"elm_total_examples_filtered: {filtered_total}")
    print(f"elm_train_examples_full: {dataset_counts['encoded_training_full']}")
    print(f"elm_dev_examples_full: {dataset_counts['encoded_dev_full']}")
    print(f"elm_test_examples_full: {dataset_counts['encoded_testing_full']}")
    print(f"elm_total_examples_full: {full_total}")

    print("\n=== Consistency Checks ===")
    print(f"embedded == pickle_text_gt_50: {embedded_count == pickle_survive_text_gt50}")
    print(f"embedded == filtered_total: {embedded_count == filtered_total if filtered_total is not None else 'N/A'}")
    print(f"embedded == full_total: {embedded_count == full_total if full_total is not None else 'N/A'}")


if __name__ == "__main__":
    main()
