#!/usr/bin/env python3
"""
Build MIMIC-III discharge-note pickle files compatible with the existing ELM pipeline.

Output contract matches the current embedding/data-prep scripts:
- one pickle per (subject_id, hadm_id)
- object type: minimal_API.Patient_ICU
- object.dsnotes includes note text and ids used downstream
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


MIMIC_MM_PATH = "/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/MIMIC-MM-Dataset-main"
if MIMIC_MM_PATH not in sys.path:
    sys.path.insert(0, MIMIC_MM_PATH)

try:
    import minimal_API
except ImportError as exc:
    raise RuntimeError(
        f"Could not import minimal_API from {MIMIC_MM_PATH}. "
        "Please verify this path and your environment."
    ) from exc


@dataclass
class BuildConfig:
    mimic3_dir: Path
    output_dir: Path
    notes_path: Path
    admissions_path: Optional[Path]
    category: str
    description: Optional[str]
    min_text_len: int
    chunksize: int
    max_hadm: Optional[int]
    overwrite: bool


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(
        description="Create MIMIC-III discharge-note pickle_ds files for embedding + ELM training"
    )
    parser.add_argument(
        "--mimic3_dir",
        default="/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/physionet.org/files/mimiciii/1.4",
        help="Directory containing MIMIC-III raw tables (NOTEEVENTS/ADMISSIONS)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for pickle files (default: <mimic3_dir>/pickle_ds)",
    )
    parser.add_argument(
        "--notes_path",
        default=None,
        help="Path to NOTEEVENTS.csv or NOTEEVENTS.csv.gz (auto-detected if omitted)",
    )
    parser.add_argument(
        "--admissions_path",
        default=None,
        help="Path to ADMISSIONS.csv or ADMISSIONS.csv.gz (auto-detected if omitted)",
    )
    parser.add_argument(
        "--category",
        default="Discharge summary",
        help="NOTEEVENTS CATEGORY to keep (default: Discharge summary)",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Optional NOTEEVENTS DESCRIPTION exact match filter (e.g., Report)",
    )
    parser.add_argument(
        "--min_text_len",
        type=int,
        default=50,
        help="Minimum note text length after stripping",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250000,
        help="Chunk size when reading NOTEEVENTS",
    )
    parser.add_argument(
        "--max_hadm",
        type=int,
        default=None,
        help="Optional max number of (subject_id, hadm_id) groups to export (for testing)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing pickle files if present",
    )

    args = parser.parse_args()

    mimic3_dir = Path(args.mimic3_dir)
    if args.notes_path:
        notes_path = Path(args.notes_path)
    else:
        notes_path = _resolve_table_path(mimic3_dir, "NOTEEVENTS")

    if args.admissions_path:
        admissions_path = Path(args.admissions_path)
    else:
        admissions_path = _resolve_table_path(mimic3_dir, "ADMISSIONS", required=False)

    output_dir = Path(args.output_dir) if args.output_dir else mimic3_dir / "pickle_ds"

    return BuildConfig(
        mimic3_dir=mimic3_dir,
        output_dir=output_dir,
        notes_path=notes_path,
        admissions_path=admissions_path,
        category=args.category,
        description=args.description,
        min_text_len=args.min_text_len,
        chunksize=args.chunksize,
        max_hadm=args.max_hadm,
        overwrite=args.overwrite,
    )


def _resolve_table_path(base_dir: Path, stem: str, required: bool = True) -> Optional[Path]:
    candidates = [base_dir / f"{stem}.csv", base_dir / f"{stem}.csv.gz"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(
            f"Could not find {stem}.csv or {stem}.csv.gz under {base_dir}"
        )
    return None


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def load_admissions_lookup(admissions_path: Optional[Path]) -> Dict[Tuple[str, str], pd.DataFrame]:
    if admissions_path is None:
        print("[info] ADMISSIONS table not found, continuing without admissions/core data")
        return {}

    print(f"[info] Loading admissions lookup from {admissions_path}")
    usecols = ["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"]
    admissions = pd.read_csv(admissions_path, compression="infer", usecols=usecols, dtype=str)
    admissions = admissions.rename(
        columns={
            "SUBJECT_ID": "subject_id",
            "HADM_ID": "hadm_id",
            "ADMITTIME": "admittime",
            "DISCHTIME": "dischtime",
        }
    )
    admissions["subject_id"] = admissions["subject_id"].astype(str).str.strip()
    admissions["hadm_id"] = admissions["hadm_id"].astype(str).str.strip()
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")

    lookup: Dict[Tuple[str, str], pd.DataFrame] = {}
    for (sid, hadm), grp in admissions.groupby(["subject_id", "hadm_id"], sort=False):
        lookup[(sid, hadm)] = grp.head(1).copy()

    print(f"[info] Built admissions lookup for {len(lookup)} admissions")
    return lookup


def load_filtered_discharge_notes(cfg: BuildConfig) -> pd.DataFrame:
    print(f"[info] Reading NOTEEVENTS from {cfg.notes_path}")
    usecols = [
        "ROW_ID",
        "SUBJECT_ID",
        "HADM_ID",
        "CHARTDATE",
        "CHARTTIME",
        "STORETIME",
        "CATEGORY",
        "DESCRIPTION",
        "CGID",
        "ISERROR",
        "TEXT",
    ]

    pieces: List[pd.DataFrame] = []
    n_total = 0
    n_kept = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            cfg.notes_path,
            compression="infer",
            usecols=usecols,
            dtype=str,
            chunksize=cfg.chunksize,
        ),
        start=1,
    ):
        n_total += len(chunk)

        category_mask = chunk["CATEGORY"].fillna("").str.lower() == cfg.category.lower()
        filtered = chunk[category_mask].copy()

        if cfg.description is not None:
            desc_mask = filtered["DESCRIPTION"].fillna("").str.lower() == cfg.description.lower()
            filtered = filtered[desc_mask]

        # Keep only non-error notes (ISERROR==1 means marked as error).
        filtered = filtered[(filtered["ISERROR"].isna()) | (filtered["ISERROR"] != "1")]

        filtered["HADM_ID"] = filtered["HADM_ID"].fillna("").astype(str).str.strip()
        filtered = filtered[filtered["HADM_ID"] != ""]

        filtered["SUBJECT_ID"] = filtered["SUBJECT_ID"].fillna("").astype(str).str.strip()
        filtered = filtered[filtered["SUBJECT_ID"] != ""]

        filtered["TEXT"] = filtered["TEXT"].fillna("").astype(str).str.strip()
        filtered = filtered[filtered["TEXT"].str.len() >= cfg.min_text_len]

        if filtered.empty:
            if chunk_idx % 10 == 0:
                print(f"[info] chunk={chunk_idx} rows={len(chunk)} kept_so_far={n_kept}")
            continue

        filtered["CHARTTIME"] = filtered["CHARTTIME"].fillna("")
        filtered["CHARTDATE"] = filtered["CHARTDATE"].fillna("")
        chart_time = pd.to_datetime(filtered["CHARTTIME"], errors="coerce")
        chart_date = pd.to_datetime(filtered["CHARTDATE"], errors="coerce")
        filtered["charttime"] = chart_time.fillna(chart_date)
        filtered["storetime"] = pd.to_datetime(filtered["STORETIME"], errors="coerce")

        filtered = filtered.rename(
            columns={
                "ROW_ID": "row_id",
                "SUBJECT_ID": "subject_id",
                "HADM_ID": "hadm_id",
                "TEXT": "text",
            }
        )

        # String note_id avoids downstream float/int CSV coercion mismatch.
        filtered["note_id"] = (
            filtered["subject_id"].astype(str)
            + "-DS-"
            + filtered["row_id"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
        )

        keep_cols = [
            "note_id",
            "subject_id",
            "hadm_id",
            "charttime",
            "storetime",
            "text",
            "CATEGORY",
            "DESCRIPTION",
            "CGID",
            "row_id",
        ]
        filtered = filtered[keep_cols]

        pieces.append(filtered)
        n_kept += len(filtered)

        if chunk_idx % 10 == 0:
            print(f"[info] chunk={chunk_idx} rows={len(chunk)} kept_so_far={n_kept}")

    if not pieces:
        raise RuntimeError("No discharge notes found after filtering; check CATEGORY/DESCRIPTION filters")

    notes = pd.concat(pieces, ignore_index=True)
    notes = notes.sort_values(["subject_id", "hadm_id", "charttime", "note_id"]).reset_index(drop=True)

    print(f"[info] Processed total rows: {n_total}")
    print(f"[info] Filtered discharge-note rows kept: {len(notes)}")
    print(f"[info] Unique (subject_id, hadm_id): {notes[['subject_id', 'hadm_id']].drop_duplicates().shape[0]}")
    return notes


def make_patient_object(dsnotes: pd.DataFrame, admissions_row: Optional[pd.DataFrame]):
    admissions = admissions_row.copy() if admissions_row is not None else _empty_df()
    core = admissions_row.copy() if admissions_row is not None else _empty_df()

    return minimal_API.Patient_ICU(
        admissions,
        _empty_df(),
        _empty_df(),
        core,
        _empty_df(),
        _empty_df(),
        _empty_df(),
        _empty_df(),
        _empty_df(),
        _empty_df(),
        _empty_df(),
        _empty_df(),
        _empty_df(),
        _empty_df(),
        _empty_df(),
        _empty_df(),
        _empty_df(),
        _empty_df(),
        _empty_df(),
        _empty_df(),
        _empty_df(),
        _empty_df(),
        _empty_df(),
        [],
        dsnotes,
    )


def export_pickles(cfg: BuildConfig, notes: pd.DataFrame, admissions_lookup: Dict[Tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    grouped = list(notes.groupby(["subject_id", "hadm_id"], sort=True))
    if cfg.max_hadm is not None:
        grouped = grouped[: cfg.max_hadm]

    summary_rows = []
    total_groups = len(grouped)
    print(f"[info] Exporting {total_groups} pickle files to {cfg.output_dir}")

    for idx, ((sid, hadm), grp) in enumerate(grouped):
        out_file = cfg.output_dir / f"{idx:08d}.pkl"
        if out_file.exists() and not cfg.overwrite:
            summary_rows.append(
                {
                    "idx": idx,
                    "filename": out_file.name,
                    "subject_id": sid,
                    "hadm_id": hadm,
                    "n_dsnotes": len(grp),
                    "status": "skipped_exists",
                }
            )
            continue

        dsnotes = grp.copy()
        dsnotes["note_type"] = "DS"
        dsnotes["note_seq"] = np.arange(1, len(dsnotes) + 1)
        dsnotes["deltacharttime"] = np.nan

        dsnotes = dsnotes[
            [
                "note_id",
                "subject_id",
                "hadm_id",
                "note_type",
                "note_seq",
                "charttime",
                "storetime",
                "text",
                "deltacharttime",
            ]
        ].reset_index(drop=True)

        patient_obj = make_patient_object(dsnotes, admissions_lookup.get((sid, hadm)))
        with open(out_file, "wb") as f:
            pickle.dump(patient_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        summary_rows.append(
            {
                "idx": idx,
                "filename": out_file.name,
                "subject_id": sid,
                "hadm_id": hadm,
                "n_dsnotes": len(dsnotes),
                "status": "written",
            }
        )

        if (idx + 1) % 2000 == 0:
            print(f"[info] wrote {idx + 1}/{total_groups} files")

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def main() -> None:
    cfg = parse_args()

    print("=" * 72)
    print("MIMIC-III discharge-note pickle builder")
    print("=" * 72)
    print(f"mimic3_dir   : {cfg.mimic3_dir}")
    print(f"notes_path   : {cfg.notes_path}")
    print(f"admissions   : {cfg.admissions_path if cfg.admissions_path else 'not used'}")
    print(f"output_dir   : {cfg.output_dir}")
    print(f"category     : {cfg.category}")
    print(f"description  : {cfg.description if cfg.description else 'not filtered'}")
    print(f"min_text_len : {cfg.min_text_len}")
    print(f"chunksize    : {cfg.chunksize}")
    print(f"max_hadm     : {cfg.max_hadm if cfg.max_hadm is not None else 'all'}")
    print(f"overwrite    : {cfg.overwrite}")

    admissions_lookup = load_admissions_lookup(cfg.admissions_path)
    notes = load_filtered_discharge_notes(cfg)
    summary = export_pickles(cfg, notes, admissions_lookup)

    summary_path = cfg.output_dir / "export_summary_mimiciii_ds.csv"
    summary.to_csv(summary_path, index=False)

    wrote = int((summary["status"] == "written").sum()) if not summary.empty else 0
    skipped = int((summary["status"] == "skipped_exists").sum()) if not summary.empty else 0
    print("=" * 72)
    print(f"Done. Wrote {wrote} files, skipped {skipped} existing files")
    print(f"Summary CSV: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
