#!/usr/bin/env python3
"""
Build MIMIC-IV discharge-note pickle files for note/HADM-based ELM training.

Compared with the current ICU-stay-key pipeline, this script:
- starts from all rows in mimic-iv-note `note/discharge.csv`
- groups by (subject_id, hadm_id)
- writes one pickle per admission with `.dsnotes`

Output object/type contract is kept compatible with the existing embedding + ELM scripts:
- one pickle per (subject_id, hadm_id)
- object type: minimal_API.Patient_ICU
- object.dsnotes contains note text and ids used downstream
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    mimiciv_note_dir: Path
    mimiciv_core_dir: Path
    discharge_path: Path
    admissions_path: Optional[Path]
    output_dir: Path
    min_text_len: int
    chunksize: int
    clip_to_admission_window: bool
    keep_without_admission: bool
    max_hadm: Optional[int]
    overwrite: bool


def _resolve_table_path(base_dir: Path, rel_csv_path: str, required: bool = True) -> Optional[Path]:
    plain = base_dir / rel_csv_path
    gz = base_dir / f"{rel_csv_path}.gz"
    if plain.exists():
        return plain
    if gz.exists():
        return gz
    if required:
        raise FileNotFoundError(f"Could not find {plain} or {gz}")
    return None


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(
        description="Create MIMIC-IV note/HADM discharge-note pickle_ds files for embedding + ELM training"
    )
    parser.add_argument(
        "--mimiciv_note_dir",
        default="/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimic-iv-note/2.2",
        help="Root directory containing note/discharge.csv",
    )
    parser.add_argument(
        "--mimiciv_core_dir",
        default="/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1",
        help="Root directory containing hosp/admissions.csv",
    )
    parser.add_argument(
        "--discharge_path",
        default=None,
        help="Path to discharge.csv(.gz); defaults to <mimiciv_note_dir>/note/discharge.csv",
    )
    parser.add_argument(
        "--admissions_path",
        default=None,
        help="Path to admissions.csv(.gz); defaults to <mimiciv_core_dir>/hosp/admissions.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for pickle files (default: <mimiciv_core_dir>/pickle_ds_note_hadm_all)",
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
        help="Chunk size when reading discharge.csv",
    )
    parser.add_argument(
        "--clip_to_admission_window",
        action="store_true",
        help="If set, keep only notes with charttime inside [admittime, dischtime]",
    )
    parser.add_argument(
        "--drop_without_admission",
        action="store_true",
        help="If set with --clip_to_admission_window, drop rows without admissions match",
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
    mimiciv_note_dir = Path(args.mimiciv_note_dir)
    mimiciv_core_dir = Path(args.mimiciv_core_dir)

    if args.discharge_path:
        discharge_path = Path(args.discharge_path)
    else:
        discharge_path = _resolve_table_path(mimiciv_note_dir, "note/discharge.csv")

    if args.admissions_path:
        admissions_path = Path(args.admissions_path)
    else:
        admissions_path = _resolve_table_path(mimiciv_core_dir, "hosp/admissions.csv", required=False)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else mimiciv_core_dir / "pickle_ds_note_hadm_all"
    )

    return BuildConfig(
        mimiciv_note_dir=mimiciv_note_dir,
        mimiciv_core_dir=mimiciv_core_dir,
        discharge_path=discharge_path,
        admissions_path=admissions_path,
        output_dir=output_dir,
        min_text_len=args.min_text_len,
        chunksize=args.chunksize,
        clip_to_admission_window=args.clip_to_admission_window,
        keep_without_admission=(not args.drop_without_admission),
        max_hadm=args.max_hadm,
        overwrite=args.overwrite,
    )


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def load_admissions_lookup(admissions_path: Optional[Path]) -> Dict[Tuple[str, str], pd.DataFrame]:
    if admissions_path is None:
        print("[info] ADMISSIONS table not found, continuing without admissions/core data")
        return {}

    print(f"[info] Loading admissions lookup from {admissions_path}")
    usecols = ["subject_id", "hadm_id", "admittime", "dischtime"]
    admissions = pd.read_csv(admissions_path, compression="infer", usecols=usecols, dtype=str)
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
    print(f"[info] Reading discharge notes from {cfg.discharge_path}")
    usecols = ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime", "text"]

    pieces: List[pd.DataFrame] = []
    n_total = 0
    n_kept = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            cfg.discharge_path,
            compression="infer",
            usecols=usecols,
            dtype=str,
            chunksize=cfg.chunksize,
        ),
        start=1,
    ):
        n_total += len(chunk)

        chunk["subject_id"] = chunk["subject_id"].fillna("").astype(str).str.strip()
        chunk["hadm_id"] = chunk["hadm_id"].fillna("").astype(str).str.strip()
        chunk["note_id"] = chunk["note_id"].fillna("").astype(str).str.strip()
        chunk["text"] = chunk["text"].fillna("").astype(str).str.strip()

        filtered = chunk[
            (chunk["subject_id"] != "")
            & (chunk["hadm_id"] != "")
            & (chunk["note_id"] != "")
            & (chunk["text"].str.len() >= cfg.min_text_len)
        ].copy()

        if filtered.empty:
            if chunk_idx % 10 == 0:
                print(f"[info] chunk={chunk_idx} rows={len(chunk)} kept_so_far={n_kept}")
            continue

        filtered["charttime"] = pd.to_datetime(filtered["charttime"], errors="coerce")
        filtered["storetime"] = pd.to_datetime(filtered["storetime"], errors="coerce")

        pieces.append(filtered)
        n_kept += len(filtered)

        if chunk_idx % 10 == 0:
            print(f"[info] chunk={chunk_idx} rows={len(chunk)} kept_so_far={n_kept}")

    if not pieces:
        raise RuntimeError("No discharge notes found after filtering")

    notes = pd.concat(pieces, ignore_index=True)

    if cfg.clip_to_admission_window:
        if cfg.admissions_path is None or (not cfg.admissions_path.exists()):
            print("[warn] --clip_to_admission_window requested but admissions file is unavailable; skipping clip")
        else:
            admissions = pd.read_csv(
                cfg.admissions_path,
                compression="infer",
                usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
                dtype=str,
            )
            admissions["subject_id"] = admissions["subject_id"].astype(str).str.strip()
            admissions["hadm_id"] = admissions["hadm_id"].astype(str).str.strip()
            admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
            admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
            admissions = admissions.drop_duplicates(subset=["subject_id", "hadm_id"])

            n_before = len(notes)
            notes = notes.merge(
                admissions[["subject_id", "hadm_id", "admittime", "dischtime"]],
                on=["subject_id", "hadm_id"],
                how="left",
            )

            has_adm = notes["admittime"].notna() & notes["dischtime"].notna()
            in_window = notes["charttime"].notna() & has_adm & (notes["charttime"] >= notes["admittime"]) & (
                notes["charttime"] <= notes["dischtime"]
            )

            if cfg.keep_without_admission:
                keep_mask = in_window | (~has_adm)
            else:
                keep_mask = in_window

            notes = notes[keep_mask].copy()
            notes = notes.drop(columns=["admittime", "dischtime"])
            print(f"[info] Admission-window clip kept {len(notes)}/{n_before} rows")

    notes = notes.sort_values(["subject_id", "hadm_id", "charttime", "note_id"]).reset_index(drop=True)
    notes = notes.drop_duplicates(subset=["note_id"], keep="first")

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


def export_pickles(
    cfg: BuildConfig,
    notes: pd.DataFrame,
    admissions_lookup: Dict[Tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
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
        dsnotes["note_type"] = dsnotes["note_type"].fillna("DS").replace("", "DS")
        dsnotes["note_seq"] = pd.to_numeric(dsnotes["note_seq"], errors="coerce")
        if dsnotes["note_seq"].isna().any():
            dsnotes["note_seq"] = np.arange(1, len(dsnotes) + 1)
        dsnotes["note_seq"] = dsnotes["note_seq"].astype(int)
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

        if (idx + 1) % 5000 == 0:
            print(f"[info] wrote {idx + 1}/{total_groups} files")

    return pd.DataFrame(summary_rows)


def main() -> None:
    cfg = parse_args()

    print("=" * 72)
    print("MIMIC-IV note/HADM discharge-note pickle builder")
    print("=" * 72)
    print(f"mimiciv_note_dir           : {cfg.mimiciv_note_dir}")
    print(f"mimiciv_core_dir           : {cfg.mimiciv_core_dir}")
    print(f"discharge_path             : {cfg.discharge_path}")
    print(f"admissions_path            : {cfg.admissions_path if cfg.admissions_path else 'not used'}")
    print(f"output_dir                 : {cfg.output_dir}")
    print(f"min_text_len               : {cfg.min_text_len}")
    print(f"chunksize                  : {cfg.chunksize}")
    print(f"clip_to_admission_window   : {cfg.clip_to_admission_window}")
    print(f"keep_without_admission     : {cfg.keep_without_admission}")
    print(f"max_hadm                   : {cfg.max_hadm if cfg.max_hadm is not None else 'all'}")
    print(f"overwrite                  : {cfg.overwrite}")

    admissions_lookup = load_admissions_lookup(cfg.admissions_path)
    notes = load_filtered_discharge_notes(cfg)
    summary = export_pickles(cfg, notes, admissions_lookup)

    summary_path = cfg.output_dir / "export_summary_mimiciv_note_hadm.csv"
    summary.to_csv(summary_path, index=False)

    wrote = int((summary["status"] == "written").sum()) if not summary.empty else 0
    skipped = int((summary["status"] == "skipped_exists").sum()) if not summary.empty else 0
    print("=" * 72)
    print(f"Done. Wrote {wrote} files, skipped {skipped} existing files")
    print(f"Summary CSV: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
