#!/usr/bin/env python3
"""
Verify note/HADM-based MIMIC-IV pickle_ds quality before embedding generation.

Checks:
1) file-level outputs exist and are internally consistent
2) summary CSV statistics look sane
3) sampled pickle objects have expected type + dsnotes schema
4) sampled dsnotes text quality is suitable for embedding script
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List

import pandas as pd


MIMIC_MM_PATH = "/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/MIMIC-MM-Dataset-main"
if MIMIC_MM_PATH not in sys.path:
    sys.path.insert(0, MIMIC_MM_PATH)

try:
    import minimal_API  # noqa: F401
except ImportError as exc:
    raise RuntimeError(
        f"Could not import minimal_API from {MIMIC_MM_PATH}. "
        "Please verify this path and your environment."
    ) from exc


REQUIRED_DSNOTE_COLS = {
    "note_id",
    "subject_id",
    "hadm_id",
    "note_type",
    "note_seq",
    "charttime",
    "storetime",
    "text",
    "deltacharttime",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify generated note/HADM pickle_ds quality")
    parser.add_argument(
        "--pickle_dir",
        default="/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/pickle_ds_note_hadm_all",
        help="Directory containing generated pickle files",
    )
    parser.add_argument(
        "--summary_csv",
        default=None,
        help="Optional explicit summary CSV path (default: <pickle_dir>/export_summary_mimiciv_note_hadm.csv)",
    )
    parser.add_argument(
        "--raw_discharge_csv",
        default="/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimic-iv-note/2.2/note/discharge.csv",
        help="Raw discharge CSV for optional count comparison",
    )
    parser.add_argument(
        "--sample_pickles",
        type=int,
        default=200,
        help="How many pickle files to sample for structural checks",
    )
    parser.add_argument(
        "--expected_min_text_len",
        type=int,
        default=50,
        help="Expected minimum text length used in prep step",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pickle_dir = Path(args.pickle_dir)
    summary_csv = Path(args.summary_csv) if args.summary_csv else pickle_dir / "export_summary_mimiciv_note_hadm.csv"
    raw_discharge_csv = Path(args.raw_discharge_csv)

    if not pickle_dir.exists():
        raise FileNotFoundError(f"pickle_dir does not exist: {pickle_dir}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary CSV does not exist: {summary_csv}")

    pkl_files: List[Path] = sorted(pickle_dir.glob("*.pkl"))
    if not pkl_files:
        raise RuntimeError(f"No .pkl files found under {pickle_dir}")

    summary = pd.read_csv(summary_csv, dtype="object")
    if summary.empty:
        raise RuntimeError(f"Summary CSV is empty: {summary_csv}")

    summary["n_dsnotes_num"] = pd.to_numeric(summary.get("n_dsnotes", 0), errors="coerce").fillna(0).astype(int)
    wrote_rows = int((summary.get("status", pd.Series(dtype="object")) == "written").sum()) if "status" in summary.columns else len(summary)
    skip_rows = int((summary.get("status", pd.Series(dtype="object")) == "skipped_exists").sum()) if "status" in summary.columns else 0
    total_dsnotes_summary = int(summary["n_dsnotes_num"].sum())

    print("=== File-Level Checks ===")
    print(f"pickle_dir: {pickle_dir}")
    print(f"summary_csv: {summary_csv}")
    print(f"pkl_files_found: {len(pkl_files)}")
    print(f"summary_rows: {len(summary)}")
    print(f"summary_written_rows: {wrote_rows}")
    print(f"summary_skipped_rows: {skip_rows}")
    print(f"summary_total_dsnotes: {total_dsnotes_summary}")

    if raw_discharge_csv.exists():
        raw_count = int(pd.read_csv(raw_discharge_csv, usecols=["note_id"], dtype="object").shape[0])
        print(f"raw_discharge_notes: {raw_count}")
        print(f"summary_total_matches_raw: {total_dsnotes_summary == raw_count}")
    else:
        print("raw_discharge_notes: N/A (raw_discharge_csv missing)")

    sample_n = min(args.sample_pickles, len(pkl_files))
    sample_files = pkl_files[:sample_n]

    bad_object = 0
    missing_dsnotes = 0
    bad_columns = 0
    bad_text = 0
    total_sample_rows = 0
    min_text_len_observed = None
    max_text_len_observed = 0

    for fp in sample_files:
        with open(fp, "rb") as f:
            obj = pickle.load(f)

        if not hasattr(obj, "dsnotes"):
            missing_dsnotes += 1
            continue

        ds = obj.dsnotes
        if not isinstance(ds, pd.DataFrame):
            bad_object += 1
            continue

        cols = set(ds.columns)
        if not REQUIRED_DSNOTE_COLS.issubset(cols):
            bad_columns += 1
            continue

        total_sample_rows += len(ds)
        text_len = ds["text"].fillna("").astype(str).str.strip().str.len()
        if len(text_len) > 0:
            curr_min = int(text_len.min())
            curr_max = int(text_len.max())
            min_text_len_observed = curr_min if min_text_len_observed is None else min(min_text_len_observed, curr_min)
            max_text_len_observed = max(max_text_len_observed, curr_max)
            bad_text += int((text_len < args.expected_min_text_len).sum())

    print("\n=== Sampled Pickle Quality ===")
    print(f"sampled_pickles: {sample_n}")
    print(f"sample_total_dsnote_rows: {total_sample_rows}")
    print(f"missing_dsnotes_attr: {missing_dsnotes}")
    print(f"dsnotes_not_dataframe: {bad_object}")
    print(f"dsnotes_missing_required_columns: {bad_columns}")
    print(f"text_rows_below_min_len_{args.expected_min_text_len}: {bad_text}")
    print(f"sample_text_len_min: {min_text_len_observed}")
    print(f"sample_text_len_max: {max_text_len_observed}")

    ok = (missing_dsnotes == 0) and (bad_object == 0) and (bad_columns == 0) and (bad_text == 0)
    print("\n=== Verdict ===")
    print(f"embedding_ready: {ok}")
    if not ok:
        print("issues_found: True (see metrics above)")


if __name__ == "__main__":
    main()
