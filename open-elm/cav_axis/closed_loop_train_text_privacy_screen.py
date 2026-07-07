#!/usr/bin/env python3
"""
Posthoc train-text privacy screen for closed-loop candidate manifests.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MIMIC_MM_PATH = "/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/MIMIC-MM-Dataset-main"
if MIMIC_MM_PATH not in sys.path:
    sys.path.insert(0, MIMIC_MM_PATH)
try:
    import minimal_API  # noqa: F401
except Exception:
    minimal_API = None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def text_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def lexical_similarity_ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return float(SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio())


def infer_base_dir(dataset_path: Path) -> Path | None:
    for parent in dataset_path.resolve().parents:
        if parent.name == "3.1":
            return parent
    return None


def infer_pickle_dir(
    dataset_path: Path,
    source_dataset_path: Path | None = None,
    explicit_pickle_dir: Path | None = None,
) -> Path | None:
    if explicit_pickle_dir is not None:
        return explicit_pickle_dir if explicit_pickle_dir.exists() else None
    base_dir = infer_base_dir(source_dataset_path or dataset_path)
    if base_dir is None:
        return None
    candidate = base_dir / "pickle_ds_note_hadm_all"
    return candidate if candidate.exists() else None


def load_split_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "dataset_row_id" in df.columns:
        df["dataset_row_id"] = pd.to_numeric(df["dataset_row_id"], errors="raise").astype(int)
    return df


def load_note_texts_for_rows(rows_df: pd.DataFrame, pickle_dir: Path) -> dict[int, str]:
    needed_by_file: dict[str, dict[str, int]] = {}
    for _, row in rows_df.iterrows():
        filename = row.get("filename")
        note_id = row.get("note_id")
        dataset_row_id = row.get("dataset_row_id")
        if pd.isna(filename) or pd.isna(note_id) or pd.isna(dataset_row_id):
            continue
        filename = str(filename)
        note_id = str(note_id)
        needed_by_file.setdefault(filename, {})[note_id] = int(dataset_row_id)

    dataset_row_to_text: dict[int, str] = {}
    for filename, note_map in needed_by_file.items():
        file_path = pickle_dir / filename
        if not file_path.exists():
            continue
        try:
            with file_path.open("rb") as handle:
                patient_obj = pickle.load(handle)
        except Exception:
            continue
        dsnotes = getattr(patient_obj, "dsnotes", None)
        if dsnotes is None or getattr(dsnotes, "empty", True):
            continue
        for _, note_row in dsnotes.iterrows():
            note_id = str(note_row.get("note_id", ""))
            if note_id not in note_map:
                continue
            dataset_row_to_text[note_map[note_id]] = str(note_row.get("text", "")).strip()
    return dataset_row_to_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train-text privacy screening on a closed-loop candidate manifest.")
    parser.add_argument("--manifest_path", required=True, help="Path to closed_loop_candidate_manifest.jsonl")
    parser.add_argument("--split_manifest_path", required=True, help="Path to split_manifest_note_level.csv")
    parser.add_argument("--output_dir", required=True, help="Directory for privacy screen outputs")
    parser.add_argument("--dataset_path", default=None, help="Optional dataset path for pickle-dir inference")
    parser.add_argument("--pickle_dir", default=None, help="Optional explicit path to pickle_ds_note_hadm_all")
    parser.add_argument("--max_train_texts", type=int, default=50000, help="Max train texts to load")
    parser.add_argument("--top_k_lexical_checks", type=int, default=200, help="Rows to keep for lexical overlap diagnostics")
    return parser.parse_args()


def summary_block(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "n_rows": 0,
            "exact_duplicate_vs_train_count": 0,
            "high_10gram_overlap_vs_train_count": 0,
            "lexical_similarity_ge_0p8_count": 0,
            "max_lexical_similarity": None,
            "mean_lexical_similarity": None,
        }
    lexical = pd.to_numeric(df["nearest_train_lexical_similarity"], errors="coerce")
    return {
        "n_rows": int(len(df)),
        "exact_duplicate_vs_train_count": int(df["exact_duplicate_vs_train_text"].fillna(False).sum()),
        "high_10gram_overlap_vs_train_count": int((pd.to_numeric(df["nearest_train_10gram_overlap_count"], errors="coerce").fillna(0) > 0).sum()),
        "lexical_similarity_ge_0p8_count": int((lexical >= 0.8).fillna(False).sum()),
        "max_lexical_similarity": None if lexical.dropna().empty else float(lexical.max()),
        "mean_lexical_similarity": None if lexical.dropna().empty else float(lexical.mean()),
    }


def ten_grams(text: str) -> set[str]:
    tokens = normalize_text(text).lower().split()
    if len(tokens) < 10:
        return set()
    return {" ".join(tokens[i : i + 10]) for i in range(len(tokens) - 10 + 1)}


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    split_manifest_path = Path(args.split_manifest_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_df = pd.read_json(manifest_path, lines=True)
    split_df = load_split_manifest(split_manifest_path)

    explicit_pickle_dir = Path(args.pickle_dir).resolve() if args.pickle_dir else None
    dataset_path = Path(args.dataset_path).resolve() if args.dataset_path else manifest_path.parent
    resolved_pickle_dir = infer_pickle_dir(dataset_path=dataset_path, source_dataset_path=None, explicit_pickle_dir=explicit_pickle_dir)
    if resolved_pickle_dir is None:
        raise FileNotFoundError("Could not resolve pickle_ds_note_hadm_all for train-text screening")

    train_rows = split_df.loc[split_df["split"].astype(str) == "train"].copy().sort_values("dataset_row_id").reset_index(drop=True)
    if args.max_train_texts and len(train_rows) > int(args.max_train_texts):
        train_rows = train_rows.head(int(args.max_train_texts)).copy()

    train_text_map = load_note_texts_for_rows(train_rows, resolved_pickle_dir)
    train_hashes = set()
    train_10gram_map: dict[int, set[str]] = {}
    for row_id, text in train_text_map.items():
        norm = normalize_text(str(text))
        if not norm:
            continue
        train_hashes.add(text_hash(norm))
        train_10gram_map[int(row_id)] = ten_grams(norm)

    candidate_df["generated_text_hash_recomputed"] = candidate_df["generated_text"].fillna("").astype(str).map(text_hash)
    candidate_df["exact_duplicate_vs_train_text"] = candidate_df["generated_text_hash_recomputed"].isin(train_hashes)

    suspicious = candidate_df.sort_values("source_synthetic_cosine", ascending=False).head(int(args.top_k_lexical_checks)).copy()
    suspicious["nearest_train_dataset_row_id"] = -1
    suspicious["nearest_train_10gram_overlap_count"] = np.nan
    suspicious["nearest_train_lexical_similarity"] = np.nan

    train_text_items = list(train_text_map.items())
    for idx, row in suspicious.iterrows():
        text = normalize_text(str(row.get("generated_text", "")))
        if not text:
            continue
        row_10grams = ten_grams(text)
        best_row_id = -1
        best_overlap = -1
        best_lexical = np.nan
        for train_row_id, train_text in train_text_items:
            overlap = len(row_10grams & train_10gram_map.get(int(train_row_id), set())) if row_10grams else 0
            if overlap > best_overlap:
                best_overlap = overlap
                best_row_id = int(train_row_id)
                best_lexical = lexical_similarity_ratio(text, str(train_text))
        suspicious.at[idx, "nearest_train_dataset_row_id"] = best_row_id
        suspicious.at[idx, "nearest_train_10gram_overlap_count"] = best_overlap if best_overlap >= 0 else np.nan
        suspicious.at[idx, "nearest_train_lexical_similarity"] = best_lexical

    candidate_df = candidate_df.merge(
        suspicious[
            [
                "candidate_id",
                "nearest_train_dataset_row_id",
                "nearest_train_10gram_overlap_count",
                "nearest_train_lexical_similarity",
            ]
        ],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    candidate_df["high_10gram_overlap_vs_train_flag"] = pd.to_numeric(
        candidate_df["nearest_train_10gram_overlap_count"], errors="coerce"
    ).fillna(0) > 0

    out_csv = output_dir / "closed_loop_train_text_privacy_screen.csv"
    candidate_df.to_csv(out_csv, index=False)

    accepted_df = candidate_df.loc[candidate_df["accepted_flag"].fillna(False)].copy()
    rejected_df = candidate_df.loc[~candidate_df["accepted_flag"].fillna(False)].copy()
    summary = {
        "manifest_path": str(manifest_path),
        "split_manifest_path": str(split_manifest_path),
        "pickle_dir": str(resolved_pickle_dir),
        "n_train_texts_loaded": int(len(train_text_map)),
        "overall": summary_block(candidate_df),
        "accepted": summary_block(accepted_df),
        "rejected": summary_block(rejected_df),
    }
    out_json = output_dir / "closed_loop_train_text_privacy_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    flagged = candidate_df.loc[
        candidate_df["exact_duplicate_vs_train_text"].fillna(False)
        | candidate_df["high_10gram_overlap_vs_train_flag"].fillna(False)
    ].copy()
    flagged_out = output_dir / "closed_loop_train_text_privacy_flagged.csv"
    flagged.to_csv(flagged_out, index=False)

    print(f"Saved privacy screen CSV to: {out_csv}")
    print(f"Saved privacy summary JSON to: {out_json}")
    print(f"Saved flagged rows CSV to: {flagged_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
