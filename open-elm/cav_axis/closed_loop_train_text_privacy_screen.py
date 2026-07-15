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


def load_note_texts_for_rows(rows_df: pd.DataFrame, pickle_dir: Path, progress_every: int = 0) -> dict[int, str]:
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
    total_files = len(needed_by_file)
    for file_index, (filename, note_map) in enumerate(needed_by_file.items(), start=1):
        if progress_every and (file_index == 1 or file_index % progress_every == 0 or file_index == total_files):
            print(f"Loading source-note pickle {file_index}/{total_files}", flush=True)
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
    parser.add_argument("--train_dataset_path", default=None, help="Optional encoded training dataset for semantic nearest-neighbor retrieval")
    parser.add_argument("--embedding_model_name", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--embedding_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--embedding_batch_size", type=int, default=128)
    parser.add_argument("--semantic_top_k", type=int, default=20, help="Nearest train embeddings to lexical-screen per generated note")
    parser.add_argument("--pickle_dir", default=None, help="Optional explicit path to pickle_ds_note_hadm_all")
    parser.add_argument("--max_train_texts", type=int, default=50000, help="Max train texts to load")
    parser.add_argument("--top_k_lexical_checks", type=int, default=200, help="Rows to keep for lexical overlap diagnostics")
    return parser.parse_args()


def extract_embedding(example: dict[str, Any]) -> np.ndarray:
    values = example.get("domain_embeddings")
    if values is None:
        raise KeyError("Dataset is missing domain_embeddings")
    return np.asarray(values[0] if isinstance(values, list) and values and isinstance(values[0], list) else values, dtype=np.float32)


def semantic_shortlist(
    candidate_texts: list[str],
    train_dataset_path: Path,
    model_name: str,
    device: str,
    batch_size: int,
    top_k: int,
) -> dict[int, list[tuple[int, float]]]:
    """Return top-k real-train embedding neighbors for each generated note."""
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer

    print(f"Loading train embeddings from: {train_dataset_path}", flush=True)
    dataset = Dataset.load_from_disk(str(train_dataset_path))
    train_ids = np.asarray(dataset["dataset_row_id"], dtype=int) if "dataset_row_id" in dataset.column_names else np.arange(len(dataset), dtype=int)
    train_embeddings = np.vstack([extract_embedding(example) for example in dataset])
    train_embeddings /= np.clip(np.linalg.norm(train_embeddings, axis=1, keepdims=True), 1e-12, None)
    print(f"Encoding {len(candidate_texts)} generated notes for semantic shortlist on {device}", flush=True)
    model = SentenceTransformer(model_name, device=device)
    generated_embeddings = model.encode(candidate_texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    top_k = min(max(1, top_k), len(train_embeddings))
    result: dict[int, list[tuple[int, float]]] = {}
    for idx, embedding in enumerate(np.asarray(generated_embeddings, dtype=np.float32)):
        scores = train_embeddings @ embedding
        positions = np.argpartition(scores, -top_k)[-top_k:]
        positions = positions[np.argsort(scores[positions])[::-1]]
        result[idx] = [(int(train_ids[pos]), float(scores[pos])) for pos in positions]
    return result


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

    print(f"Loading up to {len(train_rows)} train texts for exact-hash screening", flush=True)
    train_text_map = load_note_texts_for_rows(train_rows, resolved_pickle_dir, progress_every=5000)
    print(f"Loaded {len(train_text_map)} non-empty train texts", flush=True)
    train_hashes = set()
    for row_id, text in train_text_map.items():
        norm = normalize_text(str(text))
        if not norm:
            continue
        train_hashes.add(text_hash(norm))

    candidate_df["generated_text_hash_recomputed"] = candidate_df["generated_text"].fillna("").astype(str).map(text_hash)
    candidate_df["exact_duplicate_vs_train_text"] = candidate_df["generated_text_hash_recomputed"].isin(train_hashes)

    n_lexical = len(candidate_df) if int(args.top_k_lexical_checks) == 0 else min(len(candidate_df), int(args.top_k_lexical_checks))
    suspicious = candidate_df.sort_values("source_synthetic_cosine", ascending=False).head(n_lexical).copy()
    suspicious["nearest_train_dataset_row_id"] = -1
    suspicious["nearest_train_10gram_overlap_count"] = np.nan
    suspicious["nearest_train_lexical_similarity"] = np.nan

    semantic_neighbors: dict[int, list[tuple[int, float]]] = {}
    if args.train_dataset_path:
        semantic_neighbors = semantic_shortlist(
            candidate_df["generated_text"].fillna("").astype(str).tolist(),
            Path(args.train_dataset_path).resolve(),
            args.embedding_model_name,
            args.embedding_device,
            int(args.embedding_batch_size),
            int(args.semantic_top_k),
        )
        candidate_df["semantic_nearest_train_dataset_row_id"] = [semantic_neighbors.get(i, [(-1, np.nan)])[0][0] for i in range(len(candidate_df))]
        candidate_df["semantic_nearest_train_cosine"] = [semantic_neighbors.get(i, [(-1, np.nan)])[0][1] for i in range(len(candidate_df))]
    # With semantic retrieval, construct 10-grams only for shortlisted rows.
    # Building them for every full-train note is unnecessary and can exhaust RAM.
    train_10gram_cache: dict[int, set[str]] = {}
    train_text_items = list(train_text_map.items())
    for idx, row in suspicious.iterrows():
        text = normalize_text(str(row.get("generated_text", "")))
        if not text:
            continue
        row_10grams = ten_grams(text)
        best_row_id = -1
        best_overlap = -1
        best_lexical = np.nan
        shortlist = semantic_neighbors.get(int(idx))
        lexical_items = [(row_id, train_text_map[row_id]) for row_id, _ in shortlist if row_id in train_text_map] if shortlist else train_text_items
        for train_row_id, train_text in lexical_items:
            train_row_id = int(train_row_id)
            if train_row_id not in train_10gram_cache:
                train_10gram_cache[train_row_id] = ten_grams(str(train_text))
            overlap = len(row_10grams & train_10gram_cache[train_row_id]) if row_10grams else 0
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
        "max_train_texts": int(args.max_train_texts),
        "semantic_shortlist_enabled": bool(args.train_dataset_path),
        "semantic_top_k": int(args.semantic_top_k) if args.train_dataset_path else None,
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
