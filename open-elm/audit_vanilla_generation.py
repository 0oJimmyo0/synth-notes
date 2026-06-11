#!/usr/bin/env python3
"""
Phase 1 audit pipeline for manifest-driven vanilla ELM generation.

This script is intentionally limited to vanilla generation auditing:
- manifest integrity checks
- basic quality checks
- faithfulness checks via re-embedding
- leakage-stratified summaries
- lightweight privacy / memorization screens
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer


MIMIC_MM_PATH = "/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/MIMIC-MM-Dataset-main"
if MIMIC_MM_PATH not in sys.path:
    sys.path.insert(0, MIMIC_MM_PATH)
try:
    import minimal_API  # noqa: F401
except Exception:
    minimal_API = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit manifest-driven vanilla generation outputs.")
    parser.add_argument("--manifest_path", required=True, help="Path to the vanilla generation JSONL manifest")
    parser.add_argument("--dataset_path", required=True, help="Path to encoded_testing_filtered")
    parser.add_argument(
        "--split_manifest_path",
        default=None,
        help="Path to split_manifest_note_level.csv from leakage audit",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for audit outputs")
    parser.add_argument(
        "--embedding_model_name",
        default="BAAI/bge-large-en-v1.5",
        help="Sentence embedding model used for re-embedding generated notes",
    )
    parser.add_argument(
        "--sample_size_for_manual_review",
        type=int,
        default=50,
        help="Number of generated notes to sample for manual review",
    )
    return parser.parse_args()


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_commit(script_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(script_dir.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sentence_transformers

        versions["sentence_transformers"] = sentence_transformers.__version__
    except Exception:
        pass
    try:
        import datasets

        versions["datasets"] = datasets.__version__
    except Exception:
        pass
    return versions


def infer_base_dir(dataset_path: Path) -> Path | None:
    for parent in dataset_path.resolve().parents:
        if parent.name == "3.1":
            return parent
    return None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def collapse_flag(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    lines = [line.strip().lower() for line in stripped.splitlines() if line.strip()]
    if lines:
        line_counts = Counter(lines)
        if max(line_counts.values()) >= 3:
            return True

    tokens = re.findall(r"\b\w+\b", stripped.lower())
    if len(tokens) < 80:
        return False

    unique_ratio = len(set(tokens)) / len(tokens)
    if unique_ratio < 0.2:
        return True

    ngram_size = 8
    ngrams = [tuple(tokens[i : i + ngram_size]) for i in range(len(tokens) - ngram_size + 1)]
    if ngrams:
        counts = Counter(ngrams)
        if max(counts.values()) >= 3:
            return True
    return False


SECTION_HEADERS = [
    "Chief Complaint:",
    "History of Present Illness:",
    "Past Medical History:",
    "Physical Exam:",
    "Discharge Diagnosis:",
    "Discharge Medications:",
    "Allergies:",
    "Service:",
]


PHI_PATTERNS = {
    "ssn_like": r"\b\d{3}-\d{2}-\d{4}\b",
    "phone_like": r"\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b",
    "email_like": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "mrn_like": r"\b(?:MRN|Medical Record Number)[:\s#-]*\d{5,}\b",
    "id_like": r"\b(?:ID|Acct|Account)[:\s#-]*\d{5,}\b",
}


def section_sanity(text: str) -> dict[str, Any]:
    present = [header for header in SECTION_HEADERS if header.lower() in text.lower()]
    return {
        "section_header_count": len(present),
        "has_minimum_section_structure": len(present) >= 3,
        "section_headers_present": "|".join(present) if present else "",
    }


def quality_metrics_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    words = stripped.split() if stripped else []
    chars = len(stripped)
    return {
        "generated_word_count_recomputed": len(words),
        "generated_char_count_recomputed": chars,
        "empty_output_flag_recomputed": chars == 0,
        "too_short_flag_recomputed": len(words) < 100,
        "repetition_or_collapse_flag_recomputed": collapse_flag(stripped),
        **section_sanity(stripped),
    }


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_json(manifest_path, lines=True, dtype=False)
    df = df.reset_index(drop=True)
    return df


def ensure_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapped = series.map(lambda x: True if x is True or str(x).lower() == "true" else False if x is False or str(x).lower() == "false" else pd.NA)
    return mapped.astype("boolean")


def load_split_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "dataset_row_id" in df.columns:
        df["dataset_row_id"] = pd.to_numeric(df["dataset_row_id"], errors="raise").astype(int)
    return df


def integrity_checks(manifest_df: pd.DataFrame, dataset_len: int, split_manifest_df: pd.DataFrame | None) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []

    if len(manifest_df) != dataset_len:
        issues.append(f"Manifest row count {len(manifest_df)} does not match dataset row count {dataset_len}.")

    if manifest_df["generation_id"].duplicated().any():
        issues.append("Duplicate generation_id values detected.")

    if manifest_df["generated_text"].isna().any():
        issues.append("Manifest contains null generated_text values.")

    for col in ["note_id", "subject_id", "hadm_id"]:
        if col in manifest_df.columns:
            if (manifest_df[col].astype("string").str.strip() == "").any():
                issues.append(f"Manifest contains empty-string values in {col}.")

    if "split" in manifest_df.columns and not (manifest_df["split"] == "test").all():
        issues.append("Manifest contains rows where split != test.")

    if "generation_condition" in manifest_df.columns and not (manifest_df["generation_condition"] == "vanilla").all():
        issues.append("Manifest contains rows where generation_condition != vanilla.")

    leakage_cols = [
        "patient_disjoint_from_train",
        "hadm_disjoint_from_train",
        "note_disjoint_from_train",
    ]
    if split_manifest_df is not None:
        missing_leakage = [col for col in leakage_cols if col not in manifest_df.columns]
        if missing_leakage:
            issues.append(f"Manifest is missing leakage flag columns: {missing_leakage}")

    disallowed_text_cols = [
        "source_text",
        "source_note_text",
        "real_note_text",
        "original_note_text",
        "ground_truth_text",
    ]
    stored_source_text_cols = [col for col in disallowed_text_cols if col in manifest_df.columns and manifest_df[col].notna().any()]
    if stored_source_text_cols:
        issues.append(f"Manifest stores disallowed source text columns: {stored_source_text_cols}")

    param_cols = ["temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens", "seed"]
    param_consistency = {}
    for col in param_cols:
        if col in manifest_df.columns:
            unique_vals = sorted(manifest_df[col].dropna().astype(str).unique().tolist())
            param_consistency[col] = unique_vals
            if len(unique_vals) > 1:
                warnings.append(f"Manifest has multiple values for {col}: {unique_vals}")

    if "dataset_row_id" in manifest_df.columns:
        expected = list(range(len(manifest_df)))
        actual = manifest_df["dataset_row_id"].astype(int).tolist()
        if actual != expected:
            issues.append("dataset_row_id order does not match manifest row order.")

    if split_manifest_df is not None:
        split_subset = split_manifest_df.loc[split_manifest_df["split"] == "test"].copy()
        split_subset = split_subset.sort_values("dataset_row_id").reset_index(drop=True)
        if len(split_subset) != len(manifest_df):
            warnings.append(
                f"Split manifest test subset has {len(split_subset)} rows, manifest has {len(manifest_df)} rows."
            )
        else:
            compare_cols = [col for col in leakage_cols if col in manifest_df.columns and col in split_subset.columns]
            for col in compare_cols:
                left = ensure_bool_series(manifest_df[col]).reset_index(drop=True)
                right = ensure_bool_series(split_subset[col]).reset_index(drop=True)
                mismatch = (left != right).fillna(False)
                if mismatch.any():
                    issues.append(f"Leakage flag mismatch for column {col} against split manifest.")

    status = "PASS" if not issues else "FAIL"
    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "parameter_consistency": param_consistency,
        "row_count": int(len(manifest_df)),
        "dataset_row_count": int(dataset_len),
    }


def encode_generated_notes(model_name: str, texts: list[str]) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def extract_source_embeddings(dataset_path: Path) -> np.ndarray:
    dataset = Dataset.load_from_disk(str(dataset_path))
    rows = []
    for example in dataset:
        emb = example["domain_embeddings"][0]
        if isinstance(emb, np.ndarray):
            arr = emb
        else:
            arr = np.asarray(emb)
        rows.append(arr)
    matrix = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def iqr(values: np.ndarray) -> float:
    if len(values) == 0:
        return math.nan
    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def summarize_numeric(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {
            "mean": math.nan,
            "median": math.nan,
            "iqr": math.nan,
            "p05": math.nan,
            "p25": math.nan,
            "p75": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "iqr": iqr(values),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def compute_topk_self_retrieval(
    source_embeddings: np.ndarray,
    generated_embeddings: np.ndarray,
    top_k: int = 10,
) -> pd.DataFrame:
    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(source_embeddings.shape[1])
        index.add(source_embeddings.astype(np.float32))
        scores, indices = index.search(generated_embeddings.astype(np.float32), top_k)
    except Exception:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=top_k, metric="cosine", algorithm="brute")
        nn.fit(source_embeddings)
        distances, indices = nn.kneighbors(generated_embeddings, return_distance=True)
        scores = 1.0 - distances

    rows = []
    for i in range(len(generated_embeddings)):
        hits = indices[i].tolist()
        rank = hits.index(i) + 1 if i in hits else None
        rows.append(
            {
                "dataset_row_id": i,
                "source_retrieval_rank_top10": rank,
                "source_in_top1": bool(rank == 1),
                "source_in_top5": bool(rank is not None and rank <= 5),
                "source_in_top10": bool(rank is not None and rank <= 10),
                "retrieved_top1_score": float(scores[i][0]),
            }
        )
    return pd.DataFrame(rows)


def infer_training_dataset_path(test_dataset_path: Path) -> Path | None:
    sibling = test_dataset_path.parent / "encoded_training_filtered"
    return sibling if sibling.exists() else None


def infer_pickle_dir(dataset_path: Path) -> Path | None:
    base_dir = infer_base_dir(dataset_path)
    if base_dir is None:
        return None
    candidate = base_dir / "pickle_ds_note_hadm_all"
    return candidate if candidate.exists() else None


def load_note_texts_for_rows(rows_df: pd.DataFrame, pickle_dir: Path) -> dict[int, str]:
    needed_by_file: dict[str, dict[str, int]] = defaultdict(dict)
    for _, row in rows_df.iterrows():
        filename = row.get("filename")
        note_id = row.get("note_id")
        dataset_row_id = row.get("dataset_row_id")
        if pd.isna(filename) or pd.isna(note_id) or pd.isna(dataset_row_id):
            continue
        needed_by_file[str(filename)][str(note_id)] = int(dataset_row_id)

    dataset_row_to_text: dict[int, str] = {}
    for filename, note_map in needed_by_file.items():
        file_path = pickle_dir / filename
        if not file_path.exists():
            continue
        try:
            with file_path.open("rb") as f:
                patient_obj = pickle.load(f)
        except Exception:
            continue

        dsnotes = getattr(patient_obj, "dsnotes", None)
        if dsnotes is None or getattr(dsnotes, "empty", True):
            continue
        for _, note_row in dsnotes.iterrows():
            note_id = str(note_row.get("note_id", ""))
            if note_id not in note_map:
                continue
            text = str(note_row.get("text", "")).strip()
            dataset_row_to_text[note_map[note_id]] = text
    return dataset_row_to_text


def build_train_hash_set(train_split_df: pd.DataFrame, pickle_dir: Path) -> set[str]:
    row_texts = load_note_texts_for_rows(train_split_df, pickle_dir)
    return {text_hash(text) for text in row_texts.values() if text.strip()}


def max_ngram_overlap_count(a: str, b: str, n: int = 10) -> int:
    tokens_a = normalize_text(a).lower().split()
    tokens_b = normalize_text(b).lower().split()
    if len(tokens_a) < n or len(tokens_b) < n:
        return 0
    ngrams_a = Counter(tuple(tokens_a[i : i + n]) for i in range(len(tokens_a) - n + 1))
    ngrams_b = Counter(tuple(tokens_b[i : i + n]) for i in range(len(tokens_b) - n + 1))
    overlap = 0
    for key in set(ngrams_a) & set(ngrams_b):
        overlap += min(ngrams_a[key], ngrams_b[key])
    return overlap


def lexical_similarity_ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return float(SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio())


def group_label(disjoint_value: Any) -> str:
    if disjoint_value is True or str(disjoint_value).lower() == "true":
        return "patient_disjoint"
    if disjoint_value is False or str(disjoint_value).lower() == "false":
        return "patient_overlap"
    return "unknown"


def compute_group_summary(df: pd.DataFrame, group_name: str) -> dict[str, Any]:
    return {
        "group": group_name,
        "n_rows": int(len(df)),
        "empty_rate": float(df["empty_output_flag_recomputed"].mean()) if len(df) else math.nan,
        "too_short_rate": float(df["too_short_flag_recomputed"].mean()) if len(df) else math.nan,
        "collapse_rate": float(df["repetition_or_collapse_flag_recomputed"].mean()) if len(df) else math.nan,
        "mean_word_count": float(df["generated_word_count_recomputed"].mean()) if len(df) else math.nan,
        "median_word_count": float(df["generated_word_count_recomputed"].median()) if len(df) else math.nan,
        "mean_source_cosine": float(df["source_to_generated_cosine"].mean()) if len(df) else math.nan,
        "median_source_cosine": float(df["source_to_generated_cosine"].median()) if len(df) else math.nan,
        "p05_source_cosine": float(df["source_to_generated_cosine"].quantile(0.05)) if len(df) else math.nan,
        "top1_recovery_rate": float(df["source_in_top1"].mean()) if "source_in_top1" in df and len(df) else math.nan,
        "top5_recovery_rate": float(df["source_in_top5"].mean()) if "source_in_top5" in df and len(df) else math.nan,
        "top10_recovery_rate": float(df["source_in_top10"].mean()) if "source_in_top10" in df and len(df) else math.nan,
    }


def phi_scan(text: str) -> dict[str, int]:
    counts = {}
    for name, pattern in PHI_PATTERNS.items():
        counts[name] = len(re.findall(pattern, text))
    return counts


def determine_readiness(integrity: dict[str, Any], quality_df: pd.DataFrame, privacy_summary: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if integrity["issues"]:
        reasons.extend(integrity["issues"])
        return "FAIL", reasons

    empty_rate = float(quality_df["empty_output_flag_recomputed"].mean())
    collapse_rate = float(quality_df["repetition_or_collapse_flag_recomputed"].mean())
    median_cosine = float(quality_df["source_to_generated_cosine"].median())

    if empty_rate > 0.02:
        reasons.append(f"Empty output rate is high ({empty_rate:.2%}).")
    if collapse_rate > 0.10:
        reasons.append(f"Repetition/collapse rate is high ({collapse_rate:.2%}).")
    if median_cosine < 0.75:
        reasons.append(f"Median source cosine is low ({median_cosine:.4f}).")
    if privacy_summary.get("exact_duplicates_vs_train", 0) > 0:
        reasons.append("Exact duplicate(s) against train text detected.")

    if reasons:
        return "CAUTION", reasons
    return "PASS", ["Vanilla generation is ready for coverage mapping."]


def markdown_report(
    audit_summary: dict[str, Any],
    integrity: dict[str, Any],
    group_df: pd.DataFrame,
    privacy_summary: dict[str, Any],
) -> str:
    lines = []
    lines.append("# Vanilla Generation Audit")
    lines.append("")
    lines.append(f"- Created at: `{audit_summary['created_at']}`")
    lines.append(f"- Manifest: `{audit_summary['manifest_path']}`")
    lines.append(f"- Dataset: `{audit_summary['dataset_path']}`")
    lines.append(f"- Status: **{audit_summary['readiness_status']}**")
    lines.append("")
    lines.append("## Manifest Integrity")
    lines.append(f"- Status: **{integrity['status']}**")
    lines.append(f"- Manifest rows: `{integrity['row_count']}`")
    lines.append(f"- Dataset rows: `{integrity['dataset_row_count']}`")
    if integrity["issues"]:
        lines.append("- Issues:")
        for issue in integrity["issues"]:
            lines.append(f"  - {issue}")
    if integrity["warnings"]:
        lines.append("- Warnings:")
        for warning in integrity["warnings"]:
            lines.append(f"  - {warning}")
    lines.append("")
    lines.append("## Leakage-Stratified Summary")
    for _, row in group_df.iterrows():
        lines.append(
            f"- `{row['group']}`: n={int(row['n_rows'])}, "
            f"median cosine={row['median_source_cosine']:.4f}, "
            f"empty={row['empty_rate']:.2%}, "
            f"collapse={row['collapse_rate']:.2%}"
        )
    lines.append("")
    lines.append("## Privacy / Memorization")
    lines.append(f"- Exact duplicate generated notes: `{privacy_summary['exact_duplicate_generated_notes']}`")
    lines.append(f"- Exact duplicates vs train text: `{privacy_summary['exact_duplicates_vs_train']}`")
    lines.append(f"- PHI-like flagged notes: `{privacy_summary['phi_flagged_note_count']}`")
    if privacy_summary.get("train_text_checks_skipped_reason"):
        lines.append(f"- Train text screen: `{privacy_summary['train_text_checks_skipped_reason']}`")
    lines.append("")
    lines.append("## Decision")
    for reason in audit_summary["readiness_reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest_path)
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = load_manifest(manifest_path)
    dataset = Dataset.load_from_disk(str(dataset_path))
    dataset_len = len(dataset)

    split_manifest_df = None
    if args.split_manifest_path and Path(args.split_manifest_path).exists():
        split_manifest_df = load_split_manifest(Path(args.split_manifest_path))

    integrity = integrity_checks(manifest_df, dataset_len, split_manifest_df)

    # Augment / validate leakage fields against split manifest if available.
    if split_manifest_df is not None and "dataset_row_id" in manifest_df.columns:
        split_subset = split_manifest_df.loc[split_manifest_df["split"] == "test"].copy()
        split_subset = split_subset.sort_values("dataset_row_id").reset_index(drop=True)
        manifest_df = manifest_df.merge(
            split_subset[
                [
                    "dataset_row_id",
                    "patient_disjoint_from_train",
                    "hadm_disjoint_from_train",
                    "note_disjoint_from_train",
                    "patient_overlap_with_train",
                    "hadm_overlap_with_train",
                    "note_overlap_with_train",
                ]
            ],
            on="dataset_row_id",
            how="left",
            suffixes=("", "_expected"),
        )
        for col in [
            "patient_disjoint_from_train",
            "hadm_disjoint_from_train",
            "note_disjoint_from_train",
            "patient_overlap_with_train",
            "hadm_overlap_with_train",
            "note_overlap_with_train",
        ]:
            expected_col = f"{col}_expected"
            if expected_col in manifest_df.columns:
                if col not in manifest_df.columns or manifest_df[col].isna().all():
                    manifest_df[col] = manifest_df[expected_col]
                manifest_df.drop(columns=[expected_col], inplace=True)

    quality_rows = [quality_metrics_from_text(text) for text in manifest_df["generated_text"].fillna("").tolist()]
    quality_df = pd.concat([manifest_df.copy(), pd.DataFrame(quality_rows)], axis=1)

    source_embeddings = extract_source_embeddings(dataset_path)
    generated_embeddings = encode_generated_notes(
        args.embedding_model_name,
        quality_df["generated_text"].fillna("").tolist(),
    )

    source_cosine = np.sum(source_embeddings * generated_embeddings, axis=1)
    retrieval_df = compute_topk_self_retrieval(source_embeddings, generated_embeddings, top_k=10)
    quality_df["source_to_generated_cosine"] = source_cosine
    quality_df = quality_df.merge(retrieval_df, on="dataset_row_id", how="left")

    quality_df["leakage_group"] = quality_df["patient_disjoint_from_train"].map(group_label)

    # Train embedding nearest-neighbor privacy screen.
    training_dataset_path = infer_training_dataset_path(dataset_path)
    nearest_train_cosine = np.full(len(quality_df), np.nan, dtype=np.float32)
    nearest_train_index = np.full(len(quality_df), -1, dtype=np.int64)
    if training_dataset_path and training_dataset_path.exists():
        train_embeddings = extract_source_embeddings(training_dataset_path)
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(train_embeddings.shape[1])
            index.add(train_embeddings.astype(np.float32))
            scores, indices = index.search(generated_embeddings.astype(np.float32), 1)
            nearest_train_cosine = scores[:, 0].astype(np.float32)
            nearest_train_index = indices[:, 0].astype(np.int64)
        except Exception:
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
            nn.fit(train_embeddings)
            distances, indices = nn.kneighbors(generated_embeddings, return_distance=True)
            nearest_train_cosine = (1.0 - distances[:, 0]).astype(np.float32)
            nearest_train_index = indices[:, 0].astype(np.int64)

    quality_df["nearest_train_embedding_cosine"] = nearest_train_cosine
    quality_df["nearest_train_dataset_row_id"] = nearest_train_index

    # Exact duplicate within generated corpus.
    normalized_generated = quality_df["generated_text"].fillna("").map(normalize_text)
    generated_hashes = normalized_generated.map(text_hash)
    quality_df["generated_text_hash"] = generated_hashes
    quality_df["exact_duplicate_within_generated"] = generated_hashes.duplicated(keep=False)

    # PHI-like regex screen.
    phi_counts = [phi_scan(text) for text in quality_df["generated_text"].fillna("").tolist()]
    phi_df = pd.DataFrame(phi_counts)
    for col in phi_df.columns:
        quality_df[f"phi_{col}_count"] = phi_df[col]
    phi_count_cols = [col for col in quality_df.columns if col.startswith("phi_") and col.endswith("_count")]
    quality_df["phi_like_flag"] = quality_df[phi_count_cols].sum(axis=1) > 0

    # Train text checks, if accessible.
    train_text_hashes: set[str] | None = None
    train_text_checks_skipped_reason = None
    train_candidate_rows_df = None
    pickle_dir = infer_pickle_dir(dataset_path)
    if split_manifest_df is not None and pickle_dir is not None and training_dataset_path is not None:
        try:
            train_rows = split_manifest_df.loc[split_manifest_df["split"] == "train"].copy()
            train_rows = train_rows.sort_values("dataset_row_id").reset_index(drop=True)
            train_text_hashes = build_train_hash_set(train_rows, pickle_dir)
            quality_df["exact_duplicate_vs_train_text"] = quality_df["generated_text_hash"].isin(train_text_hashes)

            suspicious = quality_df.sort_values("nearest_train_embedding_cosine", ascending=False).head(100).copy()
            suspicious_ids = suspicious["nearest_train_dataset_row_id"].loc[suspicious["nearest_train_dataset_row_id"] >= 0].unique().tolist()
            if suspicious_ids:
                candidate_rows = train_rows.loc[train_rows["dataset_row_id"].isin(suspicious_ids)].copy()
                candidate_text_map = load_note_texts_for_rows(candidate_rows, pickle_dir)
                overlap_counts = []
                lexical_scores = []
                for _, row in quality_df.iterrows():
                    train_row_id = int(row["nearest_train_dataset_row_id"]) if row["nearest_train_dataset_row_id"] >= 0 else -1
                    candidate_text = candidate_text_map.get(train_row_id, "")
                    if candidate_text:
                        overlap_counts.append(max_ngram_overlap_count(row["generated_text"], candidate_text, n=10))
                        lexical_scores.append(lexical_similarity_ratio(row["generated_text"], candidate_text))
                    else:
                        overlap_counts.append(np.nan)
                        lexical_scores.append(np.nan)
                quality_df["nearest_train_10gram_overlap_count"] = overlap_counts
                quality_df["nearest_train_lexical_similarity"] = lexical_scores
                train_candidate_rows_df = candidate_rows
        except Exception as exc:
            train_text_checks_skipped_reason = f"Train text screen failed: {exc}"
            quality_df["exact_duplicate_vs_train_text"] = False
    else:
        train_text_checks_skipped_reason = "Train text resources were not fully accessible."
        quality_df["exact_duplicate_vs_train_text"] = False

    group_rows = [compute_group_summary(quality_df, "full")]
    for group_name in ["patient_disjoint", "patient_overlap"]:
        subset = quality_df.loc[quality_df["leakage_group"] == group_name].copy()
        group_rows.append(compute_group_summary(subset, group_name))
    group_df = pd.DataFrame(group_rows)

    privacy_summary = {
        "exact_duplicate_generated_notes": int(quality_df["exact_duplicate_within_generated"].sum()),
        "exact_duplicates_vs_train": int(quality_df["exact_duplicate_vs_train_text"].sum()),
        "phi_flagged_note_count": int(quality_df["phi_like_flag"].sum()),
        "high_train_embedding_neighbor_count_gt_0_95": int((quality_df["nearest_train_embedding_cosine"] > 0.95).fillna(False).sum()),
        "train_text_checks_skipped_reason": train_text_checks_skipped_reason,
    }

    readiness_status, readiness_reasons = determine_readiness(integrity, quality_df, privacy_summary)

    audit_summary = {
        "created_at": current_timestamp(),
        "script_path": str(Path(__file__).resolve()),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "manifest_path": str(manifest_path.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "split_manifest_path": str(Path(args.split_manifest_path).resolve()) if args.split_manifest_path else None,
        "output_dir": str(output_dir.resolve()),
        "embedding_model_name": args.embedding_model_name,
        "sample_size_for_manual_review": args.sample_size_for_manual_review,
        "package_versions": package_versions(),
        "integrity": integrity,
        "quality_summary": {
            "empty_output_rate": float(quality_df["empty_output_flag_recomputed"].mean()),
            "too_short_rate": float(quality_df["too_short_flag_recomputed"].mean()),
            "repetition_or_collapse_rate": float(quality_df["repetition_or_collapse_flag_recomputed"].mean()),
            "word_count": summarize_numeric(quality_df["generated_word_count_recomputed"].to_numpy(dtype=float)),
            "char_count": summarize_numeric(quality_df["generated_char_count_recomputed"].to_numpy(dtype=float)),
            "section_header_count": summarize_numeric(quality_df["section_header_count"].to_numpy(dtype=float)),
            "minimum_section_structure_rate": float(quality_df["has_minimum_section_structure"].mean()),
        },
        "faithfulness_summary": {
            "source_to_generated_cosine": summarize_numeric(quality_df["source_to_generated_cosine"].to_numpy(dtype=float)),
            "source_in_top1_rate": float(quality_df["source_in_top1"].mean()),
            "source_in_top5_rate": float(quality_df["source_in_top5"].mean()),
            "source_in_top10_rate": float(quality_df["source_in_top10"].mean()),
        },
        "privacy_summary": privacy_summary,
        "readiness_status": readiness_status,
        "readiness_reasons": readiness_reasons,
        "config_snapshot": vars(args),
    }

    # Save tables and report.
    quality_columns = [
        "generation_id",
        "dataset_row_id",
        "note_id",
        "subject_id",
        "hadm_id",
        "patient_disjoint_from_train",
        "leakage_group",
        "generated_word_count",
        "generated_char_count",
        "generated_word_count_recomputed",
        "generated_char_count_recomputed",
        "empty_output_flag",
        "empty_output_flag_recomputed",
        "too_short_flag",
        "too_short_flag_recomputed",
        "repetition_or_collapse_flag",
        "repetition_or_collapse_flag_recomputed",
        "section_header_count",
        "has_minimum_section_structure",
        "phi_like_flag",
    ]
    quality_df.loc[:, [col for col in quality_columns if col in quality_df.columns]].to_csv(
        output_dir / "vanilla_quality_table.csv",
        index=False,
    )

    faithfulness_columns = [
        "generation_id",
        "dataset_row_id",
        "note_id",
        "subject_id",
        "hadm_id",
        "patient_disjoint_from_train",
        "leakage_group",
        "source_to_generated_cosine",
        "source_retrieval_rank_top10",
        "source_in_top1",
        "source_in_top5",
        "source_in_top10",
        "nearest_train_embedding_cosine",
        "nearest_train_dataset_row_id",
        "nearest_train_10gram_overlap_count",
        "nearest_train_lexical_similarity",
        "exact_duplicate_vs_train_text",
    ]
    quality_df.loc[:, [col for col in faithfulness_columns if col in quality_df.columns]].to_csv(
        output_dir / "vanilla_faithfulness_table.csv",
        index=False,
    )

    group_df.to_csv(output_dir / "patient_disjoint_vs_full_metrics.csv", index=False)

    manual_review_df = quality_df.sample(
        n=min(args.sample_size_for_manual_review, len(quality_df)),
        random_state=42,
    ).loc[
        :,
        [
            "generation_id",
            "note_id",
            "subject_id",
            "hadm_id",
            "patient_disjoint_from_train",
            "leakage_group",
            "source_to_generated_cosine",
            "generated_word_count_recomputed",
            "repetition_or_collapse_flag_recomputed",
            "phi_like_flag",
            "generated_text",
        ],
    ]
    manual_review_df.to_csv(output_dir / "manual_review_sample.csv", index=False)

    (output_dir / "generation_audit_baseline.json").write_text(json.dumps(audit_summary, indent=2))
    (output_dir / "generation_audit_baseline.md").write_text(
        markdown_report(audit_summary, integrity, group_df, privacy_summary)
    )

    print("Saved audit outputs to:", output_dir)
    print("Readiness status:", readiness_status)
    for reason in readiness_reasons:
        print("-", reason)


if __name__ == "__main__":
    main()
