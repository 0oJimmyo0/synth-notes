#!/usr/bin/env python3
"""Compare frozen legacy, chunked, and section-balanced BGE representations.

This is a no-generation diagnostic. It emits embeddings and derived metrics only,
never clinical text. It does not fit target geometry or assign target regions.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


SECTION_ALIASES = {
    "principal_diagnosis": ("discharge diagnosis", "principal diagnosis", "diagnosis"),
    "hospital_course_events": ("brief hospital course", "hospital course"),
    "discharge_medications": ("discharge medications",),
    "disposition": ("disposition",),
    "instructions": ("discharge instructions", "instructions"),
}
HEADING_PATTERN = re.compile(r"(?im)^\s*([A-Za-z][A-Za-z /&-]{2,80}):\s*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_reference_csv", required=True)
    parser.add_argument("--canonical_manifest_path", required=True)
    parser.add_argument("--hybrid_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--embedding_model_name", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_tokens", type=int, default=448)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    return parser.parse_args()


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def read_jsonl(path: str) -> pd.DataFrame:
    return pd.read_json(Path(path).resolve(), lines=True)


def chunk_text(text: str, tokenizer, chunk_tokens: int, overlap: int) -> list[tuple[str, int]]:
    if chunk_tokens <= 0 or overlap < 0 or overlap >= chunk_tokens:
        raise ValueError("Require chunk_tokens > 0 and 0 <= chunk_overlap < chunk_tokens.")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return [("", 0)]
    stride = chunk_tokens - overlap
    chunks = []
    for start in range(0, len(token_ids), stride):
        current = token_ids[start : start + chunk_tokens]
        chunks.append((tokenizer.decode(current, skip_special_tokens=True), len(current)))
        if start + chunk_tokens >= len(token_ids):
            break
    return chunks


def embed_chunked(texts: list[str], tokenizer, model, args: argparse.Namespace) -> np.ndarray:
    chunks: list[str] = []
    owners: list[int] = []
    weights: list[int] = []
    for owner, text in enumerate(texts):
        for chunk, weight in chunk_text(str(text), tokenizer, args.chunk_tokens, args.chunk_overlap):
            chunks.append(chunk)
            owners.append(owner)
            weights.append(weight)
    embeddings = model.encode(
        chunks,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    result = np.zeros((len(texts), embeddings.shape[1]), dtype=np.float32)
    totals = np.zeros(len(texts), dtype=np.float32)
    for embedding, owner, weight in zip(embeddings, owners, weights):
        result[owner] += embedding * weight
        totals[owner] += weight
    return normalize(result / np.clip(totals[:, None], 1.0, None))


def split_sections(text: str) -> dict[str, str]:
    result = {field: "" for field in SECTION_ALIASES}
    recognized = []
    for match in HEADING_PATTERN.finditer(text):
        heading = match.group(1).strip().lower()
        for field, aliases in SECTION_ALIASES.items():
            if heading in aliases:
                recognized.append((match, field))
                break
    # Restrict delimiters to actual section labels so colons in clinical prose
    # cannot truncate a canonical field.
    for index, (match, field) in enumerate(recognized):
        end = recognized[index + 1][0].start() if index + 1 < len(recognized) else len(text)
        result[field] = text[match.end() : end].strip()
    return result


def embed_section_balanced(texts: list[str], tokenizer, model, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    fields = list(SECTION_ALIASES)
    by_field = {field: [] for field in fields}
    valid = np.ones(len(texts), dtype=bool)
    for index, text in enumerate(texts):
        sections = split_sections(str(text))
        for field in fields:
            by_field[field].append(sections[field])
            valid[index] &= bool(sections[field])
    output = None
    for field in fields:
        vectors = embed_chunked(by_field[field], tokenizer, model, args)
        output = vectors if output is None else output + vectors
    assert output is not None
    return normalize(output), valid


def paired_metrics(
    real: np.ndarray,
    real_cases: list[str],
    hybrid: np.ndarray,
    hybrid_cases: list[str],
) -> pd.DataFrame:
    similarities = hybrid @ real.T
    real_index = {case_id: index for index, case_id in enumerate(real_cases)}
    rows = []
    for index, case_id in enumerate(hybrid_cases):
        anchor = real_index[case_id]
        order = np.argsort(-similarities[index])
        rank = int(np.where(order == anchor)[0][0]) + 1
        rows.append({
            "case_id": case_id,
            "same_anchor_cosine": float(similarities[index, anchor]),
            "same_anchor_retrieval_rank": rank,
            "same_anchor_top1": bool(rank == 1),
        })
    return pd.DataFrame(rows)


def neighbor_overlap(left: np.ndarray, right: np.ndarray, case_ids: list[str], k: int = 5) -> pd.DataFrame:
    left_sim = left @ left.T
    right_sim = right @ right.T
    rows = []
    for index, case_id in enumerate(case_ids):
        left_order = [item for item in np.argsort(-left_sim[index]) if item != index][:k]
        right_order = [item for item in np.argsort(-right_sim[index]) if item != index][:k]
        overlap = len(set(left_order) & set(right_order)) / k
        rows.append({"case_id": case_id, "k": k, "neighbor_overlap": overlap})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    source = pd.read_csv(Path(args.source_reference_csv).resolve())
    canonical = read_jsonl(args.canonical_manifest_path)
    hybrid = read_jsonl(args.hybrid_manifest_path)
    if {"case_id", "source_real_note"}.difference(source.columns):
        raise KeyError("source reference must include case_id and source_real_note")
    for name, frame in {"canonical": canonical, "hybrid": hybrid}.items():
        if {"case_id", "generated_text"}.difference(frame.columns):
            raise KeyError(f"{name} manifest must include case_id and generated_text")

    source = source[["case_id", "source_real_note"]].drop_duplicates("case_id")
    canonical = canonical.drop_duplicates("case_id")
    common = set(source.case_id.astype(str)) & set(canonical.case_id.astype(str)) & set(hybrid.case_id.astype(str))
    if not common:
        raise ValueError("No common case IDs across the three frozen inputs.")
    source = source[source.case_id.astype(str).isin(common)].copy().sort_values("case_id")
    canonical = canonical[canonical.case_id.astype(str).isin(common)].copy().sort_values("case_id")
    hybrid = hybrid[hybrid.case_id.astype(str).isin(common)].copy().sort_values(["case_id", "candidate_index"])
    source.case_id = source.case_id.astype(str)
    canonical.case_id = canonical.case_id.astype(str)
    hybrid.case_id = hybrid.case_id.astype(str)

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name, local_files_only=True)
    if tokenizer.model_max_length < args.chunk_tokens + 2:
        raise ValueError("chunk_tokens leaves no room for model special tokens.")
    # The tokenizer plans manual chunks; SentenceTransformer only receives chunks.
    tokenizer.model_max_length = 10**9
    model = SentenceTransformer(args.embedding_model_name, device=args.device)
    real_raw = source.source_real_note.fillna("").astype(str).tolist()
    real_canonical = canonical.generated_text.fillna("").astype(str).tolist()
    hybrid_text = hybrid.generated_text.fillna("").astype(str).tolist()

    # Legacy calls intentionally match the historical whole-document API.
    legacy_real = model.encode(real_raw, batch_size=args.batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    legacy_hybrid = model.encode(hybrid_text, batch_size=args.batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    chunked_real = embed_chunked(real_raw, tokenizer, model, args)
    chunked_hybrid = embed_chunked(hybrid_text, tokenizer, model, args)
    canonical_real, canonical_real_valid = embed_section_balanced(real_canonical, tokenizer, model, args)
    canonical_hybrid, canonical_hybrid_valid = embed_section_balanced(hybrid_text, tokenizer, model, args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "representation_audit_embeddings.npz",
        legacy_real=legacy_real,
        legacy_hybrid=legacy_hybrid,
        chunked_real=chunked_real,
        chunked_hybrid=chunked_hybrid,
        canonical_real=canonical_real,
        canonical_hybrid=canonical_hybrid,
    )
    real_cases = source.case_id.tolist()
    hybrid_cases = hybrid.case_id.tolist()
    all_metrics = []
    for representation, real, candidate, valid in (
        ("legacy_prefix", legacy_real, legacy_hybrid, np.ones(len(hybrid), dtype=bool)),
        ("chunked_full_note", chunked_real, chunked_hybrid, np.ones(len(hybrid), dtype=bool)),
        ("canonical_section_balanced", canonical_real, canonical_hybrid, canonical_hybrid_valid),
    ):
        metrics = paired_metrics(real, real_cases, candidate, hybrid_cases)
        metrics["representation"] = representation
        metrics["representation_valid"] = valid
        metrics["candidate_id"] = hybrid.get("rescue_id", pd.Series(range(len(hybrid)))).astype(str).tolist()
        metrics["candidate_index"] = hybrid.get("candidate_index", pd.Series(0, index=hybrid.index)).tolist()
        metrics["patient_disjoint_from_train"] = hybrid.get("patient_disjoint_from_train", pd.Series(False, index=hybrid.index)).astype(bool).tolist()
        all_metrics.append(metrics)
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(output_dir / "representation_audit_candidate_metrics.csv", index=False)
    summary = metrics.groupby("representation", as_index=False).agg(
        n_candidates=("case_id", "size"),
        valid_rate=("representation_valid", "mean"),
        mean_same_anchor_cosine=("same_anchor_cosine", "mean"),
        median_same_anchor_cosine=("same_anchor_cosine", "median"),
        mean_retrieval_rank=("same_anchor_retrieval_rank", "mean"),
        top1_retrieval_rate=("same_anchor_top1", "mean"),
    )
    summary.to_csv(output_dir / "representation_audit_summary.csv", index=False)
    neighbors = pd.concat([
        neighbor_overlap(legacy_real, chunked_real, real_cases).assign(comparison="legacy_prefix_vs_chunked_full_note"),
        neighbor_overlap(chunked_real, canonical_real, real_cases).assign(comparison="chunked_full_note_vs_canonical_section_balanced"),
    ], ignore_index=True)
    neighbors.to_csv(output_dir / "representation_audit_neighbor_overlap.csv", index=False)
    metadata = {
        "scope": "frozen_28_anchor_no_generation_diagnostic",
        "embedding_model_name": args.embedding_model_name,
        "tokenizer_max_length": int(tokenizer.model_max_length),
        "chunk_tokens": args.chunk_tokens,
        "chunk_overlap": args.chunk_overlap,
        "aggregation": "token_count_weighted_chunk_mean_then_l2_normalize",
        "canonical_aggregation": "equal_weight_required_section_means_then_l2_normalize",
        "n_real_anchors": len(source),
        "n_hybrid_candidates": len(hybrid),
        "geometry_claim": "none; refit real train/dev geometry before any target-region evaluation",
    }
    (output_dir / "representation_audit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
