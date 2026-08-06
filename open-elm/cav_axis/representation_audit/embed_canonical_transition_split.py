#!/usr/bin/env python3
"""Embed a canonical transition split with frozen section-balanced BGE pooling.

The input manifest contains source-derived text and remains on approved project
storage. Output metadata intentionally excludes that text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


FIELDS = (
    "principal_diagnosis",
    "hospital_course_events",
    "discharge_medications",
    "disposition",
    "instructions",
)
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
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--spec_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--embedding_model_name", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--document_batch_size", type=int, default=64)
    parser.add_argument("--chunk_tokens", type=int, default=448)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    return parser.parse_args()


def normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


def split_sections(text: str) -> dict[str, str]:
    values = {field: "" for field in FIELDS}
    recognized = []
    for match in HEADING_PATTERN.finditer(text):
        heading = match.group(1).strip().lower()
        for field, aliases in SECTION_ALIASES.items():
            if heading in aliases:
                recognized.append((match, field))
                break
    # Only canonical section labels delimit content. Clinical prose such as
    # "Coronary Artery Disease:" must remain inside the diagnosis section.
    for index, (match, field) in enumerate(recognized):
        end = recognized[index + 1][0].start() if index + 1 < len(recognized) else len(text)
        values[field] = text[match.end() : end].strip()
    return values


def chunks_for_text(text: str, tokenizer, chunk_tokens: int, overlap: int) -> list[tuple[str, int]]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError("Canonical required section is empty.")
    stride = chunk_tokens - overlap
    chunks = []
    for start in range(0, len(token_ids), stride):
        current = token_ids[start : start + chunk_tokens]
        chunks.append((tokenizer.decode(current, skip_special_tokens=True), len(current)))
        if start + chunk_tokens >= len(token_ids):
            break
    return chunks


def embed_field(texts: list[str], tokenizer, model, args: argparse.Namespace) -> np.ndarray:
    chunks, owners, weights = [], [], []
    for owner, text in enumerate(texts):
        for chunk, weight in chunks_for_text(text, tokenizer, args.chunk_tokens, args.chunk_overlap):
            chunks.append(chunk)
            owners.append(owner)
            weights.append(weight)
    encoded = model.encode(
        chunks,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    output = np.zeros((len(texts), encoded.shape[1]), dtype=np.float32)
    totals = np.zeros(len(texts), dtype=np.float32)
    for vector, owner, weight in zip(encoded, owners, weights):
        output[owner] += vector * weight
        totals[owner] += weight
    return normalize(output / totals[:, None])


def count_rows(path: Path, max_rows: int | None) -> int:
    with path.open() as handle:
        count = sum(1 for line in handle if line.strip())
    return min(count, max_rows) if max_rows else count


def flush_batch(batch, embeddings, processed, tokenizer, model, args, spec, spec_sha256, metadata) -> int:
    texts_by_field = {field: [] for field in FIELDS}
    for source_index, row in batch:
        sections = split_sections(str(row["generated_text"]))
        missing = [field for field in FIELDS if not sections[field]]
        if missing:
            raise ValueError(f"{row.get('case_id')} is missing canonical sections: {missing}")
        for field in FIELDS:
            texts_by_field[field].append(sections[field])
    vectors = normalize(sum(embed_field(texts_by_field[field], tokenizer, model, args) for field in FIELDS))
    embeddings[processed : processed + len(batch)] = vectors
    for source_index, row in batch:
        metadata.write(json.dumps({
            key: row[key] for key in (
                "rescue_id", "candidate_index", "dataset_row_id", "note_id", "case_id",
                "source_split", "review_stratum", "patient_disjoint_from_train",
            ) if key in row
        } | {
            "source_index": source_index,
            "representation_id": spec["representation_id"],
            "representation_spec_sha256": spec_sha256,
        }) + "\n")
    return processed + len(batch)


def main() -> None:
    args = parse_args()
    if args.chunk_tokens <= 0 or args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_tokens:
        raise ValueError("Require chunk_tokens > 0 and 0 <= chunk_overlap < chunk_tokens.")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require shard_count > 0 and 0 <= shard_index < shard_count.")
    manifest_path = Path(args.manifest_path).resolve()
    spec_path = Path(args.spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    if tuple(spec["required_sections"]) != FIELDS:
        raise ValueError("Spec required_sections do not match the frozen five-section representation.")
    n_source_rows = count_rows(manifest_path, args.max_rows)
    if not n_source_rows:
        raise ValueError("No rows found in canonical manifest.")
    n_rows = (n_source_rows + args.shard_count - 1 - args.shard_index) // args.shard_count
    if n_rows <= 0:
        raise ValueError("This shard has no rows to embed.")
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name, local_files_only=True)
    if tokenizer.model_max_length < args.chunk_tokens + 2:
        raise ValueError("chunk_tokens leaves no room for model special tokens.")
    # Tokenization here only plans manual chunks; model inputs remain <= 448 tokens.
    tokenizer.model_max_length = 10**9
    model = SentenceTransformer(args.embedding_model_name, device=args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dimension = model.get_sentence_embedding_dimension()
    embeddings_path = output_dir / "canonical_section_balanced_embeddings.npy"
    embeddings = np.lib.format.open_memmap(embeddings_path, mode="w+", dtype=np.float32, shape=(n_rows, dimension))
    metadata_path = output_dir / "canonical_section_balanced_metadata.jsonl"
    spec_sha256 = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    processed = 0
    with manifest_path.open() as source, metadata_path.open("w") as metadata:
        batch = []
        source_index = 0
        for line in source:
            if not line.strip():
                continue
            if source_index >= n_source_rows:
                break
            row_index = source_index
            source_index += 1
            if row_index % args.shard_count != args.shard_index:
                continue
            batch.append((row_index, json.loads(line)))
            if len(batch) == args.document_batch_size:
                processed = flush_batch(batch, embeddings, processed, tokenizer, model, args, spec, spec_sha256, metadata)
                print(f"shard {args.shard_index}: embedded {processed}/{n_rows}", flush=True)
                batch = []
        if batch:
            processed = flush_batch(batch, embeddings, processed, tokenizer, model, args, spec, spec_sha256, metadata)
            print(f"shard {args.shard_index}: embedded {processed}/{n_rows}", flush=True)
    embeddings.flush()
    summary = {
        "representation_id": spec["representation_id"],
        "representation_spec_sha256": spec_sha256,
        "embedding_model_name": args.embedding_model_name,
        "n_rows": processed,
        "n_source_rows": n_source_rows,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "embedding_dimension": dimension,
        "chunk_tokens": args.chunk_tokens,
        "chunk_overlap": args.chunk_overlap,
        "aggregation": "token_count_weighted_chunk_mean_then_l2_normalize",
        "canonical_aggregation": "equal_weight_required_section_means_then_l2_normalize",
        "security_note": "Embeddings and metadata exclude source-derived note text.",
    }
    (output_dir / "canonical_section_balanced_embedding_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
