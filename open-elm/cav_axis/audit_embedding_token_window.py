#!/usr/bin/env python3
"""Audit BGE token-window exposure without exporting clinical text."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


HEADING_PATTERN = re.compile(r"(?im)^\s*([A-Za-z][A-Za-z /&-]{2,80}):")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_reference_csv", required=True)
    parser.add_argument("--canonical_manifest_path", required=True)
    parser.add_argument("--hybrid_manifest_path", required=True)
    parser.add_argument("--embedding_model_name", default="BAAI/bge-large-en-v1.5")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Frozen BGE input-window length for this audit (default: 512).",
    )
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def section_at_offset(matches: list[re.Match[str]], offset: int) -> str:
    if not matches:
        return "unlabeled"
    preceding = [match for match in matches if match.start() <= offset]
    return preceding[-1].group(1).strip().lower() if preceding else "preamble"


def audit_texts(frame: pd.DataFrame, tokenizer, max_tokens: int) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        text = str(row.text or "")
        encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True, truncation=False)
        token_count = len(encoded["input_ids"])
        offsets = encoded["offset_mapping"]
        used = min(token_count, max_tokens)
        usable_offsets = [item for item in offsets[:used] if item != (0, 0)]
        start_offset = usable_offsets[0][0] if usable_offsets else 0
        end_offset = usable_offsets[-1][1] if usable_offsets else 0
        heading_matches = list(HEADING_PATTERN.finditer(text))
        headings = sorted({
            match.group(1).strip().lower()
            for match in heading_matches
            if start_offset <= match.start() < end_offset
        })
        rows.append({
            "case_id": str(row.case_id),
            "representation": str(row.representation),
            "document_token_count": int(token_count),
            "tokens_used_by_bge": int(used),
            "truncated_by_bge": bool(token_count > max_tokens),
            "section_at_token_1": section_at_offset(heading_matches, start_offset),
            "section_at_last_bge_token": section_at_offset(heading_matches, end_offset),
            "sections_represented_before_bge_limit": "|".join(headings),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    source = pd.read_csv(Path(args.source_reference_csv).resolve())
    if {"case_id", "source_real_note"}.difference(source.columns):
        raise KeyError("source reference must include case_id and source_real_note")
    canonical = read_jsonl(Path(args.canonical_manifest_path).resolve())
    hybrid = read_jsonl(Path(args.hybrid_manifest_path).resolve())
    for name, frame in {"canonical": canonical, "hybrid": hybrid}.items():
        if {"case_id", "generated_text"}.difference(frame.columns):
            raise KeyError(f"{name} manifest must include case_id and generated_text")
    source_rows = source[["case_id", "source_real_note"]].drop_duplicates("case_id").rename(columns={"source_real_note": "text"})
    canonical_rows = canonical[["case_id", "generated_text"]].drop_duplicates("case_id").rename(columns={"generated_text": "text"})
    hybrid_rows = hybrid[["case_id", "generated_text"]].drop_duplicates("case_id").rename(columns={"generated_text": "text"})
    common_case_ids = set(source_rows.case_id.astype(str)) & set(canonical_rows.case_id.astype(str)) & set(hybrid_rows.case_id.astype(str))
    if not common_case_ids:
        raise ValueError("No case IDs are shared by raw, canonical, and hybrid inputs.")
    source_rows = source_rows[source_rows.case_id.astype(str).isin(common_case_ids)].copy()
    canonical_rows = canonical_rows[canonical_rows.case_id.astype(str).isin(common_case_ids)].copy()
    hybrid_rows = hybrid_rows[hybrid_rows.case_id.astype(str).isin(common_case_ids)].copy()
    source_rows["representation"] = "raw_real_note"
    canonical_rows["representation"] = "canonical_transition_note"
    hybrid_rows["representation"] = "hybrid_note"
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name, local_files_only=True)
    max_tokens = args.max_tokens
    audited = audit_texts(pd.concat([source_rows, canonical_rows, hybrid_rows], ignore_index=True), tokenizer, max_tokens)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audited.to_csv(output_dir / "embedding_token_window_audit.csv", index=False)
    summary = audited.groupby("representation", as_index=False).agg(
        n_documents=("case_id", "size"),
        median_document_token_count=("document_token_count", "median"),
        truncated_count=("truncated_by_bge", "sum"),
        truncated_rate=("truncated_by_bge", "mean"),
        median_tokens_used_by_bge=("tokens_used_by_bge", "median"),
    )
    summary.to_csv(output_dir / "embedding_token_window_summary.csv", index=False)
    metadata = {
        "embedding_model_name": args.embedding_model_name,
        "effective_max_tokens": max_tokens,
        "tokenizer_model_max_length": int(tokenizer.model_max_length),
        "n_common_cases": len(common_case_ids),
        "local_files_only": True,
    }
    (output_dir / "embedding_token_window_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
