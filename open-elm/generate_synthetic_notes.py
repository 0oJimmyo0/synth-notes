#!/usr/bin/env python3
"""
Generate synthetic clinic notes from embeddings using a trained ELM checkpoint.

Phase 1 requirement:
- Keep the plain-text note dump for quick inspection.
- Also write a row-level JSONL manifest incrementally during generation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
import datasets as datasets_lib
from transformers import AutoTokenizer

from src.model import LlamaForEmbeddingLM
from src.utils import batch_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic clinic notes from embeddings using a trained ELM model"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained checkpoint directory (e.g., checkpoint-8215)",
    )
    parser.add_argument(
        "--backbone_model_path",
        type=str,
        default="initial_elm_model",
        help="Path to the backbone model (default: initial_elm_model)",
    )
    parser.add_argument(
        "--embeddings_file",
        type=str,
        default=None,
        help="Path to .npy file containing embeddings (shape: [N, 1024])",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to a HuggingFace dataset directory (one dataset). Use this or --dataset_paths.",
    )
    parser.add_argument(
        "--dataset_paths",
        type=str,
        default=None,
        help="Comma-separated dataset paths to generate from. Minimal manifest support is intended for split-based dataset generation.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="synthetic_notes_improved.txt",
        help="Output file for plain-text generated notes",
    )
    parser.add_argument(
        "--manifest_output",
        type=str,
        default=None,
        help="JSONL manifest output path. Defaults next to --output_file.",
    )
    parser.add_argument(
        "--generation_condition",
        type=str,
        default="vanilla",
        help="Generation condition label written to the manifest (default: vanilla)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Logical split label for single-dataset generation (e.g., test/dev/train). If omitted, inferred from dataset path when possible.",
    )
    parser.add_argument(
        "--split_manifest_path",
        type=str,
        default=None,
        help="Path to split_manifest_note_level.csv for provenance and leakage flags",
    )
    parser.add_argument(
        "--append_manifest",
        action="store_true",
        help="Append to an existing JSONL manifest instead of overwriting it",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Repetition penalty for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to generate (None = all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return parser.parse_args()


def resolve_manifest_output(output_file: str, manifest_output: str | None) -> str:
    if manifest_output:
        return manifest_output
    output_path = Path(output_file)
    stem = output_path.stem
    return str(output_path.with_name(f"{stem}_manifest.jsonl"))


def infer_split_from_path(path: str) -> str | None:
    name = Path(path).name.lower()
    if "testing" in name or "test" in name:
        return "test"
    if "dev" in name or "valid" in name:
        return "dev"
    if "train" in name:
        return "train"
    return None


def infer_split_manifest_path(dataset_paths: list[str], explicit_path: str | None) -> str | None:
    if explicit_path:
        return explicit_path
    if not dataset_paths:
        return None
    candidate = Path(dataset_paths[0]).resolve().parent / "leakage_audit" / "split_manifest_note_level.csv"
    if candidate.exists():
        return str(candidate)
    return None


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit(script_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(script_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_package_versions() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets_lib.__version__,
        "pandas": pd.__version__,
    }


def normalize_missing(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def load_split_manifest_records(split_manifest_path: str, split_label: str) -> list[dict[str, Any]]:
    df = pd.read_csv(split_manifest_path)
    if "split" not in df.columns:
        raise ValueError(f"split manifest missing 'split' column: {split_manifest_path}")
    subset = df.loc[df["split"] == split_label].copy()
    if subset.empty:
        raise ValueError(f"No rows found for split='{split_label}' in {split_manifest_path}")
    if "dataset_row_id" in subset.columns:
        subset["dataset_row_id"] = pd.to_numeric(subset["dataset_row_id"], errors="raise").astype(int)
        subset = subset.sort_values("dataset_row_id").reset_index(drop=True)
    records = []
    for row in subset.to_dict(orient="records"):
        records.append({key: normalize_missing(value) for key, value in row.items()})
    return records


def extract_embedding(example: dict[str, Any]) -> np.ndarray:
    emb = example.get("domain_embeddings")
    if not isinstance(emb, list) or not emb:
        raise ValueError("Example missing expected domain_embeddings list")
    first = emb[0]
    if isinstance(first, torch.Tensor):
        return first.cpu().numpy()
    if isinstance(first, np.ndarray):
        return first
    return np.asarray(first)


def repetition_or_collapse_flag(text: str) -> bool:
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
        ngram_counts = Counter(ngrams)
        if max(ngram_counts.values()) >= 3:
            return True

    return False


def quality_flags(note_text: str) -> dict[str, Any]:
    stripped = note_text.strip()
    word_count = len(stripped.split()) if stripped else 0
    char_count = len(stripped)
    empty_output = char_count == 0
    too_short = word_count < 100
    collapse = repetition_or_collapse_flag(stripped)
    return {
        "generated_word_count": word_count,
        "generated_char_count": char_count,
        "generation_success": not empty_output,
        "empty_output_flag": empty_output,
        "too_short_flag": too_short,
        "repetition_or_collapse_flag": collapse,
    }


def build_source_record(
    source_row: dict[str, Any] | None,
    dataset_path: str | None,
    dataset_row_id: int,
    split_label: str | None,
) -> dict[str, Any]:
    source_row = source_row or {}
    embedding_row_id = source_row.get("embedding_row_id")
    source_row_id = source_row.get("source_row_id", embedding_row_id if embedding_row_id is not None else dataset_row_id)
    source_embedding_id = (
        str(embedding_row_id) if embedding_row_id is not None else str(source_row_id)
    )

    return {
        "source_row_id": source_row_id,
        "dataset_row_id": source_row.get("dataset_row_id", dataset_row_id),
        "embedding_row_id": embedding_row_id,
        "note_id": source_row.get("note_id"),
        "subject_id": source_row.get("subject_id"),
        "hadm_id": source_row.get("hadm_id"),
        "split": source_row.get("split", split_label),
        "source_embedding_id": source_embedding_id,
        "patient_disjoint_from_train": source_row.get("patient_disjoint_from_train"),
        "hadm_disjoint_from_train": source_row.get("hadm_disjoint_from_train"),
        "note_disjoint_from_train": source_row.get("note_disjoint_from_train"),
        "patient_overlap_with_train": source_row.get("patient_overlap_with_train"),
        "hadm_overlap_with_train": source_row.get("hadm_overlap_with_train"),
        "note_overlap_with_train": source_row.get("note_overlap_with_train"),
        "dataset_path": dataset_path,
    }


def validate_source_records(
    source_records: list[dict[str, Any]],
    embeddings: list[np.ndarray],
    split_manifest_path: str | None,
) -> None:
    if len(source_records) != len(embeddings):
        raise ValueError(
            f"Source record count ({len(source_records)}) must equal embedding count ({len(embeddings)})"
        )

    if split_manifest_path:
        for idx, record in enumerate(source_records):
            dataset_row_id = record.get("dataset_row_id")
            if dataset_row_id is not None and int(dataset_row_id) != idx:
                raise ValueError(
                    f"Row alignment mismatch at generation index {idx}: "
                    f"manifest dataset_row_id={dataset_row_id}"
                )


def build_run_metadata(
    args: argparse.Namespace,
    manifest_output: str,
    resolved_dataset_paths: list[str],
    split_manifest_path: str | None,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    run_created_at = datetime.now(timezone.utc).isoformat()

    resolved_config = {
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "backbone_model_path": os.path.abspath(args.backbone_model_path),
        "embeddings_file": os.path.abspath(args.embeddings_file) if args.embeddings_file else None,
        "dataset_path": os.path.abspath(args.dataset_path) if args.dataset_path else None,
        "dataset_paths": [os.path.abspath(path) for path in resolved_dataset_paths],
        "output_file": os.path.abspath(args.output_file),
        "manifest_output": os.path.abspath(manifest_output),
        "generation_condition": args.generation_condition,
        "split": args.split,
        "split_manifest_path": os.path.abspath(split_manifest_path) if split_manifest_path else None,
        "append_manifest": args.append_manifest,
        "batch_size": args.batch_size,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "max_samples": args.max_samples,
        "seed": args.seed,
    }

    return {
        "created_at": run_created_at,
        "script_path": str(script_path),
        "git_commit": get_git_commit(script_dir),
        "cli_args_json": json.dumps(vars(args), sort_keys=True, default=str),
        "resolved_config_json": json.dumps(resolved_config, sort_keys=True, default=str),
        "package_versions_json": json.dumps(get_package_versions(), sort_keys=True),
        "output_text_path": resolved_config["output_file"],
        "manifest_output_path": resolved_config["manifest_output"],
        "split_manifest_path": resolved_config["split_manifest_path"],
    }


def row_non_null_if_available(row: dict[str, Any], fields: list[str]) -> None:
    for field in fields:
        if field in row and row[field] == "":
            raise ValueError(f"Field '{field}' must not be an empty string")


def main() -> None:
    args = parse_args()

    if not args.embeddings_file and not args.dataset_path and not args.dataset_paths:
        raise ValueError("Must provide one of: --embeddings_file, --dataset_path, or --dataset_paths")
    if args.embeddings_file and (args.dataset_path or args.dataset_paths):
        raise ValueError("Cannot combine --embeddings_file with --dataset_path or --dataset_paths")
    if args.dataset_path and args.dataset_paths:
        raise ValueError("Use either --dataset_path or --dataset_paths, not both")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")

    set_random_seed(args.seed)

    resolved_dataset_paths = (
        [args.dataset_path]
        if args.dataset_path
        else [path.strip() for path in (args.dataset_paths or "").split(",") if path.strip()]
    )
    manifest_output = resolve_manifest_output(args.output_file, args.manifest_output)
    split_manifest_path = infer_split_manifest_path(resolved_dataset_paths, args.split_manifest_path)
    run_metadata = build_run_metadata(args, manifest_output, resolved_dataset_paths, split_manifest_path)
    run_id = f"gen-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-seed{args.seed}"

    print("=" * 60)
    print("Loading Model")
    print("=" * 60)
    print(f"Backbone model: {args.backbone_model_path}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Seed: {args.seed}")
    print("")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone_model_path)
    model = LlamaForEmbeddingLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    print("✓ Model loaded successfully")
    print("")

    print("=" * 60)
    print("Loading Embeddings")
    print("=" * 60)

    embeddings: list[np.ndarray] = []
    source_records: list[dict[str, Any]] = []

    if args.embeddings_file:
        print(f"Loading embeddings from: {args.embeddings_file}")
        emb_array = np.load(args.embeddings_file)
        if emb_array.ndim == 1:
            emb_rows = [emb_array]
        elif emb_array.ndim == 2:
            emb_rows = [emb_array[i] for i in range(emb_array.shape[0])]
        else:
            raise ValueError(f"Unexpected embedding shape: {emb_array.shape}. Expected [N, 1024] or [1024]")

        if args.max_samples is not None:
            emb_rows = emb_rows[: args.max_samples]

        embeddings.extend(emb_rows)
        for idx in range(len(emb_rows)):
            source_records.append(
                build_source_record(
                    source_row={"source_row_id": idx, "embedding_row_id": idx},
                    dataset_path=args.embeddings_file,
                    dataset_row_id=idx,
                    split_label=args.split,
                )
            )
        print(f"✓ Loaded {len(embeddings)} embeddings")
    else:
        print(f"Loading dataset(s) from {len(resolved_dataset_paths)} path(s)...")
        for dataset_path in resolved_dataset_paths:
            if not os.path.exists(dataset_path):
                print(f"  Warning: skip (not found): {dataset_path}")
                continue

            dataset = Dataset.load_from_disk(dataset_path)
            split_label = args.split if len(resolved_dataset_paths) == 1 and args.split else infer_split_from_path(dataset_path)
            split_rows = None
            if split_manifest_path and split_label:
                split_rows = load_split_manifest_records(split_manifest_path, split_label)
                if len(split_rows) != len(dataset):
                    raise ValueError(
                        f"Split manifest row count ({len(split_rows)}) does not match dataset row count ({len(dataset)}) "
                        f"for split='{split_label}' and dataset='{dataset_path}'."
                    )

            remaining = None
            if args.max_samples is not None:
                remaining = args.max_samples - len(embeddings)
                if remaining <= 0:
                    break
                if len(dataset) > remaining:
                    dataset = dataset.select(range(remaining))
                    if split_rows is not None:
                        split_rows = split_rows[:remaining]

            print(f"  {dataset_path}: {len(dataset)} rows")
            for idx, example in enumerate(dataset):
                embeddings.append(extract_embedding(example))
                source_records.append(
                    build_source_record(
                        source_row=split_rows[idx] if split_rows is not None else None,
                        dataset_path=dataset_path,
                        dataset_row_id=idx,
                        split_label=split_label,
                    )
                )

        print(f"✓ Extracted {len(embeddings)} embeddings")

    if not embeddings:
        raise ValueError("No embeddings loaded")

    validate_source_records(source_records, embeddings, split_manifest_path)

    print(f"Embedding dimension: {len(embeddings[0])}")
    print("")

    print("=" * 60)
    print("Generating Synthetic Clinic Notes")
    print("=" * 60)
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Batch size: {args.batch_size}")
    print("Generation parameters:")
    print(f"  Repetition penalty: {args.repetition_penalty}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print("")

    output_path = Path(args.output_file)
    manifest_path = Path(manifest_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()
    if manifest_path.exists() and not args.append_manifest:
        manifest_path.unlink()

    run_metadata_path = manifest_path.with_suffix(manifest_path.suffix + ".run.json")
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2))

    all_generated_notes: list[str] = []
    generation_ids: set[str] = set()
    manifest_row_count = 0
    manifest_rows: list[dict[str, Any]] = []

    with output_path.open("a", encoding="utf-8") as text_fh, manifest_path.open("a", encoding="utf-8") as manifest_fh:
        for batch_start in range(0, len(embeddings), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(embeddings))
            batch_embs = embeddings[batch_start:batch_end]

            generated_notes = batch_inference(
                model,
                tokenizer,
                batch_embs,
                args.device,
                task="clinic_note",
                repetition_penalty=args.repetition_penalty,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
            )

            if len(generated_notes) != len(batch_embs):
                raise ValueError(
                    f"Generated note count ({len(generated_notes)}) does not match batch size ({len(batch_embs)})"
                )

            for offset, note in enumerate(generated_notes):
                generation_index = batch_start + offset
                generation_id = f"{run_id}-{generation_index:08d}"
                if generation_id in generation_ids:
                    raise ValueError(f"Duplicate generation_id detected: {generation_id}")
                generation_ids.add(generation_id)

                source = source_records[generation_index]
                split_value = source.get("split") or args.split
                row = {
                    "generation_id": generation_id,
                    "created_at": run_metadata["created_at"],
                    "generation_index": generation_index,
                    "source_row_id": source.get("source_row_id"),
                    "dataset_row_id": source.get("dataset_row_id"),
                    "embedding_row_id": source.get("embedding_row_id"),
                    "note_id": source.get("note_id"),
                    "subject_id": source.get("subject_id"),
                    "hadm_id": source.get("hadm_id"),
                    "split": split_value,
                    "generation_condition": args.generation_condition,
                    "source_embedding_id": source.get("source_embedding_id"),
                    "checkpoint_path": os.path.abspath(args.checkpoint_path),
                    "checkpoint_name": Path(args.checkpoint_path).name,
                    "backbone_path": os.path.abspath(args.backbone_model_path),
                    "backbone_name": Path(args.backbone_model_path).name,
                    "seed": args.seed,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "repetition_penalty": args.repetition_penalty,
                    "max_new_tokens": args.max_new_tokens,
                    "generated_text": note,
                    "patient_disjoint_from_train": source.get("patient_disjoint_from_train"),
                    "hadm_disjoint_from_train": source.get("hadm_disjoint_from_train"),
                    "note_disjoint_from_train": source.get("note_disjoint_from_train"),
                    "patient_overlap_with_train": source.get("patient_overlap_with_train"),
                    "hadm_overlap_with_train": source.get("hadm_overlap_with_train"),
                    "note_overlap_with_train": source.get("note_overlap_with_train"),
                    "axis_id": None,
                    "axis_label": None,
                    "alpha": None,
                    "normalized_after_steering": None,
                    "random_shift_norm": None,
                    "editor_model": None,
                    "edited_text": None,
                    "post_edit_source_cosine": None,
                    "script_path": run_metadata["script_path"],
                    "git_commit": run_metadata["git_commit"],
                    "cli_args_json": run_metadata["cli_args_json"],
                    "package_versions_json": run_metadata["package_versions_json"],
                    "resolved_config_json": run_metadata["resolved_config_json"],
                    "output_text_path": run_metadata["output_text_path"],
                    "manifest_output_path": run_metadata["manifest_output_path"],
                    "run_metadata_path": str(run_metadata_path),
                    "dataset_path": source.get("dataset_path"),
                }
                row.update(quality_flags(note))
                row_non_null_if_available(row, ["generation_id", "generation_condition", "checkpoint_path", "backbone_path"])

                manifest_fh.write(json.dumps(row, default=str) + "\n")
                manifest_fh.flush()
                manifest_rows.append(row)
                manifest_row_count += 1

                text_fh.write(f"=== Note {generation_index + 1} ===\n")
                text_fh.write(note)
                text_fh.write("\n\n")
                text_fh.flush()

            all_generated_notes.extend(generated_notes)
            print(f"Generated {batch_end}/{len(embeddings)} notes... (saved to file)", end="\r")

    print(f"\n✓ Generated and saved {len(all_generated_notes)} synthetic clinic notes")
    print(f"✓ Final note file: {output_path}")
    print(f"✓ Manifest file: {manifest_path}")
    print("")

    if manifest_row_count != len(all_generated_notes):
        raise ValueError(
            f"Manifest row count ({manifest_row_count}) must equal generated note count ({len(all_generated_notes)})"
        )
    if len(generation_ids) != manifest_row_count:
        raise ValueError("Duplicate generation_id detected during validation")
    for idx, row in enumerate(manifest_rows):
        if row["generation_index"] != idx:
            raise ValueError(f"Manifest row order mismatch at index {idx}")
        if split_manifest_path and row.get("dataset_row_id") is not None and int(row["dataset_row_id"]) != idx:
            raise ValueError(f"dataset_row_id mismatch at manifest index {idx}")
        if row["seed"] != args.seed:
            raise ValueError(f"Seed mismatch in manifest at index {idx}")
        if row["temperature"] != args.temperature or row["top_p"] != args.top_p or row["top_k"] != args.top_k:
            raise ValueError(f"Decoding parameter mismatch in manifest at index {idx}")

    note_lengths = [len(note.split()) for note in all_generated_notes]
    print("=" * 60)
    print("Generation Statistics")
    print("=" * 60)
    print(f"Total notes: {len(all_generated_notes)}")
    print(f"Average words per note: {sum(note_lengths) / len(note_lengths):.0f}")
    print(f"Min words: {min(note_lengths)}")
    print(f"Max words: {max(note_lengths)}")
    print("")

    print("=" * 60)
    print("Sample Generated Note (First Note)")
    print("=" * 60)
    if all_generated_notes:
        sample = all_generated_notes[0]
        print(sample[:500] + "..." if len(sample) > 500 else sample)
        print("")

    validation_summary = {
        "generated_note_count": len(all_generated_notes),
        "manifest_row_count": manifest_row_count,
        "unique_generation_ids": len(generation_ids),
        "output_file": str(output_path),
        "manifest_output": str(manifest_path),
        "run_metadata_path": str(run_metadata_path),
        "split_manifest_path": split_manifest_path,
    }
    validation_path = manifest_path.with_suffix(manifest_path.suffix + ".validation.json")
    validation_path.write_text(json.dumps(validation_summary, indent=2))

    print("=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Total notes generated: {len(all_generated_notes)}")
    print(f"Output saved to: {output_path}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Validation summary: {validation_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
