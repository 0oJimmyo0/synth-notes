#!/usr/bin/env python3
"""
Re-embed generated synthetic notes from a manifest and save a row-aligned .npy matrix.

This is intentionally separate from the vanilla audit so coverage mapping can be rerun
without recomputing the entire audit pipeline.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-embed generated notes from a manifest.")
    parser.add_argument("--manifest_path", required=True, help="Path to generation manifest JSONL")
    parser.add_argument("--output_path", required=True, help="Path to output .npy embedding matrix")
    parser.add_argument(
        "--embedding_model_name",
        default="BAAI/bge-large-en-v1.5",
        help="Sentence embedding model used for generated-note re-embedding",
    )
    parser.add_argument(
        "--embedding_device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for sentence-transformer inference",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=256,
        help="Batch size for sentence-transformer inference",
    )
    parser.add_argument(
        "--text_column",
        default="generated_text",
        help="Manifest column containing generated note text",
    )
    parser.add_argument(
        "--metadata_output_path",
        default=None,
        help="Optional path for run metadata JSON",
    )
    return parser.parse_args()


def now_iso() -> str:
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


def resolve_embedding_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


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
    return versions


def load_manifest(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True, dtype=False).reset_index(drop=True)


def validate_manifest(df: pd.DataFrame, text_column: str) -> None:
    required = ["generation_id", "dataset_row_id", text_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if df[text_column].isna().any():
        raise ValueError(f"Manifest contains null values in {text_column}")
    if df["generation_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate generation_id values")


def encode_texts(
    texts: list[str],
    model_name: str,
    requested_device: str,
    batch_size: int,
) -> tuple[np.ndarray, str]:
    resolved_device = resolve_embedding_device(requested_device)
    model = SentenceTransformer(model_name, device=resolved_device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32), resolved_device


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest_path)
    output_path = Path(args.output_path)
    metadata_output_path = Path(args.metadata_output_path) if args.metadata_output_path else output_path.with_suffix(
        output_path.suffix + ".meta.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_df = load_manifest(manifest_path)
    validate_manifest(manifest_df, args.text_column)

    texts = manifest_df[args.text_column].fillna("").astype(str).tolist()
    embeddings, resolved_device = encode_texts(
        texts=texts,
        model_name=args.embedding_model_name,
        requested_device=args.embedding_device,
        batch_size=args.embedding_batch_size,
    )

    if embeddings.shape[0] != len(manifest_df):
        raise ValueError(
            f"Embedding row count {embeddings.shape[0]} does not match manifest row count {len(manifest_df)}"
        )

    np.save(output_path, embeddings)

    metadata: dict[str, Any] = {
        "created_at": now_iso(),
        "script_path": str(Path(__file__).resolve()),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "manifest_path": str(manifest_path.resolve()),
        "output_path": str(output_path.resolve()),
        "metadata_output_path": str(metadata_output_path.resolve()),
        "manifest_rows": int(len(manifest_df)),
        "embedding_shape": list(embeddings.shape),
        "embedding_model_name": args.embedding_model_name,
        "embedding_device_requested": args.embedding_device,
        "embedding_device_resolved": resolved_device,
        "embedding_batch_size": int(args.embedding_batch_size),
        "text_column": args.text_column,
        "package_versions": package_versions(),
    }
    metadata_output_path.write_text(json.dumps(metadata, indent=2))

    print("Saved generated-note embeddings to:", output_path)
    print("Saved re-embedding metadata to:", metadata_output_path)


if __name__ == "__main__":
    main()
