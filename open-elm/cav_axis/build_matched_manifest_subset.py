#!/usr/bin/env python3
"""
Build a matched subset of a candidate synthetic manifest (and optional embeddings)
using the unique dataset_row_id set from a reference manifest.

Primary use:
- compare a shifted pilot against a matched vanilla baseline on the same source rows
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a matched manifest/embedding subset by dataset_row_id.")
    parser.add_argument("--reference_manifest_path", required=True, help="Reference manifest defining source dataset_row_id set")
    parser.add_argument("--candidate_manifest_path", required=True, help="Candidate manifest to filter, e.g. vanilla full manifest")
    parser.add_argument("--output_manifest_path", required=True, help="Output path for filtered manifest JSONL")
    parser.add_argument("--candidate_embeddings_path", default=None, help="Optional .npy embeddings aligned to candidate manifest rows")
    parser.add_argument("--output_embeddings_path", default=None, help="Optional output .npy path for filtered embeddings")
    parser.add_argument("--output_metadata_path", default=None, help="Optional output metadata JSON path")
    parser.add_argument(
        "--require_candidate_generation_condition",
        default=None,
        help="Optional expected generation_condition for the candidate manifest",
    )
    parser.add_argument(
        "--output_generation_condition",
        default=None,
        help="Optional replacement generation_condition value for the subset manifest",
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


def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True, dtype=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()

    reference_manifest_path = Path(args.reference_manifest_path)
    candidate_manifest_path = Path(args.candidate_manifest_path)
    output_manifest_path = Path(args.output_manifest_path)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    reference_df = load_jsonl(reference_manifest_path)
    candidate_df = load_jsonl(candidate_manifest_path)

    if "dataset_row_id" not in reference_df.columns or "dataset_row_id" not in candidate_df.columns:
        raise ValueError("Both reference and candidate manifests must include dataset_row_id.")

    reference_ids = sorted(pd.to_numeric(reference_df["dataset_row_id"], errors="raise").astype(int).unique().tolist())
    candidate_df["dataset_row_id"] = pd.to_numeric(candidate_df["dataset_row_id"], errors="raise").astype(int)

    if args.require_candidate_generation_condition:
        conditions = sorted(candidate_df["generation_condition"].dropna().astype(str).unique().tolist())
        if conditions != [args.require_candidate_generation_condition]:
            raise ValueError(
                f"Candidate manifest generation_condition must be [{args.require_candidate_generation_condition!r}], got {conditions}"
            )

    subset_df = candidate_df.loc[candidate_df["dataset_row_id"].isin(reference_ids)].copy()
    subset_df = subset_df.sort_values(["dataset_row_id", "generation_index"]).reset_index(drop=True)

    matched_ids = sorted(subset_df["dataset_row_id"].astype(int).unique().tolist())
    if matched_ids != reference_ids:
        missing = sorted(set(reference_ids) - set(matched_ids))
        raise ValueError(
            f"Candidate manifest did not cover all reference dataset_row_id values. Missing count={len(missing)}"
        )

    if args.output_generation_condition:
        subset_df["generation_condition"] = args.output_generation_condition

    subset_df.to_json(output_manifest_path, orient="records", lines=True)

    embedding_summary: dict[str, Any] | None = None
    if args.candidate_embeddings_path:
        if not args.output_embeddings_path:
            raise ValueError("--output_embeddings_path is required when --candidate_embeddings_path is used.")
        candidate_embeddings_path = Path(args.candidate_embeddings_path)
        output_embeddings_path = Path(args.output_embeddings_path)
        output_embeddings_path.parent.mkdir(parents=True, exist_ok=True)

        embeddings = np.load(candidate_embeddings_path)
        if embeddings.shape[0] != len(candidate_df):
            raise ValueError(
                f"Candidate embeddings row count {embeddings.shape[0]} does not match candidate manifest row count {len(candidate_df)}."
            )

        subset_positions = subset_df.index.to_numpy()
        # Use candidate_df positions before sorting/resetting.
        subset_df_unsorted = candidate_df.loc[candidate_df["dataset_row_id"].isin(reference_ids)].copy()
        subset_df_unsorted = subset_df_unsorted.sort_values(["dataset_row_id", "generation_index"])
        subset_positions = subset_df_unsorted.index.to_numpy(dtype=int)
        subset_embeddings = embeddings[subset_positions]
        np.save(output_embeddings_path, subset_embeddings.astype(np.float32))

        embedding_summary = {
            "candidate_embeddings_path": str(candidate_embeddings_path.resolve()),
            "output_embeddings_path": str(output_embeddings_path.resolve()),
            "output_embedding_shape": list(subset_embeddings.shape),
        }

    metadata = {
        "created_at": now_iso(),
        "script_path": str(Path(__file__).resolve()),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "reference_manifest_path": str(reference_manifest_path.resolve()),
        "candidate_manifest_path": str(candidate_manifest_path.resolve()),
        "output_manifest_path": str(output_manifest_path.resolve()),
        "reference_unique_dataset_row_ids": len(reference_ids),
        "output_manifest_rows": int(len(subset_df)),
        "output_unique_dataset_row_ids": int(subset_df["dataset_row_id"].nunique()),
        "require_candidate_generation_condition": args.require_candidate_generation_condition,
        "output_generation_condition": args.output_generation_condition,
        "embedding_summary": embedding_summary,
    }

    metadata_path = (
        Path(args.output_metadata_path)
        if args.output_metadata_path
        else output_manifest_path.with_suffix(output_manifest_path.suffix + ".meta.json")
    )
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("Saved matched manifest subset to:", output_manifest_path)
    print("Matched unique dataset_row_id count:", len(reference_ids))
    print("Output manifest rows:", len(subset_df))
    if embedding_summary is not None:
        print("Saved matched embeddings to:", embedding_summary["output_embeddings_path"])
    print("Saved metadata to:", metadata_path)


if __name__ == "__main__":
    main()
