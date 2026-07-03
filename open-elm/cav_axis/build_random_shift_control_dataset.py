#!/usr/bin/env python3
"""
Build a norm-matched random-shift control dataset from a reference shifted dataset.

Each control row reuses the same source anchor and the same shift norm as the reference
row, but replaces the CAV direction with a random unit vector. The output keeps the same
dataset structure consumed by generate_synthetic_notes.py.
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset

from common import normalize_rows, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a norm-matched random-shift control dataset.")
    parser.add_argument(
        "--reference_shifted_dataset_path",
        required=True,
        help="Path to an existing shifted HF dataset whose source rows / norms should be matched",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where the random-shift HF dataset and sidecar metadata will be written",
    )
    parser.add_argument(
        "--source_dataset_path",
        default=None,
        help="Optional override for the original source HF dataset path. If omitted, inferred from reference rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed for random shift directions",
    )
    parser.add_argument(
        "--output_stem",
        default="random_shift_control",
        help="Stem for metadata sidecar files written next to the saved HF dataset",
    )
    return parser


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


def normalize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def maybe_int(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        return normalize_scalar(value)


def extract_embedding_vector(example: dict[str, Any]) -> np.ndarray:
    emb = example["domain_embeddings"]
    if not isinstance(emb, list) or not emb:
        raise ValueError("Expected each dataset row to carry a non-empty domain_embeddings list.")
    first = emb[0]
    return np.asarray(first, dtype=np.float32)


def sample_random_unit_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        raise ValueError("Encountered zero-norm random vector, which should be practically impossible.")
    return vec / norm


def main() -> None:
    args = build_parser().parse_args()

    reference_dataset_path = Path(args.reference_shifted_dataset_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_dataset = Dataset.load_from_disk(str(reference_dataset_path))
    if len(reference_dataset) == 0:
        raise ValueError("Reference shifted dataset is empty.")

    reference_df = reference_dataset.to_pandas()
    required_cols = [
        "dataset_row_id",
        "random_shift_norm",
        "normalized_after_steering",
        "source_dataset_path",
    ]
    missing = [col for col in required_cols if col not in reference_df.columns]
    if missing:
        raise ValueError(f"Reference shifted dataset is missing required columns: {missing}")

    source_dataset_path = (
        Path(args.source_dataset_path).resolve()
        if args.source_dataset_path
        else Path(str(reference_df["source_dataset_path"].dropna().iloc[0])).resolve()
    )
    source_dataset = Dataset.load_from_disk(str(source_dataset_path))

    run_metadata_path = output_dir / f"{args.output_stem}_run_metadata.json"
    rng = np.random.default_rng(args.seed)

    control_rows: list[dict[str, Any]] = []
    dataset_manifest_rows: list[dict[str, Any]] = []

    for ref_idx, ref_row in reference_df.reset_index(drop=True).iterrows():
        dataset_row_id = int(ref_row["dataset_row_id"])
        shift_norm = float(ref_row["random_shift_norm"])
        normalize_after_steering = bool(ref_row["normalized_after_steering"])

        example = source_dataset[dataset_row_id]
        source_embedding = extract_embedding_vector(example)
        random_direction = sample_random_unit_vector(source_embedding.shape[0], rng)
        shifted_embedding = source_embedding + (shift_norm * random_direction)
        if normalize_after_steering:
            shifted_embedding = normalize_rows(shifted_embedding.reshape(1, -1))[0].astype(np.float32)
        else:
            shifted_embedding = shifted_embedding.astype(np.float32)

        cosine = float(
            np.dot(source_embedding, shifted_embedding)
            / (np.linalg.norm(source_embedding) * np.linalg.norm(shifted_embedding))
        )

        control_row = {
            "input_ids": example["input_ids"],
            "domain_embeddings": [shifted_embedding.tolist()],
            "source_row_id": maybe_int(ref_row.get("source_row_id", ref_row.get("dataset_row_id"))),
            "dataset_row_id": maybe_int(ref_row.get("dataset_row_id")),
            "embedding_row_id": maybe_int(ref_row.get("embedding_row_id")),
            "note_id": normalize_scalar(ref_row.get("note_id")),
            "subject_id": maybe_int(ref_row.get("subject_id")),
            "hadm_id": maybe_int(ref_row.get("hadm_id")),
            "split": normalize_scalar(ref_row.get("split")),
            "source_embedding_id": str(
                maybe_int(ref_row.get("embedding_row_id", ref_row.get("dataset_row_id")))
            ),
            "patient_disjoint_from_train": normalize_scalar(ref_row.get("patient_disjoint_from_train")),
            "hadm_disjoint_from_train": normalize_scalar(ref_row.get("hadm_disjoint_from_train")),
            "note_disjoint_from_train": normalize_scalar(ref_row.get("note_disjoint_from_train")),
            "patient_overlap_with_train": normalize_scalar(ref_row.get("patient_overlap_with_train")),
            "hadm_overlap_with_train": normalize_scalar(ref_row.get("hadm_overlap_with_train")),
            "note_overlap_with_train": normalize_scalar(ref_row.get("note_overlap_with_train")),
            "axis_id": None,
            "axis_label": "random_shift_control",
            "alpha": normalize_scalar(ref_row.get("alpha")),
            "normalized_after_steering": normalize_after_steering,
            "random_shift_norm": shift_norm,
            "editor_model": None,
            "edited_text": None,
            "post_edit_source_cosine": cosine,
            "source_dataset_path": str(source_dataset_path),
            "source_split": normalize_scalar(ref_row.get("source_split", ref_row.get("split"))),
            "selection_query": normalize_scalar(ref_row.get("selection_query")),
            "steering_run_metadata_path": str(run_metadata_path),
            "random_shift_seed": int(args.seed),
            "random_shift_reference_dataset_path": str(reference_dataset_path),
            "random_shift_reference_row_index": int(ref_idx),
        }
        control_rows.append(control_row)
        dataset_manifest_rows.append(
            {
                "shifted_dataset_row_id": len(dataset_manifest_rows),
                "source_row_id": control_row["source_row_id"],
                "dataset_row_id": control_row["dataset_row_id"],
                "embedding_row_id": control_row["embedding_row_id"],
                "note_id": control_row["note_id"],
                "subject_id": control_row["subject_id"],
                "hadm_id": control_row["hadm_id"],
                "split": control_row["split"],
                "axis_id": control_row["axis_id"],
                "axis_label": control_row["axis_label"],
                "alpha": control_row["alpha"],
                "normalized_after_steering": control_row["normalized_after_steering"],
                "random_shift_norm": control_row["random_shift_norm"],
                "post_edit_source_cosine": control_row["post_edit_source_cosine"],
                "random_shift_reference_row_index": control_row["random_shift_reference_row_index"],
            }
        )

    control_dataset = Dataset.from_list(control_rows)
    control_dataset.save_to_disk(str(output_dir))

    manifest_csv = output_dir / f"{args.output_stem}_dataset_manifest.csv"
    pd.DataFrame(dataset_manifest_rows).to_csv(manifest_csv, index=False)

    summary_payload = {
        "created_at": now_iso(),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "script_path": str(Path(__file__).resolve()),
        "reference_shifted_dataset_path": str(reference_dataset_path),
        "source_dataset_path": str(source_dataset_path),
        "output_dir": str(output_dir),
        "output_dataset_path": str(output_dir),
        "output_manifest_csv": str(manifest_csv),
        "seed": int(args.seed),
        "n_reference_rows": int(len(reference_df)),
        "n_control_rows": int(len(control_rows)),
        "example_output_columns": control_dataset.column_names,
        "cli_args": vars(args),
    }
    save_json(run_metadata_path, summary_payload)

    print(f"Saved random-shift control dataset to: {output_dir}")
    print(f"Reference rows: {len(reference_df)}")
    print(f"Control dataset rows: {len(control_rows)}")
    print(f"Run metadata: {run_metadata_path}")
    print(f"Dataset manifest CSV: {manifest_csv}")


if __name__ == "__main__":
    main()
