#!/usr/bin/env python3
"""
Build a shifted Hugging Face embedding dataset for CAV-steered ELM generation.

The output dataset keeps the same `input_ids` / `domain_embeddings` structure used by
`generate_synthetic_notes.py`, while attaching row-level steering provenance so the
existing generation manifest can record it without a separate generation path.
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
from datasets import Dataset

from common import normalize_rows, parse_csv_list, parse_float_list, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a shifted HF dataset for CAV-steered note generation.")
    parser.add_argument("--dataset_path", required=True, help="Source HF dataset path, e.g. encoded_testing_filtered")
    parser.add_argument(
        "--bank_dir",
        default=None,
        help="Axis-bank directory created by fit_axis_bank.py",
    )
    parser.add_argument(
        "--direction_bank_path",
        default=None,
        help="Optional local direction bank .npz created by build_local_steering_directions.py",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where the shifted HF dataset and metadata files will be written",
    )
    parser.add_argument(
        "--axis_indices",
        default=None,
        help="Comma-separated axis indices to apply, e.g. 15 or 11,15",
    )
    parser.add_argument(
        "--direction_indices",
        default=None,
        help="Comma-separated direction indices to apply from --direction_bank_path",
    )
    parser.add_argument(
        "--direction_labels",
        default=None,
        help="Comma-separated direction labels to apply from --direction_bank_path",
    )
    parser.add_argument(
        "--alphas",
        required=True,
        help="Comma-separated steering strengths, e.g. 0.5,1.0",
    )
    parser.add_argument(
        "--factors_path",
        default=None,
        help="Optional factor/metadata CSV used for source selection (recommended)",
    )
    parser.add_argument(
        "--split_manifest_path",
        default=None,
        help="Optional filtered-aligned split manifest to join leakage flags/provenance",
    )
    parser.add_argument(
        "--join_cols",
        default="source_row_id,embedding_row_id,dataset_row_id,note_id,subject_id,hadm_id",
        help="Comma-separated preferred join columns for metadata table merges",
    )
    parser.add_argument(
        "--source_split",
        default=None,
        help="Optional split filter (e.g. test/dev/train) applied before selection_query",
    )
    parser.add_argument(
        "--selection_query",
        default=None,
        help="Optional pandas query over merged metadata, e.g. 'cluster_target_25 == 1'",
    )
    parser.add_argument(
        "--max_source_rows",
        type=int,
        default=None,
        help="Optional cap on selected source rows before axis/alpha expansion",
    )
    parser.add_argument(
        "--normalize_after_steering",
        action="store_true",
        help="L2-normalize shifted embeddings after adding alpha * axis",
    )
    parser.add_argument(
        "--output_stem",
        default="shifted_dataset",
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


def load_dataset_rows(dataset_path: Path) -> tuple[Dataset, pd.DataFrame]:
    dataset = Dataset.load_from_disk(str(dataset_path))
    base_df = pd.DataFrame({"dataset_row_id": np.arange(len(dataset), dtype=int)})
    metadata_cols = [col for col in dataset.column_names if col not in {"input_ids", "domain_embeddings"}]
    if metadata_cols:
        metadata_df = dataset.select_columns(metadata_cols).to_pandas()
        base_df = pd.concat([base_df, metadata_df.reset_index(drop=True)], axis=1)
    return dataset, base_df


def choose_join_keys(frames: list[pd.DataFrame], preferred_keys: list[str]) -> list[str]:
    if not frames:
        raise ValueError("No frames were provided for join-key detection.")
    common_cols = set(frames[0].columns)
    for frame in frames[1:]:
        common_cols &= set(frame.columns)

    preferred_groups = [
        ["split", "dataset_row_id"],
        ["source_row_id"],
        ["embedding_row_id"],
        ["dataset_row_id"],
        ["note_id", "subject_id", "hadm_id"],
        ["note_id"],
        ["subject_id", "hadm_id"],
    ]
    for key in preferred_keys:
        if key in {"note_id", "subject_id", "hadm_id"}:
            continue
        if [key] not in preferred_groups:
            preferred_groups.append([key])
    if all(key in preferred_keys for key in ["note_id", "subject_id", "hadm_id"]):
        if ["note_id", "subject_id", "hadm_id"] not in preferred_groups:
            preferred_groups.append(["note_id", "subject_id", "hadm_id"])
    if all(key in preferred_keys for key in ["subject_id", "hadm_id"]):
        if ["subject_id", "hadm_id"] not in preferred_groups:
            preferred_groups.append(["subject_id", "hadm_id"])
    for keys in preferred_groups:
        if all(key in common_cols for key in keys):
            return keys
    raise ValueError(f"Could not detect stable join keys. Shared columns were: {sorted(common_cols)}")


def normalize_join_cols(df: pd.DataFrame, join_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in join_cols:
        if col not in out.columns:
            raise KeyError(f"Join column missing: {col}")
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().all():
            out[col] = numeric.astype(int)
        else:
            out[col] = out[col].astype(str).str.strip()
    return out


def merge_optional_metadata(
    base_df: pd.DataFrame,
    factors_path: str | None,
    split_manifest_path: str | None,
    preferred_join_cols: list[str],
    source_split: str | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    merged = base_df.copy()
    join_report: dict[str, list[str]] = {}
    active_join_cols = ["dataset_row_id"]

    if split_manifest_path:
        split_df = pd.read_csv(split_manifest_path)
        join_cols = choose_join_keys([merged, split_df], preferred_join_cols)
        if join_cols == ["dataset_row_id"] and "split" in split_df.columns:
            if source_split:
                split_df = split_df.loc[split_df["split"].astype(str) == str(source_split)].copy()
                join_report["split_manifest_filtered_to_split"] = [str(source_split)]
            elif split_df.duplicated(subset=join_cols, keep=False).any():
                raise ValueError(
                    "Split manifest reuses dataset_row_id across multiple splits. "
                    "Pass --source_split (for example: --source_split test) so the join can be filtered first."
                )
        merged = normalize_join_cols(merged, join_cols)
        split_df = normalize_join_cols(split_df, join_cols)
        duplicate_mask = split_df.duplicated(subset=join_cols, keep=False)
        if duplicate_mask.any():
            dup_count = int(duplicate_mask.sum())
            raise ValueError(
                f"Split manifest has {dup_count} duplicate rows for join keys {join_cols}; deduplicate it first."
            )
        merged = merged.merge(
            split_df,
            on=join_cols,
            how="left",
            validate="one_to_one",
            suffixes=("", "_split"),
        )
        active_join_cols = join_cols
        join_report["split_manifest_join_cols"] = join_cols

    if factors_path:
        factors_df = pd.read_csv(factors_path)
        join_cols = choose_join_keys([merged, factors_df], preferred_join_cols)
        merged = normalize_join_cols(merged, join_cols)
        factors_df = normalize_join_cols(factors_df, join_cols)
        duplicate_mask = factors_df.duplicated(subset=join_cols, keep=False)
        if duplicate_mask.any():
            dup_count = int(duplicate_mask.sum())
            raise ValueError(
                f"Factor table has {dup_count} duplicate rows for join keys {join_cols}; deduplicate it first."
            )
        merged = merged.merge(
            factors_df,
            on=join_cols,
            how="left",
            validate="one_to_one",
            suffixes=("", "_factor"),
        )
        active_join_cols = join_cols
        join_report["factors_join_cols"] = join_cols

    return merged, active_join_cols, join_report


def infer_axis_label(summary: dict[str, Any], axis_idx: int) -> str:
    top_targets = summary.get("top_targets_by_axis", [])
    if isinstance(top_targets, list) and axis_idx < len(top_targets):
        axis_info = top_targets[axis_idx]
        targets = axis_info.get("top_targets", [])
        if targets:
            top_target = targets[0]
            target_column = str(top_target.get("target_column", f"axis_{axis_idx}"))
            return f"axis_{axis_idx}__{target_column}"
    return f"axis_{axis_idx}"


def load_bank_directions(
    bank_dir: Path | None,
    direction_bank_path: Path | None,
    axis_indices: list[int],
    direction_indices: list[int],
    direction_labels: list[str],
) -> tuple[np.ndarray, dict[int, str], dict[str, Any]]:
    if (bank_dir is None) == (direction_bank_path is None):
        raise ValueError("Provide exactly one of --bank_dir or --direction_bank_path.")

    if bank_dir is not None:
        if not axis_indices:
            raise ValueError("--axis_indices must specify at least one axis when using --bank_dir.")
        bank = np.load(bank_dir / "axis_bank.npz")
        axes = np.asarray(bank["axes"], dtype=np.float32)
        if axes.ndim != 2:
            raise ValueError(f"Expected axes to be 2D, got shape {axes.shape}")
        summary = json.loads((bank_dir / "axis_bank_summary.json").read_text())
        labels = {axis_idx: infer_axis_label(summary, axis_idx) for axis_idx in axis_indices}
        bank_metadata = {
            "bank_type": "axis_bank",
            "bank_dir": str(bank_dir),
            "summary_path": str(bank_dir / "axis_bank_summary.json"),
        }
        return axes, labels, bank_metadata

    if axis_indices:
        raise ValueError("--axis_indices is only valid with --bank_dir.")
    if direction_indices and direction_labels:
        raise ValueError("Provide only one of --direction_indices or --direction_labels.")

    bank = np.load(direction_bank_path, allow_pickle=True)
    directions = np.asarray(bank["directions"], dtype=np.float32)
    if directions.ndim != 2:
        raise ValueError(f"Expected directions to be 2D, got shape {directions.shape}")

    bank_labels_raw = bank.get("direction_labels")
    if bank_labels_raw is None:
        bank_labels = [f"direction_{idx}" for idx in range(directions.shape[1])]
    else:
        bank_labels = [str(item) for item in bank_labels_raw.tolist()]
    label_to_idx = {label: idx for idx, label in enumerate(bank_labels)}

    selected_indices = direction_indices
    if direction_labels:
        missing = [label for label in direction_labels if label not in label_to_idx]
        if missing:
            raise KeyError(f"Direction labels not found in bank: {missing}")
        selected_indices = [label_to_idx[label] for label in direction_labels]
    if not selected_indices:
        raise ValueError(
            "Specify at least one direction via --direction_indices or --direction_labels when using --direction_bank_path."
        )

    labels = {direction_idx: bank_labels[direction_idx] for direction_idx in selected_indices}
    bank_metadata = {
        "bank_type": "local_direction_bank",
        "direction_bank_path": str(direction_bank_path),
        "available_direction_labels": bank_labels,
    }
    return directions, labels, bank_metadata


def extract_embedding_vector(example: dict[str, Any]) -> np.ndarray:
    emb = example["domain_embeddings"]
    if not isinstance(emb, list) or not emb:
        raise ValueError("Expected each dataset row to carry a non-empty domain_embeddings list.")
    first = emb[0]
    return np.asarray(first, dtype=np.float32)


def maybe_int(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        return normalize_scalar(value)


def main() -> None:
    args = build_parser().parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    bank_dir = Path(args.bank_dir).resolve() if args.bank_dir else None
    direction_bank_path = Path(args.direction_bank_path).resolve() if args.direction_bank_path else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    axis_indices = [int(item) for item in parse_csv_list(args.axis_indices)]
    direction_indices = [int(item) for item in parse_csv_list(args.direction_indices)]
    direction_labels = parse_csv_list(args.direction_labels)
    alphas = parse_float_list(args.alphas)
    preferred_join_cols = parse_csv_list(args.join_cols)

    if not alphas:
        raise ValueError("--alphas must specify at least one steering strength")

    dataset, dataset_df = load_dataset_rows(dataset_path)
    if args.source_split and "split" not in dataset_df.columns:
        dataset_df["split"] = args.source_split

    directions, selected_labels, bank_metadata = load_bank_directions(
        bank_dir=bank_dir,
        direction_bank_path=direction_bank_path,
        axis_indices=axis_indices,
        direction_indices=direction_indices,
        direction_labels=direction_labels,
    )
    sample_example = dataset[0]
    if directions.shape[0] != len(sample_example["domain_embeddings"][0]):
        raise ValueError(
            f"Direction dimension {directions.shape[0]} does not match dataset embedding dim "
            f"{len(sample_example['domain_embeddings'][0])}"
        )

    merged_df, active_join_cols, join_report = merge_optional_metadata(
        dataset_df,
        factors_path=args.factors_path,
        split_manifest_path=args.split_manifest_path,
        preferred_join_cols=preferred_join_cols,
        source_split=args.source_split,
    )

    if args.source_split:
        if "split" not in merged_df.columns:
            raise ValueError("--source_split was provided, but the merged metadata has no 'split' column.")
        merged_df = merged_df.loc[merged_df["split"] == args.source_split].copy()

    if args.selection_query:
        merged_df = merged_df.query(args.selection_query, engine="python").copy()

    if args.max_source_rows is not None:
        merged_df = merged_df.head(args.max_source_rows).copy()

    if merged_df.empty:
        raise ValueError("No source rows remained after applying split/query filters.")

    run_metadata_path = output_dir / f"{args.output_stem}_run_metadata.json"
    selected_direction_ids = sorted(selected_labels.keys())

    shifted_rows: list[dict[str, Any]] = []
    dataset_manifest_rows: list[dict[str, Any]] = []

    for _, source_row in merged_df.reset_index(drop=True).iterrows():
        dataset_row_id = int(source_row["dataset_row_id"])
        example = dataset[dataset_row_id]
        source_embedding = extract_embedding_vector(example)
        source_norm = float(np.linalg.norm(source_embedding))
        if source_norm <= 0:
            raise ValueError("Encountered zero-norm source embedding; cannot steer it safely.")

        for direction_idx in selected_direction_ids:
            if direction_idx < 0 or direction_idx >= directions.shape[1]:
                raise IndexError(
                    f"Direction index {direction_idx} is out of range for directions shape {directions.shape}"
                )
            axis_vector = directions[:, direction_idx]

            for alpha in alphas:
                shifted_embedding = source_embedding + (alpha * axis_vector)
                raw_shift_norm = float(np.linalg.norm(alpha * axis_vector))
                if args.normalize_after_steering:
                    shifted_embedding = normalize_rows(shifted_embedding.reshape(1, -1))[0].astype(np.float32)
                else:
                    shifted_embedding = shifted_embedding.astype(np.float32)

                cosine = float(
                    np.dot(source_embedding, shifted_embedding)
                    / (np.linalg.norm(source_embedding) * np.linalg.norm(shifted_embedding))
                )

                shifted_row = {
                    "input_ids": example["input_ids"],
                    "domain_embeddings": [shifted_embedding.tolist()],
                    "source_row_id": maybe_int(source_row.get("source_row_id", source_row.get("dataset_row_id"))),
                    "dataset_row_id": maybe_int(source_row.get("dataset_row_id")),
                    "embedding_row_id": maybe_int(source_row.get("embedding_row_id")),
                    "note_id": normalize_scalar(source_row.get("note_id")),
                    "subject_id": maybe_int(source_row.get("subject_id")),
                    "hadm_id": maybe_int(source_row.get("hadm_id")),
                    "split": normalize_scalar(source_row.get("split", args.source_split)),
                    "source_embedding_id": str(
                        maybe_int(source_row.get("embedding_row_id", source_row.get("dataset_row_id")))
                    ),
                    "patient_disjoint_from_train": normalize_scalar(source_row.get("patient_disjoint_from_train")),
                    "hadm_disjoint_from_train": normalize_scalar(source_row.get("hadm_disjoint_from_train")),
                    "note_disjoint_from_train": normalize_scalar(source_row.get("note_disjoint_from_train")),
                    "patient_overlap_with_train": normalize_scalar(source_row.get("patient_overlap_with_train")),
                    "hadm_overlap_with_train": normalize_scalar(source_row.get("hadm_overlap_with_train")),
                    "note_overlap_with_train": normalize_scalar(source_row.get("note_overlap_with_train")),
                    "axis_id": direction_idx,
                    "axis_label": selected_labels[direction_idx],
                    "alpha": float(alpha),
                    "normalized_after_steering": bool(args.normalize_after_steering),
                    "random_shift_norm": raw_shift_norm,
                    "editor_model": None,
                    "edited_text": None,
                    "post_edit_source_cosine": cosine,
                    "source_dataset_path": str(dataset_path),
                    "source_split": normalize_scalar(source_row.get("split", args.source_split)),
                    "selection_query": args.selection_query,
                    "steering_run_metadata_path": str(run_metadata_path),
                }
                shifted_rows.append(shifted_row)
                dataset_manifest_rows.append(
                    {
                        "shifted_dataset_row_id": len(dataset_manifest_rows),
                        "source_row_id": shifted_row["source_row_id"],
                        "dataset_row_id": shifted_row["dataset_row_id"],
                        "embedding_row_id": shifted_row["embedding_row_id"],
                        "note_id": shifted_row["note_id"],
                        "subject_id": shifted_row["subject_id"],
                        "hadm_id": shifted_row["hadm_id"],
                        "split": shifted_row["split"],
                        "axis_id": shifted_row["axis_id"],
                        "axis_label": shifted_row["axis_label"],
                        "alpha": shifted_row["alpha"],
                        "normalized_after_steering": shifted_row["normalized_after_steering"],
                        "random_shift_norm": shifted_row["random_shift_norm"],
                        "post_edit_source_cosine": shifted_row["post_edit_source_cosine"],
                    }
                )

    shifted_dataset = Dataset.from_list(shifted_rows)
    shifted_dataset.save_to_disk(str(output_dir))

    manifest_csv = output_dir / f"{args.output_stem}_dataset_manifest.csv"
    pd.DataFrame(dataset_manifest_rows).to_csv(manifest_csv, index=False)

    summary_payload = {
        "created_at": now_iso(),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "script_path": str(Path(__file__).resolve()),
        "dataset_path": str(dataset_path),
        "bank_dir": str(bank_dir) if bank_dir else None,
        "direction_bank_path": str(direction_bank_path) if direction_bank_path else None,
        "output_dir": str(output_dir),
        "output_dataset_path": str(output_dir),
        "output_manifest_csv": str(manifest_csv),
        "factors_path": str(Path(args.factors_path).resolve()) if args.factors_path else None,
        "split_manifest_path": str(Path(args.split_manifest_path).resolve()) if args.split_manifest_path else None,
        "join_report": join_report,
        "active_join_cols": active_join_cols,
        "axis_indices": axis_indices if axis_indices else None,
        "direction_indices": selected_direction_ids,
        "axis_labels": selected_labels,
        "direction_bank_metadata": bank_metadata,
        "alphas": alphas,
        "source_split": args.source_split,
        "selection_query": args.selection_query,
        "normalize_after_steering": bool(args.normalize_after_steering),
        "n_source_rows": int(len(merged_df)),
        "n_shifted_rows": int(len(shifted_rows)),
        "source_dataset_columns": dataset.column_names,
        "example_output_columns": shifted_dataset.column_names,
        "cli_args": vars(args),
    }
    save_json(run_metadata_path, summary_payload)

    print(f"Saved shifted dataset to: {output_dir}")
    print(f"Selected source rows: {len(merged_df)}")
    print(f"Shifted dataset rows: {len(shifted_rows)}")
    print(f"Run metadata: {run_metadata_path}")
    print(f"Dataset manifest CSV: {manifest_csv}")


if __name__ == "__main__":
    main()
