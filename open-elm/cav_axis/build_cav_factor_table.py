#!/usr/bin/env python3
"""
Build a CAV-ready factor table from subgroup metadata and real-manifold cluster assignments.

This is intended for Phase 2 prep after whole-real manifold discovery:
- keep canonical subgroup factors for interpretable axes
- add explicit binary candidate-cluster target columns such as cluster_target_11
- preserve stable join keys so fit_axis_bank.py can merge safely
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_SUBGROUP_METADATA_PATH = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/subgroup_metadata/subgroup_metadata_filtered.csv"
)
DEFAULT_CLUSTER_ASSIGNMENTS_PATH = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/coverage/real_all_filtered_precompute_with_subgroups/"
    "real_all_filtered_cluster_assignments.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/cav_axis_inputs"
)
DEFAULT_CLUSTER_IDS = "11,20,25"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CAV-ready factor table for selected candidate clusters.")
    parser.add_argument("--subgroup_metadata_path", default=DEFAULT_SUBGROUP_METADATA_PATH)
    parser.add_argument("--cluster_assignments_path", default=DEFAULT_CLUSTER_ASSIGNMENTS_PATH)
    parser.add_argument("--cluster_ids", default=DEFAULT_CLUSTER_IDS, help="Comma-separated cluster ids to expose as binary targets")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output_stem",
        default="cav_factor_table_clusters_11_20_25",
        help="Prefix for generated files",
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


def parse_csv_ints(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("No cluster ids were provided.")
    return values


def detect_join_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    preferred = [
        ["source_row_id"],
        ["embedding_row_id"],
        ["dataset_row_id"],
        ["note_id", "subject_id", "hadm_id"],
        ["note_id"],
        ["subject_id", "hadm_id"],
    ]
    left_cols = set(left.columns)
    right_cols = set(right.columns)
    for keys in preferred:
        if all(key in left_cols and key in right_cols for key in keys):
            return keys
    raise ValueError("Could not find a stable shared join key set between subgroup metadata and cluster assignments.")


def main() -> None:
    args = parse_args()
    cluster_ids = parse_csv_ints(args.cluster_ids)

    subgroup_df = pd.read_csv(args.subgroup_metadata_path)
    cluster_df = pd.read_csv(args.cluster_assignments_path)

    join_keys = detect_join_keys(subgroup_df, cluster_df)

    keep_cluster_cols = join_keys + ["cluster_id"]
    if "split" in cluster_df.columns:
        keep_cluster_cols.append("split")
    if "patient_disjoint_from_train" in cluster_df.columns:
        keep_cluster_cols.append("patient_disjoint_from_train")
    cluster_df = cluster_df[keep_cluster_cols].drop_duplicates(subset=join_keys)

    merged = subgroup_df.merge(
        cluster_df,
        on=join_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_subgroup", "_cluster"),
    )

    if "split" not in merged.columns:
        split_candidates = [col for col in ["split_subgroup", "split_cluster"] if col in merged.columns]
        if split_candidates:
            merged["split"] = merged[split_candidates[0]]
            for extra in split_candidates[1:]:
                merged["split"] = merged["split"].where(merged["split"].notna(), merged[extra])

    if "patient_disjoint_from_train" not in merged.columns:
        leakage_candidates = [
            col
            for col in ["patient_disjoint_from_train_subgroup", "patient_disjoint_from_train_cluster"]
            if col in merged.columns
        ]
        if leakage_candidates:
            merged["patient_disjoint_from_train"] = merged[leakage_candidates[0]]
            for extra in leakage_candidates[1:]:
                merged["patient_disjoint_from_train"] = merged["patient_disjoint_from_train"].where(
                    merged["patient_disjoint_from_train"].notna(), merged[extra]
                )

    for cluster_id in cluster_ids:
        merged[f"cluster_target_{cluster_id}"] = (merged["cluster_id"].astype(int) == cluster_id).astype(int)

    merged["candidate_cluster_any"] = merged[[f"cluster_target_{cluster_id}" for cluster_id in cluster_ids]].max(axis=1)
    merged["candidate_cluster_label"] = "other"
    for cluster_id in cluster_ids:
        merged.loc[merged["cluster_id"].astype(int) == cluster_id, "candidate_cluster_label"] = f"cluster_{cluster_id}"

    ordered_cols = [
        "source_row_id",
        "embedding_row_id",
        "dataset_row_id",
        "split",
        "note_id",
        "subject_id",
        "hadm_id",
        "patient_disjoint_from_train",
        "age_bin",
        "sex_gender",
        "race_ethnicity",
        "insurance",
        "admission_type",
        "service",
        "los_bin",
        "icu_flag",
        "cluster_id",
        *[f"cluster_target_{cluster_id}" for cluster_id in cluster_ids],
        "candidate_cluster_any",
        "candidate_cluster_label",
    ]
    ordered_cols = [col for col in ordered_cols if col in merged.columns]
    factor_df = merged[ordered_cols].copy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_stem}.csv"
    json_path = output_dir / f"{args.output_stem}_summary.json"
    factor_df.to_csv(csv_path, index=False)

    summary = {
        "created_at": now_iso(),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "script_path": str(Path(__file__).resolve()),
        "subgroup_metadata_path": str(Path(args.subgroup_metadata_path).resolve()),
        "cluster_assignments_path": str(Path(args.cluster_assignments_path).resolve()),
        "join_keys": join_keys,
        "cluster_ids": cluster_ids,
        "n_rows": int(len(factor_df)),
        "n_candidate_rows": int(factor_df["candidate_cluster_any"].sum()),
        "cluster_target_counts": {
            f"cluster_target_{cluster_id}": int(factor_df[f"cluster_target_{cluster_id}"].sum())
            for cluster_id in cluster_ids
        },
        "split_counts": factor_df["split"].value_counts(dropna=False).to_dict() if "split" in factor_df.columns else {},
        "missing_rates": factor_df.isna().mean().to_dict(),
        "output_csv": str(csv_path.resolve()),
    }
    json_path.write_text(json.dumps(summary, indent=2))

    print("Saved CAV factor table to:", csv_path)
    print("Saved factor-table summary to:", json_path)


if __name__ == "__main__":
    main()
