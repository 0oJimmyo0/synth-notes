#!/usr/bin/env python3
"""
Decoder-feasibility audit for the ELM round-trip map:

    T(e) = BGE(ELM_decode(e))

This script has two subcommands:

1. prepare_subset
   Select real held-out embeddings from sparse target basin clusters and matched
   common clusters, then save a manifest-aware HF dataset that can be decoded by
   generate_synthetic_notes.py.

2. audit_roundtrip
   Given the generated manifest and re-embedded synthetic notes, compute whether
   real target-basin embeddings remain in the basin after decode -> re-embed.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    def _default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def maybe_int(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        return value


def load_dataset(dataset_path: Path) -> tuple[Dataset, pd.DataFrame, np.ndarray]:
    dataset = Dataset.load_from_disk(str(dataset_path))
    metadata_cols = [col for col in dataset.column_names if col not in {"input_ids", "domain_embeddings"}]
    if metadata_cols:
        meta_df = dataset.select_columns(metadata_cols).to_pandas().reset_index(drop=True)
    else:
        meta_df = pd.DataFrame(index=np.arange(len(dataset), dtype=int))
    if "dataset_row_id" not in meta_df.columns:
        meta_df.insert(0, "dataset_row_id", np.arange(len(dataset), dtype=int))
    if "dataset_local_row_id" not in meta_df.columns:
        meta_df.insert(0, "dataset_local_row_id", np.arange(len(dataset), dtype=int))

    embs = []
    for row in dataset["domain_embeddings"]:
        arr = np.asarray(row[0], dtype=np.float32)
        while arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr[0]
        embs.append(arr)
    emb_matrix = normalize_rows(np.vstack(embs))
    return dataset, meta_df, emb_matrix


def normalize_join_cols(df: pd.DataFrame, join_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in join_cols:
        if col not in out.columns:
            raise KeyError(f"Missing join column: {col}")
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().all():
            out[col] = numeric.astype(int)
        else:
            out[col] = out[col].astype(str).str.strip()
    return out


def choose_join_keys(frames: list[pd.DataFrame], preferred_keys: list[str]) -> list[str]:
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
    for keys in preferred_groups:
        if all(key in common_cols for key in keys):
            return keys
    raise ValueError(f"Could not detect stable join keys. Shared columns were: {sorted(common_cols)}")


def merge_metadata(
    base_df: pd.DataFrame,
    cluster_assignments_path: Path,
    split_manifest_path: Path | None,
    preferred_join_cols: list[str],
    source_split: str | None,
) -> pd.DataFrame:
    merged = base_df.copy()
    if split_manifest_path:
        split_df = pd.read_csv(split_manifest_path)
        join_cols = choose_join_keys([merged, split_df], preferred_join_cols)
        if join_cols == ["dataset_row_id"] and "split" in split_df.columns and source_split:
            split_df = split_df.loc[split_df["split"].astype(str) == str(source_split)].copy()
        merged = normalize_join_cols(merged, join_cols)
        split_df = normalize_join_cols(split_df, join_cols).drop_duplicates(subset=join_cols)
        merged = merged.merge(split_df, on=join_cols, how="left", validate="many_to_one", suffixes=("", "_split"))

    assignments_df = pd.read_csv(cluster_assignments_path)
    join_cols = choose_join_keys([merged, assignments_df], preferred_join_cols)
    merged = normalize_join_cols(merged, join_cols)
    assignments_df = normalize_join_cols(assignments_df, join_cols).drop_duplicates(subset=join_cols)
    merged = merged.merge(assignments_df, on=join_cols, how="left", validate="many_to_one", suffixes=("", "_assign"))
    return merged


def build_centroids(real_embeddings: np.ndarray, assignments_df: pd.DataFrame, cluster_ids: list[int]) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for cid in cluster_ids:
        ids = pd.to_numeric(
            assignments_df.loc[pd.to_numeric(assignments_df["cluster_id"], errors="coerce") == cid, "dataset_row_id"],
            errors="coerce",
        ).dropna().astype(int).to_numpy()
        if len(ids) == 0:
            continue
        centroids[cid] = normalize_rows(real_embeddings[ids].mean(axis=0, keepdims=True))[0]
    return centroids


def sample_rows(df: pd.DataFrame, n_rows: int, rng: random.Random) -> pd.DataFrame:
    if len(df) <= n_rows:
        return df.copy()
    idx = list(df.index)
    rng.shuffle(idx)
    return df.loc[idx[:n_rows]].copy()


def choose_common_clusters_auto(
    assignments_df: pd.DataFrame,
    cluster_summary_path: Path,
    target_cluster_ids: list[int],
    n_common_clusters: int,
) -> list[int]:
    summary_df = pd.read_csv(cluster_summary_path)
    summary_df["cluster_id"] = pd.to_numeric(summary_df["cluster_id"], errors="coerce")
    summary_df["cluster_size"] = pd.to_numeric(summary_df["cluster_size"], errors="coerce")
    summary_df = summary_df.dropna(subset=["cluster_id", "cluster_size"]).copy()
    summary_df["cluster_id"] = summary_df["cluster_id"].astype(int)
    summary_df = summary_df.loc[~summary_df["cluster_id"].isin(set(target_cluster_ids))].copy()
    summary_df = summary_df.sort_values("cluster_size", ascending=False).head(int(n_common_clusters)).copy()
    common_ids = summary_df["cluster_id"].tolist()
    if not common_ids:
        raise ValueError("Could not infer common clusters automatically.")
    return common_ids


def prepare_subset(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset_path).resolve()
    cluster_assignments_path = Path(args.cluster_assignments_path).resolve()
    cluster_summary_path = Path(args.cluster_summary_path).resolve()
    split_manifest_path = Path(args.split_manifest_path).resolve() if args.split_manifest_path else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preferred_join_cols = parse_csv_list(args.join_cols)
    target_cluster_ids = parse_int_list(args.target_cluster_ids)
    if not target_cluster_ids:
        raise ValueError("--target_cluster_ids must not be empty")

    dataset, dataset_meta, _ = load_dataset(dataset_path)
    merged = merge_metadata(dataset_meta, cluster_assignments_path, split_manifest_path, preferred_join_cols, args.source_split)
    if args.source_split and "split" in merged.columns:
        merged = merged.loc[merged["split"].astype(str) == str(args.source_split)].copy()

    if "cluster_id" not in merged.columns:
        raise KeyError("Merged metadata must contain cluster_id")
    merged["cluster_id"] = pd.to_numeric(merged["cluster_id"], errors="coerce")
    merged = merged.dropna(subset=["cluster_id"]).copy()
    merged["cluster_id"] = merged["cluster_id"].astype(int)

    common_cluster_ids = parse_int_list(args.common_cluster_ids)
    if not common_cluster_ids:
        common_cluster_ids = choose_common_clusters_auto(merged, cluster_summary_path, target_cluster_ids, args.n_common_clusters)

    rng = random.Random(int(args.seed))
    selected_frames: list[pd.DataFrame] = []

    for cid in target_cluster_ids:
        cluster_df = merged.loc[merged["cluster_id"] == cid].copy()
        cluster_df = sample_rows(cluster_df, int(args.target_rows_per_cluster), rng)
        cluster_df["decoder_group_family"] = "target_basin"
        cluster_df["decoder_group_label"] = f"target_cluster_{cid}"
        selected_frames.append(cluster_df)

    for cid in common_cluster_ids:
        cluster_df = merged.loc[merged["cluster_id"] == cid].copy()
        cluster_df = sample_rows(cluster_df, int(args.common_rows_per_cluster), rng)
        cluster_df["decoder_group_family"] = "common_cluster"
        cluster_df["decoder_group_label"] = f"common_cluster_{cid}"
        selected_frames.append(cluster_df)

    selection_df = pd.concat(selected_frames, ignore_index=True)
    selection_df = selection_df.drop_duplicates(subset=["dataset_row_id"], keep="first").reset_index(drop=True)
    selection_df["selection_rank"] = np.arange(len(selection_df), dtype=int)
    selection_df["target_basin_cluster_ids"] = ",".join(str(x) for x in target_cluster_ids)
    selection_df["common_cluster_ids"] = ",".join(str(x) for x in common_cluster_ids)

    subset_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for _, row in selection_df.iterrows():
        dataset_row_id = int(row["dataset_row_id"])
        example = dataset[dataset_row_id]
        out_row = {
            "input_ids": example["input_ids"],
            "domain_embeddings": example["domain_embeddings"],
            "source_row_id": maybe_int(row.get("source_row_id", row.get("dataset_row_id"))),
            "dataset_row_id": maybe_int(row.get("dataset_row_id")),
            "embedding_row_id": maybe_int(row.get("embedding_row_id")),
            "note_id": row.get("note_id"),
            "subject_id": maybe_int(row.get("subject_id")),
            "hadm_id": maybe_int(row.get("hadm_id")),
            "split": row.get("split", args.source_split),
            "source_embedding_id": str(maybe_int(row.get("embedding_row_id", row.get("dataset_row_id")))),
            "patient_disjoint_from_train": row.get("patient_disjoint_from_train"),
            "hadm_disjoint_from_train": row.get("hadm_disjoint_from_train"),
            "note_disjoint_from_train": row.get("note_disjoint_from_train"),
            "patient_overlap_with_train": row.get("patient_overlap_with_train"),
            "hadm_overlap_with_train": row.get("hadm_overlap_with_train"),
            "note_overlap_with_train": row.get("note_overlap_with_train"),
            "source_dataset_path": str(dataset_path),
            "source_split": row.get("split", args.source_split),
            "selection_query": "decoder_feasibility_subset",
        }
        subset_rows.append(out_row)
        manifest_rows.append(
            {
                "subset_row_id": len(manifest_rows),
                "dataset_row_id": out_row["dataset_row_id"],
                "source_row_id": out_row["source_row_id"],
                "embedding_row_id": out_row["embedding_row_id"],
                "note_id": out_row["note_id"],
                "subject_id": out_row["subject_id"],
                "hadm_id": out_row["hadm_id"],
                "split": out_row["split"],
                "cluster_id": int(row["cluster_id"]),
                "decoder_group_family": row["decoder_group_family"],
                "decoder_group_label": row["decoder_group_label"],
                "patient_disjoint_from_train": row.get("patient_disjoint_from_train"),
                "distance_to_centroid": row.get("distance_to_centroid"),
            }
        )

    subset_dataset = Dataset.from_list(subset_rows)
    subset_dataset_path = output_dir / "decoder_feasibility_subset_dataset"
    subset_dataset.save_to_disk(str(subset_dataset_path))

    selection_manifest_path = output_dir / "decoder_feasibility_selection_manifest.csv"
    dataset_manifest_path = output_dir / "decoder_feasibility_subset_dataset_manifest.csv"
    selection_df.to_csv(selection_manifest_path, index=False)
    pd.DataFrame(manifest_rows).to_csv(dataset_manifest_path, index=False)

    summary = {
        "created_at": now_iso(),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "script_path": str(Path(__file__).resolve()),
        "dataset_path": str(dataset_path),
        "cluster_assignments_path": str(cluster_assignments_path),
        "cluster_summary_path": str(cluster_summary_path),
        "split_manifest_path": str(split_manifest_path) if split_manifest_path else None,
        "output_dir": str(output_dir),
        "subset_dataset_path": str(subset_dataset_path),
        "selection_manifest_path": str(selection_manifest_path),
        "target_cluster_ids": target_cluster_ids,
        "common_cluster_ids": common_cluster_ids,
        "source_split": args.source_split,
        "n_selected_rows": int(len(selection_df)),
        "group_counts": {
            str(k): int(v)
            for k, v in selection_df["decoder_group_label"].value_counts().sort_index().to_dict().items()
        },
        "family_counts": {
            str(k): int(v)
            for k, v in selection_df["decoder_group_family"].value_counts().sort_index().to_dict().items()
        },
        "cli_args": vars(args),
    }
    save_json(output_dir / "decoder_feasibility_selection_summary.json", summary)

    print(f"Saved subset dataset to: {subset_dataset_path}")
    print(f"Saved selection manifest to: {selection_manifest_path}")
    print(json.dumps(summary["group_counts"], indent=2))


@dataclass
class RoundtripSummary:
    full_target_retention: float
    pd_target_retention: float
    full_common_entry: float
    pd_common_entry: float
    decision: str
    rationale: str


def decide_decoder_feasibility(
    retention_df: pd.DataFrame,
    retention_pass_threshold: float,
    retention_advantage_margin: float,
) -> RoundtripSummary:
    target_full = retention_df.loc[
        (retention_df["source_group_family"] == "target_basin") & (retention_df["analysis_group"] == "full"),
        "target_basin_rate_after",
    ]
    target_pd = retention_df.loc[
        (retention_df["source_group_family"] == "target_basin") & (retention_df["analysis_group"] == "patient_disjoint"),
        "target_basin_rate_after",
    ]
    common_full = retention_df.loc[
        (retention_df["source_group_family"] == "common_cluster") & (retention_df["analysis_group"] == "full"),
        "target_basin_rate_after",
    ]
    common_pd = retention_df.loc[
        (retention_df["source_group_family"] == "common_cluster") & (retention_df["analysis_group"] == "patient_disjoint"),
        "target_basin_rate_after",
    ]

    tf = float(target_full.iloc[0]) if len(target_full) else float("nan")
    tpd = float(target_pd.iloc[0]) if len(target_pd) else float("nan")
    cf = float(common_full.iloc[0]) if len(common_full) else float("nan")
    cpd = float(common_pd.iloc[0]) if len(common_pd) else float("nan")

    if np.isnan(tf):
        return RoundtripSummary(tf, tpd, cf, cpd, "CAUTION", "target-basin rows missing from audit selection")

    if tf < retention_pass_threshold or (not np.isnan(tpd) and tpd < retention_pass_threshold):
        return RoundtripSummary(
            tf,
            tpd,
            cf,
            cpd,
            "STOP_INFERENCE_TIME_STEERING",
            "real target-basin embeddings do not retain basin identity strongly enough after decode -> re-embed",
        )

    if not np.isnan(cf) and (tf - cf) < retention_advantage_margin:
        return RoundtripSummary(
            tf,
            tpd,
            cf,
            cpd,
            "CAUTION",
            "target-basin retention exists, but it is not clearly separated from common-cluster basin entry",
        )

    return RoundtripSummary(
        tf,
        tpd,
        cf,
        cpd,
        "PROMISING_FOR_DECODER_COMPATIBLE_TRANSPORT",
        "real target-basin embeddings retain basin identity after decode -> re-embed better than common clusters",
    )


def audit_roundtrip(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset_path).resolve()
    cluster_assignments_path = Path(args.cluster_assignments_path).resolve()
    selection_manifest_path = Path(args.selection_manifest_path).resolve()
    generated_manifest_path = Path(args.generated_manifest_path).resolve()
    generated_embeddings_path = Path(args.generated_embeddings_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_cluster_ids = parse_int_list(args.target_cluster_ids)
    preferred_join_cols = parse_csv_list(args.join_cols)

    _, source_meta, source_embeddings = load_dataset(dataset_path)
    assignments_df = pd.read_csv(cluster_assignments_path)
    merged_source = merge_metadata(source_meta, cluster_assignments_path, None, preferred_join_cols, args.source_split)
    if args.source_split and "split" in merged_source.columns:
        merged_source = merged_source.loc[merged_source["split"].astype(str) == str(args.source_split)].copy()

    selection_df = pd.read_csv(selection_manifest_path)
    generated_manifest_df = pd.read_json(generated_manifest_path, lines=True)
    generated_embeddings = normalize_rows(np.load(generated_embeddings_path))

    if len(generated_manifest_df) != len(generated_embeddings):
        raise ValueError(
            f"Generated manifest rows ({len(generated_manifest_df)}) do not match embedding rows ({len(generated_embeddings)})"
        )

    join_cols = choose_join_keys([selection_df, generated_manifest_df], preferred_join_cols)
    selection_df = normalize_join_cols(selection_df, join_cols)
    generated_manifest_df = normalize_join_cols(generated_manifest_df, join_cols)
    selection_df = selection_df.drop_duplicates(subset=join_cols)
    roundtrip_df = generated_manifest_df.merge(
        selection_df,
        on=join_cols,
        how="left",
        validate="many_to_one",
        suffixes=("", "_sel"),
    )
    if roundtrip_df["decoder_group_family"].isna().any():
        raise ValueError("Failed to align generated manifest with selection manifest.")

    all_cluster_ids = sorted(
        pd.to_numeric(assignments_df["cluster_id"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    centroids = build_centroids(source_embeddings, merged_source, all_cluster_ids)
    target_centroid = normalize_rows(
        np.mean(np.stack([centroids[cid] for cid in target_cluster_ids if cid in centroids], axis=0), axis=0, keepdims=True)
    )[0]

    after_scores = []
    for cid in all_cluster_ids:
        after_scores.append(generated_embeddings @ centroids[cid])
    after_score_matrix = np.stack(after_scores, axis=1)
    best_after_idx = np.argmax(after_score_matrix, axis=1)
    after_cluster_ids = np.asarray([all_cluster_ids[i] for i in best_after_idx], dtype=int)

    dataset_row_ids = pd.to_numeric(roundtrip_df["dataset_row_id"], errors="coerce").astype(int).to_numpy()
    source_roundtrip_embeddings = source_embeddings[dataset_row_ids]

    roundtrip_df["source_cosine"] = np.sum(source_roundtrip_embeddings * generated_embeddings, axis=1)
    roundtrip_df["source_cluster_before"] = pd.to_numeric(roundtrip_df["cluster_id"], errors="coerce").astype(int)
    roundtrip_df["cluster_after"] = after_cluster_ids
    roundtrip_df["before_in_target_basin"] = roundtrip_df["source_cluster_before"].isin(set(target_cluster_ids)).astype(int)
    roundtrip_df["after_in_target_basin"] = roundtrip_df["cluster_after"].isin(set(target_cluster_ids)).astype(int)
    roundtrip_df["distance_to_target_centroid_before"] = 1.0 - (source_roundtrip_embeddings @ target_centroid)
    roundtrip_df["distance_to_target_centroid_after"] = 1.0 - (generated_embeddings @ target_centroid)
    roundtrip_df["distance_delta_after_minus_before"] = (
        roundtrip_df["distance_to_target_centroid_after"] - roundtrip_df["distance_to_target_centroid_before"]
    )

    if "patient_disjoint_from_train" in roundtrip_df.columns:
        flags = roundtrip_df["patient_disjoint_from_train"].astype(str).str.lower()
        roundtrip_df["analysis_group"] = np.where(flags == "true", "patient_disjoint", np.where(flags == "false", "patient_overlap", "unknown"))
    else:
        roundtrip_df["analysis_group"] = "unknown"

    transition_df = (
        roundtrip_df.groupby(
            ["decoder_group_family", "decoder_group_label", "analysis_group", "source_cluster_before", "cluster_after"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    transition_df["fraction_within_group"] = transition_df.groupby(
        ["decoder_group_family", "decoder_group_label", "analysis_group", "source_cluster_before"]
    )["count"].transform(lambda s: s / max(float(s.sum()), 1.0))

    retention_rows = []
    for family in ["target_basin", "common_cluster"]:
        fam_df = roundtrip_df.loc[roundtrip_df["decoder_group_family"] == family].copy()
        if fam_df.empty:
            continue
        for group_name in ["full", "patient_disjoint", "patient_overlap"]:
            subset = fam_df if group_name == "full" else fam_df.loc[fam_df["analysis_group"] == group_name].copy()
            if subset.empty:
                continue
            target_source_subset = subset.loc[subset["before_in_target_basin"] == 1].copy()
            non_target_source_subset = subset.loc[subset["before_in_target_basin"] == 0].copy()
            retention_rows.append(
                {
                    "source_group_family": family,
                    "analysis_group": group_name,
                    "n_rows": int(len(subset)),
                    "source_in_target_basin_rate_before": float(subset["before_in_target_basin"].mean()),
                    "target_basin_rate_after": float(subset["after_in_target_basin"].mean()),
                    "retention_rate_if_source_target": float(target_source_subset["after_in_target_basin"].mean()) if len(target_source_subset) else np.nan,
                    "entry_rate_if_source_non_target": float(non_target_source_subset["after_in_target_basin"].mean()) if len(non_target_source_subset) else np.nan,
                    "mean_source_cosine": float(subset["source_cosine"].mean()),
                }
            )
    retention_df = pd.DataFrame(retention_rows)

    centroid_rows = []
    for family in ["target_basin", "common_cluster"]:
        fam_df = roundtrip_df.loc[roundtrip_df["decoder_group_family"] == family].copy()
        if fam_df.empty:
            continue
        for group_name in ["full", "patient_disjoint", "patient_overlap"]:
            subset = fam_df if group_name == "full" else fam_df.loc[fam_df["analysis_group"] == group_name].copy()
            if subset.empty:
                continue
            centroid_rows.append(
                {
                    "source_group_family": family,
                    "analysis_group": group_name,
                    "n_rows": int(len(subset)),
                    "mean_distance_to_target_centroid_before": float(subset["distance_to_target_centroid_before"].mean()),
                    "mean_distance_to_target_centroid_after": float(subset["distance_to_target_centroid_after"].mean()),
                    "mean_distance_delta_after_minus_before": float(subset["distance_delta_after_minus_before"].mean()),
                    "median_distance_to_target_centroid_before": float(subset["distance_to_target_centroid_before"].median()),
                    "median_distance_to_target_centroid_after": float(subset["distance_to_target_centroid_after"].median()),
                }
            )
    centroid_df = pd.DataFrame(centroid_rows)

    decision = decide_decoder_feasibility(
        retention_df=retention_df,
        retention_pass_threshold=float(args.retention_pass_threshold),
        retention_advantage_margin=float(args.retention_advantage_margin),
    )

    cluster_transition_path = output_dir / "cluster_transition_matrix.csv"
    retention_path = output_dir / "target_basin_retention.csv"
    centroid_path = output_dir / "before_after_centroid_distance.csv"
    json_path = output_dir / "decoder_feasibility_audit.json"
    md_path = output_dir / "decoder_feasibility_audit.md"

    transition_df.to_csv(cluster_transition_path, index=False)
    retention_df.to_csv(retention_path, index=False)
    centroid_df.to_csv(centroid_path, index=False)

    payload = {
        "created_at": now_iso(),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "script_path": str(Path(__file__).resolve()),
        "dataset_path": str(dataset_path),
        "cluster_assignments_path": str(cluster_assignments_path),
        "selection_manifest_path": str(selection_manifest_path),
        "generated_manifest_path": str(generated_manifest_path),
        "generated_embeddings_path": str(generated_embeddings_path),
        "output_dir": str(output_dir),
        "target_cluster_ids": target_cluster_ids,
        "n_rows": int(len(roundtrip_df)),
        "decision": decision.decision,
        "rationale": decision.rationale,
        "target_basin_retention_full": decision.full_target_retention,
        "target_basin_retention_patient_disjoint": decision.pd_target_retention,
        "common_cluster_target_basin_entry_full": decision.full_common_entry,
        "common_cluster_target_basin_entry_patient_disjoint": decision.pd_common_entry,
        "retention_summary": retention_df.to_dict(orient="records"),
        "centroid_distance_summary": centroid_df.to_dict(orient="records"),
    }
    save_json(json_path, payload)

    lines = [
        "# Decoder Feasibility Audit",
        "",
        "This audit tests whether real held-out embeddings preserve target-basin identity after the ELM round trip:",
        "",
        "`T(e) = BGE(ELM_decode(e))`",
        "",
        "## Decision",
        "",
        f"- status: `{decision.decision}`",
        f"- rationale: {decision.rationale}",
        f"- target-basin retention (full): `{decision.full_target_retention:.4f}`" if not np.isnan(decision.full_target_retention) else "- target-basin retention (full): `nan`",
        f"- target-basin retention (patient-disjoint): `{decision.pd_target_retention:.4f}`" if not np.isnan(decision.pd_target_retention) else "- target-basin retention (patient-disjoint): `nan`",
        f"- common-cluster target-basin entry (full): `{decision.full_common_entry:.4f}`" if not np.isnan(decision.full_common_entry) else "- common-cluster target-basin entry (full): `nan`",
        "",
        "## Files",
        "",
        "- `decoder_feasibility_audit.json`",
        "- `cluster_transition_matrix.csv`",
        "- `target_basin_retention.csv`",
        "- `before_after_centroid_distance.csv`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved decoder feasibility audit to: {json_path}")
    print(f"Saved cluster transitions to: {cluster_transition_path}")
    print(f"Saved retention table to: {retention_path}")
    print(f"Saved centroid distance table to: {centroid_path}")
    print(json.dumps({"decision": decision.decision, "rationale": decision.rationale}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decoder feasibility audit for ELM round-trip behavior.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare_subset", help="Build a manifest-aware held-out real subset for decoder audit generation.")
    prepare.add_argument("--dataset_path", required=True)
    prepare.add_argument("--cluster_assignments_path", required=True)
    prepare.add_argument("--cluster_summary_path", required=True)
    prepare.add_argument("--split_manifest_path", default=None)
    prepare.add_argument("--output_dir", required=True)
    prepare.add_argument("--target_cluster_ids", required=True, help="Comma-separated sparse target basin clusters")
    prepare.add_argument("--common_cluster_ids", default="", help="Optional explicit common cluster IDs")
    prepare.add_argument("--n_common_clusters", type=int, default=3, help="Used only when common_cluster_ids is omitted")
    prepare.add_argument("--target_rows_per_cluster", type=int, default=64)
    prepare.add_argument("--common_rows_per_cluster", type=int, default=64)
    prepare.add_argument("--source_split", default="test")
    prepare.add_argument(
        "--join_cols",
        default="source_row_id,embedding_row_id,dataset_row_id,note_id,subject_id,hadm_id",
    )
    prepare.add_argument("--seed", type=int, default=42)

    audit = subparsers.add_parser("audit_roundtrip", help="Audit decoder round-trip behavior after generation + re-embedding.")
    audit.add_argument("--dataset_path", required=True)
    audit.add_argument("--cluster_assignments_path", required=True)
    audit.add_argument("--selection_manifest_path", required=True)
    audit.add_argument("--generated_manifest_path", required=True)
    audit.add_argument("--generated_embeddings_path", required=True)
    audit.add_argument("--output_dir", required=True)
    audit.add_argument("--target_cluster_ids", required=True)
    audit.add_argument(
        "--join_cols",
        default="source_row_id,embedding_row_id,dataset_row_id,note_id,subject_id,hadm_id",
    )
    audit.add_argument("--source_split", default="test")
    audit.add_argument("--retention_pass_threshold", type=float, default=0.70)
    audit.add_argument("--retention_advantage_margin", type=float, default=0.15)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare_subset":
        prepare_subset(args)
    elif args.command == "audit_roundtrip":
        audit_roundtrip(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
