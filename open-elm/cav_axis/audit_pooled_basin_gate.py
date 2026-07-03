#!/usr/bin/env python3
"""
Pre-decode gate for pooled-basin entry and low-density basin enrichment.

This is the Phase 2b gate that better matches the project goal when exact
single-cluster crossing is too strict. It compares source, candidate, and
control embedding datasets on:

- pooled-basin win rate vs external clusters
- pooled-basin margin vs best external cluster
- low-density basin win rate
- low-density basin margin vs best non-low-density competitor
- source preservation
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit pooled-basin and low-density enrichment before decode.")
    parser.add_argument("--source_dataset_path", required=True)
    parser.add_argument("--candidate_dataset_paths", required=True)
    parser.add_argument("--candidate_labels", required=True)
    parser.add_argument("--cluster_assignments_path", required=True)
    parser.add_argument("--cluster_summary_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--basin_cluster_ids", required=True)
    parser.add_argument("--comparison_cluster_ids", required=True)
    parser.add_argument("--low_density_basin_cluster_ids", required=True)
    parser.add_argument("--split_manifest_path", default=None)
    parser.add_argument(
        "--join_cols",
        default="source_row_id,embedding_row_id,dataset_row_id,note_id,subject_id,hadm_id",
        help="Preferred join columns for metadata merges",
    )
    parser.add_argument("--source_split", default=None)
    parser.add_argument("--anchor_manifest_path", default=None, help="Optional CSV/JSONL anchor manifest with dataset_row_id")
    parser.add_argument("--control_labels", default="")
    parser.add_argument("--pass_basin_win_delta", type=float, default=0.05)
    parser.add_argument("--pass_low_density_win_delta", type=float, default=0.03)
    parser.add_argument("--pass_basin_margin_delta", type=float, default=0.005)
    parser.add_argument("--min_source_cosine", type=float, default=0.97)
    return parser


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


def load_dataset_rows(dataset_path: Path) -> tuple[Dataset, pd.DataFrame, np.ndarray]:
    dataset = Dataset.load_from_disk(str(dataset_path))
    metadata_cols = [col for col in dataset.column_names if col not in {"input_ids", "domain_embeddings"}]
    if metadata_cols:
        base_df = dataset.select_columns(metadata_cols).to_pandas().reset_index(drop=True)
    else:
        base_df = pd.DataFrame(index=np.arange(len(dataset), dtype=int))
    if "dataset_row_id" not in base_df.columns:
        base_df.insert(0, "dataset_row_id", np.arange(len(dataset), dtype=int))
    if "dataset_local_row_id" not in base_df.columns:
        base_df.insert(0, "dataset_local_row_id", np.arange(len(dataset), dtype=int))

    embeddings = []
    for emb in dataset["domain_embeddings"]:
        arr = np.asarray(emb[0], dtype=np.float32)
        while arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr[0]
        embeddings.append(arr)
    return dataset, base_df, normalize_rows(np.vstack(embeddings))


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


def normalize_join_cols(df: pd.DataFrame, join_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in join_cols:
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().all():
            out[col] = numeric.astype(int)
        else:
            out[col] = out[col].astype(str).str.strip()
    return out


def merge_metadata(
    base_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    split_manifest_path: str | None,
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

    join_cols = choose_join_keys([merged, assignments_df], preferred_join_cols)
    merged = normalize_join_cols(merged, join_cols)
    assignments_df = normalize_join_cols(assignments_df, join_cols).drop_duplicates(subset=join_cols)
    merged = merged.merge(assignments_df, on=join_cols, how="left", validate="many_to_one", suffixes=("", "_assign"))
    return merged


def build_centroids(real_embeddings: np.ndarray, assignments_df: pd.DataFrame, cluster_ids: list[int]) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for cluster_id in cluster_ids:
        ids = assignments_df.loc[pd.to_numeric(assignments_df["cluster_id"], errors="coerce") == cluster_id, "dataset_row_id"].astype(int).to_numpy()
        if len(ids) == 0:
            raise ValueError(f"No rows found for cluster_id={cluster_id}")
        centroids[cluster_id] = normalize_rows(real_embeddings[ids].mean(axis=0, keepdims=True))[0]
    return centroids


def load_anchor_ids(path: Path) -> set[int]:
    if path.suffix.lower() == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)
    if "dataset_row_id" not in df.columns:
        raise KeyError("Anchor manifest must contain dataset_row_id")
    return set(pd.to_numeric(df["dataset_row_id"], errors="coerce").dropna().astype(int).tolist())


def leakage_group(values: pd.Series) -> pd.Series:
    normed = values.astype(str).str.lower()
    out = pd.Series(index=values.index, dtype=object)
    out.loc[normed == "true"] = "patient_disjoint"
    out.loc[normed == "false"] = "patient_overlap"
    out = out.fillna("unknown")
    return out


def evaluate_candidate(
    candidate_label: str,
    candidate_meta: pd.DataFrame,
    candidate_embeddings: np.ndarray,
    source_embeddings: np.ndarray,
    centroids: dict[int, np.ndarray],
    basin_cluster_ids: list[int],
    comparison_cluster_ids: list[int],
    low_density_basin_cluster_ids: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    row_df = candidate_meta.copy().reset_index(drop=True)

    for cid in comparison_cluster_ids:
        row_df[f"cos_to_cluster_{cid}"] = candidate_embeddings @ centroids[cid]

    basin_cols = [f"cos_to_cluster_{cid}" for cid in basin_cluster_ids]
    external_cluster_ids = [cid for cid in comparison_cluster_ids if cid not in set(basin_cluster_ids)]
    external_cols = [f"cos_to_cluster_{cid}" for cid in external_cluster_ids]
    low_density_cols = [f"cos_to_cluster_{cid}" for cid in low_density_basin_cluster_ids]
    non_low_density_cols = [f"cos_to_cluster_{cid}" for cid in comparison_cluster_ids if cid not in set(low_density_basin_cluster_ids)]

    row_df["best_basin_cosine"] = row_df[basin_cols].max(axis=1)
    basin_col_to_id = {f"cos_to_cluster_{cid}": cid for cid in basin_cluster_ids}
    row_df["best_basin_cluster"] = row_df[basin_cols].idxmax(axis=1).map(basin_col_to_id).astype(int)

    if external_cols:
        row_df["best_external_cosine"] = row_df[external_cols].max(axis=1)
        external_col_to_id = {f"cos_to_cluster_{cid}": cid for cid in external_cluster_ids}
        row_df["best_external_cluster"] = row_df[external_cols].idxmax(axis=1).map(external_col_to_id).astype(int)
    else:
        row_df["best_external_cosine"] = np.nan
        row_df["best_external_cluster"] = pd.NA

    row_df["basin_margin_vs_external"] = row_df["best_basin_cosine"] - row_df["best_external_cosine"]
    row_df["wins_pooled_basin"] = (row_df["basin_margin_vs_external"] > 0).astype(int)

    row_df["best_low_density_cosine"] = row_df[low_density_cols].max(axis=1)
    low_col_to_id = {f"cos_to_cluster_{cid}": cid for cid in low_density_basin_cluster_ids}
    row_df["best_low_density_cluster"] = row_df[low_density_cols].idxmax(axis=1).map(low_col_to_id).astype(int)

    row_df["best_non_low_density_cosine"] = row_df[non_low_density_cols].max(axis=1)
    row_df["low_density_margin"] = row_df["best_low_density_cosine"] - row_df["best_non_low_density_cosine"]
    row_df["wins_low_density_basin"] = (row_df["low_density_margin"] > 0).astype(int)

    source_row_ids = row_df["dataset_row_id"].astype(int).to_numpy()
    resolved_source_ids = row_df.get("source_row_id_resolved")
    if resolved_source_ids is not None and pd.to_numeric(resolved_source_ids, errors="coerce").notna().any():
        source_row_ids = pd.to_numeric(resolved_source_ids, errors="coerce").fillna(pd.Series(source_row_ids)).astype(int).to_numpy()
    row_df["source_cosine"] = np.sum(candidate_embeddings * source_embeddings[source_row_ids], axis=1)

    row_df["candidate_label"] = candidate_label
    row_df["analysis_group"] = leakage_group(row_df.get("patient_disjoint_from_train", pd.Series([pd.NA] * len(row_df))))

    summary = {
        "candidate_label": candidate_label,
        "n_rows": int(len(row_df)),
        "mean_source_cosine": float(row_df["source_cosine"].mean()),
        "pooled_basin_win_rate": float(row_df["wins_pooled_basin"].mean()),
        "mean_basin_margin_vs_external": float(row_df["basin_margin_vs_external"].mean()),
        "low_density_basin_win_rate": float(row_df["wins_low_density_basin"].mean()),
        "mean_low_density_margin": float(row_df["low_density_margin"].mean()),
        "patient_disjoint_pooled_basin_win_rate": float(row_df.loc[row_df["analysis_group"] == "patient_disjoint", "wins_pooled_basin"].mean()),
        "patient_disjoint_low_density_basin_win_rate": float(row_df.loc[row_df["analysis_group"] == "patient_disjoint", "wins_low_density_basin"].mean()),
    }
    return row_df, summary


def assign_gate_status(
    summary_df: pd.DataFrame,
    control_labels: list[str],
    pass_basin_win_delta: float,
    pass_low_density_win_delta: float,
    pass_basin_margin_delta: float,
    min_source_cosine: float,
) -> pd.DataFrame:
    out = summary_df.copy()
    out["gate_status"] = "UNSET"
    out["gate_reason"] = ""

    controls = out.loc[out["candidate_label"].isin(control_labels)].copy()
    if controls.empty:
        raise ValueError("No controls matched --control_labels")

    best_basin_win = float(controls["pooled_basin_win_rate"].max())
    best_low_density_win = float(controls["low_density_basin_win_rate"].max())
    best_basin_margin = float(controls["mean_basin_margin_vs_external"].max())

    for idx, row in out.iterrows():
        if row["candidate_label"] in control_labels:
            out.at[idx, "gate_status"] = "CONTROL"
            out.at[idx, "gate_reason"] = "control baseline"
            continue

        basin_win_gain = float(row["pooled_basin_win_rate"] - best_basin_win)
        low_density_win_gain = float(row["low_density_basin_win_rate"] - best_low_density_win)
        basin_margin_gain = float(row["mean_basin_margin_vs_external"] - best_basin_margin)
        source_cos = float(row["mean_source_cosine"])

        if (
            basin_win_gain >= pass_basin_win_delta
            and low_density_win_gain >= pass_low_density_win_delta
            and basin_margin_gain >= pass_basin_margin_delta
            and source_cos >= min_source_cosine
        ):
            status = "PASS"
        elif source_cos < min_source_cosine:
            status = "FAIL"
        else:
            status = "CAUTION"

        out.at[idx, "gate_status"] = status
        out.at[idx, "gate_reason"] = (
            f"basin_win_gain_vs_best_control={basin_win_gain:.6f}; "
            f"low_density_win_gain_vs_best_control={low_density_win_gain:.6f}; "
            f"basin_margin_gain_vs_best_control={basin_margin_gain:.6f}; "
            f"mean_source_cosine={source_cos:.6f}"
        )
    return out


def main() -> None:
    args = build_parser().parse_args()

    source_dataset_path = Path(args.source_dataset_path).resolve()
    candidate_dataset_paths = [Path(x).resolve() for x in parse_csv_list(args.candidate_dataset_paths)]
    candidate_labels = parse_csv_list(args.candidate_labels)
    assignments_path = Path(args.cluster_assignments_path).resolve()
    cluster_summary_path = Path(args.cluster_summary_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(candidate_dataset_paths) != len(candidate_labels):
        raise ValueError("candidate_dataset_paths and candidate_labels must have the same length")

    basin_cluster_ids = parse_int_list(args.basin_cluster_ids)
    comparison_cluster_ids = parse_int_list(args.comparison_cluster_ids)
    low_density_basin_cluster_ids = parse_int_list(args.low_density_basin_cluster_ids)
    control_labels = parse_csv_list(args.control_labels)
    preferred_join_cols = parse_csv_list(args.join_cols)

    assignments_df = pd.read_csv(assignments_path)
    if args.source_split and "split" in assignments_df.columns:
        assignments_df = assignments_df.loc[assignments_df["split"].astype(str) == str(args.source_split)].copy()

    _, source_df, source_embeddings = load_dataset_rows(source_dataset_path)
    if args.source_split and "split" not in source_df.columns:
        source_df["split"] = args.source_split
    source_meta = merge_metadata(source_df, assignments_df, args.split_manifest_path, preferred_join_cols, args.source_split)
    if args.source_split and "split" in source_meta.columns:
        source_meta = source_meta.loc[source_meta["split"] == args.source_split].copy()
    centroid_meta = source_meta.copy()

    anchor_ids: set[int] | None = None
    if args.anchor_manifest_path:
        anchor_ids = load_anchor_ids(Path(args.anchor_manifest_path).resolve())

    cluster_summary_df = pd.read_csv(cluster_summary_path)
    centroids = build_centroids(source_embeddings, centroid_meta, comparison_cluster_ids)

    row_frames = []
    summary_rows = []
    for dataset_path, label in zip(candidate_dataset_paths, candidate_labels):
        _, candidate_df, candidate_embeddings = load_dataset_rows(dataset_path)
        candidate_meta = merge_metadata(candidate_df, assignments_df, args.split_manifest_path, preferred_join_cols, args.source_split)
        if args.source_split and "split" in candidate_meta.columns:
            candidate_meta = candidate_meta.loc[candidate_meta["split"] == args.source_split].copy()
        if anchor_ids is not None:
            candidate_meta = candidate_meta.loc[pd.to_numeric(candidate_meta["dataset_row_id"], errors="coerce").isin(anchor_ids)].copy()
            candidate_embeddings = candidate_embeddings[candidate_meta.index.to_numpy()]
            candidate_meta = candidate_meta.reset_index(drop=True)
        row_df, summary = evaluate_candidate(
            candidate_label=label,
            candidate_meta=candidate_meta,
            candidate_embeddings=candidate_embeddings,
            source_embeddings=source_embeddings,
            centroids=centroids,
            basin_cluster_ids=basin_cluster_ids,
            comparison_cluster_ids=comparison_cluster_ids,
            low_density_basin_cluster_ids=low_density_basin_cluster_ids,
        )
        row_frames.append(row_df)
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = assign_gate_status(
        summary_df,
        control_labels=control_labels,
        pass_basin_win_delta=float(args.pass_basin_win_delta),
        pass_low_density_win_delta=float(args.pass_low_density_win_delta),
        pass_basin_margin_delta=float(args.pass_basin_margin_delta),
        min_source_cosine=float(args.min_source_cosine),
    )
    row_table = pd.concat(row_frames, ignore_index=True)

    row_path = output_dir / "pooled_basin_gate_row_table.csv"
    summary_path = output_dir / "pooled_basin_gate_summary.csv"
    json_path = output_dir / "pooled_basin_gate_summary.json"
    md_path = output_dir / "pooled_basin_gate_summary.md"

    row_table.to_csv(row_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    payload = {
        "created_at": now_iso(),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "script_path": str(Path(__file__).resolve()),
        "source_dataset_path": str(source_dataset_path),
        "candidate_dataset_paths": [str(p) for p in candidate_dataset_paths],
        "candidate_labels": candidate_labels,
        "cluster_assignments_path": str(assignments_path),
        "cluster_summary_path": str(cluster_summary_path),
        "output_dir": str(output_dir),
        "basin_cluster_ids": basin_cluster_ids,
        "comparison_cluster_ids": comparison_cluster_ids,
        "low_density_basin_cluster_ids": low_density_basin_cluster_ids,
        "control_labels": control_labels,
        "source_split": args.source_split,
        "anchor_manifest_path": args.anchor_manifest_path,
        "summary_rows": summary_df.to_dict(orient="records"),
        "cluster_summary_rows": int(len(cluster_summary_df)),
    }
    save_json(json_path, payload)

    lines = [
        "# Pooled Basin Gate",
        "",
        f"- basin_cluster_ids: `{','.join(str(x) for x in basin_cluster_ids)}`",
        f"- comparison_cluster_ids: `{','.join(str(x) for x in comparison_cluster_ids)}`",
        f"- low_density_basin_cluster_ids: `{','.join(str(x) for x in low_density_basin_cluster_ids)}`",
        f"- source_split: `{args.source_split or '<none>'}`",
        "",
        "## Summary",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['candidate_label']}`: pooled_basin_win_rate={row['pooled_basin_win_rate']:.4f}, "
            f"low_density_basin_win_rate={row['low_density_basin_win_rate']:.4f}, "
            f"mean_basin_margin_vs_external={row['mean_basin_margin_vs_external']:.4f}, "
            f"gate_status={row['gate_status']}"
        )
    lines += [
        "",
        "## Files",
        "",
        "- `pooled_basin_gate_summary.csv`",
        "- `pooled_basin_gate_row_table.csv`",
        "- `pooled_basin_gate_summary.json`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved pooled basin gate summary to: {summary_path}")
    print(f"Saved pooled basin gate row table to: {row_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
