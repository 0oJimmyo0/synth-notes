#!/usr/bin/env python3
"""
Generic pre-decode audit for basin-targeted steering.

This script is the generic replacement for cluster-specific steering checks.
It scores one or more candidate embedding datasets against:

- target-vs-best-competitor margin
- target win rate within the local competitor set
- source preservation
- optional basin-vs-external win rate

The intended workflow is:
1. define a clinically coherent sparse basin
2. choose the target cluster and its main local competitors
3. run this pre-decode audit on steering candidates and controls
4. only launch note generation when a candidate clearly beats controls
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
    parser = argparse.ArgumentParser(description="Audit target-vs-competitor margin for generic basin steering.")
    parser.add_argument("--source_dataset_path", required=True, help="Original HF dataset, e.g. encoded_testing_filtered")
    parser.add_argument(
        "--candidate_dataset_paths",
        required=True,
        help="Comma-separated HF dataset paths to compare, e.g. source,candidate,random",
    )
    parser.add_argument(
        "--candidate_labels",
        required=True,
        help="Comma-separated labels aligned to candidate_dataset_paths",
    )
    parser.add_argument("--factors_path", required=True, help="Factor/metadata CSV containing cluster_id and optional target columns")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--target_cluster_id", type=int, required=True, help="Official target cluster_id, e.g. 29")
    parser.add_argument(
        "--competitor_cluster_ids",
        required=True,
        help="Comma-separated official competitor cluster_id values, e.g. 9,17,45",
    )
    parser.add_argument(
        "--basin_cluster_ids",
        default=None,
        help="Optional comma-separated basin cluster IDs. Defaults to target + competitors.",
    )
    parser.add_argument(
        "--external_cluster_ids",
        default="",
        help="Optional comma-separated external cluster IDs to monitor off-basin drift, e.g. 7",
    )
    parser.add_argument("--split_manifest_path", default=None, help="Optional filtered-aligned split manifest")
    parser.add_argument(
        "--join_cols",
        default="source_row_id,embedding_row_id,dataset_row_id,note_id,subject_id,hadm_id",
        help="Preferred join columns for metadata merges",
    )
    parser.add_argument("--source_split", default=None, help="Optional split filter such as test/dev/train")
    parser.add_argument(
        "--target_selection_query",
        default=None,
        help="Optional query defining the real target pool. Defaults to `cluster_id == target_cluster_id`.",
    )
    parser.add_argument(
        "--source_selection_query",
        default=None,
        help="Optional query restricting audited source anchors, e.g. `cluster_target_29 == 0`.",
    )
    parser.add_argument(
        "--anchor_manifest_path",
        default=None,
        help="Optional JSONL/CSV carrying dataset_row_id values to use as the exact shared anchor set across candidates.",
    )
    parser.add_argument(
        "--control_labels",
        default="",
        help="Optional comma-separated candidate labels treated as controls for PASS/CAUTION/FAIL gating.",
    )
    parser.add_argument(
        "--pass_margin_delta",
        type=float,
        default=0.002,
        help="Required improvement in mean target margin versus the best control to PASS.",
    )
    parser.add_argument(
        "--pass_target_win_delta",
        type=float,
        default=0.02,
        help="Required improvement in target win rate versus the best control to PASS.",
    )
    parser.add_argument(
        "--min_source_cosine",
        type=float,
        default=0.97,
        help="Minimum mean source cosine for a clean PASS.",
    )
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


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    def _json_default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


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
        if not isinstance(emb, list) or not emb:
            raise ValueError("Expected each dataset row to carry a non-empty domain_embeddings list.")
        arr = np.asarray(emb[0], dtype=np.float32)
        while arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise ValueError(f"Expected each domain embedding to resolve to 1D, got shape {arr.shape}")
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
    factors_path: Path,
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
        split_df = normalize_join_cols(split_df, join_cols)
        split_df = split_df.drop_duplicates(subset=join_cols)
        merged = merged.merge(split_df, on=join_cols, how="left", validate="one_to_one", suffixes=("", "_split"))

    factors_df = pd.read_csv(factors_path)
    join_cols = choose_join_keys([merged, factors_df], preferred_join_cols)
    merged = normalize_join_cols(merged, join_cols)
    factors_df = normalize_join_cols(factors_df, join_cols)
    factors_df = factors_df.drop_duplicates(subset=join_cols)
    merged = merged.merge(factors_df, on=join_cols, how="left", validate="one_to_one", suffixes=("", "_factor"))
    return merged


def compute_cluster_centroids(real_embeddings: np.ndarray, meta_df: pd.DataFrame, cluster_ids: list[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for cid in cluster_ids:
        ids = meta_df.loc[pd.to_numeric(meta_df["cluster_id"], errors="coerce") == cid, "dataset_row_id"].astype(int).to_numpy()
        if len(ids) == 0:
            raise ValueError(f"No rows found for cluster_id={cid}")
        out[cid] = normalize_rows(real_embeddings[ids].mean(axis=0, keepdims=True))[0]
    return out


def load_anchor_ids(path: Path) -> set[int]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_json(path, lines=True)
    if "dataset_row_id" not in df.columns:
        raise KeyError(f"Anchor file must contain dataset_row_id: {path}")
    return set(pd.to_numeric(df["dataset_row_id"], errors="coerce").dropna().astype(int).unique().tolist())


def resolve_source_row_ids(candidate_df: pd.DataFrame, n_source_rows: int) -> np.ndarray:
    candidates = []
    for col in ["source_row_id", "dataset_row_id", "dataset_local_row_id", "embedding_row_id"]:
        if col in candidate_df.columns:
            numeric = pd.to_numeric(candidate_df[col], errors="coerce")
            if numeric.notna().all():
                vals = numeric.astype(int).to_numpy()
                if ((vals >= 0) & (vals < n_source_rows)).all():
                    candidates.append((col, vals))
    if not candidates:
        raise ValueError("Could not resolve source row ids from candidate dataset columns.")
    return candidates[0][1]


def leakage_group(values: pd.Series) -> pd.Series:
    normed = values.astype(str).str.lower()
    out = pd.Series(index=values.index, dtype=object)
    out.loc[normed == "true"] = "patient_disjoint"
    out.loc[normed == "false"] = "patient_overlap"
    out = out.fillna("unknown")
    return out


def evaluate_candidate(
    label: str,
    candidate_df: pd.DataFrame,
    candidate_embeddings: np.ndarray,
    source_embeddings: np.ndarray,
    source_meta: pd.DataFrame,
    target_cluster_id: int,
    competitor_cluster_ids: list[int],
    basin_cluster_ids: list[int],
    external_cluster_ids: list[int],
    centroids: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_row_ids = resolve_source_row_ids(candidate_df, len(source_embeddings))
    row_df = candidate_df.copy()
    row_df["candidate_label"] = label
    row_df["source_row_id_resolved"] = source_row_ids
    row_df["source_cosine"] = np.sum(candidate_embeddings * source_embeddings[source_row_ids], axis=1)

    target_col = f"cos_to_cluster_{target_cluster_id}"
    row_df[target_col] = candidate_embeddings @ centroids[target_cluster_id]

    competitor_cols = []
    competitor_col_to_id: dict[str, int] = {}
    for cid in competitor_cluster_ids:
        col = f"cos_to_cluster_{cid}"
        row_df[col] = candidate_embeddings @ centroids[cid]
        competitor_cols.append(col)
        competitor_col_to_id[col] = cid

    row_df["best_competitor_cosine"] = row_df[competitor_cols].max(axis=1)
    best_competitor_col = row_df[competitor_cols].idxmax(axis=1)
    row_df["best_competitor_cluster"] = best_competitor_col.map(competitor_col_to_id).astype(int)
    row_df["target_cluster_margin_vs_best_competitor"] = row_df[target_col] - row_df["best_competitor_cosine"]
    row_df["target_cluster_win"] = (row_df["target_cluster_margin_vs_best_competitor"] > 0).astype(int)

    basin_cols = [f"cos_to_cluster_{cid}" for cid in basin_cluster_ids]
    basin_col_to_id: dict[str, int] = {}
    for cid in basin_cluster_ids:
        if f"cos_to_cluster_{cid}" not in row_df.columns:
            row_df[f"cos_to_cluster_{cid}"] = candidate_embeddings @ centroids[cid]
        basin_col_to_id[f"cos_to_cluster_{cid}"] = cid
    row_df["best_basin_cosine"] = row_df[basin_cols].max(axis=1)
    best_basin_col = row_df[basin_cols].idxmax(axis=1)
    row_df["best_basin_cluster"] = best_basin_col.map(basin_col_to_id).astype(int)

    if external_cluster_ids:
        external_cols = []
        for cid in external_cluster_ids:
            col = f"cos_to_cluster_{cid}"
            row_df[col] = candidate_embeddings @ centroids[cid]
            external_cols.append(col)
        row_df["best_external_cosine"] = row_df[external_cols].max(axis=1)
        row_df["basin_margin_vs_external"] = row_df["best_basin_cosine"] - row_df["best_external_cosine"]
        row_df["wins_basin_vs_external"] = (row_df["basin_margin_vs_external"] > 0).astype(int)
    else:
        row_df["best_external_cosine"] = np.nan
        row_df["basin_margin_vs_external"] = np.nan
        row_df["wins_basin_vs_external"] = pd.NA

    source_join = source_meta.copy()
    source_join["source_row_id_resolved"] = pd.to_numeric(source_join["dataset_row_id"], errors="coerce").astype(int)
    source_join = source_join[
        ["source_row_id_resolved", "dataset_row_id"] + [c for c in ["patient_disjoint_from_train", "split"] if c in source_join.columns]
    ].drop_duplicates("source_row_id_resolved")
    row_df["dataset_row_id"] = pd.to_numeric(row_df["dataset_row_id"], errors="coerce").astype(int)
    row_df = row_df.merge(source_join, on="source_row_id_resolved", how="left", suffixes=("", "_source"))
    if "dataset_row_id_source" in row_df.columns:
        row_df["source_dataset_row_id"] = row_df["source_row_id_resolved"]
        row_df["dataset_row_id_source"] = pd.to_numeric(row_df["dataset_row_id_source"], errors="coerce")
    row_df["analysis_group"] = leakage_group(row_df["patient_disjoint_from_train"]) if "patient_disjoint_from_train" in row_df.columns else "unknown"

    summary = {
        "candidate_label": label,
        "n_rows": int(len(row_df)),
        "n_unique_anchors": int(row_df["dataset_row_id"].nunique()),
        "mean_notes_per_anchor": float(len(row_df) / max(row_df["dataset_row_id"].nunique(), 1)),
        "mean_source_cosine": float(row_df["source_cosine"].mean()),
        "mean_target_cluster_margin": float(row_df["target_cluster_margin_vs_best_competitor"].mean()),
        "median_target_cluster_margin": float(row_df["target_cluster_margin_vs_best_competitor"].median()),
        "target_cluster_win_rate": float(row_df["target_cluster_win"].mean()),
        "mean_target_cluster_cosine": float(row_df[target_col].mean()),
        "mean_best_competitor_cosine": float(row_df["best_competitor_cosine"].mean()),
        "best_competitor_counts_json": json.dumps(
            {str(k): int(v) for k, v in row_df["best_competitor_cluster"].value_counts().sort_index().to_dict().items()},
            sort_keys=True,
        ),
        "patient_disjoint_mean_target_margin": float(row_df.loc[row_df["analysis_group"] == "patient_disjoint", "target_cluster_margin_vs_best_competitor"].mean())
        if (row_df["analysis_group"] == "patient_disjoint").any()
        else float("nan"),
        "patient_disjoint_target_win_rate": float(row_df.loc[row_df["analysis_group"] == "patient_disjoint", "target_cluster_win"].mean())
        if (row_df["analysis_group"] == "patient_disjoint").any()
        else float("nan"),
    }
    if external_cluster_ids:
        summary["mean_basin_margin_vs_external"] = float(row_df["basin_margin_vs_external"].mean())
        summary["basin_win_rate_vs_external"] = float(pd.to_numeric(row_df["wins_basin_vs_external"], errors="coerce").mean())
    else:
        summary["mean_basin_margin_vs_external"] = float("nan")
        summary["basin_win_rate_vs_external"] = float("nan")
    return row_df, summary


def decide_gate(
    summary_df: pd.DataFrame,
    control_labels: list[str],
    pass_margin_delta: float,
    pass_target_win_delta: float,
    min_source_cosine: float,
) -> pd.DataFrame:
    out = summary_df.copy()
    out["gate_status"] = "UNGATED"
    out["gate_reason"] = ""
    if not control_labels:
        return out

    controls = out.loc[out["candidate_label"].isin(control_labels)].copy()
    if controls.empty:
        return out

    best_control_margin = float(controls["mean_target_cluster_margin"].max())
    best_control_target_win = float(controls["target_cluster_win_rate"].max())

    for idx, row in out.iterrows():
        if row["candidate_label"] in control_labels:
            out.at[idx, "gate_status"] = "CONTROL"
            out.at[idx, "gate_reason"] = "control baseline"
            continue
        margin_gain = float(row["mean_target_cluster_margin"] - best_control_margin)
        win_gain = float(row["target_cluster_win_rate"] - best_control_target_win)
        source_ok = float(row["mean_source_cosine"]) >= min_source_cosine
        if margin_gain >= pass_margin_delta and win_gain >= pass_target_win_delta and source_ok:
            status = "PASS"
        elif (margin_gain > 0 or win_gain > 0) and source_ok:
            status = "CAUTION"
        else:
            status = "FAIL"
        out.at[idx, "gate_status"] = status
        out.at[idx, "gate_reason"] = (
            f"margin_gain_vs_best_control={margin_gain:.6f}; "
            f"target_win_gain_vs_best_control={win_gain:.6f}; "
            f"mean_source_cosine={float(row['mean_source_cosine']):.6f}"
        )
    return out


def main() -> None:
    args = build_parser().parse_args()

    source_dataset_path = Path(args.source_dataset_path).resolve()
    factors_path = Path(args.factors_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_paths = [Path(p).resolve() for p in parse_csv_list(args.candidate_dataset_paths)]
    candidate_labels = parse_csv_list(args.candidate_labels)
    if len(candidate_paths) != len(candidate_labels):
        raise ValueError("candidate_dataset_paths and candidate_labels must have the same length")

    competitor_cluster_ids = parse_int_list(args.competitor_cluster_ids)
    if not competitor_cluster_ids:
        raise ValueError("--competitor_cluster_ids must not be empty")
    basin_cluster_ids = parse_int_list(args.basin_cluster_ids) if args.basin_cluster_ids else [args.target_cluster_id] + competitor_cluster_ids
    external_cluster_ids = parse_int_list(args.external_cluster_ids)
    control_labels = parse_csv_list(args.control_labels)
    preferred_join_cols = parse_csv_list(args.join_cols)

    _, source_df, source_embeddings = load_dataset_rows(source_dataset_path)
    source_meta = merge_metadata(source_df, factors_path, args.split_manifest_path, preferred_join_cols, args.source_split)
    if args.source_split and "split" in source_meta.columns:
        source_meta = source_meta.loc[source_meta["split"].astype(str) == str(args.source_split)].copy()

    if "cluster_id" not in source_meta.columns:
        raise KeyError("Merged factor table must contain cluster_id")

    target_query = args.target_selection_query or f"cluster_id == {int(args.target_cluster_id)}"
    target_meta = source_meta.query(target_query, engine="python").copy()
    if target_meta.empty:
        raise ValueError("No rows found in the real target pool.")

    if args.source_selection_query:
        source_subset_meta = source_meta.query(args.source_selection_query, engine="python").copy()
    else:
        source_subset_meta = source_meta.copy()
    source_subset_ids = set(source_subset_meta["dataset_row_id"].astype(int).tolist())
    if args.anchor_manifest_path:
        source_subset_ids &= load_anchor_ids(Path(args.anchor_manifest_path).resolve())

    all_cluster_ids = sorted(set([args.target_cluster_id] + competitor_cluster_ids + basin_cluster_ids + external_cluster_ids))
    centroids = compute_cluster_centroids(source_embeddings, source_meta, all_cluster_ids)

    row_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for label, dataset_path in zip(candidate_labels, candidate_paths):
        _, candidate_df, candidate_embeddings = load_dataset_rows(dataset_path)
        if "dataset_row_id" in candidate_df.columns:
            keep_mask = pd.to_numeric(candidate_df["dataset_row_id"], errors="coerce").isin(source_subset_ids).to_numpy()
            candidate_df = candidate_df.loc[keep_mask].reset_index(drop=True)
            candidate_embeddings = candidate_embeddings[keep_mask]
        row_df, summary = evaluate_candidate(
            label=label,
            candidate_df=candidate_df,
            candidate_embeddings=candidate_embeddings,
            source_embeddings=source_embeddings,
            source_meta=source_meta,
            target_cluster_id=args.target_cluster_id,
            competitor_cluster_ids=competitor_cluster_ids,
            basin_cluster_ids=basin_cluster_ids,
            external_cluster_ids=external_cluster_ids,
            centroids=centroids,
        )
        row_frames.append(row_df)
        summary_rows.append(summary)

    all_rows_df = pd.concat(row_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    gated_df = decide_gate(
        summary_df=summary_df,
        control_labels=control_labels,
        pass_margin_delta=float(args.pass_margin_delta),
        pass_target_win_delta=float(args.pass_target_win_delta),
        min_source_cosine=float(args.min_source_cosine),
    )

    row_path = output_dir / "target_competitor_margin_row_table.csv"
    summary_path = output_dir / "target_competitor_margin_summary.csv"
    json_path = output_dir / "target_competitor_margin_summary.json"
    md_path = output_dir / "target_competitor_margin_summary.md"

    all_rows_df.to_csv(row_path, index=False)
    gated_df.to_csv(summary_path, index=False)

    payload = {
        "created_at": now_iso(),
        "script_path": str(Path(__file__).resolve()),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "source_dataset_path": str(source_dataset_path),
        "candidate_dataset_paths": [str(p) for p in candidate_paths],
        "candidate_labels": candidate_labels,
        "factors_path": str(factors_path),
        "output_dir": str(output_dir),
        "target_cluster_id": int(args.target_cluster_id),
        "competitor_cluster_ids": competitor_cluster_ids,
        "basin_cluster_ids": basin_cluster_ids,
        "external_cluster_ids": external_cluster_ids,
        "source_selection_query": args.source_selection_query,
        "target_selection_query": target_query,
        "source_split": args.source_split,
        "control_labels": control_labels,
        "pass_margin_delta": float(args.pass_margin_delta),
        "pass_target_win_delta": float(args.pass_target_win_delta),
        "min_source_cosine": float(args.min_source_cosine),
    }
    save_json(json_path, payload)

    lines = [
        "# Target-Competitor Margin Audit",
        "",
        f"- target_cluster_id: `{args.target_cluster_id}`",
        f"- competitor_cluster_ids: `{','.join(str(x) for x in competitor_cluster_ids)}`",
        f"- basin_cluster_ids: `{','.join(str(x) for x in basin_cluster_ids)}`",
        f"- external_cluster_ids: `{','.join(str(x) for x in external_cluster_ids) if external_cluster_ids else '<none>'}`",
        f"- source_selection_query: `{args.source_selection_query or '<none>'}`",
        f"- target_selection_query: `{target_query}`",
        "",
        "## Candidate Summary",
        "",
    ]
    for _, row in gated_df.sort_values(["gate_status", "mean_target_cluster_margin"], ascending=[True, False]).iterrows():
        lines.append(
            f"- `{row['candidate_label']}`: "
            f"gate={row['gate_status']}, "
            f"mean_target_margin={row['mean_target_cluster_margin']:.6f}, "
            f"target_win_rate={row['target_cluster_win_rate']:.4f}, "
            f"mean_source_cosine={row['mean_source_cosine']:.6f}, "
            f"reason={row['gate_reason']}"
        )
    lines += [
        "",
        "## Files",
        "",
        "- `target_competitor_margin_summary.csv`",
        "- `target_competitor_margin_row_table.csv`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved summary CSV to: {summary_path}")
    print(f"Saved row table to: {row_path}")
    print(gated_df[[
        "candidate_label",
        "mean_target_cluster_margin",
        "target_cluster_win_rate",
        "mean_source_cosine",
        "gate_status",
        "gate_reason",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
