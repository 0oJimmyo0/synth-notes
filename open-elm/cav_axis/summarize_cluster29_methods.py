#!/usr/bin/env python3
"""
Summarize cluster29 steering families under the newer pooled-basin framing.

This script is meant to answer:
- which steering families stayed in the local basin after decode?
- which ones failed mainly on exact cluster29 preference?
- which ones were mostly quality/faithfulness failures?
- which older "failures" look less bad under the pooled-basin metric?
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_pooled_basin import (
    build_centroids,
    leakage_group,
    load_factor_table,
    load_real_embeddings,
    normalize_rows,
)


DEFAULT_INCLUDE_REGEX = (
    r"cluster29|local_cluster29|vanilla_matched_to_local_cluster29|"
    r"random_shift_local_cluster29|axis11_cluster29|projected_cluster29|"
    r"barycentric_cluster29|true_target_upperbound|basin_margin"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize cluster29 steering methods under pooled-basin evaluation.")
    parser.add_argument("--real_dataset_path", required=True)
    parser.add_argument("--factors_path", required=True)
    parser.add_argument("--generation_audit_root", required=True)
    parser.add_argument("--synthetic_notes_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--basin_cluster_ids", default="29,9,17,45")
    parser.add_argument("--comparison_cluster_ids", default="29,9,17,45,7")
    parser.add_argument("--separate_cluster_ids", default="7")
    parser.add_argument("--target_cluster_id", type=int, default=29)
    parser.add_argument("--include_regex", default=DEFAULT_INCLUDE_REGEX)
    parser.add_argument("--exclude_regex", default=r"debug")
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


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def find_manifest(synthetic_dir: Path) -> Path | None:
    candidates = sorted(
        p
        for p in synthetic_dir.glob("*.jsonl")
        if ".validation." not in p.name and "_shard" not in p.name and ".meta." not in p.name
    )
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    manifest_like = [p for p in candidates if "manifest" in p.name]
    return manifest_like[0] if manifest_like else candidates[0]


def method_family(name: str) -> str:
    if "true_target_upperbound" in name:
        return "upper_bound_true_target"
    if "basin_margin" in name:
        return "local_basin_margin"
    if "adaptercal" in name:
        return "adapter_calibrated_barycentric"
    if "decodecorr" in name:
        return "decode_reembed_corrected_barycentric"
    if "barycentric" in name:
        return "barycentric_transport"
    if "projected" in name:
        return "projected_boundary"
    if "random_shift" in name:
        return "random_shift_control"
    if "vanilla_matched" in name:
        return "vanilla_control"
    if "local_cluster29_centroid" in name:
        return "local_centroid"
    if "axis11_cluster29" in name or "cav_axis11_cluster29" in name:
        return "global_axis_bank"
    return "other"


def collect_notes_per_anchor(df: pd.DataFrame) -> tuple[int, float]:
    ids = pd.to_numeric(df["dataset_row_id"], errors="coerce").dropna().astype(int)
    if ids.empty:
        return 0, float("nan")
    vc = ids.value_counts()
    return int(vc.shape[0]), float(vc.mean())


def compute_basin_metrics(
    manifest_df: pd.DataFrame,
    synthetic_embeddings: np.ndarray,
    centroids: dict[int, np.ndarray],
    comparison_cluster_ids: list[int],
    basin_cluster_ids: list[int],
    separate_cluster_ids: set[int],
    target_cluster_id: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    row_df = manifest_df[["generation_id", "dataset_row_id", "note_id", "subject_id", "hadm_id"]].copy()
    if "patient_disjoint_from_train" in manifest_df.columns:
        row_df["patient_disjoint_from_train"] = manifest_df["patient_disjoint_from_train"]
    else:
        row_df["patient_disjoint_from_train"] = pd.NA
    row_df["analysis_group"] = leakage_group(row_df["patient_disjoint_from_train"])

    cosine_cols: list[str] = []
    for cluster_id in comparison_cluster_ids:
        col = f"cos_to_cluster_{cluster_id}"
        row_df[col] = synthetic_embeddings @ centroids[cluster_id]
        cosine_cols.append(col)

    row_df["best_cluster_among_candidates"] = (
        row_df[cosine_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)
    )

    basin_cols = [f"cos_to_cluster_{cluster_id}" for cluster_id in basin_cluster_ids]
    basin_cluster_set = set(basin_cluster_ids)
    external_cluster_ids = [cluster_id for cluster_id in comparison_cluster_ids if cluster_id not in basin_cluster_set]
    external_cols = [f"cos_to_cluster_{cluster_id}" for cluster_id in external_cluster_ids]

    row_df["best_basin_cosine"] = row_df[basin_cols].max(axis=1)
    row_df["best_basin_cluster"] = row_df[basin_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)
    if external_cols:
        row_df["best_external_cosine"] = row_df[external_cols].max(axis=1)
    else:
        row_df["best_external_cosine"] = np.nan

    target_col = f"cos_to_cluster_{target_cluster_id}"
    other_cols = [col for col in cosine_cols if col != target_col]
    row_df["target_cluster_margin_vs_best_other"] = row_df[target_col] - row_df[other_cols].max(axis=1)
    row_df["pooled_basin_margin_vs_best_external"] = row_df["best_basin_cosine"] - row_df["best_external_cosine"]
    row_df["wins_pooled_basin"] = row_df["best_cluster_among_candidates"].isin(basin_cluster_set)

    def _summary(df: pd.DataFrame) -> dict[str, Any]:
        counts = df["best_cluster_among_candidates"].value_counts().sort_index().to_dict()
        out = {
            "n_rows": int(len(df)),
            "pooled_basin_win_rate": float(df["wins_pooled_basin"].mean()) if len(df) else float("nan"),
            "exact_target_cluster_win_rate": float((df["best_cluster_among_candidates"] == target_cluster_id).mean())
            if len(df)
            else float("nan"),
            "mean_basin_margin_vs_best_external": float(df["pooled_basin_margin_vs_best_external"].mean()) if len(df) else float("nan"),
            "mean_target_cluster_margin_vs_best_other": float(df["target_cluster_margin_vs_best_other"].mean()) if len(df) else float("nan"),
            "best_cluster_counts": {str(k): int(v) for k, v in counts.items()},
        }
        for cluster_id in sorted(separate_cluster_ids):
            out[f"win_rate_cluster_{cluster_id}"] = float((df["best_cluster_among_candidates"] == cluster_id).mean()) if len(df) else float("nan")
        return out

    full_summary = _summary(row_df)
    pd_summary = _summary(row_df.loc[row_df["analysis_group"] == "patient_disjoint"].copy())
    return row_df, full_summary, pd_summary


def compute_anchor_metrics(row_df: pd.DataFrame) -> dict[str, Any]:
    anchor = (
        row_df.groupby("dataset_row_id", dropna=True)
        .agg(
            n_notes=("generation_id", "count"),
            mean_basin_margin=("pooled_basin_margin_vs_best_external", "mean"),
            mean_target_margin=("target_cluster_margin_vs_best_other", "mean"),
            any_win_basin=("wins_pooled_basin", "max"),
            frac_win_basin=("wins_pooled_basin", "mean"),
            frac_win_target=("best_cluster_among_candidates", lambda s: (s == 29).mean()),
        )
        .reset_index()
    )
    return {
        "n_unique_anchors": int(anchor["dataset_row_id"].nunique()),
        "mean_notes_per_anchor": float(anchor["n_notes"].mean()) if len(anchor) else float("nan"),
        "anchor_any_basin_win_rate": float(anchor["any_win_basin"].mean()) if len(anchor) else float("nan"),
        "anchor_mean_basin_win_rate": float(anchor["frac_win_basin"].mean()) if len(anchor) else float("nan"),
        "anchor_mean_target_cluster_win_rate": float(anchor["frac_win_target"].mean()) if len(anchor) else float("nan"),
        "anchor_mean_basin_margin": float(anchor["mean_basin_margin"].mean()) if len(anchor) else float("nan"),
        "anchor_mean_target_cluster_margin": float(anchor["mean_target_margin"].mean()) if len(anchor) else float("nan"),
    }


def nested_get(obj: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def classify_failure(row: dict[str, Any]) -> str:
    labels: list[str] = []
    readiness = row.get("readiness_status")
    collapse = row.get("collapse_rate")
    source_cos = row.get("faithfulness_median_cosine")
    basin = row.get("pooled_basin_win_rate")
    target = row.get("exact_target_cluster_win_rate")
    target_margin = row.get("mean_target_cluster_margin_vs_best_other")

    if readiness not in {None, "PASS", "CAUTION"}:
        labels.append("audit_failure")
    if collapse is not None and collapse >= 0.10:
        labels.append("quality_collapse_risk")
    if source_cos is not None and source_cos < 0.78:
        labels.append("faithfulness_drift")
    if basin is not None and basin < 0.80:
        labels.append("off_basin_after_decode")
    elif basin is not None and basin >= 0.80 and target is not None and target < 0.12:
        labels.append("stays_in_basin_but_not_target_cluster")
    if target_margin is not None and target_margin < 0:
        labels.append("target_cluster_loses_local_competition")
    if not labels:
        labels.append("partial_positive_signal")
    return ";".join(labels)


def main() -> None:
    args = build_parser().parse_args()

    real_dataset_path = Path(args.real_dataset_path).resolve()
    factors_path = Path(args.factors_path).resolve()
    generation_audit_root = Path(args.generation_audit_root).resolve()
    synthetic_notes_root = Path(args.synthetic_notes_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_re = re.compile(args.include_regex)
    exclude_re = re.compile(args.exclude_regex) if args.exclude_regex else None

    basin_cluster_ids = parse_int_list(args.basin_cluster_ids)
    comparison_cluster_ids = sorted(parse_int_list(args.comparison_cluster_ids))
    separate_cluster_ids = set(parse_int_list(args.separate_cluster_ids))
    target_cluster_id = int(args.target_cluster_id)

    real_embeddings = load_real_embeddings(real_dataset_path)
    factors_df = load_factor_table(factors_path, len(real_embeddings))
    centroids = build_centroids(real_embeddings, factors_df, comparison_cluster_ids)

    rows: list[dict[str, Any]] = []
    per_method_dir = output_dir / "per_method"
    per_method_dir.mkdir(parents=True, exist_ok=True)

    for audit_dir in sorted(p for p in generation_audit_root.iterdir() if p.is_dir()):
        name = audit_dir.name
        if not include_re.search(name):
            continue
        if exclude_re and exclude_re.search(name):
            continue

        audit_json = audit_dir / "generation_audit_baseline.json"
        embeddings_path = audit_dir / "generated_note_embeddings_bge_large.npy"
        synthetic_dir = synthetic_notes_root / name
        manifest_path = find_manifest(synthetic_dir) if synthetic_dir.exists() else None
        if not (embeddings_path.exists() and manifest_path and manifest_path.exists()):
            continue

        audit_obj = json.loads(audit_json.read_text()) if audit_json.exists() else {}
        manifest_df = pd.read_json(manifest_path, lines=True)
        synthetic_embeddings = normalize_rows(np.load(embeddings_path))
        if len(manifest_df) != len(synthetic_embeddings):
            continue

        row_df, full_summary, pd_summary = compute_basin_metrics(
            manifest_df=manifest_df,
            synthetic_embeddings=synthetic_embeddings,
            centroids=centroids,
            comparison_cluster_ids=comparison_cluster_ids,
            basin_cluster_ids=basin_cluster_ids,
            separate_cluster_ids=separate_cluster_ids,
            target_cluster_id=target_cluster_id,
        )
        anchor_summary = compute_anchor_metrics(row_df)
        n_unique_rows, mean_notes_per_anchor = collect_notes_per_anchor(manifest_df)

        row_record = {
            "label": name,
            "method_family": method_family(name),
            "manifest_path": str(manifest_path),
            "audit_dir": str(audit_dir),
            "n_rows": int(len(manifest_df)),
            "n_unique_anchors": n_unique_rows,
            "mean_notes_per_anchor": mean_notes_per_anchor,
            "readiness_status": audit_obj.get("readiness_status"),
            "collapse_rate": nested_get(audit_obj, ["quality_summary", "repetition_or_collapse_rate"]),
            "empty_output_rate": nested_get(audit_obj, ["quality_summary", "empty_output_rate"]),
            "too_short_rate": nested_get(audit_obj, ["quality_summary", "too_short_rate"]),
            "section_structure_rate": nested_get(audit_obj, ["quality_summary", "minimum_section_structure_rate"]),
            "faithfulness_mean_cosine": nested_get(audit_obj, ["faithfulness_summary", "source_to_generated_cosine", "mean"]),
            "faithfulness_median_cosine": nested_get(audit_obj, ["faithfulness_summary", "source_to_generated_cosine", "median"]),
            "pooled_basin_win_rate": full_summary["pooled_basin_win_rate"],
            "exact_target_cluster_win_rate": full_summary["exact_target_cluster_win_rate"],
            "mean_basin_margin_vs_best_external": full_summary["mean_basin_margin_vs_best_external"],
            "mean_target_cluster_margin_vs_best_other": full_summary["mean_target_cluster_margin_vs_best_other"],
            "patient_disjoint_pooled_basin_win_rate": pd_summary["pooled_basin_win_rate"],
            "patient_disjoint_exact_target_cluster_win_rate": pd_summary["exact_target_cluster_win_rate"],
            "anchor_any_basin_win_rate": anchor_summary["anchor_any_basin_win_rate"],
            "anchor_mean_basin_win_rate": anchor_summary["anchor_mean_basin_win_rate"],
            "anchor_mean_target_cluster_win_rate": anchor_summary["anchor_mean_target_cluster_win_rate"],
            "anchor_mean_basin_margin": anchor_summary["anchor_mean_basin_margin"],
            "anchor_mean_target_cluster_margin": anchor_summary["anchor_mean_target_cluster_margin"],
            "win_rate_cluster_7": full_summary.get("win_rate_cluster_7"),
            "best_cluster_counts_json": json.dumps(full_summary["best_cluster_counts"], sort_keys=True),
        }
        row_record["diagnosis"] = classify_failure(row_record)
        rows.append(row_record)

        method_payload = {
            "label": name,
            "method_family": row_record["method_family"],
            "full_summary": full_summary,
            "patient_disjoint_summary": pd_summary,
            "anchor_summary": anchor_summary,
            "audit_summary": {
                "readiness_status": row_record["readiness_status"],
                "collapse_rate": row_record["collapse_rate"],
                "faithfulness_median_cosine": row_record["faithfulness_median_cosine"],
            },
            "diagnosis": row_record["diagnosis"],
        }
        (per_method_dir / f"{name}.json").write_text(json.dumps(method_payload, indent=2), encoding="utf-8")

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise SystemExit("No qualifying cluster29 methods were found.")

    summary_df = summary_df.sort_values(
        ["method_family", "anchor_mean_basin_win_rate", "exact_target_cluster_win_rate"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    output_csv = output_dir / "cluster29_method_basin_summary.csv"
    output_md = output_dir / "cluster29_method_basin_summary.md"
    output_json = output_dir / "cluster29_method_basin_summary.json"
    summary_df.to_csv(output_csv, index=False)

    family_best = (
        summary_df.sort_values(
            ["method_family", "anchor_mean_basin_win_rate", "exact_target_cluster_win_rate"],
            ascending=[True, False, False],
        )
        .groupby("method_family", as_index=False)
        .first()
    )

    report_lines = [
        "# Cluster29 Steering Method Summary",
        "",
        "This table re-scores prior cluster29 runs under the pooled-basin framing.",
        "",
        "## Main lessons",
        "",
    ]
    for _, row in family_best.iterrows():
        report_lines.append(
            f"- `{row['method_family']}` best run `{row['label']}`: "
            f"anchor_mean_basin_win_rate={row['anchor_mean_basin_win_rate']:.4f}, "
            f"exact_target_cluster_win_rate={row['exact_target_cluster_win_rate']:.4f}, "
            f"diagnosis={row['diagnosis']}"
        )
    report_lines += [
        "",
        "## Notes",
        "",
        "- Older cluster29 failures may look better under pooled-basin coverage than under exact cluster29 occupancy.",
        "- Runs with different `mean_notes_per_anchor` are not perfectly apples-to-apples; use the summary for diagnosis first, not final ranking.",
        "- `diagnosis` focuses on where each family failed: quality, faithfulness, off-basin drift, or weak target-cluster preference inside the basin.",
        "",
        "## Files",
        "",
        "- `cluster29_method_basin_summary.csv`",
        "- `cluster29_method_basin_summary.json`",
        "- `per_method/*.json`",
    ]
    output_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    payload = {
        "created_at": now_iso(),
        "script_path": str(Path(__file__).resolve()),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "real_dataset_path": str(real_dataset_path),
        "factors_path": str(factors_path),
        "generation_audit_root": str(generation_audit_root),
        "synthetic_notes_root": str(synthetic_notes_root),
        "output_dir": str(output_dir),
        "basin_cluster_ids": basin_cluster_ids,
        "comparison_cluster_ids": comparison_cluster_ids,
        "separate_cluster_ids": sorted(separate_cluster_ids),
        "target_cluster_id": target_cluster_id,
        "n_methods": int(len(summary_df)),
        "labels": summary_df["label"].tolist(),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved summary CSV to: {output_csv}")
    print(f"Saved summary MD to: {output_md}")
    print(f"Saved summary JSON to: {output_json}")
    print(summary_df[[
        "label",
        "method_family",
        "anchor_mean_basin_win_rate",
        "exact_target_cluster_win_rate",
        "diagnosis",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
