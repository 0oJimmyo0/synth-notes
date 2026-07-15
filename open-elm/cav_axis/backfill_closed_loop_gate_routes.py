#!/usr/bin/env python3
"""Backfill explicit exact-cluster and centroid gate routes for legacy manifests.

This writes corrected copies and never modifies the frozen generation artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill explicit closed-loop gate routes.")
    parser.add_argument("--manifest_paths", required=True, help="Comma-separated JSONL manifests to backfill")
    parser.add_argument("--target_cluster_ids", required=True, help="Comma-separated pooled target cluster ids")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def parse_ids(value: str) -> list[int]:
    return sorted({int(part.strip()) for part in value.split(",") if part.strip()})


def backfill(df: pd.DataFrame, target_ids: list[int]) -> pd.DataFrame:
    if "nearest_cluster_id" not in df.columns:
        raise KeyError("Manifest is missing nearest_cluster_id")
    out = df.copy()
    nearest = pd.to_numeric(out["nearest_cluster_id"], errors="coerce")
    out["exact_pooled_cluster_pass"] = nearest.isin(target_ids)
    if "target_centroid_distance_pass" in out.columns:
        centroid = out["target_centroid_distance_pass"].fillna(False).astype(bool)
    elif {"target_centroid_distance", "centroid_distance_threshold"}.issubset(out.columns):
        centroid = pd.to_numeric(out["target_centroid_distance"], errors="coerce") <= pd.to_numeric(out["centroid_distance_threshold"], errors="coerce")
    else:
        raise KeyError("Manifest needs target_centroid_distance_pass or a per-row centroid threshold")
    out["centroid_proximity_pass"] = centroid
    out["gate_route"] = np.select(
        [out["exact_pooled_cluster_pass"] & centroid, out["exact_pooled_cluster_pass"], centroid],
        ["exact_plus_centroid", "exact_only", "centroid_only"],
        default="neither",
    )
    out["target_gate_pass_corrected"] = out["exact_pooled_cluster_pass"] | out["centroid_proximity_pass"]
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_ids = parse_ids(args.target_cluster_ids)
    summary_rows = []
    for raw_path in [Path(part.strip()).resolve() for part in args.manifest_paths.split(",") if part.strip()]:
        df = pd.read_json(raw_path, lines=True)
        out = backfill(df, target_ids)
        # Candidate manifests from separate runs share the same basename.
        # Preserve both corrected copies by adding their immediate run directory.
        out_path = output_dir / f"{raw_path.parent.name}__{raw_path.stem}_gate_routes_backfilled.jsonl"
        out.to_json(out_path, orient="records", lines=True)
        summary_rows.append({
            "input_manifest": str(raw_path),
            "output_manifest": str(out_path),
            "n_rows": len(out),
            "exact_pooled_cluster_pass_rate": float(out["exact_pooled_cluster_pass"].mean()),
            "centroid_proximity_pass_rate": float(out["centroid_proximity_pass"].mean()),
            **{f"n_{route}": int((out["gate_route"] == route).sum()) for route in ["exact_plus_centroid", "exact_only", "centroid_only", "neither"]},
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "gate_route_backfill_summary.csv", index=False)
    (output_dir / "gate_route_backfill_summary.json").write_text(
        json.dumps({"target_cluster_ids": target_ids, "manifests": summary_rows}, indent=2), encoding="utf-8"
    )
    print("Saved corrected manifests and summary to:", output_dir)


if __name__ == "__main__":
    main()
