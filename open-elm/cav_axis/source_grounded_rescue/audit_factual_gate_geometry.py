#!/usr/bin/env python3
"""Paired geometry audit for factually gated source-grounded synthetic notes.

This is a bounded feasibility audit. It does not estimate whole-cohort coverage.
Real-test cluster centers are refit using the frozen Phase 1 configuration, then
source, raw-ELM, and fact-only BGE vectors are evaluated on the same anchors.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.cluster import MiniBatchKMeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factual_manifest_path", required=True)
    parser.add_argument("--factual_embeddings_path", required=True)
    parser.add_argument("--raw_manifest_path", required=True)
    parser.add_argument("--raw_embeddings_path", required=True)
    parser.add_argument("--real_dataset_path", required=True)
    parser.add_argument("--real_cluster_assignments_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_cluster_ids", default="9,17,29,45")
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)


def load_embeddings(dataset_path: Path) -> np.ndarray:
    dataset = Dataset.load_from_disk(str(dataset_path))
    return normalize(np.vstack([np.asarray(row["domain_embeddings"][0], dtype=np.float32) for row in dataset]))


def get_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def condition_summary(rows: pd.DataFrame, condition: str, target_ids: set[int]) -> dict[str, object]:
    subset = rows.loc[rows["condition"] == condition]
    output_member = subset["output_cluster_id"].isin(target_ids)
    return {
        "condition": condition,
        "n_rows": int(len(subset)),
        "mean_source_output_cosine": float(subset["source_output_cosine"].mean()),
        "median_source_output_cosine": float(subset["source_output_cosine"].median()),
        "target_basin_retention_rate": float((subset["source_in_target_basin"] & output_member).mean()),
        "target_basin_output_rate": float(output_member.mean()),
        "source_target_basin_rate": float(subset["source_in_target_basin"].mean()),
        "mean_target_centroid_distance_delta": float(subset["target_centroid_distance_delta"].mean()),
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small report table without requiring pandas' optional tabulate dependency."""
    columns = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = [len(column) for column in columns]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    header = "| " + " | ".join(column.ljust(width) for column, width in zip(columns, widths)) + " |"
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def main() -> None:
    args = parse_args()
    target_ids = {int(x) for x in args.target_cluster_ids.split(",") if x.strip()}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    factual = pd.read_json(args.factual_manifest_path, lines=True).sort_values("case_id").reset_index(drop=True)
    raw = pd.read_json(args.raw_manifest_path, lines=True).sort_values("case_id").reset_index(drop=True)
    factual_emb = normalize(np.load(args.factual_embeddings_path))
    raw_emb = normalize(np.load(args.raw_embeddings_path))
    for name, df, emb in [("factual", factual, factual_emb), ("raw", raw, raw_emb)]:
        if len(df) != len(emb):
            raise ValueError(f"{name} manifest rows ({len(df)}) != embedding rows ({len(emb)})")
        if df["case_id"].duplicated().any() or df["dataset_row_id"].duplicated().any():
            raise ValueError(f"{name} manifest must be one row per frozen case.")
    if not factual["case_id"].equals(raw["case_id"]) or not factual["dataset_row_id"].equals(raw["dataset_row_id"]):
        raise ValueError("Factual and raw manifests are not aligned on case_id and dataset_row_id.")

    assignments = pd.read_csv(args.real_cluster_assignments_path)
    assignments["dataset_row_id"] = pd.to_numeric(assignments["dataset_row_id"], errors="raise").astype(int)
    if assignments["dataset_row_id"].duplicated().any():
        raise ValueError("Real cluster assignments contain duplicate dataset_row_id values.")
    anchor_meta = factual.merge(
        assignments[["dataset_row_id", "cluster_id", "distance_to_centroid", "patient_disjoint_from_train"]],
        on="dataset_row_id", how="left", validate="one_to_one", suffixes=("", "_real"),
    )
    if anchor_meta["cluster_id"].isna().any():
        raise ValueError("Some frozen anchors are absent from real cluster assignments.")

    real_embeddings = load_embeddings(Path(args.real_dataset_path))
    if len(real_embeddings) != len(assignments):
        raise ValueError("Real dataset row count does not match real cluster assignments.")
    kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=args.random_seed, batch_size=2048, n_init="auto")
    labels = kmeans.fit_predict(real_embeddings)
    if not np.array_equal(labels, assignments.sort_values("dataset_row_id")["cluster_id"].to_numpy()):
        raise ValueError("Refit cluster labels differ from frozen real assignments; refusing incomparable audit.")
    centers = normalize(kmeans.cluster_centers_)
    target_center = normalize(centers[sorted(target_ids)].mean(axis=0))[0]

    source_emb = real_embeddings[factual["dataset_row_id"].astype(int).to_numpy()]
    records: list[pd.DataFrame] = []
    for condition, output_emb in [
        ("fact_only_checkpoint8215", factual_emb),
        ("closed_loop_selected_raw_elm", raw_emb),
    ]:
        similarities = output_emb @ centers.T
        output_cluster = similarities.argmax(axis=1).astype(int)
        result = anchor_meta[["case_id", "anchor_id", "dataset_row_id", "note_id", "review_stratum", "patient_disjoint_from_train", "cluster_id"]].copy()
        result["condition"] = condition
        result["source_cluster_id"] = result.pop("cluster_id").astype(int)
        result["output_cluster_id"] = output_cluster
        result["source_in_target_basin"] = result["source_cluster_id"].isin(target_ids)
        result["output_in_target_basin"] = result["output_cluster_id"].isin(target_ids)
        result["source_output_cosine"] = np.sum(source_emb * output_emb, axis=1)
        result["source_target_centroid_distance"] = 1.0 - (source_emb @ target_center)
        result["output_target_centroid_distance"] = 1.0 - (output_emb @ target_center)
        result["target_centroid_distance_delta"] = result["output_target_centroid_distance"] - result["source_target_centroid_distance"]
        records.append(result)
    rows = pd.concat(records, ignore_index=True)
    rows.to_csv(output_dir / "factual_gate_geometry_row_table.csv", index=False)

    transition = (rows.groupby(["condition", "source_cluster_id", "output_cluster_id"], dropna=False).size()
                  .rename("n_rows").reset_index().sort_values(["condition", "source_cluster_id", "output_cluster_id"]))
    transition.to_csv(output_dir / "factual_gate_cluster_transition.csv", index=False)
    summary = pd.DataFrame([condition_summary(rows, condition, target_ids) for condition in rows["condition"].unique()])
    by_pd_rows: list[dict[str, object]] = []
    grouped = rows.assign(
        patient_group=np.where(rows["patient_disjoint_from_train"].fillna(False), "patient_disjoint", "patient_overlap")
    ).groupby(["condition", "patient_group"], dropna=False)
    for (condition, patient_group), group in grouped:
        result = condition_summary(group, str(condition), target_ids)
        result["patient_group"] = str(patient_group)
        by_pd_rows.append(result)
    by_pd = pd.DataFrame(by_pd_rows)
    by_pd.to_csv(output_dir / "factual_gate_geometry_by_patient_disjoint.csv", index=False)
    summary.to_csv(output_dir / "factual_gate_geometry_summary.csv", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_commit(),
        "purpose": "bounded post-review geometry feasibility audit; not whole-cohort coverage",
        "target_cluster_ids": sorted(target_ids),
        "factual_rows": int(len(factual)),
        "raw_rows": int(len(raw)),
        "patient_disjoint_rows": int(factual["patient_disjoint_from_train"].fillna(False).sum()),
        "conditions": summary.to_dict(orient="records"),
        "input_paths": {k: str(Path(v).resolve()) for k, v in vars(args).items() if k.endswith("_path")},
    }
    (output_dir / "factual_gate_geometry_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Factual-Gate Geometry Pilot", "", "This is a 30-case paired feasibility audit, not a cohort-wide coverage claim.", "", dataframe_to_markdown(summary), "", "## Interpretation boundary", "", "The raw comparator is a frozen closed-loop-selected raw ELM output, not a one-draw vanilla output. A higher output target-basin rate or lower target-centroid distance than this selected comparator supports compatibility of factual fact-only generation with the selected local region. It does not establish enrichment at cohort scale."]
    (output_dir / "factual_gate_geometry_summary.md").write_text("\n".join(lines) + "\n")
    print(summary.to_string(index=False))
    print("Saved geometry audit to:", output_dir)


if __name__ == "__main__":
    main()
