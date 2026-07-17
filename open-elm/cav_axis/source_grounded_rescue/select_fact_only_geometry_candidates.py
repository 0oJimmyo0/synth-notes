#!/usr/bin/env python3
"""Select one source-grounded candidate per anchor using final BGE target-basin landing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.cluster import MiniBatchKMeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_manifest_path", required=True)
    parser.add_argument("--candidate_embeddings_path", required=True)
    parser.add_argument("--real_dataset_path", required=True)
    parser.add_argument("--real_cluster_assignments_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_cluster_ids", default="9,17,29,45")
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)


def main() -> None:
    args = parse_args()
    target_ids = {int(value) for value in args.target_cluster_ids.split(",") if value.strip()}
    candidate = pd.read_json(args.candidate_manifest_path, lines=True).reset_index(drop=True)
    embeddings = normalize(np.load(args.candidate_embeddings_path))
    if len(candidate) != len(embeddings):
        raise ValueError("Candidate manifest and embedding matrix row counts differ.")
    if candidate["rescue_id"].duplicated().any() or candidate["generated_text"].astype(str).str.strip().eq("").any():
        raise ValueError("Candidate manifest has duplicate IDs or empty notes.")
    real = Dataset.load_from_disk(args.real_dataset_path)
    real_embeddings = normalize(np.vstack([np.asarray(row["domain_embeddings"][0], dtype=np.float32) for row in real]))
    assignments = pd.read_csv(args.real_cluster_assignments_path).sort_values("dataset_row_id")
    kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=args.random_seed, batch_size=2048, n_init="auto")
    labels = kmeans.fit_predict(real_embeddings)
    if not np.array_equal(labels, assignments["cluster_id"].to_numpy()):
        raise ValueError("Refit clusters differ from frozen real cluster assignments.")
    centers = normalize(kmeans.cluster_centers_)
    scores = embeddings @ centers.T
    candidate["output_cluster_id"] = scores.argmax(axis=1).astype(int)
    candidate["output_in_target_basin"] = candidate["output_cluster_id"].isin(target_ids)
    candidate["target_basin_best_cosine"] = scores[:, sorted(target_ids)].max(axis=1)
    candidate["target_basin_margin"] = candidate["target_basin_best_cosine"] - np.delete(scores, sorted(target_ids), axis=1).max(axis=1)
    candidate["selection_rank"] = candidate.sort_values(
        ["anchor_id", "output_in_target_basin", "target_basin_margin", "target_basin_best_cosine", "candidate_index"],
        ascending=[True, False, False, False, True],
    ).groupby("anchor_id").cumcount() + 1
    selected = candidate.loc[candidate["selection_rank"] == 1].copy().sort_values("case_id")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidate.to_json(out / "fact_only_candidate_geometry_manifest.jsonl", orient="records", lines=True)
    selected.to_json(out / "fact_only_geometry_selected_manifest.jsonl", orient="records", lines=True)
    summary = {
        "candidate_rows": int(len(candidate)), "anchors": int(candidate["anchor_id"].nunique()),
        "target_cluster_ids": sorted(target_ids), "candidate_target_basin_rate": float(candidate["output_in_target_basin"].mean()),
        "selected_target_basin_rate": float(selected["output_in_target_basin"].mean()),
        "anchors_with_target_candidate": int(candidate.groupby("anchor_id")["output_in_target_basin"].any().sum()),
    }
    (out / "fact_only_geometry_selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
