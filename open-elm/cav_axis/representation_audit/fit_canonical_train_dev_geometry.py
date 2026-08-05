#!/usr/bin/env python3
"""Fit canonical geometry on real train embeddings and select complexity on dev.

This script never reads synthetic or held-out test embeddings.  Cluster-count
selection is based on agreement between independently seeded train fits when
assigned to the fixed real development split.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score


def parse_csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 1 for item in values):
        raise ValueError("Expected at least one integer greater than one.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_embeddings_path", required=True)
    parser.add_argument("--train_metadata_path", required=True)
    parser.add_argument("--dev_embeddings_path", required=True)
    parser.add_argument("--dev_metadata_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_clusters_grid", default="25,50,75,100")
    parser.add_argument("--seeds", default="20260805,20260806,20260807")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_iter", type=int, default=200)
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_embeddings(path: Path, metadata_path: Path) -> tuple[np.ndarray, list[dict]]:
    embeddings = np.load(path, mmap_mode="r")
    metadata = load_metadata(metadata_path)
    if embeddings.ndim != 2 or len(embeddings) != len(metadata):
        raise ValueError(f"Embedding/metadata mismatch for {path}")
    norms = np.linalg.norm(embeddings[:: max(1, len(embeddings) // 1000)], axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-3, atol=1e-3):
        raise ValueError(f"Embeddings in {path} are not L2-normalized.")
    return embeddings, metadata


def assign_in_batches(model: MiniBatchKMeans, embeddings: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.empty(len(embeddings), dtype=np.int32)
    distances = np.empty(len(embeddings), dtype=np.float32)
    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        block = np.asarray(embeddings[start:end], dtype=np.float32)
        block_labels = model.predict(block)
        labels[start:end] = block_labels
        centers = model.cluster_centers_[block_labels]
        distances[start:end] = np.linalg.norm(block - centers, axis=1)
    return labels, distances


def main() -> None:
    args = parse_args()
    cluster_grid = parse_csv_ints(args.n_clusters_grid)
    seeds = parse_csv_ints(args.seeds)
    if len(seeds) < 2:
        raise ValueError("At least two train-fit seeds are required for dev stability.")
    train, train_metadata = validate_embeddings(Path(args.train_embeddings_path).resolve(), Path(args.train_metadata_path).resolve())
    dev, dev_metadata = validate_embeddings(Path(args.dev_embeddings_path).resolve(), Path(args.dev_metadata_path).resolve())
    if train.shape[1] != dev.shape[1]:
        raise ValueError("Train and dev embedding dimensions differ.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = []
    fitted_by_k: dict[int, list[MiniBatchKMeans]] = {}
    for n_clusters in cluster_grid:
        dev_assignments = []
        fitted = []
        for seed in seeds:
            model = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=seed,
                batch_size=args.batch_size,
                n_init=1,
                max_iter=args.max_iter,
            ).fit(train)
            labels, _ = assign_in_batches(model, dev, args.batch_size)
            fitted.append(model)
            dev_assignments.append(labels)
        pairwise_ari = [
            adjusted_rand_score(left, right)
            for left, right in combinations(dev_assignments, 2)
        ]
        reference_counts = np.bincount(dev_assignments[0], minlength=n_clusters)
        diagnostics.append({
            "n_clusters": n_clusters,
            "n_train": int(len(train)),
            "n_dev": int(len(dev)),
            "n_seeds": len(seeds),
            "mean_dev_pairwise_ari": float(np.mean(pairwise_ari)),
            "min_dev_pairwise_ari": float(np.min(pairwise_ari)),
            "dev_nonempty_clusters": int(np.count_nonzero(reference_counts)),
            "dev_smallest_cluster_count": int(reference_counts[reference_counts > 0].min()),
            "dev_largest_cluster_fraction": float(reference_counts.max() / len(dev)),
        })
        fitted_by_k[n_clusters] = fitted
        print(json.dumps(diagnostics[-1]), flush=True)

    # Prespecified rule: prioritize dev assignment stability; ties prefer a
    # coarser geometry to avoid creating unsupported ultra-small regions.
    selected = sorted(diagnostics, key=lambda row: (-row["mean_dev_pairwise_ari"], row["n_clusters"]))[0]
    selected_k = int(selected["n_clusters"])
    final_model = fitted_by_k[selected_k][0]
    train_labels, train_distances = assign_in_batches(final_model, train, args.batch_size)
    dev_labels, dev_distances = assign_in_batches(final_model, dev, args.batch_size)

    np.save(output_dir / "canonical_train_centroids.npy", final_model.cluster_centers_.astype(np.float32))
    for split, metadata, labels, distances in (
        ("train", train_metadata, train_labels, train_distances),
        ("dev", dev_metadata, dev_labels, dev_distances),
    ):
        with (output_dir / f"canonical_{split}_cluster_assignments.jsonl").open("w") as handle:
            for row, label, distance in zip(metadata, labels, distances):
                handle.write(json.dumps(row | {
                    "cluster_id": int(label),
                    "euclidean_distance_to_centroid": float(distance),
                }) + "\n")
    summary = {
        "representation_id": train_metadata[0].get("representation_id"),
        "representation_spec_sha256": train_metadata[0].get("representation_spec_sha256"),
        "fit_population": "real_canonical_train_only",
        "selection_population": "real_canonical_dev_only",
        "n_train": int(len(train)),
        "n_dev": int(len(dev)),
        "embedding_dimension": int(train.shape[1]),
        "n_clusters_grid": cluster_grid,
        "seeds": seeds,
        "selection_rule": "maximum mean pairwise ARI over dev assignments; tie selects smaller n_clusters",
        "selected_n_clusters": selected_k,
        "selected_fit_seed": seeds[0],
        "selection_diagnostics": diagnostics,
        "security_note": "Outputs contain embeddings, IDs, and derived geometry only; no source-note text.",
    }
    (output_dir / "canonical_train_dev_geometry_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
