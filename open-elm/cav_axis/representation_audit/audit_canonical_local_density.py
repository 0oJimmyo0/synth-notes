#!/usr/bin/env python3
"""Audit canonical local-density stability using disjoint real-train references.

No synthetic or test embeddings are read.  Two deterministic halves of train
estimate local k-neighbor density for each real dev note; agreement determines
whether continuous neighborhoods are stable enough to replace global clusters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise ValueError("k values must be positive integers.")
    return sorted(set(result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_embeddings_path", required=True)
    parser.add_argument("--dev_embeddings_path", required=True)
    parser.add_argument("--dev_metadata_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k_grid", default="10,25,50")
    parser.add_argument("--query_batch_size", type=int, default=128)
    parser.add_argument("--reference_batch_size", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--merge_only", action="store_true")
    return parser.parse_args()


def topk_similarity(query: torch.Tensor, reference: np.ndarray, k_max: int, ref_batch: int, device: str) -> torch.Tensor:
    current = torch.full((len(query), k_max), -torch.inf, device=device)
    for start in range(0, len(reference), ref_batch):
        block = torch.as_tensor(np.array(reference[start : start + ref_batch], copy=True), device=device)
        scores = query @ block.T
        current = torch.topk(torch.cat((current, scores), dim=1), k=k_max, dim=1).values
    return current


def merge(args: argparse.Namespace, ks: list[int]) -> None:
    output_dir = Path(args.output_dir).resolve()
    train = np.load(Path(args.train_embeddings_path).resolve(), mmap_mode="r")
    dev = np.load(Path(args.dev_embeddings_path).resolve(), mmap_mode="r")
    metadata = [json.loads(line) for line in Path(args.dev_metadata_path).read_text().splitlines() if line]
    if len(dev) != len(metadata):
        raise ValueError("Dev embedding/metadata lengths differ.")
    parts = [np.load(output_dir / "shards" / f"shard_{index:02d}.npz") for index in range(args.shard_count)]
    indices = np.concatenate([part["query_indices"] for part in parts])
    if sorted(indices.tolist()) != list(range(len(dev))):
        raise ValueError("Local-density shards do not cover each dev row exactly once.")
    mean_a = np.empty((len(dev), len(ks)), dtype=np.float32)
    mean_b = np.empty_like(mean_a)
    for part in parts:
        mean_a[part["query_indices"]] = part["mean_similarity_a"]
        mean_b[part["query_indices"]] = part["mean_similarity_b"]
    diagnostics = []
    for column, k in enumerate(ks):
        ranks_a = pd.Series(mean_a[:, column]).rank(method="average").to_numpy()
        ranks_b = pd.Series(mean_b[:, column]).rank(method="average").to_numpy()
        correlation = float(np.corrcoef(ranks_a, ranks_b)[0, 1])
        n_sparse = max(1, int(np.ceil(len(dev) * 0.10)))
        sparse_a = set(np.argsort(mean_a[:, column])[:n_sparse])
        sparse_b = set(np.argsort(mean_b[:, column])[:n_sparse])
        diagnostics.append({
            "k": k,
            "dev_density_rank_spearman": correlation,
            "sparse_decile_jaccard": float(len(sparse_a & sparse_b) / len(sparse_a | sparse_b)),
            "mean_similarity_reference_a": float(mean_a[:, column].mean()),
            "mean_similarity_reference_b": float(mean_b[:, column].mean()),
        })
    with (output_dir / "canonical_dev_local_density.jsonl").open("w") as handle:
        for index, row in enumerate(metadata):
            handle.write(json.dumps({
                key: row[key] for key in ("dataset_row_id", "note_id", "case_id", "source_split") if key in row
            } | {
                f"mean_top_{k}_similarity_train_half_a": float(mean_a[index, column])
                for column, k in enumerate(ks)
            } | {
                f"mean_top_{k}_similarity_train_half_b": float(mean_b[index, column])
                for column, k in enumerate(ks)
            }) + "\n")
    summary = {
        "fit_population": "real_canonical_train_only_split_into_deterministic_halves",
        "evaluation_population": "real_canonical_dev_only",
        "n_train": int(len(train)),
        "n_dev": int(len(dev)),
        "k_grid": ks,
        "diagnostics": diagnostics,
        "security_note": "Outputs contain IDs and derived local-density values only; no source-note text.",
    }
    (output_dir / "canonical_local_density_stability_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    ks = int_list(args.k_grid)
    if args.merge_only:
        merge(args, ks)
        return
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard configuration.")
    train = np.load(Path(args.train_embeddings_path).resolve(), mmap_mode="r")
    dev = np.load(Path(args.dev_embeddings_path).resolve(), mmap_mode="r")
    if train.shape[1] != dev.shape[1]:
        raise ValueError("Train and dev embedding dimensions differ.")
    reference_a, reference_b = train[::2], train[1::2]
    k_max = max(ks)
    if min(len(reference_a), len(reference_b)) < k_max:
        raise ValueError("Reference half is smaller than requested k.")
    output_dir = Path(args.output_dir).resolve()
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    query_indices = np.arange(args.shard_index, len(dev), args.shard_count, dtype=np.int64)
    values_a, values_b = [], []
    with torch.inference_mode():
        for start in range(0, len(query_indices), args.query_batch_size):
            indices = query_indices[start : start + args.query_batch_size]
            query = torch.as_tensor(np.asarray(dev[indices]), device=args.device)
            top_a = topk_similarity(query, reference_a, k_max, args.reference_batch_size, args.device)
            top_b = topk_similarity(query, reference_b, k_max, args.reference_batch_size, args.device)
            values_a.append(torch.stack([top_a[:, :k].mean(dim=1) for k in ks], dim=1).cpu().numpy())
            values_b.append(torch.stack([top_b[:, :k].mean(dim=1) for k in ks], dim=1).cpu().numpy())
            print(f"shard {args.shard_index}: processed {min(start + len(indices), len(query_indices))}/{len(query_indices)}", flush=True)
    np.savez_compressed(
        shard_dir / f"shard_{args.shard_index:02d}.npz",
        query_indices=query_indices,
        mean_similarity_a=np.concatenate(values_a, axis=0),
        mean_similarity_b=np.concatenate(values_b, axis=0),
    )


if __name__ == "__main__":
    main()
