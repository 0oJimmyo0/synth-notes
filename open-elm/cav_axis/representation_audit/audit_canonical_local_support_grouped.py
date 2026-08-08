#!/usr/bin/env python3
"""Repeated subject-grouped local-support audit for canonical embeddings.

This script evaluates real canonical query notes only. Each frozen train
reference split is subject-disjoint between its A/B halves and contains one
representative for each exact embedding-vector class.  For every dev query,
train neighbors from the same subject are excluded before top-k support is
calculated.  No test or synthetic data are read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def int_list(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or values[0] <= 0:
        raise ValueError("k_grid must contain positive integers.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_embeddings_path", required=True)
    parser.add_argument("--train_subject_metadata_path", required=True)
    parser.add_argument("--dev_embeddings_path", required=True)
    parser.add_argument("--dev_subject_metadata_path", required=True)
    parser.add_argument("--reference_split_path", required=True)
    parser.add_argument("--query_split", choices=("dev", "test"), default="dev")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_seed", required=True, type=int)
    parser.add_argument("--k_grid", default="10,25,50,100")
    parser.add_argument("--query_batch_size", type=int, default=128)
    parser.add_argument("--reference_batch_size", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--merge_only", action="store_true")
    return parser.parse_args()


def load_metadata(path: str, expected_rows: int, label: str) -> pd.DataFrame:
    frame = pd.read_json(Path(path).resolve(), lines=True).sort_values("source_index")
    if frame.source_index.tolist() != list(range(expected_rows)):
        raise ValueError(f"{label} metadata must cover ordered source indices exactly once.")
    if "subject_id" not in frame:
        raise KeyError(f"{label} metadata is missing subject_id.")
    return frame.reset_index(drop=True)


def references(split_path: str, train_metadata: pd.DataFrame, k_max: int) -> tuple[np.ndarray, np.ndarray]:
    split = pd.read_csv(Path(split_path).resolve())
    required = {"source_index", "subject_id", "reference_half", "reference_keep"}
    missing = required - set(split)
    if missing:
        raise KeyError(f"Reference split is missing columns: {sorted(missing)}")
    split = split.sort_values("source_index").reset_index(drop=True)
    if split.source_index.tolist() != list(range(len(train_metadata))):
        raise ValueError("Reference split and train metadata do not share ordered source indices.")
    if split.subject_id.astype(str).tolist() != train_metadata.subject_id.astype(str).tolist():
        raise ValueError("Reference split subject IDs do not match train metadata.")
    kept = split.loc[split.reference_keep.astype(bool)].copy()
    if set(kept.reference_half) != {"a", "b"}:
        raise ValueError("Both reference halves must be nonempty.")
    if kept.groupby("subject_id").reference_half.nunique().max() != 1:
        raise ValueError("A retained reference subject appears in both halves.")
    index_a = kept.loc[kept.reference_half.eq("a"), "source_index"].to_numpy(dtype=np.int64)
    index_b = kept.loc[kept.reference_half.eq("b"), "source_index"].to_numpy(dtype=np.int64)
    if min(len(index_a), len(index_b)) < k_max:
        raise ValueError("A reference half is smaller than requested k.")
    return index_a, index_b


def support_for_half(
    query: torch.Tensor,
    query_subject_codes: torch.Tensor,
    reference: np.ndarray,
    reference_subject_codes: np.ndarray,
    k_max: int,
    reference_batch_size: int,
    device: str,
) -> torch.Tensor:
    best = torch.full((len(query), k_max), -torch.inf, device=device)
    query_subject_codes = query_subject_codes.to(device)
    for start in range(0, len(reference), reference_batch_size):
        end = min(start + reference_batch_size, len(reference))
        ref = torch.as_tensor(np.array(reference[start:end], copy=True), device=device)
        scores = query @ ref.T
        ref_codes = torch.as_tensor(reference_subject_codes[start:end], device=device)
        # Same-subject train examples are never eligible neighbors for a dev query.
        scores.masked_fill_(query_subject_codes[:, None].eq(ref_codes[None, :]), -torch.inf)
        best = torch.topk(torch.cat((best, scores), dim=1), k=k_max, dim=1).values
    if torch.isinf(best[:, -1]).any():
        raise ValueError("Same-subject exclusion left fewer than k eligible references.")
    return best


def rank_spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(pd.Series(left).rank().to_numpy(), pd.Series(right).rank().to_numpy())[0, 1])


def diagnostics(a: np.ndarray, b: np.ndarray, patient_disjoint: np.ndarray, ks: list[int], query_split: str) -> list[dict]:
    rows = []
    for subset_name, subset in ((f"all_{query_split}", np.ones(len(a), dtype=bool)), (f"patient_disjoint_{query_split}", patient_disjoint)):
        if int(subset.sum()) < 10:
            continue
        for column, k in enumerate(ks):
            left, right = a[subset, column], b[subset, column]
            sparse_n = max(1, int(np.ceil(len(left) * 0.10)))
            sparse_a = set(np.argsort(left)[:sparse_n])
            sparse_b = set(np.argsort(right)[:sparse_n])
            rows.append({
                "population": subset_name,
                "k": k,
                "n_queries": int(len(left)),
                "rank_spearman_a_vs_b": rank_spearman(left, right),
                "sparse_decile_jaccard_a_vs_b": float(len(sparse_a & sparse_b) / len(sparse_a | sparse_b)),
                "mean_support_a": float(left.mean()),
                "mean_support_b": float(right.mean()),
            })
    return rows


def merge(args: argparse.Namespace, ks: list[int]) -> None:
    output_dir = Path(args.output_dir).resolve()
    dev = np.load(Path(args.dev_embeddings_path).resolve(), mmap_mode="r")
    dev_metadata = load_metadata(args.dev_subject_metadata_path, len(dev), "Dev")
    parts = [np.load(output_dir / "shards" / f"shard_{index:02d}.npz") for index in range(args.shard_count)]
    indices = np.concatenate([part["query_indices"] for part in parts])
    if sorted(indices.tolist()) != list(range(len(dev))):
        raise ValueError("Shards do not cover each dev row exactly once.")
    support_a = np.empty((len(dev), len(ks)), dtype=np.float32)
    support_b = np.empty_like(support_a)
    excluded_a = np.empty(len(dev), dtype=np.int32)
    excluded_b = np.empty(len(dev), dtype=np.int32)
    for part in parts:
        query_indices = part["query_indices"]
        support_a[query_indices] = part["support_a"]
        support_b[query_indices] = part["support_b"]
        excluded_a[query_indices] = part["same_subject_reference_count_a"]
        excluded_b[query_indices] = part["same_subject_reference_count_b"]
    patient_disjoint = dev_metadata.get("patient_disjoint_from_train", pd.Series(False, index=dev_metadata.index))
    patient_disjoint = patient_disjoint.astype(bool).to_numpy()
    record_path = output_dir / f"canonical_{args.query_split}_local_support.jsonl"
    with record_path.open("w") as handle:
        for index, row in dev_metadata.iterrows():
            record = {key: row[key] for key in ("source_index", "dataset_row_id", "note_id", "case_id", "subject_id", "patient_disjoint_from_train") if key in row}
            record.update({f"mean_top_{k}_support_a": float(support_a[index, column]) for column, k in enumerate(ks)})
            record.update({f"mean_top_{k}_support_b": float(support_b[index, column]) for column, k in enumerate(ks)})
            record.update({f"mean_top_{k}_support": float((support_a[index, column] + support_b[index, column]) / 2) for column, k in enumerate(ks)})
            record["same_subject_reference_count_a"] = int(excluded_a[index])
            record["same_subject_reference_count_b"] = int(excluded_b[index])
            handle.write(json.dumps(record) + "\n")
    summary = {
        "split_seed": args.split_seed,
        "fit_population": "real_canonical_train_only_subject_grouped_exact_vector_deduplicated_references",
        "evaluation_population": f"real_canonical_{args.query_split}_only_with_same_subject_neighbors_excluded",
        f"n_{args.query_split}": int(len(dev)),
        f"n_patient_disjoint_{args.query_split}": int(patient_disjoint.sum()),
        "k_grid": ks,
        "diagnostics": diagnostics(support_a, support_b, patient_disjoint, ks, args.query_split),
        "same_subject_reference_exclusions": {
            "reference_a_total": int(excluded_a.sum()),
            "reference_b_total": int(excluded_b.sum()),
        },
        "security_note": "Outputs contain provenance IDs and derived support values only; no source-note text.",
    }
    (output_dir / "canonical_local_support_stability_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
    if train.ndim != 2 or dev.ndim != 2 or train.shape[1] != dev.shape[1]:
        raise ValueError("Train and dev embeddings must be two-dimensional with matching dimensions.")
    train_metadata = load_metadata(args.train_subject_metadata_path, len(train), "Train")
    dev_metadata = load_metadata(args.dev_subject_metadata_path, len(dev), "Dev")
    index_a, index_b = references(args.reference_split_path, train_metadata, max(ks))
    subject_values = pd.concat([train_metadata.subject_id, dev_metadata.subject_id]).astype(str)
    subject_codes, _ = pd.factorize(subject_values, sort=True)
    train_codes = subject_codes[:len(train)]
    dev_codes = subject_codes[len(train):]
    reference_a, reference_b = train[index_a], train[index_b]
    reference_codes_a, reference_codes_b = train_codes[index_a], train_codes[index_b]
    query_indices = np.arange(args.shard_index, len(dev), args.shard_count, dtype=np.int64)
    values_a, values_b, excluded_a, excluded_b = [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(query_indices), args.query_batch_size):
            current = query_indices[start:start + args.query_batch_size]
            query = torch.as_tensor(np.asarray(dev[current]), device=args.device)
            query_codes = torch.as_tensor(dev_codes[current], dtype=torch.long)
            top_a = support_for_half(query, query_codes, reference_a, reference_codes_a, max(ks), args.reference_batch_size, args.device)
            top_b = support_for_half(query, query_codes, reference_b, reference_codes_b, max(ks), args.reference_batch_size, args.device)
            values_a.append(torch.stack([top_a[:, :k].mean(dim=1) for k in ks], dim=1).cpu().numpy())
            values_b.append(torch.stack([top_b[:, :k].mean(dim=1) for k in ks], dim=1).cpu().numpy())
            excluded_a.append(np.count_nonzero(dev_codes[current, None] == reference_codes_a[None, :], axis=1))
            excluded_b.append(np.count_nonzero(dev_codes[current, None] == reference_codes_b[None, :], axis=1))
            print(f"seed {args.split_seed} shard {args.shard_index}: {min(start + len(current), len(query_indices))}/{len(query_indices)}", flush=True)
    shard_dir = Path(args.output_dir).resolve() / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        shard_dir / f"shard_{args.shard_index:02d}.npz",
        query_indices=query_indices,
        support_a=np.concatenate(values_a), support_b=np.concatenate(values_b),
        same_subject_reference_count_a=np.concatenate(excluded_a),
        same_subject_reference_count_b=np.concatenate(excluded_b),
    )


if __name__ == "__main__":
    main()
