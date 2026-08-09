#!/usr/bin/env python3
"""Audit V4 canonical-embedding proximity to source and unrelated train notes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_embeddings_path", required=True)
    parser.add_argument("--query_metadata_path", required=True)
    parser.add_argument("--source_embeddings_path", required=True)
    parser.add_argument("--source_metadata_path", required=True)
    parser.add_argument("--train_embeddings_path", required=True)
    parser.add_argument("--train_metadata_path", required=True)
    parser.add_argument("--cohort_manifest_csv", required=True)
    parser.add_argument("--split_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


def read_metadata(path: str) -> pd.DataFrame:
    return pd.read_json(Path(path).resolve(), lines=True, dtype=False)


def normalized(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def main() -> None:
    args = parse_args()
    query_embeddings = normalized(np.load(Path(args.query_embeddings_path).resolve(), mmap_mode="r"))
    source_embeddings = normalized(np.load(Path(args.source_embeddings_path).resolve(), mmap_mode="r"))
    train_embeddings = normalized(np.load(Path(args.train_embeddings_path).resolve(), mmap_mode="r"))
    query_meta = read_metadata(args.query_metadata_path)
    source_meta = read_metadata(args.source_metadata_path)
    train_meta = read_metadata(args.train_metadata_path)
    cohort = pd.read_csv(Path(args.cohort_manifest_csv).resolve(), dtype=str).fillna("")
    split = pd.read_csv(Path(args.split_manifest_path).resolve(), dtype=str).fillna("")
    for label, matrix, meta in (("query", query_embeddings, query_meta), ("source", source_embeddings, source_meta), ("train", train_embeddings, train_meta)):
        if len(matrix) != len(meta):
            raise ValueError(f"{label} embeddings and metadata have different row counts")
        if "dataset_row_id" not in meta:
            raise KeyError(f"{label} metadata lacks dataset_row_id")
    if {"case_id", "dataset_row_id", "subject_id"}.difference(cohort.columns):
        raise KeyError("cohort manifest needs case_id, dataset_row_id, and subject_id")
    if {"dataset_row_id", "subject_id"}.difference(split.columns):
        raise KeyError("split manifest needs dataset_row_id and subject_id")
    for frame in (query_meta, source_meta, train_meta, cohort, split):
        frame["dataset_row_id"] = frame.dataset_row_id.astype(str)
    source_index = {row.dataset_row_id: index for index, row in source_meta.iterrows()}
    train_subject = train_meta[["dataset_row_id"]].merge(
        split[["dataset_row_id", "subject_id"]].drop_duplicates("dataset_row_id"),
        on="dataset_row_id", how="left", validate="one_to_one",
    ).subject_id.fillna("").astype(str).to_numpy()
    if not np.all(train_subject):
        raise ValueError("some train embeddings could not be linked to a subject")
    cohort_by_row = cohort.drop_duplicates("dataset_row_id").set_index("dataset_row_id")
    top_k = max(1, int(args.top_k))
    rows, neighbors = [], []
    for query_index, meta in query_meta.iterrows():
        row_id = str(meta.dataset_row_id)
        if row_id not in cohort_by_row.index or row_id not in source_index:
            raise ValueError(f"query dataset_row_id lacks frozen cohort/source mapping: {row_id}")
        cohort_row = cohort_by_row.loc[row_id]
        subject_id = str(cohort_row.subject_id)
        scores = train_embeddings @ query_embeddings[query_index]
        scores[train_subject == subject_id] = -np.inf
        positions = np.argpartition(scores, -top_k)[-top_k:]
        positions = positions[np.argsort(scores[positions])[::-1]]
        source_cosine = float(query_embeddings[query_index] @ source_embeddings[source_index[row_id]])
        nearest_cosine = float(scores[positions[0]])
        rows.append({
            "case_id": cohort_row.case_id,
            "dataset_row_id": row_id,
            "subject_id": subject_id,
            "source_canonical_cosine": source_cosine,
            "nearest_unrelated_train_cosine": nearest_cosine,
            "source_minus_nearest_unrelated_margin": source_cosine - nearest_cosine,
        })
        for rank, position in enumerate(positions, start=1):
            neighbors.append({
                "case_id": cohort_row.case_id,
                "dataset_row_id": row_id,
                "neighbor_rank": rank,
                "nearest_train_dataset_row_id": str(train_meta.iloc[position].dataset_row_id),
                "nearest_train_cosine": float(scores[position]),
            })
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_note = pd.DataFrame(rows).sort_values("case_id")
    per_note.to_csv(output_dir / "v4_semantic_privacy_per_note.csv", index=False)
    pd.DataFrame(neighbors).sort_values(["case_id", "neighbor_rank"]).to_csv(output_dir / "v4_semantic_privacy_neighbors.csv", index=False)
    summary = {
        "n_notes": int(len(per_note)),
        "top_k": top_k,
        "reference": "canonical real-train embeddings with every same-subject train row excluded",
        "source_cosine_mean": float(per_note.source_canonical_cosine.mean()),
        "nearest_unrelated_train_cosine_mean": float(per_note.nearest_unrelated_train_cosine.mean()),
        "source_minus_nearest_unrelated_margin_min": float(per_note.source_minus_nearest_unrelated_margin.min()),
        "security_note": "Outputs contain provenance IDs and derived embedding similarities only; no note text is exported.",
    }
    (output_dir / "v4_semantic_privacy_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
