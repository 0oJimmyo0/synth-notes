#!/usr/bin/env python3
"""Merge deterministic canonical BGE shards and verify complete source coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard_root", required=True)
    parser.add_argument("--n_shards", type=int, required=True)
    parser.add_argument("--expected_rows", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    shard_root = Path(args.shard_root).resolve()
    shards = [shard_root / f"shard_{index:02d}" for index in range(args.n_shards)]
    metadata_rows = []
    dimension = None
    for shard in shards:
        embeddings = np.load(shard / "canonical_section_balanced_embeddings.npy", mmap_mode="r")
        rows = [json.loads(line) for line in (shard / "canonical_section_balanced_metadata.jsonl").read_text().splitlines() if line]
        if len(rows) != len(embeddings):
            raise ValueError(f"Metadata/embedding length mismatch in {shard}")
        if dimension is None:
            dimension = embeddings.shape[1]
        elif embeddings.shape[1] != dimension:
            raise ValueError("Shard embedding dimensions differ.")
        metadata_rows.extend((int(row["source_index"]), row, vector) for row, vector in zip(rows, embeddings))
    indices = [item[0] for item in metadata_rows]
    if sorted(indices) != list(range(args.expected_rows)):
        raise ValueError("Shards do not provide exactly one embedding for every source row.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = np.lib.format.open_memmap(
        output_dir / "canonical_section_balanced_embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(args.expected_rows, dimension),
    )
    with (output_dir / "canonical_section_balanced_metadata.jsonl").open("w") as metadata:
        for source_index, row, vector in sorted(metadata_rows, key=lambda item: item[0]):
            merged[source_index] = vector
            metadata.write(json.dumps(row) + "\n")
    merged.flush()
    summary = {
        "n_rows": args.expected_rows,
        "embedding_dimension": dimension,
        "n_shards": args.n_shards,
        "coverage_verified": True,
        "security_note": "Merged embeddings and metadata exclude source-derived note text.",
    }
    (output_dir / "canonical_section_balanced_embedding_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
