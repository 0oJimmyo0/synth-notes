#!/usr/bin/env python3
"""Audit full canonical embedding matrices before geometry analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--metadata_path", default=None)
    parser.add_argument("--fingerprint_output_path", default=None)
    parser.add_argument("--chunk_rows", type=int, default=4096)
    parser.add_argument("--norm_lower", type=float, default=0.999)
    parser.add_argument("--norm_upper", type=float, default=1.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = np.load(Path(args.embeddings_path).resolve(), mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError("Expected a two-dimensional embedding matrix.")
    metadata = None
    if args.metadata_path:
        metadata = [json.loads(line) for line in Path(args.metadata_path).read_text().splitlines() if line]
        if len(metadata) != len(embeddings):
            raise ValueError("Embedding and metadata lengths differ.")
    norms, fingerprints, duplicate_pairs, fingerprint_rows = [], {}, [], []
    nonfinite = 0
    duplicate_count = 0
    for start in range(0, len(embeddings), args.chunk_rows):
        block = np.asarray(embeddings[start : start + args.chunk_rows])
        nonfinite += int((~np.isfinite(block)).sum())
        norms.append(np.linalg.norm(block, axis=1))
        for offset, row in enumerate(block):
            fingerprint = hashlib.blake2b(row.tobytes(), digest_size=16).digest()
            fingerprint_rows.append((start + offset, fingerprint.hex()))
            if fingerprint in fingerprints:
                # Confirm equality rather than treating a hash match as a duplicate.
                if np.array_equal(row, embeddings[fingerprints[fingerprint]]):
                    duplicate_count += 1
                    duplicate_pairs.append((fingerprints[fingerprint], start + offset))
            else:
                fingerprints[fingerprint] = start + offset
    norms_array = np.concatenate(norms)
    summary = {
        "n_rows": int(len(embeddings)),
        "embedding_dimension": int(embeddings.shape[1]),
        "nonfinite_value_count": nonfinite,
        "norm_min": float(norms_array.min()),
        "norm_median": float(np.median(norms_array)),
        "norm_max": float(norms_array.max()),
        "norm_outside_tolerance_count": int(((norms_array < args.norm_lower) | (norms_array > args.norm_upper)).sum()),
        "norm_tolerance": [args.norm_lower, args.norm_upper],
        "exact_duplicate_vector_count": duplicate_count,
        "integrity_pass": bool(nonfinite == 0 and duplicate_count == 0 and ((norms_array >= args.norm_lower) & (norms_array <= args.norm_upper)).all()),
        "security_note": "Output contains derived integrity statistics only; no source-note text.",
    }
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    if metadata is not None:
        report_path = output_path.with_name(output_path.stem + "_duplicate_vectors.csv")
        with report_path.open("w") as handle:
            handle.write("first_source_index,duplicate_source_index,first_dataset_row_id,duplicate_dataset_row_id,first_note_id,duplicate_note_id,first_case_id,duplicate_case_id\n")
            for first, duplicate in duplicate_pairs:
                left, right = metadata[first], metadata[duplicate]
                handle.write(
                    f"{first},{duplicate},{left.get('dataset_row_id','')},{right.get('dataset_row_id','')},"
                    f"{left.get('note_id','')},{right.get('note_id','')},{left.get('case_id','')},{right.get('case_id','')}\n"
                )
    if args.fingerprint_output_path:
        fingerprint_path = Path(args.fingerprint_output_path).resolve()
        fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        with fingerprint_path.open("w") as handle:
            handle.write("source_index,vector_fingerprint\n")
            for source_index, fingerprint in fingerprint_rows:
                handle.write(f"{source_index},{fingerprint}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
