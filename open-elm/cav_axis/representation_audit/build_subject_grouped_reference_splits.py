#!/usr/bin/env python3
"""Create subject-grouped, exact-vector-deduplicated train reference halves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def half(value: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return "a" if digest[0] % 2 == 0 else "b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_subject_metadata_path", required=True)
    parser.add_argument("--fingerprints_path", required=True)
    parser.add_argument("--seeds", default="20260811,20260812,20260813,20260814,20260815")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    metadata = pd.read_json(Path(args.train_subject_metadata_path).resolve(), lines=True).sort_values("source_index")
    fingerprints = pd.read_csv(Path(args.fingerprints_path).resolve()).sort_values("source_index")
    if metadata.source_index.tolist() != list(range(len(metadata))) or fingerprints.source_index.tolist() != list(range(len(metadata))):
        raise ValueError("Metadata and fingerprints must cover ordered source indices exactly once.")
    frame = metadata[["source_index", "subject_id"]].merge(fingerprints, on="source_index", validate="one_to_one")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in seeds:
        out = frame.copy()
        out["subject_half"] = out.subject_id.astype(str).map(lambda value: half(value, seed))
        # A vector fingerprint contributes once, to one deterministic half. If
        # its preferred hash half has no member, retain the available half.
        out["fingerprint_preferred_half"] = out.vector_fingerprint.map(lambda value: half(value, seed))
        out["reference_half"] = ""
        out["reference_keep"] = False
        for fingerprint, group in out.groupby("vector_fingerprint", sort=False):
            preferred = group.fingerprint_preferred_half.iloc[0]
            candidates = group.loc[group.subject_half.eq(preferred)]
            if candidates.empty:
                candidates = group
            chosen = candidates.iloc[0]
            out.loc[chosen.name, "reference_half"] = chosen.subject_half
            out.loc[chosen.name, "reference_keep"] = True
        kept = out.loc[out.reference_keep]
        if set(kept.reference_half) != {"a", "b"}:
            raise ValueError("A reference half is empty.")
        subject_halves = out.groupby("subject_id").subject_half.nunique()
        if int(subject_halves.max()) != 1:
            raise ValueError("A subject appears in both reference halves.")
        path = output_dir / f"train_reference_split_seed_{seed}.csv"
        out[["source_index", "subject_id", "vector_fingerprint", "subject_half", "reference_half", "reference_keep"]].to_csv(path, index=False)
        summary = {
            "seed": seed,
            "n_train_rows": int(len(out)),
            "n_unique_subjects": int(out.subject_id.nunique()),
            "n_unique_vector_classes": int(out.vector_fingerprint.nunique()),
            "n_rows_removed_by_exact_vector_deduplication": int((~out.reference_keep).sum()),
            "n_reference_a": int((kept.reference_half == "a").sum()),
            "n_reference_b": int((kept.reference_half == "b").sum()),
            "subject_grouping_verified": True,
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    pd.DataFrame(summaries).to_csv(output_dir / "reference_split_summary.csv", index=False)


if __name__ == "__main__":
    main()
