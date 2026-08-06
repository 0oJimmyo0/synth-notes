#!/usr/bin/env python3
"""Join canonical embedding metadata to subject-level split provenance.

The emitted JSONL excludes all source-note text and text previews.  It is the
required provenance input for subject-grouped local-support diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical_metadata_path", required=True)
    parser.add_argument("--split_manifest_path", required=True)
    parser.add_argument("--source_split", choices=("train", "dev"), required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical = pd.read_json(Path(args.canonical_metadata_path).resolve(), lines=True)
    required_canonical = {"dataset_row_id", "note_id", "source_split"}
    if missing := required_canonical.difference(canonical.columns):
        raise KeyError(f"Canonical metadata missing columns: {sorted(missing)}")
    if not canonical["source_split"].astype(str).eq(args.source_split).all():
        raise ValueError("Canonical metadata includes a different source split.")
    if canonical["dataset_row_id"].duplicated().any():
        raise ValueError("Canonical metadata contains duplicate dataset_row_id values.")
    split = pd.read_csv(Path(args.split_manifest_path).resolve())
    required_split = {"dataset_row_id", "note_id", "split", "subject_id", "hadm_id", "patient_disjoint_from_train"}
    if missing := required_split.difference(split.columns):
        raise KeyError(f"Split manifest missing columns: {sorted(missing)}")
    split = split.loc[split["split"].astype(str).eq(args.source_split), list(required_split)].copy()
    if split["dataset_row_id"].duplicated().any():
        raise ValueError("Split manifest contains duplicate dataset_row_id values within split.")
    joined = canonical.merge(
        split,
        on=["dataset_row_id", "note_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Some canonical rows could not be joined to split provenance.")
    if joined["subject_id"].isna().any() or joined["hadm_id"].isna().any():
        raise ValueError("Joined provenance contains missing subject_id or hadm_id.")
    output = joined[[
        "dataset_row_id", "note_id", "case_id", "source_split", "source_index",
        "subject_id", "hadm_id", "patient_disjoint_from_train",
        "representation_id", "representation_spec_sha256",
    ]].copy()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_json(output_path, orient="records", lines=True)
    summary = {
        "source_split": args.source_split,
        "n_rows": int(len(output)),
        "n_unique_subjects": int(output["subject_id"].nunique()),
        "n_patient_disjoint": int(output["patient_disjoint_from_train"].astype(bool).sum()),
        "duplicate_subject_rows": int(output.duplicated(["subject_id", "dataset_row_id"]).sum()),
        "security_note": "Output contains provenance IDs and derived flags only; no source-note text.",
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
