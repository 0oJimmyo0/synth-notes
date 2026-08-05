#!/usr/bin/env python3
"""Summarize canonical source-completeness by split and available subgroups.

Outputs derived eligibility statistics and approved-provenance identifiers only;
it never reads or exports source-note text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_GROUPS = (
    "source_split,note_type,patient_disjoint_from_train,hadm_disjoint_from_train,"
    "age_bin,sex_gender,race_ethnicity,insurance,admission_type,service,los_bin,icu_flag"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_manifest_path", required=True)
    parser.add_argument("--extraction_audit_csv", action="append", required=True)
    parser.add_argument("--subgroup_metadata_path", default=None)
    parser.add_argument("--group_columns", default=DEFAULT_GROUPS)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_labels(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(Path(path).resolve()).fillna("")
        needed = {"dataset_row_id", "source_split", "canonical_ready", "missing_required_fields"}
        if missing := needed.difference(frame.columns):
            raise KeyError(f"extraction audit missing columns: {sorted(missing)}")
        frame["dataset_row_id"] = pd.to_numeric(frame["dataset_row_id"], errors="raise").astype(int)
        frame["canonical_ready"] = frame["canonical_ready"].astype(str).str.lower().eq("true")
        frames.append(frame[["dataset_row_id", "source_split", "canonical_ready", "missing_required_fields"]])
    labels = pd.concat(frames, ignore_index=True)
    if labels.duplicated(["dataset_row_id", "source_split"]).any():
        raise ValueError("duplicate dataset_row_id/source_split eligibility labels")
    return labels


def aggregate(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = frame.groupby(column, dropna=False, as_index=False).agg(
        n_notes=("canonical_ready", "size"),
        n_ready=("canonical_ready", "sum"),
    )
    grouped["ready_rate"] = grouped["n_ready"] / grouped["n_notes"]
    grouped.insert(0, "group_column", column)
    grouped = grouped.rename(columns={column: "group_value"})
    return grouped


def main() -> None:
    args = parse_args()
    labels = load_labels(args.extraction_audit_csv)
    split = pd.read_csv(Path(args.split_manifest_path).resolve())
    split["dataset_row_id"] = pd.to_numeric(split["dataset_row_id"], errors="raise").astype(int)
    split["source_split"] = split["split"].astype(str)
    split = split.drop_duplicates(["dataset_row_id", "source_split"])
    keep = [column for column in [
        "dataset_row_id", "source_split", "note_type", "text_length", "patient_disjoint_from_train", "hadm_disjoint_from_train",
    ] if column in split.columns]
    merged = labels.merge(split[keep], on=["dataset_row_id", "source_split"], how="left", validate="one_to_one")
    if args.subgroup_metadata_path:
        subgroup = pd.read_csv(Path(args.subgroup_metadata_path).resolve())
        subgroup["dataset_row_id"] = pd.to_numeric(subgroup["dataset_row_id"], errors="raise").astype(int)
        subgroup = subgroup.drop_duplicates("dataset_row_id")
        subgroup_columns = [column for column in [
            "dataset_row_id", "age_bin", "sex_gender", "race_ethnicity", "insurance", "admission_type", "service", "los_bin", "icu_flag",
        ] if column in subgroup.columns]
        merged = merged.merge(subgroup[subgroup_columns], on="dataset_row_id", how="left", validate="many_to_one")
    if merged["note_type"].isna().any():
        raise ValueError("some eligibility labels are absent from the split manifest")
    merged["source_split"] = merged["source_split"].astype(str)
    requested_groups = [column.strip() for column in args.group_columns.split(",") if column.strip()]
    rows = [aggregate(merged, column) for column in requested_groups if column in merged.columns]
    summary = pd.concat(rows, ignore_index=True)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "canonical_eligibility_by_group.csv", index=False)
    missing = (
        merged.loc[~merged.canonical_ready]
        .groupby(["source_split", "missing_required_fields"], as_index=False)
        .size()
        .rename(columns={"size": "n_notes"})
    )
    missing.to_csv(output_dir / "canonical_eligibility_missing_fields.csv", index=False)
    meta = {
        "n_labeled_notes": len(merged),
        "overall_ready_rate": float(merged.canonical_ready.mean()),
        "splits": merged.source_split.value_counts().to_dict(),
        "subgroup_metadata_used": bool(args.subgroup_metadata_path),
        "security_note": "Outputs contain derived aggregate eligibility statistics and no source-note text.",
    }
    (output_dir / "canonical_eligibility_summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
