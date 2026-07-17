#!/usr/bin/env python3
"""Materialize a row-aligned raw-ELM baseline for a source-grounded cohort."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor_ledger_path", required=True, help="Frozen source-grounded ledger JSONL")
    parser.add_argument("--raw_manifest_path", required=True, help="Frozen raw ELM comparison manifest JSONL")
    parser.add_argument("--output_manifest_path", required=True, help="Output matched raw manifest JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ledger = pd.read_json(args.anchor_ledger_path, lines=True)
    raw = pd.read_json(args.raw_manifest_path, lines=True)
    required_ledger = {"case_id", "anchor_id", "dataset_row_id"}
    required_raw = {"candidate_id", "anchor_id", "dataset_row_id", "generated_text"}
    if missing := required_ledger - set(ledger.columns):
        raise ValueError(f"Anchor ledger missing columns: {sorted(missing)}")
    if missing := required_raw - set(raw.columns):
        raise ValueError(f"Raw manifest missing columns: {sorted(missing)}")
    if ledger["anchor_id"].duplicated().any() or ledger["dataset_row_id"].duplicated().any():
        raise ValueError("Anchor ledger must contain one row per anchor and dataset_row_id.")
    if raw.duplicated(["anchor_id", "dataset_row_id"]).any():
        raise ValueError("Raw manifest must contain exactly one output per anchor for this comparison.")

    merged = ledger.merge(
        raw,
        on=["anchor_id", "dataset_row_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_raw"),
    )
    if merged["candidate_id"].isna().any():
        missing = merged.loc[merged["candidate_id"].isna(), ["case_id", "anchor_id", "dataset_row_id"]]
        raise ValueError(f"Missing raw ELM output for {len(missing)} frozen anchors:\n{missing.to_string(index=False)}")
    if merged["generated_text"].isna().any() or merged["generated_text"].astype(str).str.strip().eq("").any():
        raise ValueError("Matched raw manifest contains empty generated text.")

    merged.insert(0, "generation_id", "raw_elm_baseline__" + merged["case_id"].astype(str))
    # The current source is the frozen closed-loop accepted manifest, not a one-draw baseline.
    merged["comparison_condition"] = "closed_loop_selected_raw_elm_matched_comparator"
    merged["source_grounded_cohort"] = True
    merged["source_ledger_path"] = str(Path(args.anchor_ledger_path).resolve())
    merged["raw_manifest_path"] = str(Path(args.raw_manifest_path).resolve())
    merged = merged.sort_values("case_id").reset_index(drop=True)

    out = Path(args.output_manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_json(out, orient="records", lines=True)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "anchor_ledger_path": str(Path(args.anchor_ledger_path).resolve()),
        "raw_manifest_path": str(Path(args.raw_manifest_path).resolve()),
        "output_manifest_path": str(out.resolve()),
        "rows": int(len(merged)),
        "unique_case_ids": int(merged["case_id"].nunique()),
        "unique_anchor_ids": int(merged["anchor_id"].nunique()),
        "row_order": "case_id ascending; identical to frozen ledger order after sort",
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {len(merged)} matched raw ELM rows to: {out}")


if __name__ == "__main__":
    main()
