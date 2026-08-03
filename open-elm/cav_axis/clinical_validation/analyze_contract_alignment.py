#!/usr/bin/env python3
"""Summarize contract-alignment calibration without exporting clinical text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


STATUSES = {"present_supported", "missing", "unsupported", "uncertain"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human_review_csv", required=True)
    parser.add_argument("--judge_key_csv", required=True)
    parser.add_argument("--expected_repeats", type=int, default=3)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    human = pd.read_csv(Path(args.human_review_csv).resolve()).fillna("")
    key = pd.read_csv(Path(args.judge_key_csv).resolve()).fillna("")
    required_human = {"blinded_output_id", "contract_id", "human_alignment_status"}
    required_key = {"blinded_output_id", "contract_id", "repeat_index", "schema_valid", "judge_status"}
    if missing := required_human - set(human.columns):
        raise KeyError(f"human review missing columns: {sorted(missing)}")
    if missing := required_key - set(key.columns):
        raise KeyError(f"judge key missing columns: {sorted(missing)}")
    if human.duplicated(["blinded_output_id", "contract_id"]).any():
        raise ValueError("Human review has duplicate note/contract pairs.")
    human["human_alignment_status"] = human.human_alignment_status.astype(str).str.strip().str.lower()
    invalid = set(human.human_alignment_status) - STATUSES
    if invalid:
        raise ValueError(f"Invalid human statuses: {sorted(invalid)}")
    key["judge_status"] = key.judge_status.astype(str).str.strip().str.lower()
    key["schema_valid"] = key.schema_valid.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    # Schema-invalid payloads can contain alignment-like fragments (for example
    # prompt echo) but must never count as valid judge coverage.
    key = key[key.schema_valid & key.judge_status.isin(STATUSES)].copy()
    consensus = key.groupby(["blinded_output_id", "contract_id"], as_index=False).agg(
        judge_repeat_count=("judge_status", "size"),
        judge_status_consensus=("judge_status", lambda values: values.iloc[0] if len(values) == args.expected_repeats and values.nunique() == 1 else ""),
    )
    merged = human.merge(consensus, on=["blinded_output_id", "contract_id"], how="left", validate="one_to_one")
    # A left join leaves missing consensus as NaN.  NaN != "" is True, so
    # explicitly define coverage as a complete, schema-valid allowed status.
    merged["judge_status_consensus"] = merged["judge_status_consensus"].fillna("")
    merged["judge_repeat_count"] = pd.to_numeric(merged["judge_repeat_count"], errors="coerce").fillna(0).astype(int)
    merged["key_coverage"] = merged.judge_status_consensus.isin(STATUSES)
    merged["human_nonpresent"] = merged.human_alignment_status.ne("present_supported")
    merged["judge_nonpresent"] = merged.judge_status_consensus.ne("present_supported") & merged.key_coverage
    covered = merged[merged.key_coverage]
    tp = int((covered.human_nonpresent & covered.judge_nonpresent).sum())
    fn = int((covered.human_nonpresent & ~covered.judge_nonpresent).sum())
    fp = int((~covered.human_nonpresent & covered.judge_nonpresent).sum())
    tn = int((~covered.human_nonpresent & ~covered.judge_nonpresent).sum())
    safe = lambda n, d: float(n / d) if d else None
    summary = {
        "n_human_obligations": int(len(merged)),
        "human_status_counts": {str(k): int(v) for k, v in human.human_alignment_status.value_counts().items()},
        "key_coverage_count": int(merged.key_coverage.sum()),
        "key_coverage_rate": float(merged.key_coverage.mean()),
        "covered_exact_agreement": safe(int((covered.human_alignment_status == covered.judge_status_consensus).sum()), len(covered)),
        "covered_nonpresent_count": int(covered.human_nonpresent.sum()),
        "true_positive": tp, "false_negative": fn, "false_positive": fp, "true_negative": tn,
        "nonpresent_sensitivity": safe(tp, tp + fn),
        "nonpresent_specificity": safe(tn, tn + fp),
        "interpretation": "Calibration only. Agreement among present-supported obligations does not establish safety sensitivity; uncovered obligations and false negatives require human review.",
    }
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    safe_columns = [c for c in merged.columns if c not in {"synthetic_note", "contract_obligation", "reviewer_notes"}]
    merged[safe_columns].to_csv(output / "contract_alignment_calibration_matrix.csv", index=False)
    (output / "contract_alignment_calibration_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
