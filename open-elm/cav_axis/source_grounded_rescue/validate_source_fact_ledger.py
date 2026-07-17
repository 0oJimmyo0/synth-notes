#!/usr/bin/env python3
"""Validate manual verification status and coverage of source-fact ledgers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


REQUIRED_FIELDS = {
    "principal_diagnosis", "hospital_course_events", "discharge_medications",
    "disposition", "follow_up", "instructions",
}
VALID_STATUSES = {"pending", "verified", "corrected", "omitted", "rejected"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a manually reviewed source-fact ledger CSV.")
    parser.add_argument("--ledger_review_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_reference_csv", default=None, help="Optional restricted source-note reference for value-span support checks.")
    parser.add_argument(
        "--optional_fields",
        default="",
        help="Comma-separated fields permitted to be absent (for example follow_up).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_path = Path(args.ledger_review_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(review_path)
    optional_fields = {value.strip() for value in args.optional_fields.split(",") if value.strip()}
    unknown_optional = optional_fields.difference(REQUIRED_FIELDS)
    if unknown_optional:
        raise ValueError(f"--optional_fields contains non-required fields: {sorted(unknown_optional)}")
    active_required_fields = REQUIRED_FIELDS.difference(optional_fields)
    needed = {"case_id", "field", "manual_verification_status"}
    missing = needed.difference(frame.columns)
    if missing:
        raise KeyError(f"ledger review CSV missing columns: {sorted(missing)}")
    frame["manual_verification_status"] = frame["manual_verification_status"].fillna("pending").astype(str).str.strip().str.lower()
    invalid = sorted(set(frame.manual_verification_status).difference(VALID_STATUSES))
    if invalid:
        raise ValueError(f"invalid manual_verification_status values: {invalid}")
    status_counts = frame.manual_verification_status.value_counts().to_dict()
    per_case = []
    for case_id, group in frame.groupby("case_id", sort=True):
        verified = group.loc[group.manual_verification_status.isin(["verified", "corrected"])]
        verified_fields = set(verified.field.astype(str))
        per_case.append({
            "case_id": case_id,
            "n_facts": int(len(group)),
            "n_verified_or_corrected": int(len(verified)),
            "pending_or_rejected_facts": int((~group.manual_verification_status.isin(["verified", "corrected"])).sum()),
            "missing_required_fields": "|".join(sorted(active_required_fields.difference(verified_fields))),
            "ledger_ready_for_generation": not bool(active_required_fields.difference(verified_fields)),
        })
    cases = pd.DataFrame(per_case)
    cases.to_csv(output_dir / "source_fact_ledger_case_readiness.csv", index=False)
    source_support = None
    if args.source_reference_csv:
        reference = pd.read_csv(Path(args.source_reference_csv).resolve())
        if {"case_id", "source_real_note"}.difference(reference.columns):
            raise KeyError("source reference must contain case_id and source_real_note")
        source_by_case = reference.drop_duplicates("case_id").set_index("case_id").source_real_note.fillna("").astype(str).to_dict()
        verified = frame.loc[frame.manual_verification_status.isin(["verified", "corrected"])].copy()
        values = verified.get("manual_verified_value", verified.get("value", pd.Series("", index=verified.index))).fillna("").astype(str)
        supported = []
        for row, value in zip(verified.itertuples(index=False), values):
            source_text = source_by_case.get(str(row.case_id), "")
            normalized_value = re.sub(r"\s+", " ", value).strip()
            normalized_source = re.sub(r"\s+", " ", source_text).strip()
            supported.append(bool(normalized_value) and normalized_value in normalized_source)
        source_support = {
            "n_verified_or_corrected": int(len(verified)),
            "source_text_exact_span_support_count": int(sum(supported)),
            "source_text_exact_span_support_rate": float(sum(supported) / len(supported)) if supported else 0.0,
        }
    summary = {
        "n_cases": int(len(cases)),
        "status_counts": {key: int(value) for key, value in status_counts.items()},
        "ready_for_generation_count": int(cases.ledger_ready_for_generation.sum()) if len(cases) else 0,
        "ready_for_generation_rate": float(cases.ledger_ready_for_generation.mean()) if len(cases) else 0.0,
        "required_fields": sorted(active_required_fields),
        "optional_fields": sorted(optional_fields),
        "source_support": source_support,
    }
    (output_dir / "source_fact_ledger_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
