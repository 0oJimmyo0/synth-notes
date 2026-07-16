#!/usr/bin/env python3
"""Apply a portable ledger-review patch on approved storage.

The portable patch contains only decisions and source offsets. This script is
the only component that resolves corrected/additional fact values from the
restricted source-note reference and writes a completed restricted ledger.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


VALID_STATUSES = {"verified", "corrected", "rejected"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a portable fact-ledger review patch securely.")
    parser.add_argument("--ledger", required=True, help="Original restricted provisional ledger CSV.")
    parser.add_argument("--source_reference", required=True, help="Restricted one-row-per-case source note CSV.")
    parser.add_argument("--patch", required=True, help="Portable review patch CSV with no clinical text.")
    parser.add_argument("--output", required=True, help="Completed restricted ledger CSV.")
    parser.add_argument("--summary_json", default=None, help="Optional non-text summary JSON.")
    return parser.parse_args()


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def source_value(row: pd.Series, source_by_case: dict[str, str]) -> str:
    case_id = str(row.case_id)
    source = source_by_case.get(case_id)
    if source is None:
        raise ValueError(f"patch references unknown case_id: {case_id}")
    start, end = row.get("replacement_char_start"), row.get("replacement_char_end")
    if pd.isna(start) or pd.isna(end):
        raise ValueError(f"{row.fact_id} requires source offsets for action={row.action}")
    start, end = int(start), int(end)
    if start < 0 or end <= start or end > len(source):
        raise ValueError(f"invalid source offsets for {row.fact_id}: [{start}, {end})")
    value = source[start:end]
    transform = str(row.get("value_transform", "") or "")
    if transform == "normalize_whitespace":
        value = normalize_whitespace(value)
    elif transform not in {"", "nan", "None"}:
        raise ValueError(f"unsupported value_transform={transform!r} for {row.fact_id}")
    if not value:
        raise ValueError(f"empty source-derived value for {row.fact_id}")
    return value


def original_span_value(row: pd.Series, source_by_case: dict[str, str]) -> str:
    """Recover a rare empty extracted value from its original source span."""
    source = source_by_case.get(str(row.case_id))
    if source is None:
        raise ValueError(f"ledger references unknown case_id: {row.case_id}")
    start, end = row.get("source_char_start"), row.get("source_char_end")
    if pd.isna(start) or pd.isna(end):
        raise ValueError(f"empty verified value has no original source span: {row.fact_id}")
    value = normalize_whitespace(source[int(start):int(end)])
    if not value:
        raise ValueError(f"empty original source span for {row.fact_id}")
    return value


def main() -> None:
    args = parse_args()
    ledger = pd.read_csv(Path(args.ledger).resolve())
    source = pd.read_csv(Path(args.source_reference).resolve())
    patch = pd.read_csv(Path(args.patch).resolve())
    for frame, required, label in [
        (ledger, {"case_id", "fact_id", "field", "value", "source_section"}, "ledger"),
        (source, {"case_id", "source_real_note"}, "source reference"),
        (patch, {"action", "case_id", "fact_id", "field", "manual_verification_status"}, "patch"),
    ]:
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{label} missing columns: {sorted(missing)}")
    if ledger.fact_id.duplicated().any() or patch.fact_id.duplicated().any():
        raise ValueError("fact_id values must be unique in both ledger and patch")
    if source.case_id.duplicated().any():
        raise ValueError("source reference must have exactly one row per case_id")

    patch["manual_verification_status"] = patch.manual_verification_status.astype(str).str.strip().str.lower()
    invalid = set(patch.manual_verification_status).difference(VALID_STATUSES)
    if invalid:
        raise ValueError(f"invalid final statuses: {sorted(invalid)}")
    existing = patch.loc[patch.action.astype(str) == "update"].copy()
    additions = patch.loc[patch.action.astype(str) == "add"].copy()
    if set(existing.fact_id) != set(ledger.fact_id):
        missing = set(ledger.fact_id).difference(existing.fact_id)
        unexpected = set(existing.fact_id).difference(ledger.fact_id)
        raise ValueError(f"update patch must cover every original fact once; missing={len(missing)}, unexpected={len(unexpected)}")
    if set(additions.fact_id).intersection(ledger.fact_id):
        raise ValueError("added fact_id collides with original ledger")

    source_by_case = source.set_index("case_id").source_real_note.fillna("").astype(str).to_dict()
    patch_by_id = existing.set_index("fact_id")
    completed_rows = []
    for row in ledger.itertuples(index=False):
        original = pd.Series(row._asdict())
        decision = patch_by_id.loc[str(original.fact_id)]
        if str(decision.case_id) != str(original.case_id) or str(decision.field) != str(original.field):
            raise ValueError(f"patch provenance mismatch for {original.fact_id}")
        status = str(decision.manual_verification_status)
        original["manual_verification_status"] = status
        if status == "corrected":
            original["manual_verified_value"] = source_value(decision, source_by_case)
            original["source_section"] = str(decision.replacement_source_section)
            original["supporting_text"] = original["manual_verified_value"]
            original["source_char_start"] = int(decision.replacement_char_start)
            original["source_char_end"] = int(decision.replacement_char_end)
        elif status == "verified":
            existing_value = original.get("value")
            if pd.isna(existing_value) or not str(existing_value).strip():
                original["manual_verified_value"] = original_span_value(original, source_by_case)
            else:
                original["manual_verified_value"] = str(existing_value)
        else:
            original["manual_verified_value"] = None
        original["review_reason_code"] = decision.get("reason_code", "")
        completed_rows.append(original.to_dict())
    for row in additions.itertuples(index=False):
        decision = pd.Series(row._asdict())
        status = str(decision.manual_verification_status)
        if status not in {"verified", "corrected"}:
            raise ValueError(f"new fact {decision.fact_id} must be verified or corrected")
        value = source_value(decision, source_by_case)
        completed_rows.append({
            "case_id": str(decision.case_id),
            "dataset_row_id": source.loc[source.case_id.astype(str) == str(decision.case_id), "dataset_row_id"].iloc[0],
            "note_id": source.loc[source.case_id.astype(str) == str(decision.case_id), "note_id"].iloc[0],
            "fact_id": str(decision.fact_id),
            "field": str(decision.field),
            "value": value,
            "source_section": str(decision.replacement_source_section),
            "supporting_text": value,
            "source_char_start": int(decision.replacement_char_start),
            "source_char_end": int(decision.replacement_char_end),
            "extraction_confidence": 1.0,
            "manual_verification_status": status,
            "manual_verified_value": value,
            "review_reason_code": decision.get("reason_code", ""),
        })
    completed = pd.DataFrame(completed_rows)
    if completed.fact_id.duplicated().any() or len(completed) != len(patch):
        raise ValueError("completed ledger is not one-to-one with reviewed patch facts")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed.to_csv(output, index=False)
    summary = {
        "n_original_rows": int(len(ledger)),
        "n_added_rows": int(len(additions)),
        "n_completed_rows": int(len(completed)),
        "status_counts": {key: int(value) for key, value in completed.manual_verification_status.value_counts().items()},
        "security_note": "Completed ledger contains source-derived clinical text and must remain on approved project storage.",
    }
    summary_path = Path(args.summary_json).resolve() if args.summary_json else output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
