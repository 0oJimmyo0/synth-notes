#!/usr/bin/env python3
"""Validate evidence-first medication judge JSONL without exposing task text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "ledger_medications", "note_medication_claims", "ledger_to_note_alignment",
    "missing_ledger_medications", "unsupported_note_medications", "contradictions",
    "medication_reconciliation_pass", "final_reject",
}
STATUSES = {
    "supported_exact", "supported_equivalent", "missing", "unsupported_addition",
    "identity_contradiction", "dose_contradiction", "route_contradiction",
    "frequency_contradiction", "action_contradiction", "uncertain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate medication judge outputs against the stable contract.")
    parser.add_argument("--judge_output_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def validate(record: dict[str, object]) -> list[str]:
    errors: list[str] = []
    payload = record.get("judge_output") if isinstance(record.get("judge_output"), dict) else record
    if not isinstance(payload, dict):
        return ["output_not_object"]
    missing = REQUIRED - set(payload)
    if missing:
        errors.append("missing_required=" + ",".join(sorted(missing)))
    for field in REQUIRED - {"medication_reconciliation_pass", "final_reject"}:
        if field in payload and not isinstance(payload[field], list):
            errors.append(f"{field}_not_list")
    for field in ("medication_reconciliation_pass", "final_reject"):
        if field in payload and not isinstance(payload[field], bool):
            errors.append(f"{field}_not_bool")
    for item in payload.get("ledger_to_note_alignment", []) if isinstance(payload.get("ledger_to_note_alignment", []), list) else []:
        if not isinstance(item, dict) or item.get("status") not in STATUSES:
            errors.append("invalid_alignment_status")
            break
    return errors


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with Path(args.judge_output_path).resolve().open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                errors = validate(record)
                rows.append({"row_id": index, "task_id": record.get("task_id", ""), "schema_valid": not errors, "validation_errors": "|".join(errors)})
            except json.JSONDecodeError:
                rows.append({"row_id": index, "task_id": "", "schema_valid": False, "validation_errors": "invalid_json"})
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "medication_judge_json_validation.csv", index=False)
    summary = {"n_rows": len(table), "schema_valid_rate": float(table.schema_valid.mean()) if len(table) else 0.0}
    (output_dir / "medication_judge_json_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
