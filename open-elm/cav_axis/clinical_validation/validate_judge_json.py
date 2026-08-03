#!/usr/bin/env python3
"""Validate evidence-first medication judge JSONL without exposing task text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCHEMA_PATH = Path(__file__).with_name("medication_judge_schema.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate medication judge outputs against the stable contract.")
    parser.add_argument("--judge_output_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--schema_path", default=str(SCHEMA_PATH))
    return parser.parse_args()


def validate(record: dict[str, object], schema: dict[str, object] | None = None) -> list[str]:
    schema = schema or json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    payload = record.get("judge_output") if isinstance(record.get("judge_output"), dict) else record
    if not isinstance(payload, dict):
        return ["output_not_object"]
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})
    missing = required - set(payload)
    if missing:
        errors.append("missing_required=" + ",".join(sorted(missing)))
    if schema.get("additionalProperties") is False:
        unexpected = set(payload) - set(properties)
        if unexpected:
            errors.append("unexpected_keys=" + ",".join(sorted(unexpected)))
    for field, field_schema in properties.items():
        if field not in payload:
            continue
        expected_type = field_schema.get("type")
        if expected_type == "array" and not isinstance(payload[field], list):
            errors.append(f"{field}_not_list")
        if expected_type == "boolean" and not isinstance(payload[field], bool):
            errors.append(f"{field}_not_bool")
    findings_schema = properties.get("findings", {}).get("items", {})
    if isinstance(payload.get("findings"), list) and findings_schema:
        finding_required = set(findings_schema.get("required", []))
        finding_properties = findings_schema.get("properties", {})
        allowed_categories = set(finding_properties.get("category", {}).get("enum", []))
        max_items = properties.get("findings", {}).get("maxItems")
        if max_items is not None and len(payload["findings"]) > max_items:
            errors.append("too_many_findings")
        for finding in payload["findings"]:
            if not isinstance(finding, dict):
                errors.append("invalid_finding")
                break
            missing_finding = finding_required - set(finding)
            if missing_finding:
                errors.append("finding_missing_required=" + ",".join(sorted(missing_finding)))
                break
            if allowed_categories and finding.get("category") not in allowed_categories:
                errors.append("invalid_finding_category")
                break
            if not isinstance(finding.get("material"), bool):
                errors.append("finding_material_not_bool")
                break
            if not isinstance(finding.get("note_evidence"), str) or not isinstance(finding.get("rationale"), str):
                errors.append("finding_evidence_or_rationale_not_string")
                break
            if finding.get("ledger_fact_id") is not None and not isinstance(finding.get("ledger_fact_id"), str):
                errors.append("finding_ledger_fact_id_invalid")
                break
    if "final_reject" in payload and "medication_reconciliation_pass" in payload:
        if not isinstance(payload["final_reject"], bool) or not isinstance(payload["medication_reconciliation_pass"], bool):
            pass
        elif payload["final_reject"] == payload["medication_reconciliation_pass"]:
            errors.append("inconsistent_pass_reject_decision")
    if payload.get("final_reject") is True and payload.get("requires_human_review") is False:
        errors.append("reject_without_human_review")
    if payload.get("final_reject") is True and isinstance(payload.get("findings"), list):
        if not any(item.get("material") is True and item.get("category") != "none" for item in payload["findings"] if isinstance(item, dict)):
            errors.append("reject_without_material_finding")
    alignment_schema = properties.get("ledger_to_note_alignment", {}).get("items", {})
    if isinstance(payload.get("ledger_to_note_alignment"), list) and alignment_schema:
        alignment_required = set(alignment_schema.get("required", []))
        alignment_properties = alignment_schema.get("properties", {})
        allowed_statuses = set(alignment_schema.get("properties", {}).get("status", {}).get("enum", []))
        for item in payload["ledger_to_note_alignment"]:
            if not isinstance(item, dict):
                errors.append("invalid_alignment_status")
                break
            if alignment_required - set(item):
                errors.append("alignment_missing_required")
                break
            if alignment_schema.get("additionalProperties") is False and set(item) - set(alignment_properties):
                errors.append("alignment_unexpected_keys")
                break
            if allowed_statuses and item.get("status") not in allowed_statuses:
                errors.append("invalid_alignment_status")
                break
            for field in ("contract_id", "note_evidence", "rationale"):
                if field in alignment_properties and not isinstance(item.get(field), str):
                    errors.append("alignment_text_field_invalid")
                    break
    return errors


def main() -> None:
    args = parse_args()
    schema = json.loads(Path(args.schema_path).resolve().read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with Path(args.judge_output_path).resolve().open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                errors = validate(record, schema)
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
