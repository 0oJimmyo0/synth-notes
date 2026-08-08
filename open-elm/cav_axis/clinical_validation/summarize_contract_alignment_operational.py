#!/usr/bin/env python3
"""Summarize local contract-alignment routing without exporting clinical text."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from validate_judge_json import validate


NONPRESENT = {"missing", "unsupported", "uncertain"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge_output_path", required=True)
    parser.add_argument("--schema_path", required=True)
    parser.add_argument("--expected_repeats", type=int, default=3)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    schema = json.loads(Path(args.schema_path).resolve().read_text(encoding="utf-8"))
    by_output: dict[str, list[dict[str, object]]] = defaultdict(list)
    for line in Path(args.judge_output_path).resolve().read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            row["semantic_errors"] = validate(row, schema)
            by_output[str(row["blinded_output_id"])].append(row)

    task_rows = []
    for output_id, rows in sorted(by_output.items()):
        by_contract: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            payload = row.get("judge_output")
            if not isinstance(payload, dict):
                continue
            for item in payload.get("ledger_to_note_alignment", []):
                if isinstance(item, dict):
                    by_contract[str(item.get("contract_id", ""))].append(str(item.get("status", "")))
        nonpresent = any(status in NONPRESENT for statuses in by_contract.values() for status in statuses)
        unstable = any(
            len(statuses) != args.expected_repeats or len(set(statuses)) != 1
            for statuses in by_contract.values()
        )
        invalid = any(bool(row["semantic_errors"]) for row in rows)
        incomplete = len(rows) != args.expected_repeats
        reasons = []
        if incomplete:
            reasons.append("incomplete_repeats")
        if invalid:
            reasons.append("semantic_or_schema_invalid")
        if nonpresent:
            reasons.append("nonpresent_alignment")
        if unstable:
            reasons.append("alignment_instability")
        task_rows.append({
            "blinded_output_id": output_id,
            "repeat_count": len(rows),
            "contract_obligation_count": len(by_contract),
            "semantic_valid_repeats": sum(not row["semantic_errors"] for row in rows),
            "route_to_review": bool(reasons),
            "route_reasons": "|".join(reasons),
        })

    table = pd.DataFrame(task_rows)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "contract_alignment_operational_task_summary.csv", index=False)
    summary = {
        "n_tasks": int(len(table)),
        "expected_repeats": args.expected_repeats,
        "complete_repeat_task_rate": float(table.repeat_count.eq(args.expected_repeats).mean()) if len(table) else 0.0,
        "all_repeats_semantically_valid_task_rate": float(table.semantic_valid_repeats.eq(args.expected_repeats).mean()) if len(table) else 0.0,
        "route_to_review_count": int(table.route_to_review.sum()),
        "route_to_review_rate": float(table.route_to_review.mean()) if len(table) else 0.0,
        "route_rule": "Route if any repeat is invalid, any alignment is non-present, or any contract alignment is repeat-inconsistent.",
        "interpretation": "Operational development evidence only. Routing is conservative and does not establish clinical-error sensitivity or specificity without independent adjudication.",
        "security_note": "Outputs contain blinded IDs and derived statuses only; no contract or note text is exported.",
    }
    (output / "contract_alignment_operational_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
