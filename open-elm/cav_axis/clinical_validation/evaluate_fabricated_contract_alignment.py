#!/usr/bin/env python3
"""Evaluate a fabricated-only contract-alignment run without clinical content."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_path", required=True)
    parser.add_argument("--judge_output_path", required=True)
    parser.add_argument("--expected_repeats", type=int, default=3)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    tasks = {row["blinded_output_id"]: row for row in (json.loads(line) for line in Path(args.task_path).read_text().splitlines() if line.strip())}
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for line in Path(args.judge_output_path).read_text().splitlines():
        if line.strip():
            row = json.loads(line); groups[str(row["blinded_output_id"])].append(row)
    results = []
    for output_id, task in tasks.items():
        repeats = groups.get(output_id, [])
        expected = task["expected_status_by_contract_id"]
        statuses = []
        valid = []
        routes = []
        for row in repeats:
            payload = row.get("judge_output") or {}
            alignment = {str(item.get("contract_id")): item.get("status") for item in payload.get("ledger_to_note_alignment", []) if isinstance(item, dict)}
            statuses.append(alignment)
            valid.append(bool(row.get("schema_valid")) and row.get("parse_error") is None)
            routes.append(payload.get("requires_human_review"))
        correct = len(repeats) == args.expected_repeats and all(status == expected for status in statuses)
        stable = len(repeats) == args.expected_repeats and len({json.dumps(status, sort_keys=True) for status in statuses}) == 1
        results.append({"case_id": task["case_id"], "repeat_count": len(repeats), "schema_valid": all(valid), "stable": stable, "exact_status_match": correct, "routes": routes})
    summary = {
        "scope": "fabricated_only",
        "n_cases": len(results),
        "schema_valid_rate": sum(row["schema_valid"] for row in results) / len(results),
        "stable_rate": sum(row["stable"] for row in results) / len(results),
        "exact_status_accuracy": sum(row["exact_status_match"] for row in results) / len(results),
        "release_pass": all(row["schema_valid"] and row["stable"] and row["exact_status_match"] for row in results),
        "cases": results,
    }
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "fabricated_contract_alignment_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
