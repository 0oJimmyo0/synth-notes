#!/usr/bin/env python3
"""Re-evaluate saved hybrid course prose against the current versioned guard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--contract_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_hybrid_contract_generation import course_constraint_reasons, course_facts

    contracts = {str(row["case_id"]): row for row in read_jsonl(Path(args.contract_path).resolve())}
    rows = []
    for row in read_jsonl(Path(args.manifest_path).resolve()):
        case_id = str(row["case_id"])
        if case_id not in contracts:
            raise KeyError(f"No contract for generated case: {case_id}")
        reasons = course_constraint_reasons(str(row["hospital_course_text"]), course_facts(contracts[case_id]))
        rows.append({
            "rescue_id": str(row["rescue_id"]),
            "case_id": case_id,
            "candidate_index": int(row["candidate_index"]),
            "course_constraint_pass": not reasons,
            "course_constraint_rejection_reasons": reasons,
        })
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "hybrid_course_constraint_audit.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "n_outputs": len(rows),
        "pass_count": sum(row["course_constraint_pass"] for row in rows),
        "failure_count": sum(not row["course_constraint_pass"] for row in rows),
        "failure_reason_counts": {
            reason: sum(reason in row["course_constraint_rejection_reasons"] for row in rows)
            for reason in sorted({reason for row in rows for reason in row["course_constraint_rejection_reasons"]})
        },
        "security_note": "Outputs contain candidate IDs and derived guard results only; no synthetic text.",
    }
    (output_dir / "hybrid_course_constraint_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
