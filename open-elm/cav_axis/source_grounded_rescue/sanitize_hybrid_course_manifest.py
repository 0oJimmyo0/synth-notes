#!/usr/bin/env python3
"""Apply deterministic patient-neutral wording to an existing hybrid manifest.

This preserves generated clinical content while removing gendered pronouns that
violate the contract-first course style. The output retains each original
candidate and records whether its course changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_hybrid_contract_generation import (
    course_constraint_reasons,
    course_facts,
    neutralize_gendered_pronouns,
    render_note,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_manifest_path", required=True)
    parser.add_argument("--contract_path", required=True)
    parser.add_argument("--output_manifest_path", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    contracts = {row["case_id"]: row for row in read_jsonl(Path(args.contract_path).resolve())}
    rows = read_jsonl(Path(args.input_manifest_path).resolve())
    if not rows:
        raise ValueError("Input manifest is empty.")

    output = []
    for row in rows:
        case_id = str(row["case_id"])
        if case_id not in contracts:
            raise KeyError(f"Manifest case absent from contract: {case_id}")
        course = str(row["hospital_course_text"])
        sanitized = neutralize_gendered_pronouns(course)
        contract = contracts[case_id]
        reasons = course_constraint_reasons(sanitized, course_facts(contract))
        updated = dict(row)
        updated["hospital_course_text"] = sanitized
        updated["generated_text"] = render_note(contract, sanitized)
        updated["course_constraint_pass"] = not reasons
        updated["course_constraint_rejection_reasons"] = reasons
        updated["deterministic_gender_neutralization_applied"] = sanitized != course
        output.append(updated)

    output_path = Path(args.output_manifest_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row) + "\n")

    print(json.dumps({
        "n_rows": len(output),
        "n_cases": len({row["case_id"] for row in output}),
        "neutralized_rows": sum(row["deterministic_gender_neutralization_applied"] for row in output),
        "constraint_passing_rows": sum(row["course_constraint_pass"] for row in output),
    }, indent=2))


if __name__ == "__main__":
    main()
