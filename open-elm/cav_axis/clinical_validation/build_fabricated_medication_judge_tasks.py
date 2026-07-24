#!/usr/bin/env python3
"""Create fabricated-only tasks for the production compact MedGemma judge path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ledger(*facts: dict[str, str]) -> str:
    return json.dumps(list(facts), ensure_ascii=True)


def fact(fact_id: str, field: str, value: str) -> dict[str, str]:
    return {"fact_id": fact_id, "field": field, "value": value}


CASES = [
    (
        "fabricated_supported_regimen",
        ledger(fact("med_1", "discharge_medications", "acetaminophen 500 mg by mouth every 6 hours as needed.")),
        "Discharge medications: acetaminophen 500 mg by mouth every 6 hours as needed.",
        False,
    ),
    (
        "fabricated_omission",
        ledger(fact("med_1", "discharge_medications", "metformin 500 mg by mouth twice daily; lisinopril 10 mg by mouth daily.")),
        "Discharge medications: metformin 500 mg by mouth twice daily.",
        True,
    ),
    (
        "fabricated_action_contradiction",
        ledger(fact("inst_1", "instructions", "Stop warfarin at discharge because of active bleeding.")),
        "Instructions: continue warfarin at discharge.",
        True,
    ),
    (
        "fabricated_unsupported_addition",
        ledger(fact("med_1", "discharge_medications", "amoxicillin 500 mg by mouth three times daily for 5 days.")),
        "Discharge medications: amoxicillin 500 mg by mouth three times daily for 5 days; morphine 30 mg by mouth every 4 hours.",
        True,
    ),
    (
        "fabricated_no_medications",
        ledger(fact("med_1", "discharge_medications", "No discharge medications are prescribed.")),
        "Discharge medications: none.",
        False,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fabricated compact-judge smoke tasks.")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing task file: {output_path}")
    with output_path.open("w", encoding="utf-8") as handle:
        for case_id, verified_fact_ledger, synthetic_note, expected_final_reject in CASES:
            handle.write(json.dumps({
                "task_id": case_id,
                "blinded_output_id": case_id,
                "verified_fact_ledger": verified_fact_ledger,
                "synthetic_note": synthetic_note,
                "expected_final_reject": expected_final_reject,
                "scope": "fabricated_only_no_mimic_or_project_clinical_content",
            }, ensure_ascii=True) + "\n")
    print(json.dumps({"output_path": str(output_path), "n_tasks": len(CASES), "scope": "fabricated_only"}, indent=2))


if __name__ == "__main__":
    main()
