#!/usr/bin/env python3
"""Write fabricated-only regression tasks for the versioned medication judge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ledger(*facts: tuple[str, str, str]) -> str:
    return json.dumps([
        {"fact_id": fact_id, "field": field, "generation_value": value}
        for fact_id, field, value in facts
    ], ensure_ascii=True)


CASES = [
    ("supported_discharge", False, ledger(
        ("f1", "discharge_medications", "acetaminophen 500 mg by mouth every 6 hours as needed"),
    ), "Discharge Medications: acetaminophen 500 mg by mouth every 6 hours as needed."),
    ("unknown_component_not_omission", False, ledger(
        ("f1", "discharge_medications", "continue intravenous antibiotic; identity, dose, frequency, and end date not specified"),
    ), "Discharge Medications: continue the intravenous antibiotic as prescribed."),
    ("historical_inpatient_not_discharge", False, ledger(
        ("f1", "hospital_course_events", "one inpatient episode of sundowning treated with olanzapine 2.5 mg"),
        ("f2", "discharge_medications", "metformin 500 mg by mouth twice daily"),
    ), "Hospital Course: one episode of sundowning was treated with olanzapine 2.5 mg. Discharge Medications: metformin 500 mg by mouth twice daily."),
    ("explicit_active_omission", True, ledger(
        ("f1", "discharge_medications", "metformin 500 mg by mouth twice daily"),
        ("f2", "discharge_medications", "lisinopril 10 mg by mouth daily"),
    ), "Discharge Medications: metformin 500 mg by mouth twice daily."),
    ("unsupported_active_addition", True, ledger(
        ("f1", "discharge_medications", "amoxicillin 500 mg by mouth three times daily for 5 days"),
    ), "Discharge Medications: amoxicillin 500 mg by mouth three times daily for 5 days; morphine 30 mg by mouth every 4 hours."),
    ("unknown_active_component_requires_review", False, ledger(
        ("f1", "discharge_medications", "lisinopril active at discharge; dose not specified"),
    ), "Discharge Medications: lisinopril 10 mg by mouth daily."),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fabricated-only v3 medication-judge tasks.")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    rows = [
        {
            "task_id": f"fabricated_v3::{case_id}",
            "blinded_output_id": f"fabricated_v3_{case_id}",
            "case_id": case_id,
            "verified_fact_ledger": evidence,
            "synthetic_note": note,
            "expected_final_reject": expected_reject,
        }
        for case_id, expected_reject, evidence, note in CASES
    ]
    path = Path(args.output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"scope": "fabricated_only", "n_tasks": len(rows), "output_path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
