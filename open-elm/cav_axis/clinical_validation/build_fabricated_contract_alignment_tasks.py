#!/usr/bin/env python3
"""Create fabricated-only contract-alignment tasks for local LLM validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def item(contract_id: str, field: str, value: str) -> dict[str, str]:
    return {"fact_id": contract_id, "field": field, "generation_value": value}


BASE_CASES = [
    ("active_present", [item("c1", "active_discharge_obligations", "metformin 500 mg by mouth twice daily")], "Discharge Medications: metformin 500 mg by mouth twice daily.", {"c1": "present_supported"}),
    ("active_missing", [item("c1", "active_discharge_obligations", "metformin 500 mg by mouth twice daily")], "Discharge Medications: none.", {"c1": "missing"}),
    ("active_unsupported", [item("c1", "active_discharge_obligations", "amoxicillin 500 mg by mouth three times daily")], "Discharge Medications: amoxicillin 500 mg by mouth three times daily; morphine 30 mg by mouth every 4 hours.", {"c1": "unsupported"}),
    ("historical_context", [item("c1", "historical_context_only", "inpatient olanzapine 2.5 mg for one sundowning episode")], "Hospital Course: one sundowning episode was treated with olanzapine 2.5 mg.", {"c1": "present_supported"}),
    ("unknown_component", [item("c1", "unknown_components", "lisinopril active at discharge; dose not specified")], "Discharge Medications: lisinopril 10 mg by mouth daily.", {"c1": "uncertain"}),
    ("conditional_hold", [item("c1", "active_discharge_obligations", "hold lisinopril; restart only when systolic blood pressure exceeds 130 mm Hg")], "Instructions: hold lisinopril and restart only if systolic blood pressure is above 130 mm Hg.", {"c1": "present_supported"}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--negative_obligation", action="store_true", help="Add an explicit no-extra-medications contract item for v2 testing.")
    args = parser.parse_args()
    cases = list(BASE_CASES)
    if args.negative_obligation:
        cases = [
            (
                case_id,
                contract + [item("c2", "negative_discharge_constraints", "no additional active discharge medications beyond the listed obligations")],
                note,
                {"c1": "present_supported", "c2": "unsupported"} if case_id == "active_unsupported" else {**expected, "c2": "present_supported"},
            )
            for case_id, contract, note, expected in cases
        ]
    rows = []
    for case_id, contract, note, expected in cases:
        rows.append({
            "task_id": f"fabricated_contract_alignment::{case_id}",
            "blinded_output_id": f"fabricated_contract_alignment_{case_id}",
            "case_id": case_id,
            "verified_fact_ledger": json.dumps(contract, ensure_ascii=True),
            "synthetic_note": note,
            "expected_status_by_contract_id": expected,
        })
    output = Path(args.output_path).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"scope": "fabricated_only", "n_tasks": len(rows), "output_path": str(output)}, indent=2))


if __name__ == "__main__":
    main()
