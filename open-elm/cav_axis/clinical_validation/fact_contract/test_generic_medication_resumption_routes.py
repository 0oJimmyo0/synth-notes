#!/usr/bin/env python3
"""Regression test for generic active-discharge medication-resumption routes.

This fabricated-only check protects the deterministic route introduced after a
contract-alignment calibration miss. It contains no clinical source text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_contract_coverage import GENERIC_MEDICATION_RESUMPTION
from parse_note_sections import section_map


CASES = [
    ("resume_preadmission", "Discharge Instructions: Resume preadmission medications.", True),
    ("resume_home", "Discharge Medications: Resume all home medications.", True),
    ("continue_prior", "Discharge Instructions: Continue prior medications.", True),
    ("specific_regimen", "Discharge Medications: Continue metformin 500 mg twice daily.", False),
    ("historical_only", "Hospital Course: The patient resumed home medications on admission.\nDisposition: Home.", False),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    results = []
    for case_id, note, expected_route in CASES:
        sections = section_map(note)
        active_text = "\n".join(
            sections.get(name, "") for name in ("discharge_medications", "instructions")
        )
        match = GENERIC_MEDICATION_RESUMPTION.search(active_text)
        routed = match is not None
        results.append({
            "case_id": case_id,
            "expected_route": expected_route,
            "observed_route": routed,
            "matched_phrase": match.group(0) if match else "",
            "pass": routed == expected_route,
        })

    summary = {
        "scope": "fabricated_only",
        "n_cases": len(results),
        "n_passed": sum(row["pass"] for row in results),
        "release_pass": all(row["pass"] for row in results),
        "cases": results,
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generic_medication_resumption_route_regression.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
