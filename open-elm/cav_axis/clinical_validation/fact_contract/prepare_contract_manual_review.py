#!/usr/bin/env python3
"""Create a clinician-augmentable critical-fact contract review workbook CSV.

This makes explicit discharge obligations discoverable in hospital-course evidence
instead of assuming every critical treatment is already in a medication-list row.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_FIELDS = {
    "principal_diagnosis", "hospital_course_events", "discharge_medications", "disposition", "instructions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger_review_csv", required=True)
    parser.add_argument("--case_readiness_path", default=None,
                        help="Optional validated ledger readiness CSV; retain only generation-ready cases.")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--optional_fields", default="follow_up")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ledger = pd.read_csv(Path(args.ledger_review_csv).resolve())
    # Preserve clinician-authored compact wording from reviewed exports that
    # use this column name instead of the generation-ledger convention.
    if "generation_value" not in ledger.columns and "effective_reviewed_value" in ledger.columns:
        ledger["generation_value"] = ledger["effective_reviewed_value"]
    needed = {"case_id", "fact_id", "field", "generation_value", "manual_verification_status"}
    if missing := needed.difference(ledger.columns):
        raise KeyError(f"ledger review missing columns: {sorted(missing)}")
    optional = {item.strip() for item in args.optional_fields.split(",") if item.strip()}
    ledger["manual_verification_status"] = ledger.manual_verification_status.fillna("").astype(str).str.lower().str.strip()
    if args.case_readiness_path:
        readiness = pd.read_csv(Path(args.case_readiness_path).resolve())
        needed_readiness = {"case_id", "ledger_ready_for_generation"}
        if missing := needed_readiness.difference(readiness.columns):
            raise KeyError(f"case readiness missing columns: {sorted(missing)}")
        ready_ids = set(readiness.loc[
            readiness.ledger_ready_for_generation.fillna(False).astype(bool), "case_id"
        ].astype(str))
        ledger = ledger.loc[ledger.case_id.astype(str).isin(ready_ids)].copy()
    ledger = ledger.loc[ledger.manual_verification_status.isin({"verified", "corrected"})].copy()
    ledger["generation_value"] = ledger.generation_value.fillna("").astype(str).str.strip()
    ledger = ledger.loc[ledger.generation_value != ""].copy()
    ledger["contract_section"] = ledger.field
    ledger["contract_status"] = ledger.field.map(lambda field: "required" if field in REQUIRED_FIELDS else "optional")
    ledger.loc[~ledger.field.isin(REQUIRED_FIELDS.union(optional)), "contract_status"] = "historical_context_only"
    ledger["contract_generation_value"] = ledger.generation_value
    ledger["must_appear_in_section"] = ledger.contract_status.eq("required")
    ledger["reviewer_decision"] = "pending"
    ledger["reviewer_note"] = ""
    ledger["critical_discharge_obligation_to_add"] = ""
    ledger["required_medication_components"] = ""
    columns = [
        "case_id", "fact_id", "field", "contract_section", "contract_status",
        "contract_generation_value", "must_appear_in_section", "required_medication_components",
        "manual_verification_status", "reviewer_decision", "critical_discharge_obligation_to_add", "reviewer_note",
    ]
    output = Path(args.output_csv).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger[columns].to_csv(output, index=False)
    print({"n_rows": len(ledger), "n_cases": ledger.case_id.nunique(), "output_csv": str(output)})


if __name__ == "__main__":
    main()
