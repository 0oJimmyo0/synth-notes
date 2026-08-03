#!/usr/bin/env python3
"""Build transition_note_contract_v1 JSONL from a reviewed source-fact ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from normalize_medications import medication_components, reviewer_components


REQUIRED_FIELDS = {
    "principal_diagnosis", "hospital_course_events", "discharge_medications", "disposition", "instructions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ledger_review_csv")
    source.add_argument("--reviewed_contract_csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--optional_fields", default="follow_up")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.reviewed_contract_csv or args.ledger_review_csv
    frame = pd.read_csv(Path(source_path).resolve())
    is_reviewed_contract = args.reviewed_contract_csv is not None
    required_columns = (
        {"case_id", "fact_id", "field", "contract_section", "contract_status", "contract_generation_value", "reviewer_decision"}
        if is_reviewed_contract else {"case_id", "fact_id", "field", "generation_value", "manual_verification_status"}
    )
    if missing := required_columns.difference(frame.columns):
        raise KeyError(f"contract source missing columns: {sorted(missing)}")
    optional = {value.strip() for value in args.optional_fields.split(",") if value.strip()}
    if is_reviewed_contract:
        frame["reviewer_decision"] = frame.reviewer_decision.fillna("").astype(str).str.lower().str.strip()
        pending = frame.reviewer_decision.eq("pending")
        if pending.any():
            raise ValueError(f"{int(pending.sum())} contract-review rows are still pending")
        usable = frame.loc[frame.reviewer_decision.isin({"include", "required", "optional", "historical_context_only"})].copy()
        usable["generation_value"] = usable.contract_generation_value.fillna("").astype(str).str.strip()
        usable["status"] = usable.contract_status.fillna("").astype(str).str.strip()
        usable["section"] = usable.contract_section.fillna("").astype(str).str.strip()
    else:
        frame["status"] = frame.manual_verification_status.fillna("").astype(str).str.lower().str.strip()
        usable = frame.loc[frame.status.isin({"verified", "corrected"})].copy()
        usable["generation_value"] = usable.generation_value.fillna("").astype(str).str.strip()
        usable["section"] = usable.field.astype(str)
    usable = usable.loc[usable.generation_value != ""]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contracts = []
    for case_id, group in usable.groupby("case_id", sort=True):
        facts = []
        for row in group.sort_values(["field", "fact_id"], kind="stable").itertuples(index=False):
            field = str(row.field)
            status = str(row.status) if is_reviewed_contract else ("required" if field in REQUIRED_FIELDS else "optional")
            if not is_reviewed_contract and field not in REQUIRED_FIELDS and field not in optional:
                status = "historical_context_only"
            values = [str(row.generation_value)]
            reviewer_components_present = bool(
                is_reviewed_contract and str(getattr(row, "required_medication_components", "")).strip()
                and str(getattr(row, "required_medication_components", "")).strip().lower() != "nan"
            )
            if field == "discharge_medications" and not reviewer_components_present:
                # Each semicolon-delimited regimen item becomes independently auditable.
                values = [item.strip() for item in str(row.generation_value).split(";") if item.strip()]
            for value_index, value in enumerate(values, start=1):
                fact = {
                    "fact_id": str(row.fact_id) if len(values) == 1 else f"{row.fact_id}.med{value_index:03d}",
                    "field": field, "section": str(row.section), "status": status, "generation_value": value,
                }
                if field == "discharge_medications":
                    reviewer_value = getattr(row, "required_medication_components", "") if is_reviewed_contract else ""
                    fact["medication_components"] = reviewer_components(reviewer_value, value) if is_reviewed_contract else medication_components(value)
                facts.append(fact)
        # Hospital-course evidence is intentionally labeled `include`: ELM may
        # use it as constrained narrative context, but it is not rendered as a
        # deterministic transition section. All other required fields must use
        # the stricter `required` status.
        present_required = {
            fact["field"]
            for fact in facts
            if fact["status"] == "required"
            or (fact["field"] == "hospital_course_events" and fact["status"] == "include")
        }
        contracts.append({
            "case_id": str(case_id), "contract_version": "transition_note_contract_v1",
            "ready_for_hybrid_generation": REQUIRED_FIELDS.issubset(present_required), "facts": facts,
        })
    with (output_dir / "transition_note_contract_v1.jsonl").open("w", encoding="utf-8") as handle:
        for contract in contracts:
            handle.write(json.dumps(contract) + "\n")
    summary = {
        "n_contracts": len(contracts),
        "n_ready_for_hybrid_generation": sum(item["ready_for_hybrid_generation"] for item in contracts),
        "contract_version": "transition_note_contract_v1",
    }
    (output_dir / "transition_note_contract_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
