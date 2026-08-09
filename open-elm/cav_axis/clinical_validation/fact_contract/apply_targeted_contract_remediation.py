#!/usr/bin/env python3
"""Apply source-reviewed atomic remediation obligations to a reviewed contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RENDERED_SECTIONS = {
    "principal_diagnosis",
    "hospital_course_events",
    "discharge_medications",
    "disposition",
    "instructions",
    "follow_up",
}
REPAIR_ACTIONS = {
    "add_required_atomic",
    "replace_existing_required_with_complete_atomic_transition",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_reviewed_contract_csv", required=True)
    parser.add_argument("--remediation_atomic_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_path", required=True)
    return parser.parse_args()


def normalize_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    args = parse_args()
    base = pd.read_csv(Path(args.base_reviewed_contract_csv).resolve(), dtype=str).fillna("")
    remediation = pd.read_csv(Path(args.remediation_atomic_csv).resolve(), dtype=str).fillna("")
    required_base = {
        "case_id", "fact_id", "field", "contract_section", "contract_status",
        "contract_generation_value", "reviewer_decision", "case_excluded",
        "case_exclusion_reason",
    }
    required_remediation = {
        "case_id", "source_fact_id", "remediation_action", "contract_section",
        "contract_status", "contract_generation_value", "required_medication_components",
        "case_excluded_yes_no", "exclusion_reason",
    }
    if missing := required_base.difference(base.columns):
        raise KeyError(f"Base contract missing columns: {sorted(missing)}")
    if missing := required_remediation.difference(remediation.columns):
        raise KeyError(f"Remediation file missing columns: {sorted(missing)}")

    remediation["case_excluded"] = remediation.case_excluded_yes_no.map(normalize_bool)
    remediation_cases = set(remediation.case_id.astype(str))
    if not remediation_cases.issubset(set(base.case_id.astype(str))):
        raise ValueError("Remediation includes cases absent from the base reviewed contract")

    excluded = remediation.loc[remediation.case_excluded].copy()
    repaired = remediation.loc[~remediation.case_excluded].copy()
    if set(excluded.remediation_action).difference({"exclude_case"}):
        raise ValueError("Excluded remediation rows must use remediation_action=exclude_case")
    if set(repaired.remediation_action).difference(REPAIR_ACTIONS):
        raise ValueError("Unexpected remediation action for repairable case")
    if excluded.groupby("case_id").exclusion_reason.agg(lambda values: all(str(value).strip() for value in values)).eq(False).any():
        raise ValueError("Each excluded case needs an exclusion_reason")
    if repaired.empty:
        raise ValueError("No repairable atomic obligations were supplied")
    if repaired.contract_status.str.strip().str.lower().ne("required").any():
        raise ValueError("Repairable remediation rows must be required")
    if repaired.contract_section.str.strip().isin(RENDERED_SECTIONS).eq(False).any():
        raise ValueError("Repairable remediation rows must use a rendered section")
    if repaired.contract_generation_value.str.strip().eq("").any():
        raise ValueError("Repairable remediation rows need contract_generation_value")

    # Retain all reviewed base facts. The atomic rows are additive: their source
    # parent may be a broad course or instruction narrative containing other
    # obligations that cannot safely be deleted as part of a targeted repair.
    merged = base.copy()
    exclusion_reasons = excluded.groupby("case_id").exclusion_reason.first().to_dict()
    excluded_cases = set(exclusion_reasons)
    merged.loc[merged.case_id.astype(str).isin(excluded_cases), "case_excluded"] = "True"
    merged.loc[merged.case_id.astype(str).isin(excluded_cases), "case_exclusion_reason"] = merged.loc[
        merged.case_id.astype(str).isin(excluded_cases), "case_id"
    ].map(exclusion_reasons)

    additions = []
    for index, row in enumerate(repaired.itertuples(index=False), start=1):
        addition = {column: "" for column in merged.columns}
        section = str(row.contract_section).strip()
        case_id = str(row.case_id)
        addition.update({
            "case_id": case_id,
            "fact_id": f"{row.source_fact_id}__v4remediation_{index:03d}",
            "field": section,
            "contract_section": section,
            "contract_status": "required",
            "contract_generation_value": str(row.contract_generation_value).strip(),
            "must_appear_in_section": "True",
            "required_medication_components": str(row.required_medication_components).strip(),
            "manual_verification_status": "corrected",
            "reviewer_decision": "required",
            "critical_discharge_obligation_to_add": "",
            "reviewer_note": "Targeted V4 remediation obligation from source-reviewed evidence.",
            "case_excluded": "False",
            "case_exclusion_reason": "",
        })
        if "source_parent_fact_id" in addition:
            addition["source_parent_fact_id"] = str(row.source_fact_id)
        if "atomic_requirement_type" in addition:
            addition["atomic_requirement_type"] = "targeted_v4_remediation"
        additions.append(addition)
    merged = pd.concat([merged, pd.DataFrame(additions, columns=merged.columns)], ignore_index=True)

    output = Path(args.output_csv).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    summary = {
        "base_rows": int(len(base)),
        "remediation_rows": int(len(remediation)),
        "repairable_cases": int(repaired.case_id.nunique()),
        "excluded_cases": int(len(excluded_cases)),
        "atomic_required_rows_added": int(len(additions)),
        "merged_rows": int(len(merged)),
        "security_note": "Output contains source-derived reviewed contract values and must remain on approved project storage.",
    }
    summary_path = Path(args.summary_path).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
