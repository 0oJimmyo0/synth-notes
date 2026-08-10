#!/usr/bin/env python3
"""Build a conservative draft transition contract from a validated source ledger.

This is a deterministic triage tool, not a source-review replacement or a
clinical decision system.  It emits fully automated draft contracts only when
no generic safety-review rule fires; all other cases are routed to manual
review rather than reconstructed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


FACT_CONTRACT_DIR = Path(__file__).resolve().parent / "fact_contract"
if str(FACT_CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(FACT_CONTRACT_DIR))
from normalize_medications import medication_components


VERSION = "contract_automation_v1_1"
REQUIRED_FIELDS = {
    "principal_diagnosis",
    "hospital_course_events",
    "discharge_medications",
    "disposition",
    "instructions",
}
RENDERED_SECTIONS = REQUIRED_FIELDS | {"follow_up"}
HIGH_RISK = re.compile(
    r"\b(?:"
    r"insulin|sliding scale|warfarin|coumadin|heparin|enoxaparin|apixaban|"
    r"rivaroxaban|dabigatran|anticoag|tacrolimus|cyclosporine|chemotherapy|"
    r"methotrexate|opioid|oxycodone|hydromorphone|morphine|fentanyl|"
    r"dialysis|vancomycin|aminoglycoside|steroid taper|predni(?:sone|solone)"
    r")\b",
    flags=re.IGNORECASE,
)
UNKNOWN = re.compile(r"\b(?:not specified|unknown|unavailable|redacted)\b|_{3,}", re.IGNORECASE)
TRUNCATED = re.compile(
    r"(?:\.\.\.|\b(?:and the|one of the|any of the|following|as directed)\s*$)",
    re.IGNORECASE,
)
ANTIMICROBIAL = re.compile(
    r"\b(?:"
    r"amoxicillin|ampicillin|azithromycin|cephalexin|ceftriaxone|ciprofloxacin|"
    r"clindamycin|doxycycline|ertapenem|fluconazole|levofloxacin|linezolid|"
    r"metronidazole|nitrofurantoin|penicillin|piperacillin|rifampin|"
    r"trimethoprim|sulfamethoxazole|vancomycin"
    r")\b",
    flags=re.IGNORECASE,
)
COURSE_ENDPOINT = re.compile(
    r"\b(?:for|x)\s+\d+\s*(?:day|days|week|weeks)|\buntil\b|\btotal\s+of\s+\d+",
    flags=re.IGNORECASE,
)
MEDICATION_LINE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger_review_csv", required=True)
    parser.add_argument("--case_manifest_csv", required=True)
    parser.add_argument("--expected_automation_split", default="development")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def clean(value: object) -> str:
    return " ".join(str(value or "").replace("\r", "").split())


def value_is_unknown(value: str) -> bool:
    return bool(UNKNOWN.search(value))


def split_medications(value: str) -> list[str]:
    """Split only explicit list separators; never infer medication boundaries."""
    lines = [MEDICATION_LINE.sub("", line).strip() for line in value.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) > 1:
        return lines
    return [item.strip() for item in value.split(";") if item.strip()] or [value]


def route_for_row(row: pd.Series) -> tuple[str, str, str]:
    """Return contract status, section, and deterministic routing rule."""
    field = clean(row["field"])
    recovered = clean(row.get("recovered_transition", "")).lower() == "true"
    transition_type = clean(row.get("transition_type", ""))
    if recovered and transition_type in RENDERED_SECTIONS:
        return "required", transition_type, "RECOVERED_TRANSITION_ROUTE"
    if field == "hospital_course_events":
        return "include", field, "COURSE_CONTEXT_ONLY"
    if field in {"principal_diagnosis", "discharge_medications", "disposition", "instructions"}:
        return "required", field, "CORE_TRANSITION_ROUTE"
    if field == "follow_up":
        return "optional", field, "DIRECT_FOLLOW_UP_OPTIONAL"
    if field == "procedures_this_admission":
        return "include", "hospital_course_events", "PROCEDURE_CONTEXT_ONLY"
    return "historical_context_only", field, "HISTORICAL_CONTEXT_ROUTE"


def case_review_reasons(group: pd.DataFrame) -> list[str]:
    reasons: list[str] = []
    fields = set(group["field"].astype(str))
    missing = sorted(REQUIRED_FIELDS - fields)
    if missing:
        return ["MISSING_REQUIRED_FIELD_" + "_".join(missing).upper()]
    for row in group.itertuples(index=False):
        field = clean(getattr(row, "field"))
        value = clean(getattr(row, "generation_value"))
        recovered = clean(getattr(row, "recovered_transition", "")).lower() == "true"
        if recovered:
            reasons.append("RECOVERED_TRANSITION_REQUIRES_REVIEW")
        if field in RENDERED_SECTIONS and TRUNCATED.search(value):
            reasons.append("POSSIBLE_TRUNCATED_TRANSITION")
        if field == "discharge_medications" and value_is_unknown(value):
            reasons.append("UNKNOWN_DISCHARGE_MEDICATION_COMPONENT")
        if field == "discharge_medications":
            if len(split_medications(value)) > 1:
                reasons.append("COMPOUND_MEDICATION_LIST_REQUIRES_REVIEW")
            for regimen in split_medications(value):
                if ANTIMICROBIAL.search(regimen) and not COURSE_ENDPOINT.search(regimen):
                    reasons.append("ANTIMICROBIAL_WITHOUT_COURSE_ENDPOINT")
        if field in {"instructions", "follow_up"} and (
            len(value) > 280 or len(value.split(";")) > 1 or "\n" in str(getattr(row, "generation_value"))
        ):
            reasons.append("COMPOUND_ACTIONABLE_TEXT_REQUIRES_REVIEW")
        if field in {"discharge_medications", "instructions", "follow_up"} and HIGH_RISK.search(value):
            reasons.append("HIGH_RISK_TRANSITION_REQUIRES_REVIEW")
    return sorted(set(reasons))


def medication_component_string(value: str) -> str:
    components = medication_components(value)
    keys = (
        ("identity", "name"),
        ("action", "action"),
        ("dose", "dose"),
        ("route", "route"),
        ("frequency_or_timing", "frequency_or_timing"),
    )
    rendered = [f"{target}={components.get(source) or 'not specified'}" for target, source in keys]
    rendered.extend(("duration=not specified", "condition=not specified"))
    return "|".join(rendered)


def output_rows(group: pd.DataFrame, disposition: str, reasons: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in group.sort_values(["field", "fact_id"], kind="stable").to_dict("records"):
        status, section, rule = route_for_row(pd.Series(row))
        value = clean(row["generation_value"])
        values = split_medications(value) if row["field"] == "discharge_medications" else [value]
        for index, item in enumerate(values, start=1):
            fact_id = str(row["fact_id"])
            if len(values) > 1:
                fact_id = f"{fact_id}.auto_med{index:03d}"
            rows.append({
                "case_id": row["case_id"],
                "fact_id": fact_id,
                "field": row["field"],
                "contract_section": section,
                "contract_status": status,
                "contract_generation_value": item,
                "required_medication_components": (
                    medication_component_string(item)
                    if row["field"] == "discharge_medications" and status == "required"
                    else ""
                ),
                "automation_decision": disposition,
                "automation_rule_id": rule,
                "automation_reason": "|".join(reasons),
                "source_fact_id": row.get("source_fact_id", ""),
                "atomic_parent_fact_id": row["fact_id"] if len(values) > 1 else "",
                "atomic_sequence": index if len(values) > 1 else "",
                "automation_version": VERSION,
            })
    return rows


def main() -> None:
    args = parse_args()
    ledger = pd.read_csv(Path(args.ledger_review_csv).resolve(), dtype=str).fillna("")
    manifest = pd.read_csv(Path(args.case_manifest_csv).resolve(), dtype=str).fillna("")
    required_ledger = {"case_id", "fact_id", "field", "generation_value", "manual_verification_status"}
    required_manifest = {"case_id", "automation_split"}
    if missing := required_ledger - set(ledger.columns):
        raise KeyError(f"ledger is missing columns: {sorted(missing)}")
    if missing := required_manifest - set(manifest.columns):
        raise KeyError(f"manifest is missing columns: {sorted(missing)}")

    selected = set(manifest.loc[
        manifest["automation_split"].eq(args.expected_automation_split), "case_id"
    ])
    if not selected:
        raise ValueError("No cases matched expected automation split")
    ledger = ledger.loc[ledger["case_id"].isin(selected)].copy()
    ledger["manual_verification_status"] = ledger["manual_verification_status"].str.lower().str.strip()
    ledger = ledger.loc[ledger["manual_verification_status"].isin({"verified", "corrected"})].copy()
    ledger["generation_value"] = ledger["generation_value"].map(clean)
    ledger = ledger.loc[ledger["generation_value"].ne("")].copy()
    found = set(ledger["case_id"])
    if found != selected:
        raise ValueError(f"Ledger/manifest case mismatch: missing={sorted(selected - found)[:10]}")

    automated_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    for case_id, group in ledger.groupby("case_id", sort=True):
        reasons = case_review_reasons(group)
        decision = "fully_automated" if not reasons else "route_to_manual_review"
        automated_rows.extend(output_rows(group, decision, reasons))
        case_rows.append({
            "case_id": case_id,
            "automation_decision": decision,
            "automation_reason": "|".join(reasons),
            "n_input_facts": len(group),
            "n_output_rows": 0,
            "automation_version": VERSION,
        })
    output = pd.DataFrame(automated_rows)
    case_summary = pd.DataFrame(case_rows)
    row_counts = output.groupby("case_id").size().rename("n_output_rows")
    case_summary["n_output_rows"] = case_summary["case_id"].map(row_counts).fillna(0).astype(int)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "automation_contract_candidates.csv", index=False)
    case_summary.to_csv(output_dir / "automation_case_summary.csv", index=False)
    config = {
        "automation_version": VERSION,
        "required_fields": sorted(REQUIRED_FIELDS),
        "high_risk_pattern": HIGH_RISK.pattern,
        "unknown_pattern": UNKNOWN.pattern,
        "truncated_pattern": TRUNCATED.pattern,
        "safety_policy": "Uncertain transition cases route to manual review; no reconstruction.",
    }
    (output_dir / "automation_config.json").write_text(json.dumps(config, indent=2) + "\n")
    summary = {
        "automation_version": VERSION,
        "n_cases": int(len(case_summary)),
        "case_decisions": case_summary["automation_decision"].value_counts().to_dict(),
        "n_output_rows": int(len(output)),
        "security_note": "Outputs contain source-derived contract candidates and remain on approved project storage.",
    }
    (output_dir / "automation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
