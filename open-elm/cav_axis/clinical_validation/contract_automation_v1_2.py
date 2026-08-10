#!/usr/bin/env python3
"""Structured, conservative source-ledger-to-contract automation v1.2.

Only explicit source fragments are emitted. Compound or uncertain transition
content is routed to manual review rather than completed by inference.
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


VERSION = "contract_automation_v1_2"
REQUIRED_FIELDS = {"principal_diagnosis", "hospital_course_events", "discharge_medications", "disposition", "instructions"}
RENDERED_SECTIONS = REQUIRED_FIELDS | {"follow_up"}
NUMBERED_ITEM = re.compile(r"(?<!\d)(?:^|\s)\d{1,2}[.)]\s+(?=[A-Za-z])")
UNKNOWN = re.compile(r"\b(?:not specified|unknown|unavailable|redacted)\b|_{3,}", re.I)
# A transition may be truncated without an ellipsis when extraction ends after a
# connector or possessive, e.g. "return to the ED or notify your".
TRUNCATED = re.compile(
    r"(?:\.\.\.|\b(?:and the|one of the|any of the|following|as directed|"
    r"and|or|to|with|for|of|the|a|an|your|their|his|her)\s*$)",
    re.I,
)
HIGH_RISK = re.compile(r"\b(?:insulin|warfarin|coumadin|heparin|enoxaparin|apixaban|rivaroxaban|dabigatran|anticoag|tacrolimus|cyclosporine|chemotherapy|opioid|oxycodone|hydromorphone|morphine|fentanyl|dialysis|vancomycin|steroid taper)\b", re.I)
ANTIMICROBIAL = re.compile(r"\b(?:amoxicillin|azithromycin|cephalexin|ceftriaxone|ciprofloxacin|clindamycin|doxycycline|ertapenem|levofloxacin|metronidazole|nitrofurantoin|penicillin|trimethoprim|sulfamethoxazole|vancomycin)\b", re.I)
COURSE = re.compile(r"\b(?:for|x)\s+\d+\s*(?:day|days|week|weeks)|\buntil\b|\btotal\s+of\s+\d+", re.I)
ACTION_START = re.compile(r"\b(?:follow up|see|call|return|continue|stop|hold|avoid|monitor|take|use|keep|wear|weight bear|schedule|obtain|check)\b", re.I)


def clean(value: object) -> str:
    return " ".join(str(value or "").replace("\r", "").split())


def parse_numbered_medications(value: str) -> list[tuple[str, int, int]]:
    """Return explicit numbered items with relative source spans.

    A list marker must be followed by a letter, so a decimal such as ``2.5 mg``
    cannot become a false item boundary.
    """
    matches = list(NUMBERED_ITEM.finditer(value))
    if not matches:
        return [(value.strip(), 0, len(value))] if value.strip() else []
    items = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        text = value[start:end].strip()
        if text:
            offset = value.find(text, start, end)
            items.append((text, offset, offset + len(text)))
    return items


def parse_action_clauses(value: str) -> tuple[list[tuple[str, int, int]], bool]:
    """Split only sentence/semicolon boundaries and explicit ``and <action>``."""
    boundaries = re.compile(r"(?<=[.!?;])\s+|\s+and\s+(?=(?:follow up|see|call|return|continue|stop|hold|avoid|monitor|take|use|keep|wear|weight bear|schedule|obtain|check)\b)", re.I)
    clauses, cursor = [], 0
    for piece in boundaries.split(value):
        text = piece.strip()
        if not text:
            continue
        start = value.find(text, cursor)
        cursor = start + len(text)
        clauses.append((text, start, cursor))
    actionable = [item for item in clauses if ACTION_START.search(item[0])]
    unresolved = bool(ACTION_START.search(value)) and not actionable
    return actionable or clauses, unresolved


def medication_component_string(value: str) -> str:
    component = medication_components(value)
    pairs = [("identity", "name"), ("action", "action"), ("dose", "dose"), ("route", "route"), ("frequency_or_timing", "frequency_or_timing")]
    return "|".join([f"{target}={component.get(source) or 'not specified'}" for target, source in pairs] + ["duration=not specified", "condition=not specified"])


def route_row(row: dict[str, str]) -> tuple[str, str, str]:
    field = clean(row["field"])
    if clean(row.get("recovered_transition", "")).lower() == "true" and clean(row.get("transition_type", "")) in RENDERED_SECTIONS:
        return "required", clean(row["transition_type"]), "RECOVERED_TRANSITION_ROUTE"
    if field == "hospital_course_events": return "include", field, "COURSE_CONTEXT_ONLY"
    if field in {"principal_diagnosis", "discharge_medications", "disposition", "instructions"}: return "required", field, "CORE_TRANSITION_ROUTE"
    if field == "follow_up": return "optional", field, "DIRECT_FOLLOW_UP_OPTIONAL"
    if field == "procedures_this_admission": return "include", "hospital_course_events", "PROCEDURE_CONTEXT_ONLY"
    return "historical_context_only", field, "HISTORICAL_CONTEXT_ROUTE"


def case_reasons(group: pd.DataFrame) -> list[str]:
    reasons = []
    fields = set(group.field)
    if missing := REQUIRED_FIELDS - fields:
        return ["MISSING_REQUIRED_FIELD_" + "_".join(sorted(missing)).upper()]
    for row in group.to_dict("records"):
        field, value = clean(row["field"]), str(row["generation_value"])
        if field in RENDERED_SECTIONS and TRUNCATED.search(clean(value)): reasons.append("POSSIBLE_TRUNCATED_TRANSITION")
        if field == "discharge_medications":
            items = parse_numbered_medications(value)
            if UNKNOWN.search(value): reasons.append("UNKNOWN_DISCHARGE_MEDICATION_COMPONENT")
            for item, _, _ in items:
                if ANTIMICROBIAL.search(item) and not COURSE.search(item): reasons.append("ANTIMICROBIAL_WITHOUT_COURSE_ENDPOINT")
                if HIGH_RISK.search(item) and any(not medication_components(item).get(key) for key in ("name", "dose", "frequency_or_timing")):
                    reasons.append("INCOMPLETE_HIGH_RISK_MEDICATION_COMPONENT")
        if field in {"instructions", "follow_up"}:
            _, unresolved = parse_action_clauses(value)
            if unresolved: reasons.append("UNRESOLVED_ACTIONABLE_FRAGMENT")
    return sorted(set(reasons))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger_review_csv", required=True)
    parser.add_argument("--case_manifest_csv", required=True)
    parser.add_argument("--expected_automation_split", default="development")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ledger = pd.read_csv(Path(args.ledger_review_csv).resolve(), dtype=str).fillna("")
    manifest = pd.read_csv(Path(args.case_manifest_csv).resolve(), dtype=str).fillna("")
    needed = {"case_id", "fact_id", "field", "generation_value", "manual_verification_status"}
    if missing := needed - set(ledger.columns): raise KeyError(f"ledger is missing columns: {sorted(missing)}")
    if {"case_id", "automation_split"} - set(manifest.columns): raise KeyError("manifest must contain case_id and automation_split")
    selected = set(manifest.loc[manifest.automation_split.eq(args.expected_automation_split), "case_id"])
    ledger = ledger.loc[ledger.case_id.isin(selected) & ledger.manual_verification_status.str.lower().isin({"verified", "corrected"})].copy()
    ledger = ledger.loc[ledger.generation_value.map(clean).ne("")].copy()
    if set(ledger.case_id) != selected: raise ValueError("ledger and selected manifest cases differ")

    outputs, cases = [], []
    for case_id, group in ledger.groupby("case_id", sort=True):
        reasons = case_reasons(group)
        decision = "fully_automated" if not reasons else "route_to_manual_review"
        for row in group.sort_values(["field", "fact_id"], kind="stable").to_dict("records"):
            status, section, rule = route_row(row)
            values = parse_numbered_medications(row["generation_value"]) if row["field"] == "discharge_medications" else parse_action_clauses(row["generation_value"])[0] if row["field"] in {"instructions", "follow_up"} else [(clean(row["generation_value"]), 0, len(str(row["generation_value"]))) ]
            for number, (value, start, end) in enumerate(values, 1):
                if not clean(value): continue
                parent = str(row["fact_id"])
                outputs.append({"case_id": case_id, "fact_id": f"{parent}.auto{number:03d}" if len(values) > 1 else parent, "field": row["field"], "contract_section": section, "contract_status": status, "contract_generation_value": clean(value), "required_medication_components": medication_component_string(value) if row["field"] == "discharge_medications" and status == "required" else "", "automation_decision": decision, "automation_reason": "|".join(reasons), "automation_rule_id": rule, "source_fact_id": row.get("source_fact_id", ""), "source_span_start": int(row.get("source_char_start") or 0) + start, "source_span_end": int(row.get("source_char_start") or 0) + end, "parser_rule": "medication_numbered_item_v1" if row["field"] == "discharge_medications" else "action_clause_v1" if row["field"] in {"instructions", "follow_up"} else "preserved_source_field_v1", "automation_version": VERSION})
        cases.append({"case_id": case_id, "automation_decision": decision, "automation_reason": "|".join(reasons), "automation_version": VERSION})
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outputs).to_csv(out / "automation_contract_candidates.csv", index=False)
    case_frame = pd.DataFrame(cases); case_frame.to_csv(out / "automation_case_summary.csv", index=False)
    config = {"automation_version": VERSION, "numbered_item_pattern": NUMBERED_ITEM.pattern, "safety_policy": "Unresolved or high-risk transition content routes to manual review; no reconstruction."}
    (out / "automation_config.json").write_text(json.dumps(config, indent=2) + "\n")
    summary = {"automation_version": VERSION, "n_cases": len(case_frame), "case_decisions": case_frame.automation_decision.value_counts().to_dict(), "n_output_rows": len(outputs), "security_note": "Outputs contain source-derived candidates and remain on approved project storage."}
    (out / "automation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
