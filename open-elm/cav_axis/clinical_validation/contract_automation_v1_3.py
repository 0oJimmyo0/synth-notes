#!/usr/bin/env python3
"""V1.3 source-linked instruction and follow-up atomization.

V1.3 retains the frozen V1.2 medication parser. It replaces only instruction
and follow-up handling with single-span actionable atoms and conservative
manual routing for unresolved fragments.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from canonicalize_obligation_v1 import canonicalize_obligation_text
from contract_automation_v1_2 import (
    REQUIRED_FIELDS,
    RENDERED_SECTIONS,
    UNKNOWN,
    case_reasons as v1_2_case_reasons,
    clean,
    medication_component_string,
    parse_numbered_medications,
    route_row,
)


VERSION = "contract_automation_v1_3"
ACTION = re.compile(
    r"\b(?:follow[-\s]+up|see|schedule|attend|return|call|seek|take|continue|stop|hold|avoid|"
    r"apply|use|wear|keep|change|clean|weight\s*bear|ambulate|elevate|remove|monitor|obtain|check)\b",
    re.I,
)
FOLLOW_UP = re.compile(
    r"\b(?:follow[-\s]+up|appointment|see\s+(?:your|a|an|the|dr\.?|clinic)|schedule|"
    r"outpatient|repeat\s+(?:ct|mri|x-?ray|radiograph|labs?|blood\s+work))\b",
    re.I,
)
RETURN_PRECAUTION = re.compile(r"\b(?:return|call|seek)\b.*\b(?:er|emergency|doctor|911|care)\b", re.I)
NON_ACTIONABLE = re.compile(
    r"^(?:expired|see instruction sheet|dear\s+(?:mr|ms|mrs)\.?|"
    r"division of .+ instructions|your .+ care team)$",
    re.I,
)
FRAGMENT = re.compile(
    r"(?:\b(?:and|or|to|with|for|of|the|a|an|your|their|his|her|dr)\.?\s*$|"
    r"\.{3}|\b(?:call|see|follow[-\s]+up with)\s+(?:dr\.?|not specified)\s*$)",
    re.I,
)
SPLIT = re.compile(
    r"(?<=[.!?;])\s+|\s+(?:[-*]|\d+[.)])\s+|\s+and\s+(?=(?:follow[-\s]+up|see|schedule|attend|return|call|seek|take|continue|stop|hold|avoid|apply|use|wear|keep|change|clean|weight\s*bear|ambulate|elevate|remove|monitor|obtain|check)\b)",
    re.I,
)


@dataclass(frozen=True)
class Atom:
    text: str
    start: int
    end: int
    section: str
    action: str
    parser_rule: str


def action_name(text: str) -> str:
    match = ACTION.search(text)
    return match.group(0).lower().replace("-", " ") if match else ""


def atomize_actionable(value: str) -> list[Atom]:
    """Split only at independent predicates, sentence boundaries, or list markers."""
    atoms: list[Atom] = []
    cursor = 0
    for piece in SPLIT.split(value):
        text = piece.strip(" \t-*")
        if not text or NON_ACTIONABLE.fullmatch(text):
            continue
        start = value.find(text, cursor)
        if start < 0:
            start = value.find(text)
        cursor = start + len(text)
        action = action_name(text)
        if not action:
            continue
        section = "instructions" if RETURN_PRECAUTION.search(text) else "follow_up" if FOLLOW_UP.search(text) else "instructions"
        atoms.append(Atom(text, start, cursor, section, action, "action_atom_v1"))
    return atoms


def has_manual_fragment(value: str) -> bool:
    text = clean(value)
    if not text:
        return False
    # A wrapper/header alone is filtered; a clipped action requires adjudication.
    if NON_ACTIONABLE.fullmatch(text):
        return False
    return bool(ACTION.search(text) and FRAGMENT.search(text))


def case_reasons(group: pd.DataFrame) -> list[str]:
    # V1.3 changes only instruction/follow-up atomization. Preserve every V1.2
    # medication and transition safety route before adding new fragment routes.
    reasons = list(v1_2_case_reasons(group))
    for row in group.to_dict("records"):
        field, value = clean(row["field"]), str(row["generation_value"])
        if field in {"instructions", "follow_up"} and has_manual_fragment(value):
            reasons.append("UNRESOLVED_INSTRUCTION_OR_FOLLOW_UP_FRAGMENT")
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
    required = {"case_id", "fact_id", "field", "generation_value", "manual_verification_status"}
    if missing := required - set(ledger.columns):
        raise KeyError(f"ledger is missing columns: {sorted(missing)}")
    selected = set(manifest.loc[manifest.automation_split.eq(args.expected_automation_split), "case_id"])
    ledger = ledger.loc[
        ledger.case_id.isin(selected)
        & ledger.manual_verification_status.str.lower().isin({"verified", "corrected"})
        & ledger.generation_value.map(clean).ne("")
    ].copy()
    if set(ledger.case_id) != selected:
        raise ValueError("ledger and selected manifest cases differ")

    outputs, cases = [], []
    for case_id, group in ledger.groupby("case_id", sort=True):
        reasons = case_reasons(group)
        decision = "fully_automated" if not reasons else "route_to_manual_review"
        for row in group.sort_values(["field", "fact_id"], kind="stable").to_dict("records"):
            field, source = row["field"], str(row["generation_value"])
            base_start = int(row.get("source_char_start") or 0)
            if field in {"instructions", "follow_up"}:
                parsed = atomize_actionable(source)
                values = [(atom.text, atom.start, atom.end, "required", atom.section, atom.parser_rule, atom.action) for atom in parsed]
            elif field == "discharge_medications":
                values = [(text, start, end, "required", "discharge_medications", "medication_numbered_item_v1", "") for text, start, end in parse_numbered_medications(source)]
            else:
                status, section, rule = route_row(row)
                values = [(clean(source), 0, len(source), status, section, rule, "")]
            for number, (value, start, end, status, section, rule, action) in enumerate(values, 1):
                if not clean(value):
                    continue
                fact_id = str(row["fact_id"])
                semantic_json = ""
                if field in {"instructions", "follow_up"}:
                    semantic = {
                        "obligation_kind": "follow_up" if section == "follow_up" else "instruction",
                        "action": action,
                        "source_start": base_start + start,
                        "source_end": base_start + end,
                        "parent_fact_id": fact_id,
                        "parser_rule": rule,
                    }
                    semantic_json = json.dumps(semantic, sort_keys=True)
                outputs.append({
                    "case_id": case_id,
                    "fact_id": f"{fact_id}.auto{number:03d}" if len(values) > 1 else fact_id,
                    "field": field,
                    "contract_section": section,
                    "contract_status": status,
                    "contract_generation_value": clean(value),
                    "required_medication_components": medication_component_string(value) if field == "discharge_medications" else "",
                    "automation_decision": decision,
                    "automation_reason": "|".join(reasons),
                    "automation_rule_id": rule,
                    "source_fact_id": row.get("source_fact_id", ""),
                    "source_span_start": base_start + start,
                    "source_span_end": base_start + end,
                    "parser_rule": rule,
                    "canonical_obligation_text": canonicalize_obligation_text(value),
                    "obligation_semantics_json": semantic_json,
                    "automation_version": VERSION,
                })
        cases.append({"case_id": case_id, "automation_decision": decision, "automation_reason": "|".join(reasons), "automation_version": VERSION})

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outputs).to_csv(out / "automation_contract_candidates.csv", index=False)
    case_frame = pd.DataFrame(cases)
    case_frame.to_csv(out / "automation_case_summary.csv", index=False)
    summary = {"automation_version": VERSION, "n_cases": len(case_frame), "case_decisions": case_frame.automation_decision.value_counts().to_dict(), "n_output_rows": len(outputs), "security_note": "Outputs contain source-derived candidates and remain on approved project storage."}
    (out / "automation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
