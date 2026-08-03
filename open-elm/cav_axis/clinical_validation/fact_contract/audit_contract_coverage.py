#!/usr/bin/env python3
"""Audit final notes against a versioned transition-note fact contract.

This is a deterministic coverage screen, not a substitute for clinical review.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from normalize_medications import component_presence
from parse_note_sections import normalize_text, section_map


# These phrases make an unbounded active-discharge medication claim. A reviewed
# contract enumerates the discharge regimen, so they must route to a human
# rather than silently extending that regimen.
GENERIC_MEDICATION_RESUMPTION = re.compile(
    r"\b(?:resume|continue)\s+(?:all\s+)?(?:pre[- ]?admission|home|prior)\s+medications?\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract_path", required=True)
    parser.add_argument("--notes_csv", required=True)
    parser.add_argument("--case_id_column", default="case_id")
    parser.add_argument("--note_column", default="synthetic_note")
    parser.add_argument("--candidate_id_column", default="rescue_id",
                        help="Unique output identifier; falls back to case ID for one-note-per-case review tables.")
    parser.add_argument(
        "--hard_fields", default="discharge_medications",
        help="Comma-separated fields that cause a backtest route when unsatisfied. "
             "Use all required fields only after deterministic hybrid rendering is implemented.",
    )
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contracts = {
        str(item["case_id"]): item
        for line in Path(args.contract_path).read_text(encoding="utf-8").splitlines() if line.strip()
        for item in [json.loads(line)]
    }
    notes_path = Path(args.notes_csv).resolve()
    notes = pd.read_json(notes_path, lines=True) if notes_path.suffix == ".jsonl" else pd.read_csv(notes_path)
    # Generated manifests use `generated_text`; completed human-review tables use
    # `synthetic_note`. Preserve the explicit override while supporting both.
    if args.note_column not in notes.columns and args.note_column == "synthetic_note" and "generated_text" in notes.columns:
        args.note_column = "generated_text"
    if args.candidate_id_column not in notes.columns:
        args.candidate_id_column = args.case_id_column
    needed = {args.case_id_column, args.note_column}
    if missing := needed.difference(notes.columns):
        raise KeyError(f"notes CSV missing columns: {sorted(missing)}")
    hard_fields = {value.strip() for value in args.hard_fields.split(",") if value.strip()}
    rows = []
    for note_row in notes.itertuples(index=False):
        case_id = str(getattr(note_row, args.case_id_column))
        candidate_id = str(getattr(note_row, args.candidate_id_column))
        if case_id not in contracts:
            continue
        sections = section_map(getattr(note_row, args.note_column))
        for fact in contracts[case_id]["facts"]:
            if fact["status"] != "required":
                continue
            section_text = sections.get(fact["section"], "")
            expected = normalize_text(fact["generation_value"])
            exact = bool(expected) and expected in normalize_text(section_text)
            coverage = "satisfied_exact" if exact else ("wrong_section" if expected and expected in normalize_text(getattr(note_row, args.note_column)) else "missing")
            missing_components = ""
            if fact["field"] == "discharge_medications" and "medication_components" in fact:
                medication_contract = fact["medication_components"]
                components = component_presence(medication_contract, section_text)
                missing_components = "|".join(key for key, present in components.items() if not present)
                if missing_components and not exact:
                    coverage = "partially_satisfied" if section_text else "missing"
                elif section_text and not exact:
                    coverage = "satisfied_normalized_equivalent"
                forbidden_phrase = medication_contract.get("forbidden_phrase", "")
                if forbidden_phrase and forbidden_phrase in normalize_text(section_text):
                    coverage = "action_contradiction"
                    missing_components = f"forbidden_phrase:{forbidden_phrase}"
            rows.append({
                "case_id": case_id, "fact_id": fact["fact_id"], "field": fact["field"],
                "candidate_id": candidate_id,
                "required_section": fact["section"], "coverage_status": coverage,
                "missing_components": missing_components,
                "hard_reject": fact["field"] in hard_fields and (coverage in {"missing", "wrong_section", "action_contradiction"} or (bool(missing_components) and not exact)),
            })
        active_discharge_text = "\n".join(
            sections.get(name, "")
            for name in ("discharge_medications", "instructions")
        )
        generic_resumption = GENERIC_MEDICATION_RESUMPTION.search(active_discharge_text)
        if generic_resumption:
            rows.append({
                "case_id": case_id,
                "fact_id": f"{case_id}__no_extra_active_discharge_medications",
                "field": "discharge_medications",
                "candidate_id": candidate_id,
                "required_section": "discharge_medications|instructions",
                "coverage_status": "generic_active_medication_claim",
                "missing_components": f"generic_resumption:{normalize_text(generic_resumption.group(0))}",
                # This is a mandatory human-review route regardless of the
                # caller's coverage-field backtest configuration.
                "hard_reject": True,
            })
    audit = pd.DataFrame(rows)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "contract_fact_coverage.csv", index=False)
    per_case = audit.groupby(["candidate_id", "case_id"], as_index=False).agg(
        required_fact_count=("fact_id", "size"),
        required_fact_pass_count=("hard_reject", lambda values: int((~values).sum())),
        contract_pass=("hard_reject", lambda values: not bool(values.any())),
    )
    per_case.to_csv(output_dir / "contract_note_coverage.csv", index=False)
    summary = {
        "n_notes_with_contract": int(len(per_case)),
        "contract_pass_rate": float(per_case.contract_pass.mean()) if len(per_case) else 0.0,
        "hard_fields": sorted(hard_fields),
        "limitation": "Exact/normalized string coverage is a conservative backtest aid. Non-hard fields are reported but do not route this free-form backtest; human review remains required until normalization is validated.",
    }
    (output_dir / "contract_coverage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
