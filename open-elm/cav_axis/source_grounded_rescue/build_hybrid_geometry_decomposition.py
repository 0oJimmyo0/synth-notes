#!/usr/bin/env python3
"""Build non-generative representations for hybrid output-space drift analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_hybrid_contract_generation import RENDERED_FIELDS, course_facts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract_path", required=True)
    parser.add_argument("--generation_ledger_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_stem", default="hybrid_geometry_decomposition")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_transition_sections(contract: dict) -> str:
    """Render only deterministic transition sections, with no course narrative."""
    sections = []
    for field, heading in RENDERED_FIELDS:
        values = [
            str(fact["generation_value"]).strip()
            for fact in contract["facts"]
            if str(fact["field"]) == field
            and str(fact["status"]) in {"required", "optional"}
            and str(fact["generation_value"]).strip()
        ]
        if values:
            sections.append(f"{heading}:\n" + "\n".join(f"- {value}" for value in values))
    return "\n\n".join(sections).strip()


def render_verified_course(contract: dict) -> str:
    facts = course_facts(contract)
    values = [str(fact["value"]).strip() for fact in facts if str(fact["value"]).strip()]
    return "\n".join(["Brief Hospital Course:"] + [f"- {value}" for value in values]).strip()


def render_contextual_scaffold(contract: dict) -> str:
    """Render reviewed non-transition context without promoting historical medications."""
    allowed_fields = {
        "admission_reason": "Admission Context",
        "procedures_this_admission": "Procedure History",
        "important_results": "Pertinent Results",
        "demographics.sex": "Patient Context",
    }
    sections = []
    for field, heading in allowed_fields.items():
        values = [
            str(fact["generation_value"]).strip()
            for fact in contract["facts"]
            if str(fact["field"]) == field
            and str(fact["status"]) in {"optional", "historical_context_only"}
            and str(fact["generation_value"]).strip()
        ]
        if values:
            sections.append(f"{heading}:\n" + "\n".join(f"- {value}" for value in values))
    return "\n\n".join(sections).strip()


def main() -> None:
    args = parse_args()
    contracts = {str(item["case_id"]): item for item in read_jsonl(Path(args.contract_path).resolve())}
    ledgers = {str(item["case_id"]): item for item in read_jsonl(Path(args.generation_ledger_path).resolve())}
    case_ids = sorted(set(contracts).intersection(ledgers))
    if not case_ids:
        raise ValueError("No overlapping contract and generation-ledger cases.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.output_stem}_manifest.jsonl"
    rows = []
    for case_id in case_ids:
        contract, ledger = contracts[case_id], ledgers[case_id]
        if not contract.get("ready_for_hybrid_generation"):
            continue
        sections = render_transition_sections(contract)
        verified_course = render_verified_course(contract)
        scaffold = render_contextual_scaffold(contract)
        variants = {
            "deterministic_transition_sections_only": sections,
            "deterministic_transition_sections_plus_verified_course": "\n\n".join(
                [sections, verified_course]
            ).strip(),
            "deterministic_transition_sections_course_plus_contextual_scaffold": "\n\n".join(
                [sections, verified_course, scaffold]
            ).strip(),
        }
        for variant, text in variants.items():
            rows.append({
                "rescue_id": f"{case_id}__geometry_decomposition__{variant}",
                "case_id": case_id,
                "anchor_id": ledger.get("anchor_id"),
                "dataset_row_id": ledger.get("dataset_row_id"),
                "review_stratum": ledger.get("review_stratum"),
                "patient_disjoint_from_train": ledger.get("patient_disjoint_from_train"),
                "arm": "geometry_decomposition",
                "representation_variant": variant,
                "generated_text": text,
            })
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "n_cases": len({row["case_id"] for row in rows}),
        "n_rows": len(rows),
        "variants": sorted({row["representation_variant"] for row in rows}),
        "purpose": "Geometry diagnosis only; these deterministic representations are not synthetic-note candidates.",
    }
    (output_dir / f"{args.output_stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
