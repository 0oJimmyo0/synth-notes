#!/usr/bin/env python3
"""Build a restricted V1.2 instruction/follow-up mismatch audit pack.

The pack is development-only and contains source-derived contract text to allow
manual taxonomy assignment. It must remain on approved project storage.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


SECTIONS = ("instructions", "follow_up")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automation_candidates_csv", required=True)
    parser.add_argument("--automation_case_summary_csv", required=True)
    parser.add_argument("--manual_gold_contract_csv", required=True)
    parser.add_argument("--case_manifest_csv", required=True)
    parser.add_argument("--expected_automation_split", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def counter_with_values(frame: pd.DataFrame) -> Counter[str]:
    return Counter(normalize(value) for value in frame.contract_generation_value if normalize(value))


def unmatched_values(frame: pd.DataFrame, other: Counter[str]) -> list[str]:
    remaining = other.copy()
    values = []
    for value in frame.contract_generation_value:
        key = normalize(value)
        if not key:
            continue
        if remaining[key]:
            remaining[key] -= 1
        else:
            values.append(str(value))
    return values


def unmatched_evidence(frame: pd.DataFrame, other: Counter[str], columns: list[str]) -> list[dict[str, str]]:
    """Return restricted evidence metadata for unmatched values only."""
    remaining = other.copy()
    records = []
    for row in frame.itertuples(index=False):
        key = normalize(getattr(row, "contract_generation_value"))
        if not key:
            continue
        if remaining[key]:
            remaining[key] -= 1
            continue
        records.append({column: str(getattr(row, column, "")) for column in columns})
    return records


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.automation_candidates_csv, dtype=str).fillna("")
    decisions = pd.read_csv(args.automation_case_summary_csv, dtype=str).fillna("")
    gold = pd.read_csv(args.manual_gold_contract_csv, dtype=str).fillna("")
    manifest = pd.read_csv(args.case_manifest_csv, dtype=str).fillna("")

    expected_cases = set(manifest.loc[
        manifest.automation_split.eq(args.expected_automation_split), "case_id"
    ])
    if set(decisions.case_id) != expected_cases or set(gold.case_id) != expected_cases:
        raise ValueError("automation, gold, and manifest case sets must match")

    automated_cases = set(decisions.loc[
        decisions.automation_decision.str.lower().eq("fully_automated"), "case_id"
    ])
    excluded = gold.groupby("case_id").case_excluded.agg(
        lambda values: {normalize(value) for value in values}
    )
    if excluded.map(len).ne(1).any():
        raise ValueError("gold case_excluded values must be uniform within case")
    accepted_cases = automated_cases & {
        case_id for case_id, values in excluded.items() if "false" in values
    }

    rows = []
    for case_id in sorted(accepted_cases):
        for section in SECTIONS:
            gold_rows = gold.loc[
                gold.case_id.eq(case_id)
                & gold.contract_status.map(normalize).eq("required")
                & gold.contract_section.map(normalize).eq(section)
                & gold.case_excluded.map(normalize).eq("false")
            ]
            candidate_rows = candidates.loc[
                candidates.case_id.eq(case_id)
                & candidates.contract_status.map(normalize).eq("required")
                & candidates.contract_section.map(normalize).eq(section)
            ]
            gold_counter, candidate_counter = counter_with_values(gold_rows), counter_with_values(candidate_rows)
            matched = sum((gold_counter & candidate_counter).values())
            if matched == sum(gold_counter.values()) == sum(candidate_counter.values()):
                continue
            rows.append({
                "case_id": case_id,
                "section": section,
                "gold_obligation_count": sum(gold_counter.values()),
                "predicted_obligation_count": sum(candidate_counter.values()),
                "matched_obligation_count": matched,
                "gold_unmatched_values_RESTRICTED": json.dumps(unmatched_values(gold_rows, candidate_counter)),
                "predicted_unmatched_values_RESTRICTED": json.dumps(unmatched_values(candidate_rows, gold_counter)),
                "gold_unmatched_fact_ids_RESTRICTED": json.dumps(
                    unmatched_evidence(gold_rows, candidate_counter, ["fact_id"])
                ),
                "predicted_unmatched_evidence_RESTRICTED": json.dumps(
                    unmatched_evidence(
                        candidate_rows,
                        gold_counter,
                        ["fact_id", "source_fact_id", "source_span_start", "source_span_end", "parser_rule"],
                    )
                ),
                "mismatch_type": "",
                "source_structure": "",
                "manual_contract_granularity": "",
                "parser_behavior": "",
                "required_fix_class": "",
                "taxonomy_reviewer_note": "",
            })

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(rows)
    audit_path = output_dir / "v1_2_instruction_followup_mismatch_audit_RESTRICTED.csv"
    audit.to_csv(audit_path, index=False)
    summary = {
        "scope": "development_only_v1_2_instruction_followup_mismatch_taxonomy",
        "automated_safe_cases": len(accepted_cases),
        "mismatched_case_section_rows": len(audit),
        "taxonomy": [
            "under_splitting", "over_splitting", "modifier_attachment",
            "follow_up_decomposition", "instruction_follow_up_section_confusion",
            "conjunction_ambiguity", "negation_constraint_attachment",
            "truncated_or_unresolved_source_fragment", "evaluation_only_normalization_mismatch",
            "true_unsupported_extraction",
        ],
        "security_note": "Audit CSV contains source-derived contract values and must remain on approved project storage.",
    }
    (output_dir / "v1_2_instruction_followup_mismatch_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
