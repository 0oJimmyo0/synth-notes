#!/usr/bin/env python3
"""Build development-only automation reflection-study artifacts.

This script does not tune or evaluate held-out gold. The independent-review
sample is selected from source-ledger complexity before prediction data are
read, so V1.3 outputs cannot influence sample selection.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


CAUSES = [
    "true_parser_omission", "true_unsupported_parser_addition", "under_splitting",
    "over_splitting", "modifier_attachment_error", "instruction_follow_up_classification_error",
    "clinically_equivalent_differently_atomized", "lexical_canonicalization_mismatch",
    "ambiguous_source", "incomplete_truncated_source_route_manual", "manual_contract_inconsistency_or_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger_review_csv", required=True)
    parser.add_argument("--case_manifest_csv", required=True)
    parser.add_argument("--manual_gold_contract_csv", required=True)
    parser.add_argument("--automation_candidates_csv", required=True)
    parser.add_argument("--automation_case_summary_csv", required=True)
    parser.add_argument("--expected_automation_split", default="development")
    parser.add_argument("--human_agreement_cases", type=int, default=24)
    parser.add_argument("--selection_seed", type=int, default=20260903)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def case_exclusion(gold: pd.DataFrame) -> pd.Series:
    values = gold.groupby("case_id").case_excluded.agg(lambda items: {normalize(item) for item in items})
    if values.map(len).ne(1).any():
        raise ValueError("manual gold case_excluded must be uniform within case")
    return values.map(lambda items: next(iter(items)))


def source_complexity_sample(ledger: pd.DataFrame, cases: set[str], count: int, seed: int) -> pd.DataFrame:
    """Select evenly from source instruction/follow-up complexity quartiles only."""
    subset = ledger.loc[ledger.case_id.isin(cases) & ledger.field.isin(["instructions", "follow_up"])].copy()
    complexity = subset.groupby("case_id").generation_value.agg(
        lambda values: sum(len(str(value)) for value in values)
    ).rename("instruction_followup_char_count").reset_index()
    complexity = pd.DataFrame({"case_id": sorted(cases)}).merge(complexity, on="case_id", how="left").fillna(0)
    complexity["complexity_quartile"] = pd.qcut(
        complexity.instruction_followup_char_count.rank(method="first"), 4, labels=[1, 2, 3, 4]
    ).astype(int)
    per_quartile, remainder = divmod(count, 4)
    selected = []
    for quartile, group in complexity.groupby("complexity_quartile", sort=True):
        n = per_quartile + (1 if quartile <= remainder else 0)
        selected.append(group.sample(n=min(n, len(group)), random_state=seed + int(quartile)))
    result = pd.concat(selected, ignore_index=True).sort_values(["complexity_quartile", "case_id"], kind="stable")
    result["selection_rank"] = range(1, len(result) + 1)
    return result


def multiset_rows(frame: pd.DataFrame, other: Counter[tuple[str, str]]) -> list[dict[str, str]]:
    remaining = other.copy()
    rows = []
    for row in frame.itertuples(index=False):
        key = (str(row.case_id), normalize(row.contract_generation_value))
        if not key[1]:
            continue
        if remaining[key]:
            remaining[key] -= 1
        else:
            rows.append(row._asdict())
    return rows


def main() -> None:
    args = parse_args()
    ledger = pd.read_csv(args.ledger_review_csv, dtype=str).fillna("")
    manifest = pd.read_csv(args.case_manifest_csv, dtype=str).fillna("")
    gold = pd.read_csv(args.manual_gold_contract_csv, dtype=str).fillna("")
    candidates = pd.read_csv(args.automation_candidates_csv, dtype=str).fillna("")
    decisions = pd.read_csv(args.automation_case_summary_csv, dtype=str).fillna("")
    expected = set(manifest.loc[manifest.automation_split.eq(args.expected_automation_split), "case_id"])
    if set(ledger.case_id) != expected or set(gold.case_id) != expected or set(decisions.case_id) != expected:
        raise ValueError("all inputs must contain exactly the expected development split")

    # This selection uses ledger + manifest only; predictions are intentionally not read above it.
    sample = source_complexity_sample(ledger, expected, args.human_agreement_cases, args.selection_seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_dir / "human_agreement_source_blind_sample_manifest.csv", index=False)
    source_columns = [column for column in [
        "case_id", "fact_id", "field", "generation_value", "source_fact_id",
        "source_char_start", "source_char_end", "manual_verification_status",
    ] if column in ledger]
    review_pack = ledger.loc[ledger.case_id.isin(set(sample.case_id)), source_columns].copy()
    for column in ["independent_contract_status", "independent_contract_section", "independent_contract_generation_value", "independent_reviewer_note"]:
        review_pack[column] = ""
    review_pack.to_csv(output_dir / "human_agreement_source_review_pack_RESTRICTED.csv", index=False)

    excluded = case_exclusion(gold)
    accepted = set(decisions.loc[decisions.automation_decision.eq("fully_automated"), "case_id"]) & {
        case_id for case_id, excluded_value in excluded.items() if excluded_value == "false"
    }
    gold_required = gold.loc[
        gold.case_id.isin(accepted) & gold.contract_status.map(normalize).eq("required") & gold.case_excluded.map(normalize).eq("false")
    ].copy()
    candidate_required = candidates.loc[
        candidates.case_id.isin(accepted) & candidates.contract_status.map(normalize).eq("required")
    ].copy()
    gold_counter = Counter((str(row.case_id), normalize(row.contract_generation_value)) for row in gold_required.itertuples(index=False))
    candidate_counter = Counter((str(row.case_id), normalize(row.contract_generation_value)) for row in candidate_required.itertuples(index=False))

    mismatch_rows = []
    for side, rows in (("gold_missing", multiset_rows(gold_required, candidate_counter)), ("predicted_extra", multiset_rows(candidate_required, gold_counter))):
        for row in rows:
            mismatch_rows.append({
                "case_id": row["case_id"], "side": side, "section": row.get("contract_section", ""),
                "obligation_value_RESTRICTED": row.get("contract_generation_value", ""),
                "fact_id": row.get("fact_id", ""), "source_fact_id": row.get("source_fact_id", ""),
                "source_span_start": row.get("source_span_start", ""), "source_span_end": row.get("source_span_end", ""),
                "parser_rule": row.get("parser_rule", ""), "primary_cause": "", "cause_reviewer_note": "",
            })
    pd.DataFrame(mismatch_rows).to_csv(output_dir / "v1_3_raw_obligation_mismatch_audit_RESTRICTED.csv", index=False)

    modes = {
        "diagnosis_and_disposition": {"principal_diagnosis", "disposition"},
        "core_plus_medications": {"principal_diagnosis", "disposition", "discharge_medications"},
        "all_required_raw": {"principal_diagnosis", "disposition", "discharge_medications", "instructions", "follow_up"},
    }
    curve = []
    all_gold = sum(gold_counter.values())
    for mode, sections in modes.items():
        gold_mode = gold_required.loc[gold_required.contract_section.isin(sections)]
        candidate_mode = candidate_required.loc[candidate_required.contract_section.isin(sections)]
        gold_mode_counter = Counter((str(row.case_id), normalize(row.contract_generation_value)) for row in gold_mode.itertuples(index=False))
        candidate_mode_counter = Counter((str(row.case_id), normalize(row.contract_generation_value)) for row in candidate_mode.itertuples(index=False))
        matched = sum((gold_mode_counter & candidate_mode_counter).values())
        curve.append({
            "mode": mode, "accepted_safe_cases": len(accepted),
            "manual_routing_case_rate": 1 - (len(accepted) / len(expected)),
            "gold_required_obligations_in_mode": sum(gold_mode_counter.values()),
            "auto_populated_obligations": sum(candidate_mode_counter.values()),
            "value_matched_obligations": matched,
            "raw_precision": matched / sum(candidate_mode_counter.values()) if candidate_mode_counter else None,
            "raw_recall_within_mode": matched / sum(gold_mode_counter.values()) if gold_mode_counter else None,
            "matched_fraction_of_all_safe_gold_obligations": matched / all_gold if all_gold else None,
        })
    pd.DataFrame(curve).to_csv(output_dir / "selective_automation_raw_precision_coverage.csv", index=False)
    summary = {
        "scope": "development_only_automation_task_reflection_study",
        "expected_development_cases": len(expected),
        "accepted_safe_cases_for_raw_mismatch_audit": len(accepted),
        "raw_mismatched_obligations": len(mismatch_rows),
        "human_agreement_sample_cases": len(sample),
        "human_agreement_selection": "source-ledger instruction/follow-up complexity quartiles; no predictions or gold contract used",
        "allowed_primary_causes": CAUSES,
        "security_note": "Mismatch and review-pack CSVs contain restricted source-derived values and must remain on approved project storage.",
    }
    (output_dir / "automation_task_reflection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
