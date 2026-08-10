#!/usr/bin/env python3
"""Evaluate deterministic contract automation against manual gold contracts.

This evaluator exports only derived agreement counts and case-level statuses;
it does not copy contract values into its outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


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


def uniform_case_values(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame.groupby("case_id")[column].agg(
        lambda items: {normalize(item) for item in items}
    )
    invalid = values.loc[values.map(len).ne(1)]
    if not invalid.empty:
        raise ValueError(f"{column} is inconsistent within cases: {invalid.index.tolist()[:10]}")
    return values.map(lambda items: next(iter(items)))


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(Path(args.automation_candidates_csv).resolve(), dtype=str).fillna("")
    automated = pd.read_csv(Path(args.automation_case_summary_csv).resolve(), dtype=str).fillna("")
    gold = pd.read_csv(Path(args.manual_gold_contract_csv).resolve(), dtype=str).fillna("")
    manifest = pd.read_csv(Path(args.case_manifest_csv).resolve(), dtype=str).fillna("")

    required_candidate = {"case_id", "contract_status", "contract_section", "contract_generation_value"}
    required_summary = {"case_id", "automation_decision"}
    required_gold = {"case_id", "contract_status", "contract_section", "contract_generation_value", "case_excluded"}
    required_manifest = {"case_id", "automation_split"}
    for label, frame, columns in (
        ("automation candidates", candidates, required_candidate),
        ("automation case summary", automated, required_summary),
        ("manual gold", gold, required_gold),
        ("case manifest", manifest, required_manifest),
    ):
        if missing := columns - set(frame.columns):
            raise KeyError(f"{label} is missing columns: {sorted(missing)}")

    expected = set(manifest.loc[
        manifest.automation_split.eq(args.expected_automation_split), "case_id"
    ])
    if not expected:
        raise ValueError("No cases matched expected automation split")
    if set(automated.case_id) != expected or set(gold.case_id) != expected:
        raise ValueError("Automation and gold inputs must contain exactly the expected split cases")
    if not set(candidates.case_id).issubset(expected):
        raise ValueError("Automation candidates include cases outside the expected split")

    gold_excluded = uniform_case_values(gold, "case_excluded")
    decisions = automated.set_index("case_id")["automation_decision"].map(normalize)
    full_auto = decisions.eq("fully_automated")
    auto_excluded = decisions.eq("automatically_excluded")
    gold_is_excluded = gold_excluded.eq("true")
    false_accept = full_auto & gold_is_excluded
    false_exclude = auto_excluded & ~gold_is_excluded

    accepted_case_ids = set(decisions.index[full_auto])
    gold_required = gold.loc[
        gold.case_id.isin(accepted_case_ids)
        & gold.contract_status.map(normalize).eq("required")
        & gold.case_excluded.map(normalize).eq("false")
    ].copy()
    candidate_required = candidates.loc[
        candidates.case_id.isin(accepted_case_ids)
        & candidates.contract_status.map(normalize).eq("required")
    ].copy()

    def obligations(frame: pd.DataFrame, include_section: bool) -> Counter[tuple[str, ...]]:
        tuples = []
        for row in frame.itertuples(index=False):
            value = normalize(getattr(row, "contract_generation_value"))
            if not value:
                continue
            base = (str(getattr(row, "case_id")), value)
            tuples.append(base + (normalize(getattr(row, "contract_section")),) if include_section else base)
        return Counter(tuples)

    gold_values = obligations(gold_required, include_section=False)
    candidate_values = obligations(candidate_required, include_section=False)
    matched_values = sum((gold_values & candidate_values).values())
    candidate_total = sum(candidate_values.values())
    gold_total = sum(gold_values.values())
    gold_sections = obligations(gold_required, include_section=True)
    candidate_sections = obligations(candidate_required, include_section=True)
    matched_sections = sum((gold_sections & candidate_sections).values())

    # Derived diagnostics deliberately retain counts only, never contract text.
    section_rows = []
    all_sections = sorted(
        set(gold_required.contract_section.map(normalize))
        | set(candidate_required.contract_section.map(normalize))
    )
    for section in all_sections:
        gold_counter = obligations(
            gold_required.loc[gold_required.contract_section.map(normalize).eq(section)],
            include_section=False,
        )
        candidate_counter = obligations(
            candidate_required.loc[candidate_required.contract_section.map(normalize).eq(section)],
            include_section=False,
        )
        matched = sum((gold_counter & candidate_counter).values())
        gold_count, candidate_count = sum(gold_counter.values()), sum(candidate_counter.values())
        section_rows.append({
            "contract_section": section,
            "gold_required_obligations": gold_count,
            "automation_required_obligations": candidate_count,
            "value_matched_obligations": matched,
            "recall": matched / gold_count if gold_count else None,
            "precision": matched / candidate_count if candidate_count else None,
        })

    case_rows = []
    for case_id in sorted(accepted_case_ids):
        gold_counter = Counter({key: count for key, count in gold_values.items() if key[0] == case_id})
        candidate_counter = Counter({key: count for key, count in candidate_values.items() if key[0] == case_id})
        matched = sum((gold_counter & candidate_counter).values())
        case_rows.append({
            "case_id": case_id,
            "gold_required_obligation_count": sum(gold_counter.values()),
            "automation_required_obligation_count": sum(candidate_counter.values()),
            "value_matched_required_obligation_count": matched,
            "missing_required_obligation_count": sum(gold_counter.values()) - matched,
            "extra_required_obligation_count": sum(candidate_counter.values()) - matched,
        })

    case_matrix = pd.DataFrame({
        "case_id": decisions.index,
        "automation_decision": decisions.values,
        "gold_case_excluded": gold_excluded.reindex(decisions.index).values,
        "false_accept": false_accept.values,
        "false_exclude": false_exclude.values,
    }).sort_values("case_id", kind="stable")

    summary = {
        "n_cases": int(len(case_matrix)),
        "case_decisions": decisions.value_counts().to_dict(),
        "gold_excluded_cases": int(gold_is_excluded.sum()),
        "false_accept_count": int(false_accept.sum()),
        "false_exclude_count": int(false_exclude.sum()),
        "automated_coverage": float(full_auto.mean()),
        "required_obligation_recall_among_fully_automated_safe_cases": (
            matched_values / gold_total if gold_total else None
        ),
        "required_obligation_precision_among_fully_automated_safe_cases": (
            matched_values / candidate_total if candidate_total else None
        ),
        "section_accuracy_among_value_matched_required_obligations": (
            matched_sections / matched_values if matched_values else None
        ),
        "denominators": {
            "gold_required_obligations": gold_total,
            "automation_required_obligations": candidate_total,
            "value_matched_required_obligations": matched_values,
        },
        "security_note": "Outputs contain derived agreement statuses only; no contract text is exported.",
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case_matrix.to_csv(output_dir / "contract_automation_case_matrix.csv", index=False)
    pd.DataFrame(section_rows).to_csv(output_dir / "contract_automation_section_agreement.csv", index=False)
    pd.DataFrame(case_rows).to_csv(output_dir / "contract_automation_case_obligation_agreement.csv", index=False)
    (output_dir / "contract_automation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
