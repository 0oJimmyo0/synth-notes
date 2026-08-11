#!/usr/bin/env python3
"""Analyze prediction-blind independent atomic contracts against manual gold.

Outputs only derived counts and agreement metrics. It never exports source text,
gold text, or independent-review text.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
from canonicalize_obligation_v1 import canonicalize_obligation_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed_atomic_addendum_csv", required=True)
    parser.add_argument("--manual_gold_contract_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def normalize(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def counter(frame: pd.DataFrame, value_column: str, normalizer) -> Counter[tuple[str, str]]:
    return Counter(
        (str(row.case_id), normalizer(getattr(row, value_column)))
        for row in frame.itertuples(index=False)
        if normalizer(getattr(row, value_column))
    )


def score(gold: pd.DataFrame, independent: pd.DataFrame, value_column: str, normalizer) -> dict[str, float | int | None]:
    gold_counter = counter(gold, "contract_generation_value", normalizer)
    independent_counter = counter(independent, value_column, normalizer)
    matched = sum((gold_counter & independent_counter).values())
    gold_total, independent_total = sum(gold_counter.values()), sum(independent_counter.values())
    return {
        "gold_required_atomic_obligations": gold_total,
        "independent_required_atomic_obligations": independent_total,
        "matched_obligations": matched,
        "recall": matched / gold_total if gold_total else None,
        "precision": matched / independent_total if independent_total else None,
    }


def main() -> None:
    args = parse_args()
    independent = pd.read_csv(args.completed_atomic_addendum_csv, dtype=str).fillna("")
    gold = pd.read_csv(args.manual_gold_contract_csv, dtype=str).fillna("")
    required = {
        "case_id", "parent_fact_id", "independent_parent_resolution",
        "independent_atomic_status", "independent_atomic_section",
        "independent_atomic_generation_value", "independent_atom_reviewer_note",
    }
    if missing := required - set(independent.columns):
        raise KeyError(f"atomic addendum is missing columns: {sorted(missing)}")
    allowed_resolution = {"atomic_created", "route_manual"}
    resolution = independent.independent_parent_resolution.str.strip().str.lower()
    if invalid := set(resolution) - allowed_resolution:
        raise ValueError(f"invalid or pending parent resolutions: {sorted(invalid)}")
    parent_resolution = independent.groupby("parent_fact_id").independent_parent_resolution.agg(
        lambda values: {value.strip().lower() for value in values}
    )
    if parent_resolution.map(len).ne(1).any():
        raise ValueError("parent resolution must be uniform across duplicated parent rows")
    created = independent.loc[resolution.eq("atomic_created")].copy()
    manual = independent.loc[resolution.eq("route_manual")].copy()
    if created.independent_atomic_status.str.strip().str.lower().ne("required").any():
        raise ValueError("atomic_created rows must have required atomic status")
    if created.independent_atomic_section.str.strip().eq("").any() or created.independent_atomic_generation_value.str.strip().eq("").any():
        raise ValueError("atomic_created rows require a section and nonblank generation value")
    if manual.independent_atomic_status.str.strip().ne("").any() or manual.independent_atomic_generation_value.str.strip().ne("").any():
        raise ValueError("route_manual rows must not contain an atomic obligation")
    if manual.independent_atom_reviewer_note.str.strip().eq("").any():
        raise ValueError("route_manual rows require a reviewer note")

    gold_excluded = gold.groupby("case_id").case_excluded.agg(lambda values: {normalize(value) for value in values})
    if gold_excluded.map(len).ne(1).any():
        raise ValueError("manual gold case_excluded must be uniform within case")
    comparable_cases = set(created.case_id) & {case for case, values in gold_excluded.items() if values == {"false"}}
    independent_required = created.loc[created.case_id.isin(comparable_cases)]
    gold_required = gold.loc[
        gold.case_id.isin(comparable_cases)
        & gold.case_excluded.map(normalize).eq("false")
        & gold.contract_status.map(normalize).eq("required")
    ]
    sections = ["principal_diagnosis", "discharge_medications", "disposition", "instructions", "follow_up"]
    section_rows = []
    for section in sections:
        section_rows.append({
            "section": section,
            **score(
                gold_required.loc[gold_required.contract_section.eq(section)],
                independent_required.loc[independent_required.independent_atomic_section.eq(section)],
                "independent_atomic_generation_value",
                normalize,
            ),
        })
    summary = {
        "scope": "prediction_blind_independent_atomic_contract_agreement",
        "independent_cases": int(independent.case_id.nunique()),
        "comparable_nonexcluded_cases": int(len(comparable_cases)),
        "manual_routed_parent_rows": int(manual.parent_fact_id.nunique()),
        "raw": score(gold_required, independent_required, "independent_atomic_generation_value", normalize),
        "safe_canonicalized": score(gold_required, independent_required, "independent_atomic_generation_value", canonicalize_obligation_text),
        "limitation": "Agreement is a development-only comparison between one manual gold contract and one prediction-blind independent atomic review; it is not an external clinical validation.",
        "security_note": "Outputs contain derived agreement counts only; no restricted obligation text is exported.",
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(section_rows).to_csv(output_dir / "independent_atomic_agreement_by_section.csv", index=False)
    (output_dir / "independent_atomic_agreement_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
