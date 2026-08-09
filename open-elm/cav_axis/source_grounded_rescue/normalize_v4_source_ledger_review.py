#!/usr/bin/env python3
"""Normalize the V4 three-file source review into the established ledger schema.

The V4 review separates row decisions, case safety decisions, and manually
recovered atomic transition obligations.  This bridge preserves the row-level
source provenance while making those decisions consumable by the existing
ledger validator, contract builder, and generation-ledger serializer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROW_STATUS_MAP = {
    "verified": "verified",
    "verified_source_block_case": "verified",
    "omit_placeholder": "omitted",
}
PASS_DECISIONS = {"pass_no_change", "pass_with_transition_repairs"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row_review_csv", required=True)
    parser.add_argument("--atomic_transitions_csv", required=True)
    parser.add_argument("--case_review_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_path", required=True)
    return parser.parse_args()


def normalized_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(Path(args.row_review_csv).resolve(), dtype=str).fillna("")
    atomic = pd.read_csv(Path(args.atomic_transitions_csv).resolve(), dtype=str).fillna("")
    cases = pd.read_csv(Path(args.case_review_csv).resolve(), dtype=str).fillna("")

    required_row_columns = {
        "case_id", "dataset_row_id", "note_id", "fact_id", "field", "value",
        "source_section", "supporting_text", "source_char_start", "source_char_end",
        "extraction_confidence", "manual_verification_status",
    }
    required_atomic_columns = {
        "case_id", "atomic_transition_id", "source_fact_id", "contract_section",
        "contract_status", "contract_generation_value", "source_only_no_inference",
    }
    required_case_columns = {"case_id", "reviewer_decision", "blocked_yes_no", "block_reason"}
    for label, frame, needed in (
        ("row review", rows, required_row_columns),
        ("atomic transitions", atomic, required_atomic_columns),
        ("case review", cases, required_case_columns),
    ):
        missing = needed.difference(frame.columns)
        if missing:
            raise KeyError(f"{label} is missing columns: {sorted(missing)}")

    if cases.case_id.duplicated().any():
        raise ValueError("case review contains duplicate case_id values")
    if rows.fact_id.duplicated().any():
        raise ValueError("row review contains duplicate fact_id values")
    if atomic.atomic_transition_id.duplicated().any():
        raise ValueError("atomic transitions contain duplicate atomic_transition_id values")

    row_cases = set(normalized_text(rows.case_id))
    case_cases = set(normalized_text(cases.case_id))
    if row_cases != case_cases:
        raise ValueError(f"row/case review case mismatch: rows_only={sorted(row_cases - case_cases)}, cases_only={sorted(case_cases - row_cases)}")
    atomic_cases = set(normalized_text(atomic.case_id))
    if not atomic_cases.issubset(case_cases):
        raise ValueError(f"atomic transitions reference unknown cases: {sorted(atomic_cases - case_cases)}")

    cases["reviewer_decision"] = normalized_text(cases.reviewer_decision).str.lower()
    cases["blocked_yes_no"] = normalized_text(cases.blocked_yes_no).str.lower()
    invalid_decisions = set(cases.reviewer_decision).difference(PASS_DECISIONS | {"block"})
    if invalid_decisions:
        raise ValueError(f"invalid case reviewer decisions: {sorted(invalid_decisions)}")
    if set(cases.blocked_yes_no).difference({"yes", "no"}):
        raise ValueError("blocked_yes_no must be yes or no")
    expected_blocked = cases.reviewer_decision.eq("block")
    if not expected_blocked.eq(cases.blocked_yes_no.eq("yes")).all():
        raise ValueError("reviewer_decision and blocked_yes_no disagree")

    rows["manual_verification_status"] = normalized_text(rows.manual_verification_status).str.lower()
    invalid_row_statuses = set(rows.manual_verification_status).difference(ROW_STATUS_MAP)
    if invalid_row_statuses:
        raise ValueError(f"invalid V4 row statuses: {sorted(invalid_row_statuses)}")
    rows["manual_verification_status"] = rows.manual_verification_status.map(ROW_STATUS_MAP)
    rows["manual_verified_value"] = normalized_text(rows.get("manual_verified_value", pd.Series("", index=rows.index)))
    rows["value"] = normalized_text(rows.value)
    retained = rows.manual_verification_status.ne("omitted")
    rows.loc[retained & rows.manual_verified_value.eq(""), "manual_verified_value"] = rows.loc[
        retained & rows.manual_verified_value.eq(""), "value"
    ]
    if (retained & rows.manual_verified_value.eq("")).any():
        raise ValueError("retained source rows have blank reviewed values")
    rows["generation_value"] = rows.manual_verified_value
    rows.loc[~retained, ["manual_verified_value", "generation_value"]] = ""

    case_metadata = cases.set_index("case_id")
    rows["case_blocked"] = rows.case_id.map(case_metadata.blocked_yes_no).eq("yes")
    rows["case_blocked_reason"] = rows.case_id.map(case_metadata.block_reason)
    rows["case_validation_status"] = rows.case_id.map(case_metadata.reviewer_decision).map(
        lambda value: "validated_for_generation" if value in PASS_DECISIONS else "blocked_by_source_review"
    )
    rows["generation_value_review_status"] = rows.manual_verification_status.replace({"omitted": "omit"})
    rows["source_parent_fact_id"] = ""
    rows["required_medication_components"] = ""
    rows["atomic_transition"] = False

    atomic["contract_status"] = normalized_text(atomic.contract_status).str.lower()
    if set(atomic.contract_status).difference({"required"}):
        raise ValueError("V4 atomic transitions must be required obligations")
    if not normalized_text(atomic.source_only_no_inference).str.lower().eq("yes").all():
        raise ValueError("every atomic transition must be explicitly source-only/no-inference")
    if atomic.case_id.map(case_metadata.blocked_yes_no).eq("yes").any():
        raise ValueError("atomic transitions cannot be attached to blocked cases")
    if normalized_text(atomic.contract_generation_value).eq("").any():
        raise ValueError("atomic transition has blank contract_generation_value")

    lookup = rows.drop_duplicates("case_id").set_index("case_id")
    atomic_rows = pd.DataFrame({
        "case_id": atomic.case_id,
        "dataset_row_id": atomic.case_id.map(lookup.dataset_row_id),
        "note_id": atomic.case_id.map(lookup.note_id),
        "fact_id": atomic.atomic_transition_id,
        "field": atomic.contract_section,
        "value": atomic.contract_generation_value,
        "source_section": "manual_source_linked_transition_recovery",
        "supporting_text": "",
        "source_char_start": "",
        "source_char_end": "",
        "extraction_confidence": "manual_source_review",
        "manual_verification_status": "corrected",
        "manual_verified_value": atomic.contract_generation_value,
        "generation_value": atomic.contract_generation_value,
        "case_blocked": False,
        "case_blocked_reason": "",
        "case_validation_status": "validated_for_generation",
        "generation_value_review_status": "corrected",
        "source_parent_fact_id": atomic.source_fact_id,
        "required_medication_components": atomic.get("required_medication_components", ""),
        "atomic_transition": True,
    })
    output = pd.concat([rows, atomic_rows], ignore_index=True, sort=False)
    if output.fact_id.duplicated().any():
        raise ValueError("normalized ledger contains duplicate fact_id values")

    output_path = Path(args.output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    summary = {
        "n_cases": int(len(cases)),
        "case_decisions": {key: int(value) for key, value in cases.reviewer_decision.value_counts().items()},
        "n_original_rows": int(len(rows)),
        "n_atomic_required_transitions": int(len(atomic_rows)),
        "n_normalized_rows": int(len(output)),
        # This is the source-review pass count.  The downstream validator is
        # authoritative for final generation readiness because it also checks
        # required canonical-section coverage.
        "n_case_review_passes": int(cases.reviewer_decision.isin(PASS_DECISIONS).sum()),
        "n_blocked_cases": int(cases.reviewer_decision.eq("block").sum()),
        "security_note": "Normalized rows preserve source-review provenance and must remain on approved project storage.",
    }
    summary_path = Path(args.summary_path).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
