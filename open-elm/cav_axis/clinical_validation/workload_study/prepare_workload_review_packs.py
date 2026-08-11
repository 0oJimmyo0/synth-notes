#!/usr/bin/env python3
"""Create separate source-only and selective-assisted workload review packs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


UNCERTAIN_DIAGNOSIS = re.compile(r"\b(?:rule out|r/o|possible|concern for|versus|\bvs\.?\b|question of)\b|_{3,}|\bnot specified\b", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload_manifest_csv", required=True)
    parser.add_argument("--workload_review_form_csv", required=True)
    parser.add_argument("--provisional_ledger_csv", required=True)
    parser.add_argument("--source_reference_csv", required=True)
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def candidate_state(row: pd.Series) -> tuple[str, str]:
    value = str(row["value"]).strip()
    if row["field"] == "principal_diagnosis" and value and not UNCERTAIN_DIAGNOSIS.search(value):
        return "AUTO_ACCEPT", "direct_principal_diagnosis_with_contiguous_source_span"
    return "MANUAL_REVIEW", "field_policy_or_uncertain_source"


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.workload_manifest_csv, dtype=str).fillna("")
    review_form = pd.read_csv(args.workload_review_form_csv, dtype=str).fillna("")
    ledger = pd.read_csv(args.provisional_ledger_csv, dtype=str).fillna("")
    reference = pd.read_csv(args.source_reference_csv, dtype=str).fillna("")
    policy = Path(args.policy_path).read_text(encoding="utf-8")
    if "policy_id: selective_contract_policy_v1" not in policy:
        raise ValueError("unexpected selective policy")
    if set(manifest.dataset_row_id) != set(ledger.dataset_row_id) or set(manifest.dataset_row_id) != set(reference.dataset_row_id):
        raise ValueError("manifest, ledger, and source reference case sets must match")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"policy_id": "selective_contract_policy_v1", "arms": {}}
    for arm in ("full_manual", "automation_assisted"):
        arm_manifest = manifest.loc[manifest.workflow_condition.eq(arm)].copy()
        case_rows = set(arm_manifest.dataset_row_id)
        arm_dir = output_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        arm_manifest.to_csv(arm_dir / "workload_case_manifest.csv", index=False)
        review_form.loc[review_form.dataset_row_id.isin(case_rows)].to_csv(arm_dir / "workload_review_form.csv", index=False)
        ledger_rows = ledger.loc[ledger.dataset_row_id.isin(case_rows)].copy()
        reference.loc[reference.dataset_row_id.isin(case_rows)].to_csv(arm_dir / "source_reference_RESTRICTED.csv", index=False)
        if arm == "full_manual":
            ledger_rows.to_csv(arm_dir / "provisional_source_ledger_RESTRICTED.csv", index=False)
            auto_count = 0
        else:
            states = ledger_rows.apply(candidate_state, axis=1, result_type="expand")
            ledger_rows[["candidate_state", "candidate_policy_reason"]] = states
            ledger_rows["candidate_contract_section"] = ledger_rows["field"].where(
                ledger_rows.field.isin(["principal_diagnosis", "discharge_medications", "disposition", "instructions", "follow_up"]),
                "",
            )
            ledger_rows.to_csv(arm_dir / "selective_candidate_ledger_RESTRICTED.csv", index=False)
            auto_count = int(ledger_rows.candidate_state.eq("AUTO_ACCEPT").sum())
            high_risk_auto = ledger_rows.loc[
                ledger_rows.candidate_state.eq("AUTO_ACCEPT") & ~ledger_rows.field.eq("principal_diagnosis")
            ]
            if not high_risk_auto.empty:
                raise ValueError("policy violation: non-diagnosis AUTO_ACCEPT candidate")
        # This template records the human-approved final contract in both arms.
        # Reviewers may duplicate a parent row for multiple atoms or add a row
        # with a cited source span for an obligation absent from extraction.
        template_columns = [column for column in [
            "case_id", "dataset_row_id", "note_id", "fact_id", "field", "value",
            "source_section", "source_char_start", "source_char_end", "supporting_text",
            "candidate_state", "candidate_policy_reason", "candidate_contract_section",
        ] if column in ledger_rows]
        contract_template = ledger_rows.loc[:, template_columns].copy()
        for column in [
            "final_parent_resolution", "final_atom_sequence", "final_contract_status",
            "final_contract_section", "final_contract_generation_value",
            "final_source_span_start", "final_source_span_end", "final_reviewer_note",
        ]:
            contract_template[column] = "pending" if column == "final_parent_resolution" else ""
        contract_template.to_csv(arm_dir / "final_contract_template_RESTRICTED.csv", index=False)
        summary["arms"][arm] = {
            "n_cases": int(len(arm_manifest)), "n_provisional_facts": int(len(ledger_rows)),
            "auto_accept_candidates": auto_count,
        }
    summary["security_note"] = "Review packs contain source-derived text and remain on approved project storage."
    (output_dir / "workload_review_pack_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
