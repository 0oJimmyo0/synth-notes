#!/usr/bin/env python3
"""Create identity-safe V2 source-only and automation-assisted review packs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


IDENTITY = ["source_split", "dataset_row_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload_manifest_csv", required=True)
    parser.add_argument("--workload_review_form_csv", required=True)
    parser.add_argument("--provisional_ledger_csv", required=True)
    parser.add_argument("--source_reference_csv", required=True)
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def add_source_split(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    if "source_split" in frame.columns:
        return frame
    lookup = manifest[["dataset_row_id", "source_split"]].drop_duplicates()
    if lookup.dataset_row_id.duplicated().any():
        raise ValueError("cannot infer source_split from a non-unique dataset_row_id")
    return frame.merge(lookup, on="dataset_row_id", how="left", validate="many_to_one")


def assert_identity(frame: pd.DataFrame, label: str) -> None:
    missing = set(IDENTITY) - set(frame.columns)
    if missing:
        raise KeyError(f"{label} is missing identity columns: {sorted(missing)}")
    if frame[IDENTITY].isna().any().any() or frame[IDENTITY].astype(str).eq("").any().any():
        raise ValueError(f"{label} has blank source identities")


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.workload_manifest_csv, dtype=str).fillna("")
    review_form = pd.read_csv(args.workload_review_form_csv, dtype=str).fillna("")
    ledger = add_source_split(pd.read_csv(args.provisional_ledger_csv, dtype=str).fillna(""), manifest)
    reference = add_source_split(pd.read_csv(args.source_reference_csv, dtype=str).fillna(""), manifest)
    policy = Path(args.policy_path).read_text(encoding="utf-8")
    if "policy_id: selective_contract_policy_v2" not in policy:
        raise ValueError("expected selective_contract_policy_v2")
    for frame, label in ((manifest, "manifest"), (review_form, "review form"), (ledger, "ledger"), (reference, "reference")):
        assert_identity(frame, label)
    selected_ids = set(map(tuple, manifest[IDENTITY].itertuples(index=False, name=None)))
    for frame, label in ((ledger, "ledger"), (reference, "reference")):
        observed = set(map(tuple, frame[IDENTITY].itertuples(index=False, name=None)))
        if observed != selected_ids:
            raise ValueError(f"{label} identities do not exactly match manifest")
    provenance = reference[[*IDENTITY, "note_id"]].drop_duplicates()
    if provenance.duplicated(IDENTITY).any():
        raise ValueError("reference has non-unique source identities")
    ledger = ledger.merge(provenance, on=IDENTITY, how="left", suffixes=("", "_reference"), validate="many_to_one")
    if "note_id_reference" in ledger:
        if ledger.note_id.astype(str).ne(ledger.note_id_reference.astype(str)).any():
            raise ValueError("ledger and source reference note IDs disagree")
        ledger = ledger.drop(columns=["note_id_reference"])

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"policy_id": "selective_contract_policy_v2", "identity_key": IDENTITY, "arms": {}}
    for arm in ("full_manual", "automation_assisted"):
        arm_manifest = manifest.loc[manifest.workflow_condition.eq(arm)].copy()
        arm_ids = set(map(tuple, arm_manifest[IDENTITY].itertuples(index=False, name=None)))
        mask = ledger.apply(lambda row: tuple(row[IDENTITY]) in arm_ids, axis=1)
        ledger_rows = ledger.loc[mask].copy()
        ref_mask = reference.apply(lambda row: tuple(row[IDENTITY]) in arm_ids, axis=1)
        reference_rows = reference.loc[ref_mask].copy()
        arm_dir = output_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        arm_manifest.to_csv(arm_dir / "workload_case_manifest.csv", index=False)
        review_form.loc[review_form.workflow_condition.eq(arm)].to_csv(arm_dir / "workload_review_form.csv", index=False)
        reference_rows.to_csv(arm_dir / "source_reference_RESTRICTED.csv", index=False)
        if arm == "full_manual":
            ledger_rows.to_csv(arm_dir / "provisional_source_ledger_RESTRICTED.csv", index=False)
        else:
            # Suggestions may facilitate navigation, but none are accepted automatically.
            ledger_rows["candidate_state"] = "MANUAL_REVIEW"
            ledger_rows["candidate_policy_reason"] = "v2_all_parent_records_require_atomic_human_validation"
            ledger_rows["candidate_contract_section"] = ""
            ledger_rows.to_csv(arm_dir / "selective_candidate_ledger_RESTRICTED.csv", index=False)
        columns = [
            "case_id", "dataset_row_id", "note_id", "source_split", "fact_id", "field", "value",
            "source_section", "source_char_start", "source_char_end", "supporting_text", "candidate_state",
            "candidate_policy_reason", "candidate_contract_section",
        ]
        template = ledger_rows[[column for column in columns if column in ledger_rows]].copy()
        for column in [
            "final_parent_resolution", "final_atom_sequence", "final_clinical_priority", "final_render_decision",
            "final_contract_section", "final_contract_generation_value", "final_primary_span_start",
            "final_primary_span_end", "final_corroborating_source_spans", "final_reviewer_note",
        ]:
            template[column] = "pending" if column == "final_parent_resolution" else ""
        template.to_csv(arm_dir / "final_contract_template_RESTRICTED.csv", index=False)
        summary["arms"][arm] = {"n_cases": int(len(arm_manifest)), "n_provisional_facts": int(len(ledger_rows))}
    summary["security_note"] = "Review packs contain source-derived text and remain on approved project storage."
    (output_dir / "workload_review_pack_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
