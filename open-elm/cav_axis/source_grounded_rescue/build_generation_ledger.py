#!/usr/bin/env python3
"""Build prompt-safe generation ledgers from reviewer-normalized fact values."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_FIELDS = {
    "principal_diagnosis", "hospital_course_events", "discharge_medications",
    "disposition", "follow_up", "instructions",
}
VALID_STATUSES = {"verified", "corrected", "omit"}


def json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serialize concise, reviewed generation ledgers.")
    parser.add_argument("--review_template_csv", required=True)
    parser.add_argument("--completed_audit_ledger_csv", default=None, help="Optional completed restricted audit ledger for case provenance.")
    parser.add_argument("--pilot_anchor_manifest", default=None, help="Optional pilot anchor manifest for anchor provenance.")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(Path(args.review_template_csv).resolve())
    required = {"case_id", "fact_id", "field", "generation_value", "generation_value_review_status"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"generation-ledger review template missing columns: {sorted(missing)}")
    frame["generation_value_review_status"] = frame.generation_value_review_status.fillna("").astype(str).str.strip().str.lower()
    invalid = set(frame.generation_value_review_status).difference(VALID_STATUSES)
    if invalid:
        raise ValueError(f"invalid generation_value_review_status values: {sorted(invalid)}")
    value = frame.generation_value.fillna("").astype(str).str.strip()
    non_omitted_missing = (frame.generation_value_review_status != "omit") & (value == "")
    if non_omitted_missing.any():
        raise ValueError(f"{int(non_omitted_missing.sum())} verified/corrected generation values are blank")
    usable = frame.loc[frame.generation_value_review_status.isin(["verified", "corrected"])].copy()
    usable["generation_value"] = usable.generation_value.fillna("").astype(str).str.strip()
    provenance = None
    if args.completed_audit_ledger_csv or args.pilot_anchor_manifest:
        if not args.completed_audit_ledger_csv or not args.pilot_anchor_manifest:
            raise ValueError("--completed_audit_ledger_csv and --pilot_anchor_manifest must be provided together")
        audit = pd.read_csv(Path(args.completed_audit_ledger_csv).resolve())
        anchors = pd.read_csv(Path(args.pilot_anchor_manifest).resolve())
        audit_required = {"case_id", "dataset_row_id", "note_id"}
        anchor_required = {"dataset_row_id", "anchor_id", "review_stratum", "patient_disjoint_from_train"}
        if audit_required.difference(audit.columns) or anchor_required.difference(anchors.columns):
            raise KeyError("provenance inputs are missing required identifiers")
        provenance = audit[["case_id", "dataset_row_id", "note_id"]].drop_duplicates("case_id").merge(
            anchors[["dataset_row_id", "anchor_id", "review_stratum", "patient_disjoint_from_train"]].drop_duplicates("dataset_row_id"),
            on="dataset_row_id", how="left", validate="one_to_one",
        )
        if provenance.anchor_id.isna().any():
            raise ValueError("some audit-ledger cases could not be joined to pilot anchors")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ledgers = []
    for case_id, group in usable.groupby("case_id", sort=True):
        fields = set(group.field.astype(str))
        missing_fields = REQUIRED_FIELDS.difference(fields)
        if missing_fields:
            raise ValueError(f"{case_id} is missing required generation fields: {sorted(missing_fields)}")
        facts = [
            {"fact_id": str(row.fact_id), "field": str(row.field), "value": str(row.generation_value)}
            for row in group.sort_values(["field", "fact_id"], kind="stable").itertuples(index=False)
        ]
        metadata = {
            column: json_scalar(group.iloc[0][column])
            for column in [
                "source_review_case_id", "anchor_id", "dataset_row_id", "note_id",
                "review_stratum", "patient_disjoint_from_train",
            ]
            if column in group.columns
        }
        if provenance is not None:
            matched = provenance.loc[provenance.case_id.astype(str) == str(case_id)]
            if len(matched) != 1:
                raise ValueError(f"missing or duplicate provenance for generation ledger case {case_id}")
            metadata.update({column: json_scalar(matched.iloc[0][column]) for column in ["anchor_id", "dataset_row_id", "note_id", "review_stratum", "patient_disjoint_from_train"]})
        serialized = json.dumps(facts, sort_keys=True, separators=(",", ":"))
        ledgers.append({
            "case_id": str(case_id),
            **metadata,
            "facts": facts,
            "generation_ledger_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        })
    with (output_dir / "generation_ledgers.jsonl").open("w", encoding="utf-8") as handle:
        for ledger in ledgers:
            handle.write(json.dumps(ledger) + "\n")
    summary = {
        "n_cases": len(ledgers),
        "n_generation_facts": int(len(usable)),
        "source_spans_in_output": False,
        "security_note": "Generation values remain source-derived facts and must remain on approved project storage.",
    }
    (output_dir / "generation_ledger_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
