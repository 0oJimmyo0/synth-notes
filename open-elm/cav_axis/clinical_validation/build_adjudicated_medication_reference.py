#!/usr/bin/env python3
"""Merge frozen human labels with label-blind judge-route adjudications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def yes(value: object) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1", "fail"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a material-discrepancy development reference table.")
    parser.add_argument("--human_review_csv", action="append", required=True)
    parser.add_argument("--adjudication_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    human = pd.concat([pd.read_csv(path).fillna("") for path in args.human_review_csv], ignore_index=True, sort=False)
    if human.blinded_output_id.duplicated().any():
        raise ValueError("Human review IDs must be unique across input files.")
    adjudication = pd.read_csv(args.adjudication_csv).fillna("")
    if adjudication.blinded_output_id.duplicated().any():
        raise ValueError("Adjudication IDs must be unique.")

    human["existing_material_discrepancy"] = human.apply(
        lambda row: yes(row.get("human_medication_error_yes_no"))
        or yes(row.get("unsupported_major_claim_yes_no"))
        or yes(row.get("critical_omission_yes_no")),
        axis=1,
    )
    merged = human.merge(
        adjudication[[
            "blinded_output_id", "finding_supported_yes_no", "finding_material_yes_no",
            "overall_judge_route_appropriate_yes_no",
        ]],
        on="blinded_output_id", how="left", validate="one_to_one",
    )
    for column in (
        "finding_supported_yes_no", "finding_material_yes_no",
        "overall_judge_route_appropriate_yes_no",
    ):
        merged[column] = merged[column].fillna("")
    merged["adjudicated_material_discrepancy"] = merged["finding_material_yes_no"].map(
        {"yes": True, "no": False, "uncertain": pd.NA, "": pd.NA}
    )
    merged["reference_material_discrepancy"] = merged["existing_material_discrepancy"] | (merged["adjudicated_material_discrepancy"] == True)
    merged["reference_label_provenance"] = merged.apply(
        lambda row: "judge-route adjudication" if str(row["finding_material_yes_no"]).strip() else "original blinded review",
        axis=1,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = merged.drop(columns=[c for c in ("verified_fact_ledger", "synthetic_note", "reviewer_notes") if c in merged], errors="ignore")
    safe.to_csv(output_dir / "adjudicated_medication_development_reference.csv", index=False)
    summary = {
        "n_notes": len(merged),
        "reference_material_discrepancy_count": int(merged.reference_material_discrepancy.sum()),
        "adjudicated_route_count": int(merged.finding_material_yes_no.astype(str).str.strip().ne("").sum()),
        "adjudicated_material_count": int(merged.finding_material_yes_no.astype(str).str.lower().eq("yes").sum()),
        "adjudicated_uncertain_count": int(merged.finding_material_yes_no.astype(str).str.lower().eq("uncertain").sum()),
        "security_note": "Derived reference excludes ledger text, synthetic notes, and reviewer free text.",
    }
    (output_dir / "adjudicated_medication_development_reference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
