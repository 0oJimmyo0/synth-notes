#!/usr/bin/env python3
"""Combine frozen final-cohort reviewed ledgers with collision-free case IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final_cohort_csv", required=True)
    parser.add_argument("--initial_review_csv", required=True)
    parser.add_argument("--initial_source_reference_csv", required=True)
    parser.add_argument("--replenishment_review_csv", required=True)
    parser.add_argument("--replenishment_source_reference_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_review(path: str, source_reference: str, wave: str, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    review = pd.read_csv(Path(path).resolve())
    reference = pd.read_csv(Path(source_reference).resolve())
    selected = selected.loc[selected.source_wave.astype(str).eq(wave)].copy()
    if selected.empty:
        return review.head(0), reference.head(0)
    selected = selected.rename(columns={"case_id": "source_ledger_case_id"})
    chosen = selected[[
        "final_case_id", "dataset_row_id", "source_ledger_case_id", "support_arm", "cohort_stratum",
        "patient_disjoint_from_train", "mean_top_50_support",
    ]]
    review = review.merge(
        chosen,
        left_on=["case_id", "dataset_row_id"],
        right_on=["source_ledger_case_id", "dataset_row_id"],
        how="inner",
        validate="many_to_one",
    )
    if review.final_case_id.nunique() != len(selected):
        raise ValueError(f"{wave} review did not resolve every selected final case.")
    review["fact_id"] = review.apply(lambda row: f"{row.final_case_id}__{row.fact_id}", axis=1)
    review["case_id"] = review.final_case_id
    reference = reference.merge(
        chosen[["final_case_id", "dataset_row_id", "source_ledger_case_id"]],
        left_on=["case_id", "dataset_row_id"],
        right_on=["source_ledger_case_id", "dataset_row_id"],
        how="inner",
        validate="one_to_one",
    )
    reference["source_ledger_case_id"] = reference.case_id
    reference["case_id"] = reference.final_case_id
    return review, reference


def main() -> None:
    args = parse_args()
    final = pd.read_csv(Path(args.final_cohort_csv).resolve())
    needed = {"final_case_id", "dataset_row_id", "case_id", "source_wave", "support_arm", "cohort_stratum"}
    if missing := needed.difference(final.columns):
        raise KeyError(f"Final cohort missing columns: {sorted(missing)}")
    if final.final_case_id.duplicated().any() or final.dataset_row_id.duplicated().any():
        raise ValueError("Final cohort must have unique final_case_id and dataset_row_id values.")
    initial_review, initial_reference = load_review(
        args.initial_review_csv, args.initial_source_reference_csv, "initial", final
    )
    replenishment_review, replenishment_reference = load_review(
        args.replenishment_review_csv, args.replenishment_source_reference_csv, "replenishment", final
    )
    review = pd.concat([initial_review, replenishment_review], ignore_index=True)
    reference = pd.concat([initial_reference, replenishment_reference], ignore_index=True)
    if review.case_id.nunique() != len(final) or reference.case_id.nunique() != len(final):
        raise ValueError("Combined outputs do not cover every final case exactly once.")
    if review.fact_id.duplicated().any() or reference.case_id.duplicated().any():
        raise ValueError("Combined fact or source-reference identifiers are not unique.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    review.to_csv(output_dir / "final40_reviewed_source_fact_ledger.csv", index=False)
    reference.to_csv(output_dir / "final40_source_fact_ledger_source_reference_RESTRICTED.csv", index=False)
    final.to_csv(output_dir / "final40_anchor_manifest.csv", index=False)
    readiness = final[["final_case_id"]].rename(columns={"final_case_id": "case_id"})
    readiness["ledger_ready_for_generation"] = True
    readiness.to_csv(output_dir / "final40_case_readiness.csv", index=False)
    summary = {
        "n_final_cases": int(len(final)),
        "n_final_facts": int(len(review)),
        "n_source_references": int(len(reference)),
        "case_id_policy": "final_case_id=dev_<dataset_row_id>; original source-ledger case ID retained separately",
        "security_note": "Combined ledger and source reference contain source-derived content and must remain on approved project storage.",
    }
    (output_dir / "final40_combined_ledger_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
