#!/usr/bin/env python3
"""Build restricted active-discharge medication alignment tasks from reviewed contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review_csv", required=True, help="Frozen blinded full-note review CSV containing synthetic_note.")
    parser.add_argument("--contract_path", required=True, help="Reviewed compiled contract JSONL.")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    review = pd.read_csv(Path(args.review_csv).resolve()).fillna("")
    required_review = {"blinded_output_id", "case_id", "synthetic_note"}
    if missing := required_review - set(review.columns):
        raise KeyError(f"review CSV missing columns: {sorted(missing)}")
    if review.blinded_output_id.duplicated().any() or review.case_id.duplicated().any():
        raise ValueError("Review CSV must contain unique blinded_output_id and case_id values.")

    contracts = {str(row["case_id"]): row for row in read_jsonl(Path(args.contract_path).resolve())}
    rows = []
    for _, row in review.iterrows():
        case_id = str(row["case_id"])
        contract = contracts.get(case_id)
        if contract is None:
            raise KeyError(f"No reviewed contract for case_id={case_id}")
        if not bool(contract.get("ready_for_hybrid_generation")):
            raise ValueError(f"Case is not contract-ready: {case_id}")
        active = [
            fact for fact in contract.get("facts", [])
            if fact.get("status") == "required" and fact.get("field") == "discharge_medications"
        ]
        if not active:
            raise ValueError(f"No active discharge medication obligations for case_id={case_id}")
        evidence = [
            {
                "fact_id": str(fact["fact_id"]),
                "field": "active_discharge_obligations",
                "generation_value": str(fact["generation_value"]),
            }
            for fact in active
        ]
        evidence.append({
            "fact_id": f"{case_id}__no_extra_active_discharge_medications",
            "field": "negative_discharge_constraints",
            "generation_value": "No active discharge medication claims beyond the explicit active discharge obligations in this contract.",
        })
        rows.append({
            "task_id": f"contract_alignment_v2::{row['blinded_output_id']}",
            "blinded_output_id": str(row["blinded_output_id"]),
            "case_id": case_id,
            "patient_disjoint_from_train": row.get("patient_disjoint_from_train", ""),
            "verified_fact_ledger": json.dumps(evidence, ensure_ascii=True),
            "synthetic_note": str(row["synthetic_note"]),
            "contract_version": contract.get("contract_version", ""),
        })
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    task_path = output_dir / "contract_alignment_tasks.jsonl"
    task_path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "n_tasks": len(rows),
        "scope": "active_discharge_medication_alignment_only",
        "contract_path": str(Path(args.contract_path).resolve()),
        "security_note": "Tasks contain reviewed contract facts and synthetic notes; retain on approved project storage.",
    }
    (output_dir / "contract_alignment_task_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
