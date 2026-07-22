#!/usr/bin/env python3
"""Build restricted, ledger-grounded medication-judge tasks from reviewed notes.

The output deliberately retains compact verified ledgers and synthetic notes, so it
must stay on approved MIMIC project storage. It is a calibration/evaluation
dataset, never a source-note export and never an automatic clinical gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"blinded_output_id", "verified_fact_ledger", "synthetic_note"}
LABEL_COLUMNS = [
    "discharge_medications_supported_yes_no",
    "unsupported_major_claim_yes_no",
    "critical_omission_yes_no",
    "overall_clinical_usability_pass_fail",
    "human_medication_error_yes_no",
    "human_error_types_pipe_delimited",
    "human_severe_medication_error_yes_no",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a restricted medication-judge task set.")
    parser.add_argument("--review_csv", action="append", required=True, help="Completed blinded review CSV(s).")
    parser.add_argument("--split", choices=("development", "evaluation", "prospective"), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--include_human_labels", action="store_true", help="Include labels for development/calibration only.")
    return parser.parse_args()


def normalized_label(value: object) -> str:
    return str(value or "").strip().lower()


def main() -> None:
    args = parse_args()
    if args.split == "evaluation" and args.include_human_labels:
        raise ValueError("Held-out evaluation tasks must not include human labels.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    template_rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for raw_path in args.review_csv:
        path = Path(raw_path).resolve()
        frame = pd.read_csv(path).fillna("")
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise KeyError(f"{path} is missing required columns: {sorted(missing)}")
        for row_index, row in frame.iterrows():
            task_id = f"{path.stem}::{row['blinded_output_id']}"
            if task_id in seen:
                raise ValueError(f"Duplicate task ID: {task_id}")
            seen.add(task_id)
            record = {
                "task_id": task_id,
                "blinded_output_id": str(row["blinded_output_id"]),
                "case_id": str(row.get("case_id", "")),
                "review_stratum": str(row.get("review_stratum", "")),
                "patient_disjoint_from_train": row.get("patient_disjoint_from_train", ""),
                "document_type": str(row.get("document_type", "discharge_transition_note")),
                "verified_fact_ledger": str(row["verified_fact_ledger"]),
                "synthetic_note": str(row["synthetic_note"]),
            }
            if args.include_human_labels:
                record["human_labels"] = {column: str(row.get(column, "")) for column in LABEL_COLUMNS}
            records.append(record)
            template_rows.append({
                "task_id": task_id,
                "human_medication_error_yes_no": "",
                "human_error_types_pipe_delimited": "",
                "human_severe_medication_error_yes_no": "",
                "annotation_notes": "",
            })

    task_path = output_dir / f"medication_judge_{args.split}_tasks.jsonl"
    with task_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    pd.DataFrame(template_rows).to_csv(output_dir / f"medication_judge_{args.split}_annotation_template.csv", index=False)
    summary = {
        "split": args.split,
        "n_tasks": len(records),
        "includes_human_labels": bool(args.include_human_labels),
        "task_path": str(task_path),
        "security_note": "Task JSONL contains compact verified ledger text and synthetic notes. Keep it on approved project storage and do not send it to external APIs.",
    }
    (output_dir / f"medication_judge_{args.split}_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
