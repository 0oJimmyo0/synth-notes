#!/usr/bin/env python3
"""Build a blinded human reference pack for contract-alignment calibration."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_path", required=True)
    parser.add_argument("--judge_output_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    tasks = [json.loads(line) for line in Path(args.task_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    outputs: dict[str, list[dict[str, object]]] = defaultdict(list)
    for line in Path(args.judge_output_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line); outputs[str(row["blinded_output_id"])].append(row)
    review_rows = []
    key_rows = []
    for task in tasks:
        output_id = str(task["blinded_output_id"])
        for fact in json.loads(str(task["verified_fact_ledger"])):
            review_rows.append({
                "blinded_output_id": output_id,
                "case_id": str(task.get("case_id", "")),
                "patient_disjoint_from_train": task.get("patient_disjoint_from_train", ""),
                "contract_id": str(fact["fact_id"]),
                "contract_field": str(fact["field"]),
                "contract_obligation": str(fact["generation_value"]),
                "synthetic_note": str(task["synthetic_note"]),
                "human_alignment_status": "",
                "reviewer_notes": "",
            })
        for repeat in outputs.get(output_id, []):
            payload = repeat.get("judge_output") or {}
            for alignment in payload.get("ledger_to_note_alignment", []) if isinstance(payload, dict) else []:
                if isinstance(alignment, dict):
                    key_rows.append({
                        "blinded_output_id": output_id,
                        "repeat_index": repeat.get("repeat_index"),
                        "schema_valid": repeat.get("schema_valid"),
                        "contract_id": alignment.get("contract_id"),
                        "judge_status": alignment.get("status"),
                    })
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(review_rows).to_csv(output_dir / "contract_alignment_human_review_BLINDED.csv", index=False)
    pd.DataFrame(key_rows).to_csv(output_dir / "contract_alignment_judge_key_AFTER_LABELS.csv", index=False)
    summary = {
        "n_notes": len(tasks), "n_contract_obligations": len(review_rows),
        "blinding": "The review CSV excludes all MedGemma labels and schema results. Do not open the key until labels are finalized.",
        "allowed_human_statuses": ["present_supported", "missing", "unsupported", "uncertain"],
        "security_note": "Contains reviewed contract facts and synthetic notes; retain on approved project storage.",
    }
    (output_dir / "contract_alignment_human_review_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
