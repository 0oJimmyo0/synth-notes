#!/usr/bin/env python3
"""Build a label-blind prospective medication-judge review pack.

The reviewer receives only compact verified medication evidence and the
synthetic note. Judge decisions, routes, and findings are kept in a separate
key so they cannot influence clinical adjudication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_evidence_fields(raw_fields: str) -> set[str]:
    fields = {field.strip() for field in raw_fields.split(",") if field.strip()}
    if not fields:
        raise ValueError("--medication_evidence_fields must name at least one ledger field.")
    return fields


def medication_evidence(ledger_text: str, evidence_fields: set[str]) -> str:
    ledger = json.loads(ledger_text)
    if not isinstance(ledger, list):
        raise ValueError("verified_fact_ledger must be a JSON list")
    relevant = [
        item
        for item in ledger
        if isinstance(item, dict) and item.get("field") in evidence_fields
    ]
    return json.dumps(relevant, ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a blinded prospective medication-review pack.")
    parser.add_argument("--task_path", required=True)
    parser.add_argument("--judge_output_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_repeats", type=int, default=3)
    parser.add_argument("--n_non_routed_sample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--medication_evidence_fields",
        default="discharge_medications,instructions",
        help="Comma-separated verified ledger fields shown to the blinded medication reviewer.",
    )
    args = parser.parse_args()
    if args.expected_repeats < 1 or args.n_non_routed_sample < 0:
        raise ValueError("expected repeats must be positive and non-routed sample size must be nonnegative")
    evidence_fields = parse_evidence_fields(args.medication_evidence_fields)

    tasks = {
        str(row["blinded_output_id"]): row
        for row in (json.loads(line) for line in Path(args.task_path).resolve().read_text(encoding="utf-8").splitlines() if line.strip())
    }
    grouped: dict[str, list[dict[str, object]]] = {}
    for line in Path(args.judge_output_path).resolve().read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped.setdefault(str(row["blinded_output_id"]), []).append(row)

    key_rows = []
    for output_id, task in tasks.items():
        repeats = grouped.get(output_id, [])
        decisions = [bool((row.get("judge_output") or {}).get("final_reject", False)) for row in repeats]
        requires_review = [bool((row.get("judge_output") or {}).get("requires_human_review", True)) for row in repeats]
        schema_valid = [bool(row.get("schema_valid")) and row.get("parse_error") is None for row in repeats]
        complete = len(repeats) == args.expected_repeats
        stable = complete and len(set(decisions)) == 1
        valid = complete and all(schema_valid)
        routed = bool(any(decisions) or any(requires_review) or not stable or not valid)
        key_rows.append({
            "blinded_output_id": output_id,
            "case_id": str(task.get("case_id", "")),
            "patient_disjoint_from_train": task.get("patient_disjoint_from_train", ""),
            "judge_final_reject_any": bool(any(decisions)),
            "judge_review_route": routed,
            "judge_repeat_count": len(repeats),
            "judge_complete_repeats": complete,
            "judge_label_stable": stable,
            "judge_schema_valid": valid,
        })
    key = pd.DataFrame(key_rows)
    routed_ids = set(key.loc[key.judge_review_route, "blinded_output_id"])
    non_routed_ids = key.loc[~key.judge_review_route, "blinded_output_id"]
    sampled_non_routed = set(non_routed_ids.sample(n=min(args.n_non_routed_sample, len(non_routed_ids)), random_state=args.seed))
    selected_ids = routed_ids | sampled_non_routed

    review_rows = []
    for output_id in sorted(selected_ids):
        task = tasks[output_id]
        review_rows.append({
            "blinded_output_id": output_id,
            "case_id": str(task.get("case_id", "")),
            "patient_disjoint_from_train": task.get("patient_disjoint_from_train", ""),
            "verified_medication_evidence": medication_evidence(str(task["verified_fact_ledger"]), evidence_fields),
            "synthetic_note": str(task["synthetic_note"]),
            "human_medication_error_yes_no": "",
            "human_error_types_pipe_delimited": "",
            "human_severe_medication_error_yes_no": "",
            "reviewer_notes": "",
        })

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(review_rows).to_csv(output_dir / "prospective_medication_review_BLINDED.csv", index=False)
    key.to_csv(output_dir / "prospective_medication_review_key_AFTER_LABELS.csv", index=False)
    summary = {
        "n_tasks": len(tasks),
        "n_routed": len(routed_ids),
        "n_non_routed_sampled": len(sampled_non_routed),
        "n_reviewed": len(review_rows),
        "expected_repeats": args.expected_repeats,
        "medication_evidence_fields": sorted(evidence_fields),
        "blinding": "The review CSV excludes all judge decisions, routes, and findings. Open the key only after labels are finalized.",
        "security_note": "The review CSV contains compact verified medication evidence and synthetic notes; retain it on approved project storage.",
    }
    (output_dir / "prospective_medication_review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
