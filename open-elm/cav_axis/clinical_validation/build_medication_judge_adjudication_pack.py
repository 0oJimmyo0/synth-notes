#!/usr/bin/env python3
"""Create a restricted label-blind adjudication pack for judge review routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def medication_evidence(ledger_text: str) -> str:
    ledger = json.loads(ledger_text)
    if not isinstance(ledger, list):
        raise ValueError("verified_fact_ledger must be a JSON list")
    relevant = [
        item for item in ledger
        if isinstance(item, dict) and item.get("field") in {"discharge_medications", "instructions"}
    ]
    return json.dumps(relevant, ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build label-blind judge-route adjudication CSV.")
    parser.add_argument("--task_path", required=True)
    parser.add_argument("--judge_output_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    tasks = {
        str(row["blinded_output_id"]): row
        for row in (json.loads(line) for line in Path(args.task_path).resolve().read_text(encoding="utf-8").splitlines() if line.strip())
    }
    outputs = [json.loads(line) for line in Path(args.judge_output_path).resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in outputs:
        grouped.setdefault(str(row["blinded_output_id"]), []).append(row)

    review_rows = []
    for output_id, repeats in grouped.items():
        decisions = [bool((row.get("judge_output") or {}).get("final_reject", False)) for row in repeats]
        stable = len(set(decisions)) == 1
        all_schema_valid = all(bool(row.get("schema_valid")) for row in repeats)
        any_reject = any(decisions)
        if not (any_reject or not stable or not all_schema_valid):
            continue
        task = tasks.get(output_id)
        if task is None:
            raise KeyError(f"No task record for {output_id}")
        reasons = []
        if any_reject:
            reasons.append("judge_any_reject")
        if not stable:
            reasons.append("repeat_disagreement")
        if not all_schema_valid:
            reasons.append("invalid_schema_or_truncated_output")
        review_rows.append({
            "blinded_output_id": output_id,
            "review_route_reason": "|".join(reasons),
            "repeat_decisions": json.dumps(decisions),
            "judge_findings_by_repeat": json.dumps([
                (row.get("judge_output") or {}).get("findings", [])
                for row in repeats
            ], ensure_ascii=True),
            "verified_medication_evidence": medication_evidence(str(task["verified_fact_ledger"])),
            "synthetic_note": str(task["synthetic_note"]),
            "finding_supported_yes_no": "",
            "finding_material_yes_no": "",
            "overall_judge_route_appropriate_yes_no": "",
            "reviewer_notes": "",
        })

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "medication_judge_route_adjudication_BLINDED_TO_PRIOR_LABELS.csv"
    pd.DataFrame(review_rows).sort_values("blinded_output_id").to_csv(output_path, index=False)
    summary = {
        "n_review_routes": len(review_rows),
        "output_path": str(output_path),
        "blinding": "Existing human labels and prior human error classifications are excluded from this pack.",
        "security_note": "Contains compact verified medication evidence and synthetic notes; retain on approved project storage.",
    }
    (output_dir / "medication_judge_route_adjudication_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
