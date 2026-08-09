#!/usr/bin/env python3
"""Build an arm-blinded paired review pack for a renderer head-to-head study."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from build_source_grounded_review_pack import REVIEW_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation_ledger_path", required=True)
    parser.add_argument("--contract_path", required=True)
    parser.add_argument("--hybrid_selected_manifest_path", required=True)
    parser.add_argument("--deterministic_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--document_type", default="discharge_transition_note")
    parser.add_argument("--optional_fields", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_one_note_per_case(path: Path, label: str) -> dict[str, dict[str, object]]:
    rows = read_jsonl(path)
    required = {"case_id", "anchor_id", "generated_text"}
    if not rows:
        raise ValueError(f"{label} manifest is empty")
    missing = required - set(rows[0])
    if missing:
        raise KeyError(f"{label} manifest missing columns: {sorted(missing)}")
    by_case = {str(row["case_id"]): row for row in rows}
    if len(by_case) != len(rows):
        raise ValueError(f"{label} manifest must contain exactly one note per case")
    if any(not str(row["generated_text"]).strip() for row in rows):
        raise ValueError(f"{label} manifest contains empty generated text")
    return by_case


def main() -> None:
    args = parse_args()
    ledgers = read_jsonl(Path(args.generation_ledger_path).resolve())
    ledger_by_case = {str(row["case_id"]): row for row in ledgers}
    if len(ledger_by_case) != len(ledgers):
        raise ValueError("Generation ledger contains duplicate case_id values")

    contracts = read_jsonl(Path(args.contract_path).resolve())
    contract_by_case = {str(row["case_id"]): row for row in contracts}
    if len(contract_by_case) != len(contracts):
        raise ValueError("Reviewed contract contains duplicate case_id values")

    hybrid_by_case = load_one_note_per_case(
        Path(args.hybrid_selected_manifest_path).resolve(), "Hybrid selected"
    )
    deterministic_by_case = load_one_note_per_case(
        Path(args.deterministic_manifest_path).resolve(), "Deterministic"
    )
    hybrid_cases = set(hybrid_by_case)
    deterministic_cases = set(deterministic_by_case)
    if hybrid_cases != deterministic_cases:
        raise ValueError("Hybrid and deterministic manifests do not contain the same cases")
    if not hybrid_cases.issubset(contract_by_case) or not hybrid_cases.issubset(ledger_by_case):
        raise ValueError("Compared cases must be present in both the contract and generation ledger")

    candidates: list[dict[str, object]] = []
    for case_id in sorted(hybrid_cases):
        for arm, condition, model_condition, row in (
            (
                "hybrid_generated_course",
                "checkpoint_8215_hybrid_contract_patient_neutralized_canonical_k50_selected",
                "checkpoint_8215",
                hybrid_by_case[case_id],
            ),
            (
                "deterministic_verified_course",
                "deterministic_verified_course_renderer_v3",
                "deterministic_renderer_v3",
                deterministic_by_case[case_id],
            ),
        ):
            candidates.append(
                {
                    "case_id": case_id,
                    "ledger": ledger_by_case[case_id],
                    "facts": contract_by_case[case_id]["facts"],
                    "arm": arm,
                    "condition": condition,
                    "model_condition": model_condition,
                    "row": row,
                }
            )

    random.Random(args.seed).shuffle(candidates)
    review_rows, key_rows = [], []
    for index, candidate in enumerate(candidates, start=1):
        row = candidate["row"]
        ledger = candidate["ledger"]
        blinded_id = f"renderer_head_to_head_blind_{index:03d}"
        review_rows.append(
            {
                "blinded_output_id": blinded_id,
                "case_id": candidate["case_id"],
                "review_stratum": ledger.get("review_stratum"),
                "patient_disjoint_from_train": ledger.get("patient_disjoint_from_train"),
                "document_type": args.document_type,
                "optional_fields": args.optional_fields,
                "verified_fact_ledger": json.dumps(candidate["facts"], ensure_ascii=True, indent=2),
                "synthetic_note": row["generated_text"],
                **{field: "" for field in REVIEW_FIELDS},
            }
        )
        key_rows.append(
            {
                "blinded_output_id": blinded_id,
                "case_id": candidate["case_id"],
                "anchor_id": row["anchor_id"],
                "condition": candidate["condition"],
                "model_condition": candidate["model_condition"],
                "arm": candidate["arm"],
                "rescue_id": row.get("rescue_id"),
                "candidate_index": row.get("candidate_index"),
                "seed": row.get("seed"),
                "document_type": args.document_type,
                "optional_fields": args.optional_fields,
            }
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(review_rows).to_csv(output_dir / "renderer_head_to_head_blinded_review.csv", index=False)
    pd.DataFrame(key_rows).to_csv(output_dir / "renderer_head_to_head_blinded_key_SEALED.csv", index=False)
    summary = {
        "n_cases": len(hybrid_cases),
        "n_blinded_outputs": len(review_rows),
        "outputs_per_case": 2,
        "review_rule": "pass only when no unsupported major claim, no critical omission, and both factual faithfulness and clinical consistency are at least 4",
        "document_type": args.document_type,
        "optional_fields": [value.strip() for value in args.optional_fields.split(",") if value.strip()],
        "review_reference": "reviewed_contract",
        "security_note": "The review CSV contains compact reviewed facts and notes only. Keep the sealed key closed until all adjudications are final.",
    }
    (output_dir / "renderer_head_to_head_review_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
