#!/usr/bin/env python3
"""Build a blinded, ledger-grounded review pack for the rescue smoke test.

The reviewer receives compact verified facts, not source-note text or evidence
spans. The separate key remains on approved project storage and must not be
shared with reviewers until labels are finalized.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


REVIEW_FIELDS = [
    "principal_diagnosis_supported_yes_no",
    "hospital_course_supported_yes_no",
    "major_procedures_supported_yes_no_not_applicable",
    "discharge_medications_supported_yes_no",
    "disposition_supported_yes_no",
    "follow_up_supported_yes_no",
    "instructions_supported_yes_no",
    "unsupported_major_claim_yes_no",
    "critical_omission_yes_no",
    "internal_clinical_consistency_score_1to5",
    "overall_factual_faithfulness_score_1to5",
    "overall_clinical_usability_pass_fail",
    "reviewer_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a blinded ledger-grounded rescue review pack.")
    parser.add_argument("--generation_ledger_path", required=True)
    parser.add_argument("--rescue_manifest_path", required=True)
    parser.add_argument("--raw_elm_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    ledger_path = Path(args.generation_ledger_path).resolve()
    rescue_path = Path(args.rescue_manifest_path).resolve()
    raw_path = Path(args.raw_elm_manifest_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ledgers = load_jsonl(ledger_path)
    ledger_by_anchor = {str(row["anchor_id"]): row for row in ledgers}
    if len(ledger_by_anchor) != len(ledgers):
        raise ValueError("generation ledger contains duplicate anchor_id values")

    rescue = pd.read_json(rescue_path, lines=True)
    raw = pd.read_json(raw_path, lines=True)
    if rescue.empty:
        raise ValueError("rescue manifest is empty")
    if raw.anchor_id.duplicated().any():
        raise ValueError("raw ELM manifest must contain one note per anchor")

    expected_rescue = len(ledgers) * 4
    if len(rescue) != expected_rescue:
        raise ValueError(f"expected {expected_rescue} rescue outputs, found {len(rescue)}")
    required_conditions = {
        ("untouched_backbone", "correction"),
        ("untouched_backbone", "fact_only"),
        ("checkpoint_8215", "correction"),
        ("checkpoint_8215", "fact_only"),
    }
    observed_conditions = set(zip(rescue.model_condition.astype(str), rescue.arm.astype(str)))
    if observed_conditions != required_conditions:
        raise ValueError(f"unexpected rescue conditions: {sorted(observed_conditions)}")
    if rescue.rescue_id.duplicated().any():
        raise ValueError("rescue manifest contains duplicate rescue_id values")

    raw_by_anchor = raw.set_index(raw.anchor_id.astype(str)).to_dict(orient="index")
    review_rows, key_rows = [], []
    for anchor_id, ledger in ledger_by_anchor.items():
        if anchor_id not in raw_by_anchor:
            raise ValueError(f"no raw ELM note for anchor_id={anchor_id}")
        case_id = str(ledger["case_id"])
        fact_json = json.dumps(ledger["facts"], ensure_ascii=True, indent=2)
        raw_row = raw_by_anchor[anchor_id]
        candidates = [{
            "condition": "raw_elm_baseline",
            "model_condition": "checkpoint_8215",
            "arm": "raw_elm_baseline",
            "candidate_id": raw_row.get("candidate_id"),
            "generated_text": raw_row.get("generated_text"),
        }]
        rescue_rows = rescue.loc[rescue.anchor_id.astype(str) == anchor_id]
        if len(rescue_rows) != 4:
            raise ValueError(f"case {case_id} has {len(rescue_rows)} rescue outputs, expected 4")
        for row in rescue_rows.to_dict(orient="records"):
            candidates.append({
                "condition": f"{row['model_condition']}__{row['arm']}",
                "model_condition": row["model_condition"],
                "arm": row["arm"],
                "candidate_id": row.get("rescue_id"),
                "generated_text": row.get("generated_text"),
            })
        for candidate in candidates:
            if not isinstance(candidate["generated_text"], str) or not candidate["generated_text"].strip():
                raise ValueError(f"empty text for {case_id}/{candidate['condition']}")
            blinded_id = f"rescue_blind_{len(review_rows) + 1:03d}"
            review_rows.append({
                "blinded_output_id": blinded_id,
                "case_id": case_id,
                "review_stratum": ledger.get("review_stratum"),
                "patient_disjoint_from_train": ledger.get("patient_disjoint_from_train"),
                "verified_fact_ledger": fact_json,
                "synthetic_note": candidate["generated_text"],
                **{field: "" for field in REVIEW_FIELDS},
            })
            key_rows.append({
                "blinded_output_id": blinded_id,
                "case_id": case_id,
                "anchor_id": anchor_id,
                "dataset_row_id": ledger.get("dataset_row_id"),
                "condition": candidate["condition"],
                "model_condition": candidate["model_condition"],
                "arm": candidate["arm"],
                "candidate_id": candidate["candidate_id"],
            })

    order = list(range(len(review_rows)))
    random.Random(args.seed).shuffle(order)
    review_df = pd.DataFrame(review_rows).iloc[order].reset_index(drop=True)
    key_df = pd.DataFrame(key_rows).iloc[order].reset_index(drop=True)
    review_df.to_csv(output_dir / "source_grounded_blinded_review.csv", index=False)
    key_df.to_csv(output_dir / "source_grounded_blinded_key.csv", index=False)
    summary = {
        "n_cases": len(ledgers),
        "n_blinded_outputs": len(review_df),
        "outputs_per_case": 5,
        "conditions": ["raw_elm_baseline", "untouched_backbone__correction", "untouched_backbone__fact_only", "checkpoint_8215__correction", "checkpoint_8215__fact_only"],
        "security_note": "The blinded review contains compact reviewed facts and synthetic notes, but no source-note text or evidence spans. Keep the key blinded until review labels are finalized.",
    }
    (output_dir / "source_grounded_review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
