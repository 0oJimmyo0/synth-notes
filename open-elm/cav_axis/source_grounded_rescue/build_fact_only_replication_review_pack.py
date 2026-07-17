#!/usr/bin/env python3
"""Build a blinded factual-review pack for the frozen fact-only replication."""

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
    parser = argparse.ArgumentParser(description="Build a blinded review pack for fact-only replication outputs.")
    parser.add_argument("--primary_ledger_path", required=True)
    parser.add_argument("--primary_manifest_path", required=True)
    parser.add_argument("--raw_elm_manifest_path", required=True)
    parser.add_argument("--control_ledger_path", required=True)
    parser.add_argument("--control_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_rows(rows: list[dict[str, object]], id_field: str, label: str) -> dict[str, dict[str, object]]:
    indexed = {str(row[id_field]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{label} has duplicate {id_field} values")
    return indexed


def main() -> None:
    args = parse_args()
    primary_ledger = load_jsonl(Path(args.primary_ledger_path).resolve())
    control_ledger = load_jsonl(Path(args.control_ledger_path).resolve())
    primary = pd.read_json(Path(args.primary_manifest_path).resolve(), lines=True)
    raw = pd.read_json(Path(args.raw_elm_manifest_path).resolve(), lines=True)
    control = pd.read_json(Path(args.control_manifest_path).resolve(), lines=True)
    primary_by_case = index_rows(primary.to_dict(orient="records"), "case_id", "primary manifest")
    raw_by_anchor = index_rows(raw.to_dict(orient="records"), "anchor_id", "raw ELM manifest")
    control_by_case = index_rows(control.to_dict(orient="records"), "case_id", "control manifest")
    primary_case_ids = {str(row["case_id"]) for row in primary_ledger}
    control_case_ids = {str(row["case_id"]) for row in control_ledger}
    if len(primary_case_ids) != 30 or len(control_case_ids) != 15:
        raise ValueError("expected frozen 30-case primary ledger and 15-case control ledger")
    if not control_case_ids.issubset(primary_case_ids):
        raise ValueError("control cases must be a subset of primary cases")
    if set(primary_by_case) != primary_case_ids or set(control_by_case) != control_case_ids:
        raise ValueError("generation manifests do not exactly match frozen ledgers")

    review_rows, key_rows = [], []
    for ledger in primary_ledger:
        case_id = str(ledger["case_id"])
        anchor_id = str(ledger["anchor_id"])
        if anchor_id not in raw_by_anchor:
            raise ValueError(f"no raw ELM baseline for anchor_id={anchor_id}")
        facts = json.dumps(ledger["facts"], ensure_ascii=True, indent=2)
        candidates = [
            ("raw_elm_baseline", raw_by_anchor[anchor_id]),
            ("checkpoint_8215_fact_only", primary_by_case[case_id]),
        ]
        if case_id in control_case_ids:
            candidates.append(("untouched_backbone_fact_only", control_by_case[case_id]))
        for condition, candidate in candidates:
            text = candidate.get("generated_text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"empty generated text for {case_id}/{condition}")
            blinded_id = f"fact_replication_blind_{len(review_rows)+1:03d}"
            review_rows.append({
                "blinded_output_id": blinded_id,
                "case_id": case_id,
                "review_stratum": ledger.get("review_stratum"),
                "patient_disjoint_from_train": ledger.get("patient_disjoint_from_train"),
                "verified_fact_ledger": facts,
                "synthetic_note": text,
                **{field: "" for field in REVIEW_FIELDS},
            })
            key_rows.append({
                "blinded_output_id": blinded_id,
                "case_id": case_id,
                "anchor_id": anchor_id,
                "dataset_row_id": ledger.get("dataset_row_id"),
                "condition": condition,
                "candidate_id": candidate.get("rescue_id", candidate.get("candidate_id")),
                "model_condition": candidate.get("model_condition", "checkpoint_8215"),
                "arm": candidate.get("arm", "raw_elm_baseline"),
            })

    order = list(range(len(review_rows)))
    random.Random(args.seed).shuffle(order)
    review_df = pd.DataFrame(review_rows).iloc[order].reset_index(drop=True)
    key_df = pd.DataFrame(key_rows).iloc[order].reset_index(drop=True)
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(output_dir / "fact_only_replication_blinded_review_RESTRICTED.csv", index=False)
    key_df.to_csv(output_dir / "fact_only_replication_blinded_key.csv", index=False)
    summary = {
        "n_primary_cases": len(primary_case_ids),
        "n_control_cases": len(control_case_ids),
        "n_blinded_outputs": len(review_df),
        "condition_counts": {"raw_elm_baseline": 30, "checkpoint_8215_fact_only": 30, "untouched_backbone_fact_only": 15},
        "security_note": "The review contains compact reviewed facts and synthetic notes but no source-note text or evidence spans. Keep the condition key blinded until labels are finalized.",
    }
    (output_dir / "fact_only_replication_review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
