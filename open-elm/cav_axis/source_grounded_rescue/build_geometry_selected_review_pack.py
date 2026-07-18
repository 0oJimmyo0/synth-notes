#!/usr/bin/env python3
"""Build a blinded factual review pack for final-output geometry-selected fact-only notes."""

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
    parser.add_argument("--selected_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--document_type", default="complete_discharge_summary")
    parser.add_argument("--optional_fields", default="")
    parser.add_argument("--allow_selected_subset", action="store_true", help="Permit output filtering to leave some frozen anchors unselected.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    ledgers = read_jsonl(Path(args.generation_ledger_path).resolve())
    ledger_by_case = {str(row["case_id"]): row for row in ledgers}
    if len(ledger_by_case) != len(ledgers):
        raise ValueError("Generation ledger contains duplicate case_id values.")
    selected = pd.read_json(Path(args.selected_manifest_path).resolve(), lines=True)
    required = {"rescue_id", "case_id", "anchor_id", "generated_text", "output_in_target_basin"}
    if missing := required - set(selected.columns):
        raise ValueError(f"Selected manifest missing columns: {sorted(missing)}")
    if selected["rescue_id"].duplicated().any() or selected["case_id"].duplicated().any():
        raise ValueError("Selected manifest must contain one unique output per case.")
    if selected["generated_text"].isna().any() or selected["generated_text"].astype(str).str.strip().eq("").any():
        raise ValueError("Selected manifest contains empty generated text.")
    selected_cases = set(selected["case_id"].astype(str))
    ledger_cases = set(ledger_by_case)
    if not selected_cases.issubset(ledger_cases):
        raise ValueError("Selected manifest includes cases absent from the frozen generation ledger.")
    if not args.allow_selected_subset and selected_cases != ledger_cases:
        raise ValueError("Selected manifest cases do not exactly match the frozen generation ledger.")

    review_rows, key_rows = [], []
    for row in selected.sort_values("case_id").to_dict(orient="records"):
        case_id = str(row["case_id"])
        ledger = ledger_by_case[case_id]
        blinded_id = f"geometry_selected_blind_{len(review_rows) + 1:03d}"
        review_rows.append(
            {
                "blinded_output_id": blinded_id,
                "case_id": case_id,
                "review_stratum": ledger.get("review_stratum"),
                "patient_disjoint_from_train": ledger.get("patient_disjoint_from_train"),
                "document_type": args.document_type,
                "optional_fields": args.optional_fields,
                "verified_fact_ledger": json.dumps(ledger["facts"], ensure_ascii=True, indent=2),
                "synthetic_note": row["generated_text"],
                **{field: "" for field in REVIEW_FIELDS},
            }
        )
        key_rows.append(
            {
                "blinded_output_id": blinded_id,
                "case_id": case_id,
                "anchor_id": row["anchor_id"],
                "condition": "checkpoint_8215_fact_only_geometry_selected",
                "model_condition": "checkpoint_8215",
                "arm": "fact_only_geometry_selected",
                "rescue_id": row["rescue_id"],
                "candidate_index": row.get("candidate_index"),
                "seed": row.get("seed"),
                "output_cluster_id": row.get("output_cluster_id"),
                "output_in_target_basin": row.get("output_in_target_basin"),
                "target_basin_margin": row.get("target_basin_margin"),
                "document_type": args.document_type,
                "optional_fields": args.optional_fields,
            }
        )
    order = list(range(len(review_rows)))
    random.Random(args.seed).shuffle(order)
    review_df = pd.DataFrame(review_rows).iloc[order].reset_index(drop=True)
    key_df = pd.DataFrame(key_rows).iloc[order].reset_index(drop=True)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(output_dir / "geometry_selected_fact_only_blinded_review.csv", index=False)
    key_df.to_csv(output_dir / "geometry_selected_fact_only_blinded_key.csv", index=False)
    summary = {
        "n_cases": int(len(review_df)),
        "n_target_basin_selected": int(key_df["output_in_target_basin"].sum()),
        "review_rule": "pass only when no unsupported major claim, no critical omission, and both factual faithfulness and clinical consistency are at least 4",
        "document_type": args.document_type,
        "optional_fields": [value.strip() for value in args.optional_fields.split(",") if value.strip()],
        "frozen_ledger_cases": int(len(ledger_cases)),
        "selected_case_count": int(len(selected_cases)),
        "unselected_case_count": int(len(ledger_cases.difference(selected_cases))),
        "security_note": "Review CSV contains compact verified facts and synthetic notes only; keep the condition/geometry key blinded until labels are final.",
    }
    (output_dir / "geometry_selected_fact_only_review_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
