#!/usr/bin/env python3
"""Filter generated notes by finalized blinded factual-review labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manifest containing only factually passing generated notes.")
    parser.add_argument("--generation_manifest_path", required=True)
    parser.add_argument("--unblinded_label_matrix_path", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = pd.read_json(Path(args.generation_manifest_path).resolve(), lines=True)
    labels = pd.read_csv(Path(args.unblinded_label_matrix_path).resolve())
    required = {"condition", "candidate_id", "rule_derived_pass", "reviewer_rule_match"}
    if missing := required.difference(labels.columns):
        raise KeyError(f"unblinded label matrix missing columns: {sorted(missing)}")
    labels = labels.loc[labels.condition.astype(str).eq(args.condition)].copy()
    if labels.empty:
        raise ValueError(f"no labels found for condition={args.condition}")
    if not labels.reviewer_rule_match.astype(bool).all():
        raise ValueError("cannot build factual gate manifest: review pass/fail rule mismatch")
    labels["candidate_id"] = labels.candidate_id.astype(str)
    if labels.candidate_id.duplicated().any():
        raise ValueError("condition labels have duplicate candidate_id values")
    if generated.rescue_id.duplicated().any():
        raise ValueError("generation manifest has duplicate rescue_id values")
    generated["candidate_id"] = generated.rescue_id.astype(str)
    merged = generated.merge(
        labels[["candidate_id", "rule_derived_pass", "overall_clinical_usability_pass_fail", "overall_factual_faithfulness_score_1to5", "unsupported_major_claim_yes_no", "critical_omission_yes_no"]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    if merged.rule_derived_pass.isna().any():
        missing = merged.loc[merged.rule_derived_pass.isna(), "rescue_id"].tolist()
        raise ValueError(f"generated notes missing finalized review labels: {missing}")
    passed = merged.loc[merged.rule_derived_pass.astype(bool)].copy().reset_index(drop=True)
    passed["factual_gate_condition"] = args.condition
    passed["factual_gate_pass"] = True
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "factual_gate_passed_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in passed.to_dict(orient="records"):
            handle.write(json.dumps(row) + "\n")
    summary = {
        "condition": args.condition,
        "n_generated": int(len(generated)),
        "n_factual_gate_passed": int(len(passed)),
        "factual_gate_pass_rate": float(len(passed) / len(generated)) if len(generated) else 0.0,
        "manifest_path": str(manifest_path),
        "security_note": "Manifest contains generated text and review-derived labels; retain on approved project storage.",
    }
    (output_dir / "factual_gate_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
