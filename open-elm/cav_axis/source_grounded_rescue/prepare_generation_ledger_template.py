#!/usr/bin/env python3
"""Select a prospective cohort and create a concise generation-ledger template.

The completed audit ledger retains source-supported spans for verification. This
template deliberately excludes those spans from the eventual prompt contract:
reviewers write a short, factual ``generation_value`` for each usable fact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a prospective concise generation-ledger review template.")
    parser.add_argument("--completed_ledger_csv", required=True)
    parser.add_argument("--pilot_anchor_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_exact_pooled", type=int, default=2)
    parser.add_argument("--n_centroid_only", type=int, default=2)
    parser.add_argument(
        "--exclude_generation_ledger_path",
        default=None,
        help="Optional prior prompt-safe ledger JSONL. Its dataset rows are excluded prospectively.",
    )
    parser.add_argument("--output_stem", default="generation_ledger_smoke")
    return parser.parse_args()


def select_cases(anchors: pd.DataFrame, stratum: str, n: int) -> pd.DataFrame:
    subset = anchors.loc[anchors.review_stratum.astype(str) == stratum].copy()
    if subset.empty:
        raise ValueError(f"no cases available for review_stratum={stratum}")
    subset["patient_disjoint_from_train"] = subset.patient_disjoint_from_train.fillna(False).astype(bool)
    # Select one patient-disjoint example first when available, then stable order.
    selected = subset.sort_values(["patient_disjoint_from_train", "case_id"], ascending=[False, True]).head(n).copy()
    selected["smoke_selection_reason"] = f"{stratum}_reviewed_case"
    return selected


def main() -> None:
    args = parse_args()
    ledger = pd.read_csv(Path(args.completed_ledger_csv).resolve())
    anchors = pd.read_csv(Path(args.pilot_anchor_manifest).resolve())
    required = {"case_id", "field", "fact_id", "manual_verification_status", "manual_verified_value"}
    missing = required.difference(ledger.columns)
    if missing:
        raise KeyError(f"completed ledger missing columns: {sorted(missing)}")
    required = {"case_id", "review_stratum", "patient_disjoint_from_train"}
    missing = required.difference(anchors.columns)
    if missing:
        raise KeyError(f"pilot anchor manifest missing columns: {sorted(missing)}")
    ledger_cases = ledger[["case_id", "dataset_row_id"]].drop_duplicates("dataset_row_id", keep="first").rename(
        columns={"case_id": "ledger_case_id"}
    )
    candidates = anchors.merge(ledger_cases, on="dataset_row_id", how="left", validate="one_to_one")
    if candidates.ledger_case_id.isna().any():
        raise ValueError("some pilot anchors could not be aligned to completed ledger cases by dataset_row_id")
    if args.exclude_generation_ledger_path:
        excluded = []
        for line in Path(args.exclude_generation_ledger_path).resolve().read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line).get("dataset_row_id")
                if value is not None:
                    excluded.append(int(value))
        candidates = candidates.loc[~candidates.dataset_row_id.isin(excluded)].copy()
        if candidates.empty:
            raise ValueError("all pilot anchors were excluded by the prior generation ledger")
    selected = pd.concat([
        select_cases(candidates, "exact_pooled", args.n_exact_pooled),
        select_cases(candidates, "centroid_only", args.n_centroid_only),
    ], ignore_index=True).drop_duplicates("case_id")
    selected = selected.rename(columns={"case_id": "source_review_case_id", "ledger_case_id": "case_id"})
    selected = selected.sort_values("case_id").reset_index(drop=True)
    usable = ledger.loc[ledger.manual_verification_status.isin(["verified", "corrected"])].copy()
    template = usable.merge(
        selected[[
            "case_id", "source_review_case_id", "anchor_id", "dataset_row_id", "note_id",
            "review_stratum", "patient_disjoint_from_train", "smoke_selection_reason",
        ]],
        on="case_id", how="inner", validate="many_to_one",
    )
    template = template[[
        "case_id", "review_stratum", "patient_disjoint_from_train", "smoke_selection_reason",
        "fact_id", "field", "manual_verification_status", "manual_verified_value",
    ]].copy()
    template = template.rename(columns={"manual_verified_value": "source_fact_value_for_reviewer"})
    template["generation_value"] = ""
    template["generation_value_review_status"] = "pending"
    template["reviewer_note"] = ""
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = str(args.output_stem).strip()
    if not output_stem:
        raise ValueError("--output_stem must not be empty")
    template.to_csv(output_dir / f"{output_stem}_review_template_RESTRICTED.csv", index=False)
    selected.to_csv(output_dir / f"{output_stem}_case_manifest.csv", index=False)
    summary = {
        "n_cases": int(selected.case_id.nunique()),
        "ledger_case_ids": selected.case_id.tolist(),
        "source_review_case_ids": selected.source_review_case_id.tolist(),
        "review_stratum_counts": {str(key): int(value) for key, value in selected.review_stratum.value_counts().items()},
        "patient_disjoint_count": int(selected.patient_disjoint_from_train.sum()),
        "n_usable_fact_rows_requiring_generation_value": int(len(template)),
        "security_note": "Template contains source-derived values and must remain on approved MIMIC-IV storage.",
    }
    summary["output_stem"] = output_stem
    summary["excluded_prior_generation_ledger"] = str(Path(args.exclude_generation_ledger_path).resolve()) if args.exclude_generation_ledger_path else None
    (output_dir / f"{output_stem}_template_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
