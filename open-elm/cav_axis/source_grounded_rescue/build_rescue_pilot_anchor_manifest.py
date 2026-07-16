#!/usr/bin/env python3
"""Select a reproducible, stratified source-grounded rescue pilot cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stratified anchor manifest from completed source-paired review.")
    parser.add_argument("--completed_source_review_csv", required=True)
    parser.add_argument("--source_review_key_csv", required=True, help="Restricted unblinding key with anchor provenance.")
    parser.add_argument("--accepted_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_exact_pooled", type=int, default=20)
    parser.add_argument("--n_centroid_only", type=int, default=20)
    parser.add_argument("--n_patient_disjoint_extra", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def yes(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().isin({"yes", "true", "1", "y"})


def stratified_take(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if frame.empty or n <= 0:
        return frame.head(0).copy()
    order = frame.sort_values(["severity_rank", "case_id"], ascending=[False, True], kind="stable")
    if len(order) <= n:
        return order
    # Preserve severe cases while randomizing ties deterministically.
    return order.groupby("severity_rank", group_keys=False).apply(
        lambda group: group.sample(frac=1, random_state=seed)
    ).head(n).copy()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    review = pd.read_csv(Path(args.completed_source_review_csv).resolve())
    key = pd.read_csv(Path(args.source_review_key_csv).resolve())
    accepted = read_jsonl(Path(args.accepted_manifest_path).resolve())
    required_review = {"case_id", "review_stratum", "source_anchor_loss_flag", "severity_tier"}
    missing = required_review.difference(review.columns)
    if missing:
        raise KeyError(f"completed source review missing columns: {sorted(missing)}")
    required_key = {"case_id", "anchor_id", "dataset_row_id"}
    missing = required_key.difference(key.columns)
    if missing:
        raise KeyError(f"source review key missing columns: {sorted(missing)}")

    review["source_anchor_loss"] = yes(review["source_anchor_loss_flag"])
    review["severity_rank"] = review["severity_tier"].fillna("").astype(str).str.lower().map({"critical": 3, "severe": 3, "high": 2, "moderate": 1, "low": 0}).fillna(0).astype(int)
    case = review.groupby("case_id", as_index=False).agg(
        review_stratum=("review_stratum", "first"),
        source_anchor_loss=("source_anchor_loss", "max"),
        severity_rank=("severity_rank", "max"),
        n_reviewed_variants=("case_id", "size"),
    )
    key = key[["case_id", "anchor_id", "dataset_row_id"]].drop_duplicates("case_id", keep="first")
    selected = case.merge(key, on="case_id", how="inner", validate="one_to_one")
    accepted_cols = [column for column in ["anchor_id", "dataset_row_id", "note_id", "subject_id", "hadm_id", "patient_disjoint_from_train", "source_cluster_id"] if column in accepted.columns]
    accepted_lookup = accepted[accepted_cols].drop_duplicates(["anchor_id", "dataset_row_id"], keep="first")
    selected = selected.merge(accepted_lookup, on=["anchor_id", "dataset_row_id"], how="left", validate="one_to_one")
    selected["patient_disjoint_from_train"] = selected.get("patient_disjoint_from_train", False).fillna(False).astype(bool)

    exact = stratified_take(selected.loc[selected.review_stratum.astype(str) == "exact_pooled"], args.n_exact_pooled, args.seed)
    centroid = stratified_take(selected.loc[selected.review_stratum.astype(str) == "centroid_only"], args.n_centroid_only, args.seed + 1)
    first = pd.concat([exact, centroid], ignore_index=True).drop_duplicates("case_id")
    remaining_pd = selected.loc[selected.patient_disjoint_from_train & ~selected.case_id.isin(first.case_id)]
    extra = stratified_take(remaining_pd, args.n_patient_disjoint_extra, args.seed + 2)
    final = pd.concat([first, extra], ignore_index=True).drop_duplicates("case_id").sort_values("case_id").reset_index(drop=True)
    final["selection_reason"] = final.apply(
        lambda row: "patient_disjoint_extra" if row.case_id in set(extra.case_id) else f"review_stratum_{row.review_stratum}", axis=1
    )
    final.to_csv(output_dir / "rescue_pilot_anchor_manifest.csv", index=False)
    summary = {
        "n_selected": int(len(final)),
        "review_stratum_counts": {str(key): int(value) for key, value in final.review_stratum.value_counts().items()},
        "patient_disjoint_count": int(final.patient_disjoint_from_train.sum()),
        "source_anchor_loss_count": int(final.source_anchor_loss.sum()),
        "security_note": "This manifest contains provenance only; source-note text remains in approved pickle storage.",
    }
    (output_dir / "rescue_pilot_anchor_selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
