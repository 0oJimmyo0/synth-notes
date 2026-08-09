#!/usr/bin/env python3
"""Freeze a subject-unique development cohort for support/vanilla calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def load_prior_manifests(path: str) -> tuple[set[int], set[str], int]:
    dataset_ids: set[int] = set()
    subject_ids: set[str] = set()
    manifest_count = 0
    for raw in Path(path).resolve().read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        frame = pd.read_csv(Path(raw.strip()).resolve())
        if "dataset_row_id" not in frame:
            raise KeyError(f"Prior anchor manifest lacks dataset_row_id: {raw}")
        dataset_ids.update(pd.to_numeric(frame.dataset_row_id, errors="raise").astype(int))
        if "subject_id" in frame:
            subject_ids.update(frame.subject_id.dropna().astype(str))
        manifest_count += 1
    return dataset_ids, subject_ids, manifest_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--n_per_arm", type=int, default=30)
    parser.add_argument("--n_patient_disjoint_per_arm", type=int, default=15)
    parser.add_argument("--prior_anchor_manifest_list")
    parser.add_argument("--exclude_subject_csv", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.n_patient_disjoint_per_arm <= args.n_per_arm:
        raise ValueError("n_patient_disjoint_per_arm must be between zero and n_per_arm.")
    frame = pd.read_csv(Path(args.diagnostic_csv).resolve())
    needed = {"source_index", "dataset_row_id", "case_id", "subject_id", "patient_disjoint_from_train",
              "stable_sparse_k50_with_adjacent", "stable_dense_k50_with_adjacent", "mean_top_50_support"}
    missing = needed - set(frame)
    if missing:
        raise KeyError(f"Diagnostic CSV missing columns: {sorted(missing)}")
    frame["dataset_row_id"] = pd.to_numeric(frame.dataset_row_id, errors="raise").astype(int)
    frame["patient_disjoint_from_train"] = frame.patient_disjoint_from_train.astype(bool)
    prior_ids, prior_subjects, manifest_count = set(), set(), 0
    if args.prior_anchor_manifest_list:
        prior_ids, prior_subjects, manifest_count = load_prior_manifests(args.prior_anchor_manifest_list)
    for path in args.exclude_subject_csv:
        excluded = pd.read_csv(Path(path).resolve())
        if "subject_id" not in excluded:
            raise KeyError(f"Exclude-subject CSV lacks subject_id: {path}")
        prior_subjects.update(excluded.subject_id.dropna().astype(str))
    frame = frame.loc[
        ~frame.dataset_row_id.isin(prior_ids)
        & ~frame.subject_id.astype(str).isin(prior_subjects)
    ].copy()
    arms = {
        "stable_sparse": frame.stable_sparse_k50_with_adjacent.astype(bool),
        "stable_dense": frame.stable_dense_k50_with_adjacent.astype(bool),
    }
    selected, used_subjects = [], set()
    for arm, mask in arms.items():
        for is_disjoint, required_n in ((True, args.n_patient_disjoint_per_arm), (False, args.n_per_arm - args.n_patient_disjoint_per_arm)):
            candidates = frame.loc[mask & frame.patient_disjoint_from_train.eq(is_disjoint)].copy()
            candidates = candidates.loc[~candidates.subject_id.astype(str).isin(used_subjects)]
            candidates["selection_hash"] = candidates.apply(
                lambda row: rank(args.seed, f"{arm}|{is_disjoint}|{row.subject_id}|{row.source_index}"), axis=1
            )
            # A subject contributes at most one development note to this cohort.
            candidates = candidates.sort_values("selection_hash").drop_duplicates("subject_id", keep="first")
            chosen = candidates.head(required_n).copy()
            if len(chosen) != required_n:
                raise ValueError(f"Only {len(chosen)} unique-subject candidates available for {arm}, patient_disjoint={is_disjoint}.")
            chosen["support_arm"] = arm
            chosen["cohort_stratum"] = "patient_disjoint" if is_disjoint else "patient_overlap"
            selected.append(chosen)
            used_subjects.update(chosen.subject_id.astype(str))
    cohort = pd.concat(selected, ignore_index=True)
    cohort["selection_rank_within_stratum"] = cohort.groupby(["support_arm", "cohort_stratum"]).cumcount() + 1
    cohort = cohort.sort_values(["support_arm", "cohort_stratum", "selection_rank_within_stratum"])
    # Match the provenance schema consumed by source-grounded generation.
    cohort["anchor_id"] = cohort["dataset_row_id"].map(lambda value: f"anchor_{int(value)}")
    cohort["review_stratum"] = cohort["cohort_stratum"]
    if cohort.subject_id.astype(str).duplicated().any():
        raise ValueError("Selected cohort repeats a subject.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    keep = [
        "source_index", "dataset_row_id", "note_id", "case_id", "subject_id", "patient_disjoint_from_train",
        "anchor_id", "review_stratum", "support_arm", "cohort_stratum", "selection_rank_within_stratum", "mean_top_50_support",
        "sparse_frequency_k25", "sparse_frequency_k50", "sparse_frequency_k100",
    ]
    cohort[[column for column in keep if column in cohort]].to_csv(output_dir / "canonical_dev_support_vanilla_cohort.csv", index=False)
    summary = {
        "scope": "development_only_support_vanilla_calibration",
        "selection_seed": args.seed,
        "support_definition": "stable sparse at k=25 and k=50 with frequency >=0.80; stable dense has zero sparse membership at k=25,50,100",
        "n_notes": int(len(cohort)),
        "n_unique_subjects": int(cohort.subject_id.nunique()),
        "prior_anchor_manifest_count": manifest_count,
        "prior_dataset_row_ids_excluded": len(prior_ids),
        "prior_subjects_excluded": len(prior_subjects),
        "counts": cohort.groupby(["support_arm", "cohort_stratum"]).size().rename("n").reset_index().to_dict(orient="records"),
        "security_note": "Output contains provenance IDs and derived support labels only; no source-note text.",
    }
    (output_dir / "canonical_dev_support_vanilla_cohort_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
