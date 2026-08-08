#!/usr/bin/env python3
"""Freeze a subject-unique held-out screening cohort from canonical support labels."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic_csv", required=True)
    parser.add_argument("--prior_anchor_manifest_list", required=True)
    parser.add_argument("--exclude_subject_csv", action="append", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--n_per_arm", type=int, default=30,
        help="Total anchors per support arm across both patient-disjoint strata.",
    )
    parser.add_argument(
        "--n_patient_disjoint_per_arm", type=int, default=15,
        help="Patient-disjoint anchors within each support arm; the remainder are patient-overlap.",
    )
    args = parser.parse_args()
    if not 0 <= args.n_patient_disjoint_per_arm <= args.n_per_arm:
        raise ValueError("n_patient_disjoint_per_arm must be between zero and n_per_arm.")
    frame = pd.read_csv(Path(args.diagnostic_csv).resolve())
    required = {
        "dataset_row_id", "case_id", "subject_id", "patient_disjoint_from_train",
        "stable_sparse_k50_with_adjacent", "stable_dense_k50_with_adjacent", "mean_top_50_support",
    }
    if missing := required - set(frame):
        raise KeyError(f"Diagnostic CSV missing columns: {sorted(missing)}")
    prior_ids, prior_subjects, manifest_count = load_prior_manifests(args.prior_anchor_manifest_list)
    for path in args.exclude_subject_csv:
        extra = pd.read_csv(Path(path).resolve())
        if "subject_id" not in extra:
            raise KeyError(f"Exclude-subject CSV lacks subject_id: {path}")
        prior_subjects.update(extra.subject_id.dropna().astype(str))
    frame["dataset_row_id"] = pd.to_numeric(frame.dataset_row_id, errors="raise").astype(int)
    frame["patient_disjoint_from_train"] = frame.patient_disjoint_from_train.astype(bool)
    eligible = frame.loc[
        ~frame.dataset_row_id.isin(prior_ids)
        & ~frame.subject_id.astype(str).isin(prior_subjects)
    ].copy()
    selections, used_subjects = [], set()
    for arm, column in (("stable_sparse", "stable_sparse_k50_with_adjacent"), ("stable_dense", "stable_dense_k50_with_adjacent")):
        for is_disjoint, n_required in ((True, args.n_patient_disjoint_per_arm), (False, args.n_per_arm - args.n_patient_disjoint_per_arm)):
            candidates = eligible.loc[
                eligible[column].astype(bool) & eligible.patient_disjoint_from_train.eq(is_disjoint)
            ].copy()
            candidates = candidates.loc[~candidates.subject_id.astype(str).isin(used_subjects)]
            candidates["selection_hash"] = candidates.apply(
                lambda row: rank(args.seed, f"{arm}|{is_disjoint}|{row.subject_id}|{row.dataset_row_id}"), axis=1
            )
            candidates = candidates.sort_values("selection_hash").drop_duplicates("subject_id", keep="first")
            chosen = candidates.head(n_required).copy()
            if len(chosen) != n_required:
                raise ValueError(f"Only {len(chosen)} candidates available for {arm}, patient_disjoint={is_disjoint}.")
            chosen["support_arm"] = arm
            chosen["cohort_stratum"] = "patient_disjoint" if is_disjoint else "patient_overlap"
            selections.append(chosen)
            used_subjects.update(chosen.subject_id.astype(str))
    cohort = pd.concat(selections, ignore_index=True)
    cohort["selection_rank_within_stratum"] = cohort.groupby(["support_arm", "cohort_stratum"]).cumcount() + 1
    # These aliases make the frozen screening manifest directly compatible with
    # the source-grounded ledger provenance schema without changing selection.
    cohort["anchor_id"] = cohort["dataset_row_id"].map(lambda value: f"anchor_{int(value)}")
    cohort["review_stratum"] = cohort["cohort_stratum"]
    if cohort.subject_id.astype(str).duplicated().any() or cohort.dataset_row_id.duplicated().any():
        raise ValueError("Frozen cohort contains duplicate subjects or dataset rows.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_stem = f"canonical_test_support_screening{len(cohort)}"
    keep = [
        "source_index", "dataset_row_id", "note_id", "case_id", "subject_id", "patient_disjoint_from_train",
        "anchor_id", "review_stratum", "support_arm", "cohort_stratum", "selection_rank_within_stratum", "mean_top_50_support",
        "sparse_frequency_k25", "sparse_frequency_k50", "sparse_frequency_k100",
    ]
    cohort[[column for column in keep if column in cohort]].to_csv(output_dir / f"{cohort_stem}.csv", index=False)
    summary = {
        "scope": "frozen_heldout_test_screening_before_source_review",
        "selection_seed": args.seed,
        "n_notes": int(len(cohort)),
        "n_unique_subjects": int(cohort.subject_id.nunique()),
        "prior_anchor_manifest_count": manifest_count,
        "prior_dataset_row_ids_excluded": len(prior_ids),
        "prior_subjects_excluded": len(prior_subjects),
        "counts": cohort.groupby(["support_arm", "cohort_stratum"]).size().rename("n").reset_index().to_dict(orient="records"),
        "security_note": "Output contains provenance IDs and derived support labels only; no source-note text.",
    }
    (output_dir / f"{cohort_stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
