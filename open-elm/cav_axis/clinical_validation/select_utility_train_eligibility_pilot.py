#!/usr/bin/env python3
"""Select one outcome-blind utility-train note per sampled patient."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def rank(seed: int, *values: object) -> str:
    return hashlib.sha256(":".join(map(str, (seed, *values))).encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility_train_dev_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_patients", type=int, default=400)
    parser.add_argument(
        "--length_quantiles",
        type=int,
        default=5,
        help="Select proportionally across note-length quantiles; use 0 for unstratified selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(Path(args.utility_train_dev_manifest).resolve(), dtype=str).fillna("")
    needed = {"dataset_row_id", "subject_id", "hadm_id", "note_id", "split"}
    if args.length_quantiles:
        needed.add("text_length")
    if missing := needed.difference(frame.columns):
        raise KeyError(f"utility manifest missing columns: {sorted(missing)}")
    frame = frame.loc[frame.split.eq("train")].copy()
    if frame.empty:
        raise ValueError("utility manifest contains no train rows")
    frame["subject_id"] = frame.subject_id.astype(str)
    frame["_note_rank"] = [rank(args.seed, "note", *row) for row in frame[["subject_id", "dataset_row_id"]].itertuples(index=False, name=None)]
    one_per_subject = frame.sort_values("_note_rank", kind="stable").groupby("subject_id", as_index=False).first()
    one_per_subject["_subject_rank"] = one_per_subject.subject_id.map(lambda value: rank(args.seed, "subject", value))
    if args.length_quantiles:
        if args.length_quantiles < 2:
            raise ValueError("length_quantiles must be at least 2 or 0")
        lengths = pd.to_numeric(one_per_subject["text_length"], errors="raise")
        one_per_subject["note_length_quintile"] = pd.qcut(
            lengths.rank(method="first"),
            q=args.length_quantiles,
            labels=False,
        ).astype(int).add(1)
        base, remainder = divmod(args.n_patients, args.length_quantiles)
        selected_parts = []
        for quantile in range(1, args.length_quantiles + 1):
            take = base + int(quantile <= remainder)
            part = one_per_subject.loc[
                one_per_subject.note_length_quintile.eq(quantile)
            ].sort_values("_subject_rank", kind="stable").head(take)
            if len(part) != take:
                raise ValueError(f"length quantile {quantile} has fewer than {take} patients")
            selected_parts.append(part)
        selected = pd.concat(selected_parts, ignore_index=True)
    else:
        selected = one_per_subject.sort_values("_subject_rank", kind="stable").head(args.n_patients).copy()
    if len(selected) != args.n_patients:
        raise ValueError(f"requested {args.n_patients} patients but found {len(selected)}")
    selected["anchor_id"] = selected.dataset_row_id.map(lambda value: f"utility_train_{value}")
    selected["review_stratum"] = "utility_train_eligibility_pilot"
    selected["patient_disjoint_from_train"] = False
    selected = selected.drop(columns=["_note_rank", "_subject_rank"])
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_name = f"utility_train_eligibility_pilot{args.n_patients}"
    selected.to_csv(output_dir / f"{cohort_name}.csv", index=False)
    summary = {
        "scope": "outcome_blind_utility_train_v4_eligibility_pilot",
        "selection_seed": args.seed,
        "n_notes": int(len(selected)),
        "n_unique_subjects": int(selected.subject_id.nunique()),
        "source_split": "utility_train",
        "selection": "one hash-selected note per patient; no outcome fields accessed",
        "length_quantiles": args.length_quantiles,
        "note_length_strata": (
            selected.note_length_quintile.value_counts().sort_index().to_dict()
            if args.length_quantiles
            else None
        ),
        "service_stratification": "not available in the frozen utility manifest; not inferred",
        "security_note": "Output contains provenance IDs and split labels only; no note text or outcomes.",
    }
    (output_dir / f"{cohort_name}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
