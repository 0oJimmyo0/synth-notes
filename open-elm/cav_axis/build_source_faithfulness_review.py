#!/usr/bin/env python3
"""Build a secure, source-paired faithfulness review pack for closed-loop notes.

The review sheet deliberately contains real source-note text and must remain in
the approved project filesystem.  Synthetic conditions A/B are blinded; the
separate key maps them to accepted closed-loop or matched vanilla.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from closed_loop_train_text_privacy_screen import infer_pickle_dir, load_note_texts_for_rows


REVIEW_FIELDS = [
    "principal_diagnosis_preserved_yes_no",
    "major_procedures_preserved_yes_no",
    "complications_preserved_yes_no",
    "discharge_disposition_preserved_yes_no",
    "important_medication_changes_preserved_yes_no",
    "unsupported_major_claim_yes_no",
    "critical_omission_yes_no",
    "overall_source_faithfulness_score_1to5",
    "overall_source_faithfulness_pass_fail",
    "reviewer_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a blinded source-faithfulness review pack.")
    parser.add_argument("--accepted_manifest_path", required=True)
    parser.add_argument("--vanilla_manifest_path", required=True)
    parser.add_argument("--split_manifest_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_cluster_ids", default="9,17,29,45")
    parser.add_argument("--centroid_only_sample_size", type=int, default=34)
    parser.add_argument("--pickle_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: str) -> pd.DataFrame:
    return pd.read_json(Path(path).resolve(), lines=True)


def as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).map({True: True, False: False, "True": True, "False": False, 1: True, 0: False}).fillna(False).astype(bool)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    target_ids = {int(v.strip()) for v in args.target_cluster_ids.split(",") if v.strip()}
    accepted = load_jsonl(args.accepted_manifest_path)
    vanilla = load_jsonl(args.vanilla_manifest_path)
    for frame, label in [(accepted, "accepted"), (vanilla, "vanilla")]:
        if frame.anchor_id.duplicated().any():
            raise ValueError(f"{label} must contain exactly one note per anchor")
        frame["nearest_cluster_id"] = pd.to_numeric(frame["nearest_cluster_id"], errors="coerce")
        frame["exact_pooled_cluster_pass"] = frame.nearest_cluster_id.isin(target_ids)
    exact = accepted.loc[accepted.exact_pooled_cluster_pass].copy()
    centroid_only = accepted.loc[~accepted.exact_pooled_cluster_pass & as_bool(accepted.target_centroid_distance_pass)].copy()
    centroid_only = centroid_only.sample(n=min(len(centroid_only), args.centroid_only_sample_size), random_state=args.seed)
    selected = pd.concat([exact.assign(review_stratum="exact_pooled"), centroid_only.assign(review_stratum="centroid_only")], ignore_index=True)
    vanilla_cols = ["anchor_id", "dataset_row_id", "generated_text", "candidate_id"]
    pairs = selected.merge(vanilla[vanilla_cols], on=["anchor_id", "dataset_row_id"], suffixes=("_accepted", "_vanilla"), validate="one_to_one")
    split = pd.read_csv(Path(args.split_manifest_path).resolve())
    split["dataset_row_id"] = pd.to_numeric(split.dataset_row_id, errors="raise").astype(int)
    if "split" in split.columns:
        split = split.loc[split["split"].astype(str) == "test"].copy()
    duplicate_rows = split.loc[split.duplicated("dataset_row_id", keep=False)]
    if not duplicate_rows.empty:
        consistency_cols = [col for col in ["note_id", "filename"] if col in duplicate_rows.columns]
        inconsistent = duplicate_rows.groupby("dataset_row_id")[consistency_cols].nunique(dropna=False).max(axis=1) > 1
        if inconsistent.any():
            raise ValueError("Test split manifest has conflicting provenance for duplicate dataset_row_id values")
        split = split.drop_duplicates("dataset_row_id", keep="first")
    source_cols = [col for col in ["dataset_row_id", "note_id", "filename"] if col in split.columns]
    source_lookup = pairs[["dataset_row_id"]].merge(
        split[source_cols], on="dataset_row_id", how="left", validate="many_to_one"
    )
    pickle_dir = infer_pickle_dir(Path(args.dataset_path).resolve(), explicit_pickle_dir=Path(args.pickle_dir).resolve() if args.pickle_dir else None)
    if pickle_dir is None:
        raise FileNotFoundError("Could not resolve pickle_ds_note_hadm_all")
    source_texts = load_note_texts_for_rows(source_lookup, pickle_dir)
    rng = random.Random(args.seed)
    review_rows, key_rows = [], []
    for index, row in pairs.reset_index(drop=True).iterrows():
        source = source_texts.get(int(row.dataset_row_id), "")
        if not source:
            continue
        accepted_text, vanilla_text = row.generated_text_accepted, row.generated_text_vanilla
        flip = bool(rng.getrandbits(1))
        a_text, b_text = (accepted_text, vanilla_text) if not flip else (vanilla_text, accepted_text)
        case_id = f"source_review_{len(review_rows)+1:03d}"
        review_rows.append({"case_id": case_id, "review_stratum": row.review_stratum, "source_real_note": source, "synthetic_note_A": a_text, "synthetic_note_B": b_text, **{field: "" for field in REVIEW_FIELDS}})
        key_rows.append({"case_id": case_id, "anchor_id": row.anchor_id, "dataset_row_id": int(row.dataset_row_id), "review_stratum": row.review_stratum, "synthetic_A_condition": "accepted_closed_loop" if not flip else "matched_vanilla", "synthetic_B_condition": "matched_vanilla" if not flip else "accepted_closed_loop", "accepted_candidate_id": row.candidate_id_accepted, "vanilla_candidate_id": row.candidate_id_vanilla})
    pd.DataFrame(review_rows).to_csv(output_dir / "source_faithfulness_blinded_review.csv", index=False)
    pd.DataFrame(key_rows).to_csv(output_dir / "source_faithfulness_blinded_key.csv", index=False)
    summary = {"target_cluster_ids": sorted(target_ids), "n_exact_pooled": int(len(exact)), "n_centroid_only_sampled": int(len(centroid_only)), "n_completed_source_pairs": int(len(review_rows)), "pickle_dir": str(pickle_dir), "security_note": "The review CSV contains real source-note text; do not move it outside the approved project filesystem."}
    (output_dir / "source_faithfulness_review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
