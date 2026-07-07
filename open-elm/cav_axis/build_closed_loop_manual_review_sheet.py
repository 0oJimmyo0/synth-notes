#!/usr/bin/env python3
"""
Build a prefilled manual-review sheet from a closed-loop accepted-note sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a manual review sheet for accepted closed-loop notes.")
    parser.add_argument("--accepted_sample_path", required=True, help="Path to accepted_note_manual_review_sample.csv")
    parser.add_argument("--output_path", required=True, help="Path to write the review CSV")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of rows to include")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_path = Path(args.accepted_sample_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sample_path)
    if args.limit:
        df = df.head(int(args.limit)).copy()
    df = df.reset_index(drop=True)

    review_df = pd.DataFrame(
        {
            "review_slot": range(1, len(df) + 1),
            "candidate_id": df.get("candidate_id"),
            "anchor_id": df.get("anchor_id"),
            "dataset_row_id": df.get("dataset_row_id"),
            "note_id": df.get("note_id"),
            "subject_id": df.get("subject_id"),
            "hadm_id": df.get("hadm_id"),
            "patient_disjoint_from_train": df.get("patient_disjoint_from_train"),
            "source_synthetic_cosine": df.get("source_synthetic_cosine"),
            "target_centroid_cosine": df.get("target_centroid_cosine"),
            "nearest_cluster_id": df.get("nearest_cluster_id"),
            "generated_word_count": df.get("generated_word_count"),
            "readability_score_1to5": "",
            "discharge_summary_structure_score_1to5": "",
            "collapse_or_repetition_flag_manual": "",
            "section_layout_score_1to5": "",
            "hallucination_flag": "",
            "phi_like_leakage_flag_manual": "",
            "source_conditioned_feel_score_1to5": "",
            "overall_pass_fail": "",
            "reviewer_notes": "",
            "generated_text": df.get("generated_text"),
        }
    )
    review_df.to_csv(output_path, index=False)
    print(f"Saved closed-loop manual review sheet to: {output_path}")
    print(f"Rows: {len(review_df)}")


if __name__ == "__main__":
    main()
