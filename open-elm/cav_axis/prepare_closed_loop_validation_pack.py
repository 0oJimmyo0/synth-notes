#!/usr/bin/env python3
"""
Prepare a Phase 2 closed-loop validation pack from a completed enrichment run.

This script freezes one closed-loop run into:

1. full accepted-note review sheets,
2. a blinded comparison set with accepted / near-miss / vanilla controls,
3. source-cluster -> output-cluster transition tables,
4. accepted-note gate-route decomposition tables,
5. a concise JSON/Markdown summary for the run-level validation step.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REVIEW_COLUMNS = [
    "completeness_score_1to5",
    "internal_clinical_consistency_score_1to5",
    "temporal_consistency_score_1to5",
    "diagnosis_procedure_consistency_score_1to5",
    "medication_plausibility_score_1to5",
    "physiologic_plausibility_score_1to5",
    "repetition_collapse_score_1to5",
    "privacy_concern_score_1to5",
    "overall_usability_score_1to5",
    "overall_pass_fail",
    "reviewer_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare validation artifacts for a completed closed-loop enrichment run.")
    parser.add_argument("--accepted_manifest_path", required=True, help="Path to closed_loop_accepted_manifest.jsonl")
    parser.add_argument("--candidate_manifest_path", required=True, help="Path to closed_loop_candidate_manifest.jsonl")
    parser.add_argument("--rejected_manifest_path", required=True, help="Path to closed_loop_rejected_manifest.jsonl")
    parser.add_argument("--summary_json_path", required=True, help="Path to closed_loop_enrichment_summary.json")
    parser.add_argument("--output_dir", required=True, help="Output directory for validation artifacts")
    parser.add_argument(
        "--vanilla_manifest_path",
        default=None,
        help="Optional vanilla manifest for matched blinded controls",
    )
    parser.add_argument("--near_miss_count", type=int, default=50, help="Number of near-miss rejected controls")
    parser.add_argument("--vanilla_count", type=int, default=50, help="Number of matched vanilla controls")
    parser.add_argument("--overlap_review_count", type=int, default=40, help="Suggested overlapping note count for two reviewers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling blinded sets")
    return parser.parse_args()


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def split_reasons(text: Any) -> list[str]:
    if pd.isna(text):
        return []
    return [item for item in str(text).split("|") if item]


def build_full_accepted_review_sheet(accepted_df: pd.DataFrame) -> pd.DataFrame:
    review_df = accepted_df.copy().reset_index(drop=True)
    review_df.insert(0, "review_slot", np.arange(1, len(review_df) + 1))
    for col in REVIEW_COLUMNS:
        review_df[col] = ""
    keep_cols = [
        "review_slot",
        "candidate_id",
        "generated_text",
        "generated_word_count",
        "source_cluster_id",
        "nearest_cluster_id",
        "source_synthetic_cosine",
        "target_centroid_distance",
        "hit_max_new_tokens",
        "ended_with_eos",
        "patient_disjoint_from_train",
    ] + REVIEW_COLUMNS
    keep_cols = [col for col in keep_cols if col in review_df.columns]
    return review_df[keep_cols]


def choose_near_miss_rejects(rejected_df: pd.DataFrame, n: int) -> pd.DataFrame:
    if rejected_df.empty or n <= 0:
        return rejected_df.head(0).copy()

    ranked = rejected_df.copy()
    ranked["reason_count"] = ranked["rejection_reasons"].map(lambda value: len(split_reasons(value)))
    ranked["soft_pass_score"] = (
        ranked["target_gate_pass"].fillna(False).astype(int)
        + ranked["source_cosine_pass"].fillna(False).astype(int)
        + ranked["privacy_pass"].fillna(False).astype(int)
        + ranked["basic_quality_pass"].fillna(False).astype(int)
        + ranked["clinical_quality_pass"].fillna(False).astype(int)
        + ranked["structure_pass"].fillna(False).astype(int)
    )
    ranked = ranked.sort_values(
        ["soft_pass_score", "reason_count", "target_score", "source_synthetic_cosine"],
        ascending=[False, True, False, False],
    )
    ranked = ranked.drop_duplicates(subset=["anchor_id"], keep="first")
    return ranked.head(n).copy()


def choose_matched_vanilla_controls(vanilla_df: pd.DataFrame, accepted_df: pd.DataFrame, n: int) -> pd.DataFrame:
    if vanilla_df.empty or accepted_df.empty or n <= 0:
        return vanilla_df.head(0).copy()

    accepted_anchor_rows = accepted_df[["dataset_row_id", "generated_word_count"]].copy()
    accepted_anchor_rows["dataset_row_id"] = pd.to_numeric(accepted_anchor_rows["dataset_row_id"], errors="coerce")
    accepted_anchor_rows = accepted_anchor_rows.dropna(subset=["dataset_row_id"]).copy()
    accepted_anchor_rows["dataset_row_id"] = accepted_anchor_rows["dataset_row_id"].astype(int)

    vanilla = vanilla_df.copy()
    vanilla["dataset_row_id"] = pd.to_numeric(vanilla["dataset_row_id"], errors="coerce")
    vanilla = vanilla.dropna(subset=["dataset_row_id"]).copy()
    vanilla["dataset_row_id"] = vanilla["dataset_row_id"].astype(int)

    matched = vanilla.merge(
        accepted_anchor_rows.rename(columns={"generated_word_count": "accepted_generated_word_count"}),
        on="dataset_row_id",
        how="inner",
    )
    if matched.empty:
        return matched

    matched["generated_word_count"] = pd.to_numeric(matched.get("generated_word_count"), errors="coerce")
    matched["word_count_gap"] = (matched["generated_word_count"] - matched["accepted_generated_word_count"]).abs()
    matched = matched.sort_values(["word_count_gap", "dataset_row_id"])
    matched = matched.drop_duplicates(subset=["dataset_row_id"], keep="first")
    return matched.head(n).copy()


def build_blinded_comparison_set(
    accepted_df: pd.DataFrame,
    near_miss_df: pd.DataFrame,
    vanilla_df: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blinded_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []

    def add_rows(frame: pd.DataFrame, condition: str) -> None:
        for _, row in frame.iterrows():
            blinded_note_id = f"blind_{len(blinded_rows) + 1:04d}"
            blinded_rows.append(
                {
                    "blinded_note_id": blinded_note_id,
                    "generated_text": row.get("generated_text"),
                    **{col: "" for col in REVIEW_COLUMNS},
                }
            )
            key_rows.append(
                {
                    "blinded_note_id": blinded_note_id,
                    "condition": condition,
                    "candidate_id": row.get("candidate_id"),
                    "anchor_id": row.get("anchor_id"),
                    "dataset_row_id": row.get("dataset_row_id"),
                    "note_id": row.get("note_id"),
                    "source_cluster_id": row.get("source_cluster_id"),
                    "nearest_cluster_id": row.get("nearest_cluster_id"),
                    "source_synthetic_cosine": row.get("source_synthetic_cosine"),
                    "target_centroid_distance": row.get("target_centroid_distance"),
                    "target_gate_pass": row.get("target_gate_pass"),
                    "accepted_flag": row.get("accepted_flag"),
                    "rejection_reasons": row.get("rejection_reasons"),
                }
            )

    add_rows(accepted_df, "accepted_closed_loop")
    add_rows(near_miss_df, "near_miss_rejected")
    add_rows(vanilla_df, "matched_vanilla")

    blinded_df = pd.DataFrame(blinded_rows)
    key_df = pd.DataFrame(key_rows)

    order = list(range(len(blinded_df)))
    rng = random.Random(seed)
    rng.shuffle(order)
    blinded_df = blinded_df.iloc[order].reset_index(drop=True)
    key_df = key_df.iloc[order].reset_index(drop=True)
    return blinded_df, key_df


def build_transition_table(candidate_df: pd.DataFrame, accepted_only: bool = False) -> pd.DataFrame:
    df = candidate_df.loc[candidate_df["accepted_flag"].fillna(False)].copy() if accepted_only else candidate_df.copy()
    if df.empty:
        return df.head(0)
    df["source_cluster_id"] = pd.to_numeric(df["source_cluster_id"], errors="coerce")
    df["nearest_cluster_id"] = pd.to_numeric(df["nearest_cluster_id"], errors="coerce")
    df = df.dropna(subset=["source_cluster_id", "nearest_cluster_id"]).copy()
    df["source_cluster_id"] = df["source_cluster_id"].astype(int)
    df["nearest_cluster_id"] = df["nearest_cluster_id"].astype(int)

    trans = (
        df.groupby(["source_cluster_id", "nearest_cluster_id"], as_index=False)
        .agg(
            n_candidates=("candidate_id", "size"),
            n_unique_anchors=("anchor_id", "nunique"),
            mean_source_cosine=("source_synthetic_cosine", "mean"),
            mean_target_centroid_distance=("target_centroid_distance", "mean"),
        )
        .sort_values(["source_cluster_id", "n_candidates"], ascending=[True, False])
    )
    trans["fraction_within_source_cluster"] = trans["n_candidates"] / trans.groupby("source_cluster_id")["n_candidates"].transform("sum")
    return trans.reset_index(drop=True)


def build_source_cluster_summary(candidate_df: pd.DataFrame, target_cluster_ids: list[int]) -> pd.DataFrame:
    df = candidate_df.copy()
    df["source_cluster_id"] = pd.to_numeric(df["source_cluster_id"], errors="coerce")
    df["nearest_cluster_id"] = pd.to_numeric(df["nearest_cluster_id"], errors="coerce")
    df = df.dropna(subset=["source_cluster_id"]).copy()
    df["source_cluster_id"] = df["source_cluster_id"].astype(int)

    rows: list[dict[str, Any]] = []
    for source_cluster_id, group_df in df.groupby("source_cluster_id", sort=True):
        accepted = group_df["accepted_flag"].fillna(False)
        rows.append(
            {
                "source_cluster_id": int(source_cluster_id),
                "n_unique_anchors": int(group_df["anchor_id"].nunique()),
                "n_candidates": int(len(group_df)),
                "n_target_gate_pass": int(group_df["target_gate_pass"].fillna(False).sum()),
                "n_accepted": int(accepted.sum()),
                "p_output_in_pooled_basin": float(group_df["nearest_cluster_in_target"].fillna(False).mean()),
                "p_accepted": float(accepted.mean()),
                "p_return_to_exact_source_cluster": float((group_df["nearest_cluster_id"] == int(source_cluster_id)).fillna(False).mean()),
                **{
                    f"p_output_cluster_{cluster_id}": float((group_df["nearest_cluster_id"] == cluster_id).fillna(False).mean())
                    for cluster_id in sorted(target_cluster_ids)
                },
            }
        )
    return pd.DataFrame(rows)


def classify_gate_route(row: pd.Series) -> str:
    if "gate_route" in row and pd.notna(row.get("gate_route")):
        return str(row["gate_route"])
    exact_pass = bool(row.get("exact_pooled_cluster_pass", row.get("nearest_cluster_in_target", False)))
    centroid_pass = bool(row.get("centroid_proximity_pass", row.get("target_centroid_distance_pass", False)))
    if exact_pass and centroid_pass:
        return "exact_plus_centroid"
    if exact_pass:
        return "exact_only"
    if centroid_pass:
        return "centroid_only"
    return "neither"


def build_gate_route_tables(accepted_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = accepted_df.copy()
    df["gate_route"] = df.apply(classify_gate_route, axis=1)
    route_summary = (
        df.groupby("gate_route", as_index=False)
        .agg(
            n_notes=("candidate_id", "size"),
            n_unique_anchors=("anchor_id", "nunique"),
            mean_source_cosine=("source_synthetic_cosine", "mean"),
            mean_target_centroid_distance=("target_centroid_distance", "mean"),
            mean_generated_word_count=("generated_word_count", "mean"),
            exact_target_cluster_rate=("nearest_cluster_in_target", "mean"),
            repetition_or_collapse_rate=("repetition_or_collapse_flag", "mean"),
        )
        .sort_values("n_notes", ascending=False)
        .reset_index(drop=True)
    )
    route_detail = df[
        [
            "candidate_id",
            "anchor_id",
            "dataset_row_id",
            "source_cluster_id",
            "nearest_cluster_id",
            "gate_route",
            "source_synthetic_cosine",
            "target_centroid_distance",
            "generated_word_count",
            "repetition_or_collapse_flag",
            "hit_max_new_tokens",
            "ended_with_eos",
            "patient_disjoint_from_train",
        ]
    ].copy()
    return route_summary, route_detail


def main() -> None:
    args = parse_args()

    accepted_manifest_path = Path(args.accepted_manifest_path).resolve()
    candidate_manifest_path = Path(args.candidate_manifest_path).resolve()
    rejected_manifest_path = Path(args.rejected_manifest_path).resolve()
    summary_json_path = Path(args.summary_json_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    accepted_df = load_jsonl(accepted_manifest_path)
    candidate_df = load_jsonl(candidate_manifest_path)
    rejected_df = load_jsonl(rejected_manifest_path)
    summary = json.loads(summary_json_path.read_text())
    vanilla_df = load_jsonl(Path(args.vanilla_manifest_path).resolve()) if args.vanilla_manifest_path else pd.DataFrame()

    accepted_review_df = build_full_accepted_review_sheet(accepted_df)
    accepted_review_path = output_dir / "accepted_106_full_review_sheet.csv"
    accepted_review_df.to_csv(accepted_review_path, index=False)

    reviewer_a_path = output_dir / "accepted_106_full_review_sheet_reviewerA.csv"
    reviewer_b_path = output_dir / "accepted_106_full_review_sheet_reviewerB.csv"
    accepted_review_df.to_csv(reviewer_a_path, index=False)
    accepted_review_df.to_csv(reviewer_b_path, index=False)

    overlap_subset = accepted_review_df.head(min(int(args.overlap_review_count), len(accepted_review_df))).copy()
    overlap_subset.to_csv(output_dir / "accepted_overlap_subset_for_dual_review.csv", index=False)

    near_miss_df = choose_near_miss_rejects(rejected_df, int(args.near_miss_count))
    vanilla_controls_df = choose_matched_vanilla_controls(vanilla_df, accepted_df, int(args.vanilla_count))

    blinded_df, blinded_key_df = build_blinded_comparison_set(
        accepted_df=accepted_df,
        near_miss_df=near_miss_df,
        vanilla_df=vanilla_controls_df,
        seed=int(args.seed),
    )
    blinded_df.to_csv(output_dir / "blinded_review_set.csv", index=False)
    blinded_key_df.to_csv(output_dir / "blinded_review_key.csv", index=False)

    all_transition_df = build_transition_table(candidate_df, accepted_only=False)
    accepted_transition_df = build_transition_table(candidate_df, accepted_only=True)
    source_cluster_summary_df = build_source_cluster_summary(candidate_df, target_cluster_ids=summary["target_cluster_ids"])
    all_transition_df.to_csv(output_dir / "source_to_output_transition_all_candidates.csv", index=False)
    accepted_transition_df.to_csv(output_dir / "source_to_output_transition_accepted_only.csv", index=False)
    source_cluster_summary_df.to_csv(output_dir / "source_cluster_landing_summary.csv", index=False)

    route_summary_df, route_detail_df = build_gate_route_tables(accepted_df)
    route_summary_df.to_csv(output_dir / "accepted_gate_route_summary.csv", index=False)
    route_detail_df.to_csv(output_dir / "accepted_gate_route_detail.csv", index=False)

    validation_summary = {
        "run_summary_json": str(summary_json_path),
        "total_accepted_notes": int(len(accepted_df)),
        "accepted_unique_anchors": int(accepted_df["anchor_id"].nunique()) if "anchor_id" in accepted_df.columns else 0,
        "near_miss_rejected_controls": int(len(near_miss_df)),
        "matched_vanilla_controls": int(len(vanilla_controls_df)),
        "blinded_review_total_notes": int(len(blinded_df)),
        "target_cluster_ids": summary.get("target_cluster_ids"),
        "accepted_exact_target_cluster_rate": float(accepted_df["nearest_cluster_in_target"].fillna(False).mean()) if len(accepted_df) else math.nan,
        "accepted_cluster_counts": {
            str(int(k)): int(v)
            for k, v in Counter(pd.to_numeric(accepted_df["nearest_cluster_id"], errors="coerce").dropna().astype(int).tolist()).items()
        },
        "notes": [
            "Exact pooled-cluster membership and centroid proximity are reported as separate gate routes.",
            "For legacy manifests, routes are backfilled from nearest_cluster_id and target_centroid_distance_pass; exact pooled membership remains the primary enrichment endpoint.",
        ],
        "output_files": {
            "accepted_full_review_sheet": str(accepted_review_path),
            "accepted_full_review_sheet_reviewerA": str(reviewer_a_path),
            "accepted_full_review_sheet_reviewerB": str(reviewer_b_path),
            "accepted_overlap_subset_for_dual_review": str(output_dir / "accepted_overlap_subset_for_dual_review.csv"),
            "blinded_review_set": str(output_dir / "blinded_review_set.csv"),
            "blinded_review_key": str(output_dir / "blinded_review_key.csv"),
            "source_to_output_transition_all_candidates": str(output_dir / "source_to_output_transition_all_candidates.csv"),
            "source_to_output_transition_accepted_only": str(output_dir / "source_to_output_transition_accepted_only.csv"),
            "source_cluster_landing_summary": str(output_dir / "source_cluster_landing_summary.csv"),
            "accepted_gate_route_summary": str(output_dir / "accepted_gate_route_summary.csv"),
            "accepted_gate_route_detail": str(output_dir / "accepted_gate_route_detail.csv"),
        },
    }

    (output_dir / "closed_loop_validation_pack_summary.json").write_text(
        json.dumps(validation_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    md_lines = [
        "# Closed-Loop Validation Pack",
        "",
        f"- Accepted notes frozen for review: `{validation_summary['total_accepted_notes']}`",
        f"- Unique accepted anchors: `{validation_summary['accepted_unique_anchors']}`",
        f"- Near-miss rejected controls: `{validation_summary['near_miss_rejected_controls']}`",
        f"- Matched vanilla controls: `{validation_summary['matched_vanilla_controls']}`",
        f"- Blinded comparison total: `{validation_summary['blinded_review_total_notes']}`",
        "",
        "## Notes",
    ]
    md_lines.extend([f"- {line}" for line in validation_summary["notes"]])
    (output_dir / "closed_loop_validation_pack_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved accepted review sheets to: {accepted_review_path}")
    print(f"Saved blinded review set to: {output_dir / 'blinded_review_set.csv'}")
    print(f"Saved transition tables to: {output_dir / 'source_to_output_transition_all_candidates.csv'}")
    print(f"Saved gate-route tables to: {output_dir / 'accepted_gate_route_summary.csv'}")
    print(f"Saved validation summary to: {output_dir / 'closed_loop_validation_pack_summary.json'}")


if __name__ == "__main__":
    main()
