#!/usr/bin/env python3
"""Select one source-grounded candidate per anchor using final BGE target-basin landing."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.cluster import MiniBatchKMeans


# These are structural presence checks, not factuality checks.  The blinded
# ledger review remains the authority for whether a stated field is supported.
REQUIRED_FIELD_HEADING_PATTERNS = {
    "principal_diagnosis": re.compile(r"(?im)^\s*(?:discharge\s+)?diagnos(?:is|es)\s*:"),
    "hospital_course_events": re.compile(r"(?im)^\s*(?:brief\s+)?hospital\s+course\s*:"),
    "discharge_medications": re.compile(r"(?im)^\s*(?:discharge\s+)?medications?\s*:"),
    "disposition": re.compile(r"(?im)^\s*(?:discharge\s+)?disposition\s*:"),
    "instructions": re.compile(r"(?im)^\s*(?:discharge\s+)?instructions?\s*:"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_manifest_path", required=True)
    parser.add_argument("--candidate_embeddings_path", required=True)
    parser.add_argument("--real_dataset_path", required=True)
    parser.add_argument("--real_cluster_assignments_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_cluster_ids", default="9,17,29,45")
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--generation_ledger_path", default=None, help="Optional prompt-safe ledger for deterministic output eligibility checks.")
    parser.add_argument("--enforce_absent_followup_omission", action="store_true")
    parser.add_argument(
        "--reject_cap_hits",
        action="store_true",
        help="Reject candidates recorded as reaching max_new_tokens before final-output geometry ranking.",
    )
    parser.add_argument(
        "--enforce_required_field_headings",
        action="store_true",
        help="Reject outputs missing a recognizable heading for a required ledger field. This is a structural screen, not factual validation.",
    )
    return parser.parse_args()


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)


def load_ledger_field_map(ledger_path: str | None) -> dict[str, set[str]]:
    if not ledger_path:
        raise ValueError("A generation ledger is required for ledger-aware output eligibility checks.")
    ledgers = [json.loads(line) for line in Path(ledger_path).read_text().splitlines() if line.strip()]
    field_map = {
        str(ledger["case_id"]): {str(fact.get("field")) for fact in ledger.get("facts", [])}
        for ledger in ledgers
    }
    if len(field_map) != len(ledgers):
        raise ValueError("Generation ledger contains duplicate case IDs.")
    return field_map


def append_rejection_reasons(candidate: pd.DataFrame, rejected: pd.Series, reason: str) -> None:
    current = candidate.loc[rejected, "selection_rejection_reason"].astype(str)
    candidate.loc[rejected, "selection_rejection_reason"] = np.where(
        current.eq(""), reason, current + "|" + reason
    )
    candidate.loc[rejected, "eligible_for_selection"] = False


def main() -> None:
    args = parse_args()
    target_ids = {int(value) for value in args.target_cluster_ids.split(",") if value.strip()}
    candidate = pd.read_json(args.candidate_manifest_path, lines=True).reset_index(drop=True)
    embeddings = normalize(np.load(args.candidate_embeddings_path))
    if len(candidate) != len(embeddings):
        raise ValueError("Candidate manifest and embedding matrix row counts differ.")
    if candidate["rescue_id"].duplicated().any() or candidate["generated_text"].astype(str).str.strip().eq("").any():
        raise ValueError("Candidate manifest has duplicate IDs or empty notes.")
    candidate["eligible_for_selection"] = True
    candidate["selection_rejection_reason"] = ""
    cap_hit_rejection_count = 0
    if args.reject_cap_hits:
        if "hit_max_new_tokens" not in candidate.columns:
            raise KeyError("--reject_cap_hits requires hit_max_new_tokens in the candidate manifest.")
        cap_hit = candidate["hit_max_new_tokens"].fillna(False).astype(bool)
        cap_hit_rejection_count = int(cap_hit.sum())
        append_rejection_reasons(candidate, cap_hit, "hit_max_new_tokens")
    ledger_fields_by_case: dict[str, set[str]] = {}
    if args.enforce_absent_followup_omission or args.enforce_required_field_headings:
        ledger_fields_by_case = load_ledger_field_map(args.generation_ledger_path)
        if set(candidate.case_id.astype(str)).difference(ledger_fields_by_case):
            raise ValueError("Candidate manifest includes cases absent from the generation ledger.")
    if args.enforce_absent_followup_omission:
        follow_up_by_case = {case_id: "follow_up" in fields for case_id, fields in ledger_fields_by_case.items()}
        candidate["ledger_has_follow_up"] = candidate.case_id.astype(str).map(follow_up_by_case).astype(bool)
        follow_up_pattern = re.compile(r"\bfollow[\s-]*up\b", flags=re.IGNORECASE)
        candidate["unsupported_follow_up_mention"] = [
            (not has_follow_up) and bool(follow_up_pattern.search(str(text)))
            for has_follow_up, text in zip(candidate["ledger_has_follow_up"], candidate["generated_text"])
        ]
        rejected = candidate["unsupported_follow_up_mention"]
        append_rejection_reasons(candidate, rejected, "unsupported_follow_up_mention_without_ledger_fact")
    required_field_rejection_counts: dict[str, int] = {}
    if args.enforce_required_field_headings:
        candidate["ledger_required_fields"] = candidate.case_id.astype(str).map(
            lambda case_id: ",".join(sorted(field for field in ledger_fields_by_case[case_id] if field in REQUIRED_FIELD_HEADING_PATTERNS))
        )
        for field, pattern in REQUIRED_FIELD_HEADING_PATTERNS.items():
            required_for_case = candidate.case_id.astype(str).map(lambda case_id: field in ledger_fields_by_case[case_id])
            has_heading = candidate["generated_text"].astype(str).map(lambda text: bool(pattern.search(text)))
            missing_heading = required_for_case & ~has_heading
            candidate[f"missing_{field}_heading"] = missing_heading
            required_field_rejection_counts[field] = int(missing_heading.sum())
            append_rejection_reasons(candidate, missing_heading, f"missing_{field}_heading")
    real = Dataset.load_from_disk(args.real_dataset_path)
    real_embeddings = normalize(np.vstack([np.asarray(row["domain_embeddings"][0], dtype=np.float32) for row in real]))
    assignments = pd.read_csv(args.real_cluster_assignments_path).sort_values("dataset_row_id")
    kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=args.random_seed, batch_size=2048, n_init="auto")
    labels = kmeans.fit_predict(real_embeddings)
    if not np.array_equal(labels, assignments["cluster_id"].to_numpy()):
        raise ValueError("Refit clusters differ from frozen real cluster assignments.")
    centers = normalize(kmeans.cluster_centers_)
    scores = embeddings @ centers.T
    candidate["output_cluster_id"] = scores.argmax(axis=1).astype(int)
    candidate["output_in_target_basin"] = candidate["output_cluster_id"].isin(target_ids)
    candidate["target_basin_best_cosine"] = scores[:, sorted(target_ids)].max(axis=1)
    candidate["target_basin_margin"] = candidate["target_basin_best_cosine"] - np.delete(scores, sorted(target_ids), axis=1).max(axis=1)
    eligible = candidate.loc[candidate["eligible_for_selection"]].copy()
    eligible["selection_rank"] = eligible.sort_values(
        ["anchor_id", "output_in_target_basin", "target_basin_margin", "target_basin_best_cosine", "candidate_index"],
        ascending=[True, False, False, False, True],
    ).groupby("anchor_id").cumcount() + 1
    candidate["selection_rank"] = pd.NA
    candidate.loc[eligible.index, "selection_rank"] = eligible["selection_rank"]
    selected = eligible.loc[eligible["selection_rank"] == 1].copy().sort_values("case_id")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    anchor_summary = candidate.groupby("anchor_id", as_index=False).agg(
        candidate_count=("rescue_id", "size"),
        eligible_candidate_count=("eligible_for_selection", "sum"),
        rejected_candidate_count=("eligible_for_selection", lambda values: int((~values).sum())),
        any_target_candidate=("output_in_target_basin", "any"),
    )
    selected_by_anchor = selected[["anchor_id", "rescue_id", "output_in_target_basin"]].rename(
        columns={"rescue_id": "selected_rescue_id", "output_in_target_basin": "selected_in_target_basin"}
    )
    anchor_summary = anchor_summary.merge(selected_by_anchor, on="anchor_id", how="left", validate="one_to_one")
    anchor_summary["has_selected_eligible_candidate"] = anchor_summary["selected_rescue_id"].notna()
    candidate.to_json(out / "fact_only_candidate_geometry_manifest.jsonl", orient="records", lines=True)
    selected.to_json(out / "fact_only_geometry_selected_manifest.jsonl", orient="records", lines=True)
    anchor_summary.to_csv(out / "fact_only_anchor_selection_eligibility.csv", index=False)
    summary = {
        "candidate_rows": int(len(candidate)), "anchors": int(candidate["anchor_id"].nunique()),
        "target_cluster_ids": sorted(target_ids), "candidate_target_basin_rate": float(candidate["output_in_target_basin"].mean()),
        "selected_target_basin_rate": float(selected["output_in_target_basin"].mean()),
        "anchors_with_target_candidate": int(candidate.groupby("anchor_id")["output_in_target_basin"].any().sum()),
        "eligible_candidates": int(candidate["eligible_for_selection"].sum()),
        "rejected_candidates": int((~candidate["eligible_for_selection"]).sum()),
        "anchors_with_eligible_candidate": int(eligible["anchor_id"].nunique()),
        "selection_filters": {
            "absent_followup_omission": bool(args.enforce_absent_followup_omission),
            "required_field_headings": bool(args.enforce_required_field_headings),
        },
        "required_field_heading_rejection_counts": required_field_rejection_counts,
        "cap_hit_rejection_count": cap_hit_rejection_count,
    }
    (out / "fact_only_geometry_selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
