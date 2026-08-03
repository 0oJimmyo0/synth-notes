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
    parser.add_argument(
        "--frozen_centroid_dataset_path",
        action="append",
        default=[],
        metavar="SPLIT=PATH",
        help=(
            "Filtered real dataset used to reconstruct frozen cluster centroids. "
            "Provide once for every split in the frozen assignment table, for example "
            "--frozen_centroid_dataset_path train=/.../encoded_training_filtered."
        ),
    )
    parser.add_argument(
        "--source_split",
        default="test",
        help="Split represented by --real_dataset_path when a full multi-split assignment table is supplied.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_cluster_ids", default="9,17,29,45")
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--centroid_batch_size",
        type=int,
        default=4096,
        help="Rows per batch while reconstructing frozen real-data centroids.",
    )
    parser.add_argument("--generation_ledger_path", default=None, help="Optional prompt-safe ledger for deterministic output eligibility checks.")
    parser.add_argument("--contract_note_coverage_csv", default=None,
                        help="Optional per-output contract audit. Only contract-passing outputs may reach geometry ranking.")
    parser.add_argument("--enforce_absent_followup_omission", action="store_true")
    parser.add_argument(
        "--enforce_absent_followup_heading",
        action="store_true",
        help="Reject only an unsupported dedicated Follow-up heading. Use for hybrid notes, where verified instructions may mention follow-up actions.",
    )
    parser.add_argument(
        "--reject_cap_hits",
        action="store_true",
        help="Reject candidates recorded as reaching max_new_tokens before final-output geometry ranking.",
    )
    parser.add_argument(
        "--reject_course_constraint_failures",
        action="store_true",
        help="Reject hybrid candidates whose generated hospital course violated its no-pronoun/no-disposition constraint.",
    )
    parser.add_argument(
        "--reject_course_format_artifacts",
        action="store_true",
        help="Reject hybrid courses containing unfinished bracketed drafting artifacts.",
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


def parse_split_dataset_paths(values: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--frozen_centroid_dataset_path must use SPLIT=PATH format.")
        split, raw_path = value.split("=", 1)
        split = split.strip()
        if not split or not raw_path.strip():
            raise ValueError("--frozen_centroid_dataset_path must include both a split and a path.")
        if split in paths:
            raise ValueError(f"Duplicate frozen centroid dataset path for split {split!r}.")
        paths[split] = Path(raw_path).resolve()
    return paths


def build_frozen_centers(
    assignments: pd.DataFrame,
    split_dataset_paths: dict[str, Path],
    n_clusters: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Reconstruct centroids in the frozen full-cohort cluster-label space.

    The frozen labels were fitted on all filtered splits jointly. Refitting K-means
    on one split gives arbitrary label identities, so centroids are instead the
    normalized mean embeddings of rows assigned to each frozen cluster.
    """
    assignment_splits = set(assignments["split"].astype(str))
    if set(split_dataset_paths) != assignment_splits:
        raise ValueError(
            "Frozen centroid datasets must cover exactly the assignment-table splits: "
            f"expected {sorted(assignment_splits)}, got {sorted(split_dataset_paths)}."
        )

    sums: np.ndarray | None = None
    counts = np.zeros(n_clusters, dtype=np.int64)
    split_counts: dict[str, int] = {}
    for split, dataset_path in sorted(split_dataset_paths.items()):
        dataset = Dataset.load_from_disk(str(dataset_path))
        split_assignments = assignments.loc[assignments["split"].astype(str) == split].copy()
        split_assignments["dataset_row_id"] = pd.to_numeric(
            split_assignments["dataset_row_id"], errors="raise"
        ).astype(int)
        if split_assignments["dataset_row_id"].duplicated().any():
            raise ValueError(f"Frozen assignments contain duplicate dataset_row_id values for split {split!r}.")
        split_assignments = split_assignments.sort_values("dataset_row_id")
        expected_ids = np.arange(len(dataset), dtype=int)
        if len(split_assignments) != len(dataset) or not np.array_equal(
            split_assignments["dataset_row_id"].to_numpy(), expected_ids
        ):
            raise ValueError(f"Frozen assignments do not align one-to-one with {split!r} dataset rows.")
        labels = pd.to_numeric(split_assignments["cluster_id"], errors="raise").astype(int).to_numpy()
        if labels.min() < 0 or labels.max() >= n_clusters:
            raise ValueError(f"Frozen cluster IDs for split {split!r} are outside [0, {n_clusters}).")
        # Stream batches to avoid materializing the full train split in the
        # interactive CPU memory cgroup. Assignment IDs were validated above,
        # so positional slices preserve the frozen label-to-embedding mapping.
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            batch_values = dataset[start:stop]["domain_embeddings"]
            embeddings = normalize(np.vstack([
                np.asarray(value[0], dtype=np.float32) for value in batch_values
            ]))
            if sums is None:
                sums = np.zeros((n_clusters, embeddings.shape[1]), dtype=np.float64)
            np.add.at(sums, labels[start:stop], embeddings)
        counts += np.bincount(labels, minlength=n_clusters)
        split_counts[split] = int(len(dataset))

    if sums is None or np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Cannot construct frozen centroids; clusters without rows: {missing}")
    return normalize(sums.astype(np.float32)), split_counts


def main() -> None:
    args = parse_args()
    target_ids = {int(value) for value in args.target_cluster_ids.split(",") if value.strip()}
    candidate = pd.read_json(args.candidate_manifest_path, lines=True).reset_index(drop=True)
    embeddings = normalize(np.load(args.candidate_embeddings_path))
    if len(candidate) != len(embeddings):
        raise ValueError("Candidate manifest and embedding matrix row counts differ.")
    if candidate["rescue_id"].duplicated().any() or candidate["generated_text"].astype(str).str.strip().eq("").any():
        raise ValueError("Candidate manifest has duplicate IDs or empty notes.")
    # Deterministic geometry-diagnostic representations have one row per
    # anchor, so they do not carry sampled-generation candidate indices.
    if "candidate_index" not in candidate.columns:
        candidate["candidate_index"] = 0
    candidate["eligible_for_selection"] = True
    candidate["selection_rejection_reason"] = ""
    if args.contract_note_coverage_csv:
        contract = pd.read_csv(Path(args.contract_note_coverage_csv).resolve())
        required_contract = {"candidate_id", "contract_pass"}
        if missing := required_contract.difference(contract.columns):
            raise KeyError(f"contract coverage is missing columns: {sorted(missing)}")
        if contract.candidate_id.astype(str).duplicated().any():
            raise ValueError("contract coverage contains duplicate candidate_id values")
        contract_map = contract.set_index(contract.candidate_id.astype(str)).contract_pass.astype(bool)
        candidate["contract_pass"] = candidate.rescue_id.astype(str).map(contract_map)
        if candidate.contract_pass.isna().any():
            missing = candidate.loc[candidate.contract_pass.isna(), "rescue_id"].astype(str).head().tolist()
            raise ValueError(f"candidate manifest has outputs absent from contract audit, for example {missing}")
        append_rejection_reasons(candidate, ~candidate.contract_pass, "contract_audit_failure")
    cap_hit_rejection_count = 0
    if args.reject_cap_hits:
        if "hit_max_new_tokens" not in candidate.columns:
            raise KeyError("--reject_cap_hits requires hit_max_new_tokens in the candidate manifest.")
        cap_hit = candidate["hit_max_new_tokens"].fillna(False).astype(bool)
        cap_hit_rejection_count = int(cap_hit.sum())
        append_rejection_reasons(candidate, cap_hit, "hit_max_new_tokens")
    course_constraint_rejection_count = 0
    if args.reject_course_constraint_failures:
        if "course_constraint_pass" not in candidate.columns:
            raise KeyError("--reject_course_constraint_failures requires course_constraint_pass in the candidate manifest.")
        course_constraint_fail = ~candidate["course_constraint_pass"].fillna(False).astype(bool)
        course_constraint_rejection_count = int(course_constraint_fail.sum())
        append_rejection_reasons(candidate, course_constraint_fail, "hospital_course_constraint_failure")
    course_format_artifact_rejection_count = 0
    if args.reject_course_format_artifacts:
        if "hospital_course_text" not in candidate.columns:
            raise KeyError("--reject_course_format_artifacts requires hospital_course_text in the candidate manifest.")
        course_format_artifact = candidate["hospital_course_text"].fillna("").astype(str).str.contains(r"[\[\]]", regex=True)
        course_format_artifact_rejection_count = int(course_format_artifact.sum())
        append_rejection_reasons(candidate, course_format_artifact, "hospital_course_format_artifact")
    ledger_fields_by_case: dict[str, set[str]] = {}
    if args.enforce_absent_followup_omission or args.enforce_absent_followup_heading or args.enforce_required_field_headings:
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
    if args.enforce_absent_followup_heading:
        follow_up_by_case = {case_id: "follow_up" in fields for case_id, fields in ledger_fields_by_case.items()}
        candidate["ledger_has_follow_up"] = candidate.case_id.astype(str).map(follow_up_by_case).astype(bool)
        follow_up_heading = re.compile(r"(?im)^\s*(?:follow[\s-]*up)\s*:")
        candidate["unsupported_follow_up_heading"] = [
            (not has_follow_up) and bool(follow_up_heading.search(str(text)))
            for has_follow_up, text in zip(candidate["ledger_has_follow_up"], candidate["generated_text"])
        ]
        append_rejection_reasons(candidate, candidate["unsupported_follow_up_heading"], "unsupported_follow_up_heading_without_ledger_fact")
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
    assignments = pd.read_csv(args.real_cluster_assignments_path)
    required_assignment_columns = {"split", "dataset_row_id", "cluster_id"}
    if missing := required_assignment_columns.difference(assignments.columns):
        raise KeyError(f"Frozen assignment table is missing columns: {sorted(missing)}")
    source_assignments = assignments.loc[assignments["split"].astype(str) == args.source_split].copy()
    source_dataset = Dataset.load_from_disk(args.real_dataset_path)
    if len(source_assignments) != len(source_dataset):
        raise ValueError(
            "Frozen assignment row count does not match --real_dataset_path for the requested source split: "
            f"{len(source_assignments)} vs {len(source_dataset)}."
        )
    centers, frozen_centroid_split_counts = build_frozen_centers(
        assignments=assignments,
        split_dataset_paths=parse_split_dataset_paths(args.frozen_centroid_dataset_path),
        n_clusters=args.n_clusters,
        batch_size=args.centroid_batch_size,
    )
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
        "frozen_centroid_split_counts": frozen_centroid_split_counts,
        "selected_target_basin_rate": float(selected["output_in_target_basin"].mean()),
        "anchors_with_target_candidate": int(candidate.groupby("anchor_id")["output_in_target_basin"].any().sum()),
        "eligible_candidates": int(candidate["eligible_for_selection"].sum()),
        "rejected_candidates": int((~candidate["eligible_for_selection"]).sum()),
        "anchors_with_eligible_candidate": int(eligible["anchor_id"].nunique()),
        "selection_filters": {
            "absent_followup_omission": bool(args.enforce_absent_followup_omission),
            "absent_followup_heading": bool(args.enforce_absent_followup_heading),
            "required_field_headings": bool(args.enforce_required_field_headings),
        },
        "required_field_heading_rejection_counts": required_field_rejection_counts,
        "cap_hit_rejection_count": cap_hit_rejection_count,
        "course_constraint_rejection_count": course_constraint_rejection_count,
        "course_format_artifact_rejection_count": course_format_artifact_rejection_count,
    }
    (out / "fact_only_geometry_selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
