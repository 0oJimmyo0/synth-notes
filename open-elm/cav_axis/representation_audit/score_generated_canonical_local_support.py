#!/usr/bin/env python3
"""Score generated canonical notes against one frozen train reference split.

This is a development-only analysis helper.  It uses the frozen, subject-grouped
train reference halves and excludes any train note from the generated note's
own subject before calculating local cosine support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_embeddings_path", required=True)
    parser.add_argument("--train_subject_metadata_path", required=True)
    parser.add_argument("--candidate_embeddings_path", required=True)
    parser.add_argument("--candidate_embedding_metadata_path", required=True)
    parser.add_argument("--candidate_subject_manifest_path", required=True)
    parser.add_argument("--reference_split_path", required=True)
    parser.add_argument("--split_seed", required=True, type=int)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--analysis_scope",
        default="generated_candidates_against_frozen_real_train_references_only",
        help="Provenance label for the scored candidate set (for example heldout_test).",
    )
    parser.add_argument("--query_batch_size", type=int, default=128)
    parser.add_argument("--reference_batch_size", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def ordered_jsonl(path: str, expected_rows: int, label: str) -> pd.DataFrame:
    frame = pd.read_json(Path(path).resolve(), lines=True).sort_values("source_index")
    if frame.source_index.tolist() != list(range(expected_rows)):
        raise ValueError(f"{label} metadata must cover ordered source indices exactly once.")
    return frame.reset_index(drop=True)


def reference_indices(path: str, train_metadata: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    split = pd.read_csv(Path(path).resolve()).sort_values("source_index").reset_index(drop=True)
    required = {"source_index", "subject_id", "reference_half", "reference_keep"}
    if missing := required.difference(split.columns):
        raise KeyError(f"Reference split is missing columns: {sorted(missing)}")
    if split.source_index.tolist() != list(range(len(train_metadata))):
        raise ValueError("Reference split does not cover ordered train rows exactly once.")
    if split.subject_id.astype(str).tolist() != train_metadata.subject_id.astype(str).tolist():
        raise ValueError("Reference split subject IDs do not match train metadata.")
    kept = split.loc[split.reference_keep.astype(bool)].copy()
    if set(kept.reference_half) != {"a", "b"}:
        raise ValueError("Both frozen reference halves must be nonempty.")
    if kept.groupby("subject_id").reference_half.nunique().max() != 1:
        raise ValueError("A retained reference subject appears in both reference halves.")
    index_a = kept.loc[kept.reference_half.eq("a"), "source_index"].to_numpy(dtype=np.int64)
    index_b = kept.loc[kept.reference_half.eq("b"), "source_index"].to_numpy(dtype=np.int64)
    if min(len(index_a), len(index_b)) < k:
        raise ValueError("A frozen reference half is smaller than k.")
    return index_a, index_b


def mean_top_k(
    queries: np.ndarray,
    query_codes: np.ndarray,
    references: np.ndarray,
    reference_codes: np.ndarray,
    k: int,
    query_batch_size: int,
    reference_batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    values, exclusions = [], []
    for start in range(0, len(queries), query_batch_size):
        query = torch.as_tensor(np.asarray(queries[start:start + query_batch_size]), device=device)
        codes = torch.as_tensor(query_codes[start:start + len(query)], dtype=torch.long, device=device)
        best = torch.full((len(query), k), -torch.inf, device=device)
        excluded = np.zeros(len(query), dtype=np.int32)
        for ref_start in range(0, len(references), reference_batch_size):
            ref_end = min(ref_start + reference_batch_size, len(references))
            ref = torch.as_tensor(np.asarray(references[ref_start:ref_end]), device=device)
            ref_codes = torch.as_tensor(reference_codes[ref_start:ref_end], dtype=torch.long, device=device)
            scores = query @ ref.T
            mask = codes[:, None].eq(ref_codes[None, :])
            excluded += mask.sum(dim=1).cpu().numpy().astype(np.int32)
            scores.masked_fill_(mask, -torch.inf)
            best = torch.topk(torch.cat((best, scores), dim=1), k=k, dim=1).values
        if torch.isinf(best[:, -1]).any():
            raise ValueError("Same-subject exclusion left fewer than k references.")
        values.append(best.mean(dim=1).cpu().numpy())
        exclusions.append(excluded)
    return np.concatenate(values), np.concatenate(exclusions)


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("k must be positive.")
    train = np.load(Path(args.train_embeddings_path).resolve(), mmap_mode="r")
    candidates = np.load(Path(args.candidate_embeddings_path).resolve(), mmap_mode="r")
    if train.ndim != 2 or candidates.ndim != 2 or train.shape[1] != candidates.shape[1]:
        raise ValueError("Train and candidate embeddings must be two-dimensional with matching dimensions.")
    train_meta = ordered_jsonl(args.train_subject_metadata_path, len(train), "Train")
    if "subject_id" not in train_meta:
        raise KeyError("Train metadata is missing subject_id.")
    candidate_meta = ordered_jsonl(args.candidate_embedding_metadata_path, len(candidates), "Candidate")
    required_candidate = {"rescue_id", "case_id", "dataset_row_id", "candidate_index"}
    if missing := required_candidate.difference(candidate_meta.columns):
        raise KeyError(f"Candidate embedding metadata missing columns: {sorted(missing)}")
    if candidate_meta.rescue_id.duplicated().any():
        raise ValueError("Candidate metadata contains duplicate rescue IDs.")
    subject_manifest = pd.read_csv(Path(args.candidate_subject_manifest_path).resolve())
    # Development manifests retain ``final_case_id`` after source review,
    # whereas frozen held-out screening manifests use their immutable ``case_id``.
    # Both are valid contract/generation identifiers.
    case_id_column = "final_case_id" if "final_case_id" in subject_manifest.columns else "case_id"
    needed_subject = {case_id_column, "dataset_row_id", "subject_id", "support_arm", "cohort_stratum", "patient_disjoint_from_train"}
    if missing := needed_subject.difference(subject_manifest.columns):
        raise KeyError(f"Candidate subject manifest missing columns: {sorted(missing)}")
    # Normalize either frozen cohort identifier to the contract/generation key.
    subject_manifest = subject_manifest[[
        case_id_column, "dataset_row_id", "subject_id", "support_arm",
        "cohort_stratum", "patient_disjoint_from_train",
    ]].rename(columns={case_id_column: "case_id"})
    if subject_manifest.case_id.duplicated().any() or subject_manifest.dataset_row_id.duplicated().any():
        raise ValueError("Candidate subject manifest must contain one row per case and dataset row.")
    # Source-ledger case IDs are reassigned to ``ledger_*`` during review,
    # while the frozen held-out cohort retains its original split case ID.
    # ``dataset_row_id`` is the immutable cross-stage provenance key.
    candidates_meta = candidate_meta.merge(
        subject_manifest[["dataset_row_id", "subject_id", "support_arm", "cohort_stratum", "patient_disjoint_from_train"]],
        on="dataset_row_id", how="left", validate="many_to_one", suffixes=("", "_frozen"),
    )
    if candidates_meta.subject_id.isna().any():
        raise ValueError("Some candidate rows could not be linked to frozen subject provenance.")
    index_a, index_b = reference_indices(args.reference_split_path, train_meta, args.k)
    subject_codes, _ = pd.factorize(pd.concat([train_meta.subject_id, candidates_meta.subject_id]).astype(str), sort=True)
    train_codes, candidate_codes = subject_codes[:len(train)], subject_codes[len(train):]
    with torch.inference_mode():
        support_a, excluded_a = mean_top_k(
            candidates, candidate_codes, train[index_a], train_codes[index_a], args.k,
            args.query_batch_size, args.reference_batch_size, args.device,
        )
        support_b, excluded_b = mean_top_k(
            candidates, candidate_codes, train[index_b], train_codes[index_b], args.k,
            args.query_batch_size, args.reference_batch_size, args.device,
        )
    output = candidates_meta[["rescue_id", "case_id", "dataset_row_id", "candidate_index", "subject_id", "support_arm", "cohort_stratum", "patient_disjoint_from_train"]].copy()
    output["mean_top_k_support_a"] = support_a
    output["mean_top_k_support_b"] = support_b
    output["mean_top_k_support"] = (support_a + support_b) / 2
    output["same_subject_reference_count_a"] = excluded_a
    output["same_subject_reference_count_b"] = excluded_b
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "generated_canonical_local_support.csv", index=False)
    summary = {
        "scope": args.analysis_scope,
        "split_seed": args.split_seed,
        "n_candidates": int(len(output)),
        "n_cases": int(output.case_id.nunique()),
        "k": args.k,
        "mean_support": float(output.mean_top_k_support.mean()),
        "security_note": "Outputs contain provenance IDs and derived support values only; no source-note text.",
    }
    (output_dir / "generated_canonical_local_support_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
