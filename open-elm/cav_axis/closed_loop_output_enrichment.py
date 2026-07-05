#!/usr/bin/env python3
"""
Closed-loop output-space enrichment for sparse-basin synthetic note generation.

This Phase 2 path does not try to control the pre-decode input embedding.
Instead, it:

1. starts from real anchor embeddings inside the target basin,
2. generates multiple candidate notes per anchor with vanilla ELM,
3. re-embeds every generated note with BGE,
4. keeps only candidates whose final embeddings land in the intended target
   region while also passing quality, faithfulness, privacy, and diversity gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

OPEN_ELM_DIR = Path(__file__).resolve().parents[1]
if str(OPEN_ELM_DIR) not in sys.path:
    sys.path.insert(0, str(OPEN_ELM_DIR))

from generate_synthetic_notes import (  # noqa: E402
    load_generation_model,
    quality_flags,
)
from src.utils import batch_inference  # noqa: E402
from decoder_feasibility_audit import (  # noqa: E402
    build_centroids,
    choose_join_keys,
    load_dataset,
    merge_metadata,
    normalize_join_cols,
    normalize_rows,
    parse_csv_list,
    parse_int_list,
)


PHI_PATTERNS = {
    "ssn_like": r"\b\d{3}-\d{2}-\d{4}\b",
    "phone_like": r"\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b",
    "email_like": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "mrn_like": r"\b(?:MRN|Medical Record Number)[:\s#-]*\d{5,}\b",
    "id_like": r"\b(?:ID|Acct|Account)[:\s#-]*\d{5,}\b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop output-space enrichment for sparse clinical regions.")
    parser.add_argument("--anchor_manifest_path", required=True, help="CSV or JSONL manifest identifying anchor rows")
    parser.add_argument("--dataset_path", required=True, help="Path to encoded anchor dataset")
    parser.add_argument("--checkpoint_path", required=True, help="Vanilla or PEFT ELM checkpoint path")
    parser.add_argument("--backbone_path", required=True, help="Backbone model path used by ELM")
    parser.add_argument(
        "--embedding_model_name",
        default="BAAI/bge-large-en-v1.5",
        help="Sentence embedding model for candidate re-embedding",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for manifests and reports")
    parser.add_argument("--cluster_assignments_path", required=True, help="Cluster assignments aligned to dataset_path")
    parser.add_argument(
        "--split_manifest_path",
        default=None,
        help="Optional filtered-aligned split manifest for leakage flags",
    )
    parser.add_argument(
        "--target_cluster_ids",
        default=None,
        help="Comma-separated target cluster ids",
    )
    parser.add_argument(
        "--target_basin_path",
        default=None,
        help="Optional CSV/JSONL describing the target basin rows or clusters",
    )
    parser.add_argument(
        "--target_centroid_path",
        default=None,
        help="Optional .npy or .json path for precomputed target centroid metadata",
    )
    parser.add_argument(
        "--join_cols",
        default="dataset_row_id,note_id,subject_id,hadm_id",
        help="Preferred join columns for manifests and dataset metadata",
    )
    parser.add_argument("--source_split", default="test", help="Logical split label for anchors")
    parser.add_argument("--n_candidates_per_anchor", type=int, default=8, help="Candidates to decode per anchor")
    parser.add_argument("--accepted_per_anchor", type=int, default=1, help="Maximum accepted candidates per anchor")
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48,49", help="Comma-separated candidate generation seeds")
    parser.add_argument("--temperature_values", default="0.8", help="Comma-separated temperatures to cycle through")
    parser.add_argument("--top_p_values", default="0.9", help="Comma-separated top-p values to cycle through")
    parser.add_argument("--top_k_values", default="50", help="Comma-separated top-k values to cycle through")
    parser.add_argument("--batch_size", type=int, default=4, help="ELM generation batch size")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens per candidate")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Generation repetition penalty")
    parser.add_argument("--device", default="cuda", help="Generation device for ELM")
    parser.add_argument("--embedding_device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--embedding_batch_size", type=int, default=256)
    parser.add_argument("--min_word_count", type=int, default=100)
    parser.add_argument("--min_source_cosine", type=float, default=0.78)
    parser.add_argument(
        "--target_centroid_distance_threshold",
        type=float,
        default=None,
        help="Optional explicit max cosine-distance to target centroid for acceptance",
    )
    parser.add_argument(
        "--target_centroid_threshold_quantile",
        type=float,
        default=0.90,
        help="If explicit threshold is not given, use this anchor-distance quantile",
    )
    parser.add_argument(
        "--candidate_near_duplicate_cosine",
        type=float,
        default=0.995,
        help="Reject accepted candidates that are embedding-near-duplicates of earlier accepted candidates",
    )
    parser.add_argument(
        "--anchor_diversity_cosine",
        type=float,
        default=0.992,
        help="Per-anchor diversity ceiling among selected accepted candidates",
    )
    parser.add_argument(
        "--manual_review_sample_size",
        type=int,
        default=50,
        help="Number of accepted notes to sample for manual review",
    )
    parser.add_argument(
        "--train_text_path",
        default=None,
        help="Optional CSV/JSONL/Parquet/Pickle table with training-note text for overlap screening",
    )
    parser.add_argument(
        "--max_train_texts_for_overlap",
        type=int,
        default=50000,
        help="Cap train texts loaded for lexical overlap screening",
    )
    parser.add_argument(
        "--selection_strategy",
        default="best_target_score",
        choices=["best_target_score", "maxmin_diverse"],
        help="How to select among multiple passing candidates per anchor",
    )
    parser.add_argument("--shard_id", type=int, default=0, help="Zero-based shard id for anchor-parallel closed-loop runs")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of anchor shards")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_commit(script_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(script_dir.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sentence_transformers

        versions["sentence_transformers"] = sentence_transformers.__version__
    except Exception:
        pass
    return versions


def save_json(path: Path, payload: dict[str, Any]) -> None:
    def _default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        raise TypeError(f"Cannot serialize object of type {type(value)}")

    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=json_default) + "\n")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize object of type {type(value)}")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def resolve_embedding_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_float_list(value: str | None) -> list[float]:
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported table format: {path}")


def ensure_anchor_table(anchor_manifest_path: Path) -> pd.DataFrame:
    df = load_table(anchor_manifest_path)
    if df.empty:
        raise ValueError(f"Anchor manifest is empty: {anchor_manifest_path}")
    return df.reset_index(drop=True)


def infer_target_cluster_ids(target_cluster_ids: list[int], target_basin_path: Path | None) -> list[int]:
    if target_cluster_ids:
        return sorted(set(target_cluster_ids))
    if target_basin_path is None:
        raise ValueError("Must provide --target_cluster_ids or --target_basin_path")
    basin_df = load_table(target_basin_path)
    if "cluster_id" not in basin_df.columns:
        raise ValueError("target_basin_path must contain a cluster_id column if target_cluster_ids are omitted")
    inferred = pd.to_numeric(basin_df["cluster_id"], errors="coerce").dropna().astype(int).unique().tolist()
    if not inferred:
        raise ValueError("No valid target cluster ids found in target_basin_path")
    return sorted(inferred)


def merge_anchor_metadata(
    anchor_df: pd.DataFrame,
    dataset_meta_df: pd.DataFrame,
    preferred_join_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    join_cols = choose_join_keys([anchor_df, dataset_meta_df], preferred_join_cols)
    left = normalize_join_cols(anchor_df, join_cols)
    right = normalize_join_cols(dataset_meta_df, join_cols)
    merged = left.merge(right, on=join_cols, how="left", validate="many_to_one", suffixes=("", "_dataset"))
    return merged, join_cols


def prepare_anchor_rows(
    anchor_df: pd.DataFrame,
    target_cluster_ids: list[int],
) -> pd.DataFrame:
    out = anchor_df.copy()
    if "decoder_group_family" in out.columns:
        out = out.loc[out["decoder_group_family"].astype(str) == "target_basin"].copy()
    if "cluster_id" in out.columns:
        out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="coerce")
        out = out.loc[out["cluster_id"].isin(target_cluster_ids)].copy()
    if "dataset_local_row_id" not in out.columns:
        raise KeyError("Merged anchor metadata is missing dataset_local_row_id")
    out["dataset_local_row_id"] = pd.to_numeric(out["dataset_local_row_id"], errors="raise").astype(int)
    out = out.drop_duplicates(subset=["dataset_local_row_id"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No anchor rows remained after target-basin filtering")
    if "anchor_id" not in out.columns:
        out.insert(0, "anchor_id", [f"anchor_{i:06d}" for i in range(len(out))])
    if "patient_disjoint_from_train" in out.columns:
        out["patient_disjoint_from_train"] = out["patient_disjoint_from_train"].astype("boolean")
    return out


def apply_anchor_shard(anchor_df: pd.DataFrame, shard_id: int, num_shards: int) -> pd.DataFrame:
    if num_shards <= 1:
        return anchor_df.reset_index(drop=True)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard_id={shard_id} for num_shards={num_shards}")
    shard_df = anchor_df.iloc[shard_id::num_shards].copy().reset_index(drop=True)
    if shard_df.empty:
        raise ValueError(f"Anchor shard {shard_id}/{num_shards} is empty")
    return shard_df


def maybe_load_train_texts(path: Path | None, cap: int) -> tuple[list[str], str | None]:
    if path is None:
        return [], "train_text_path_not_provided"
    df = load_table(path)
    candidate_cols = ["text", "note_text", "CLEAN_TEXT", "generated_text", "source_text"]
    text_col = next((col for col in candidate_cols if col in df.columns), None)
    if text_col is None:
        return [], f"no_text_column_found_in_{path.name}"
    texts = df[text_col].fillna("").astype(str).tolist()
    texts = [normalize_text(text) for text in texts if normalize_text(text)]
    if cap and len(texts) > cap:
        texts = texts[:cap]
        return texts, f"loaded_first_{cap}_train_texts_only"
    return texts, None


def build_train_overlap_reference(train_texts: list[str]) -> tuple[set[str], set[str]]:
    text_hashes = {text_hash(text) for text in train_texts}
    union_ngrams: set[str] = set()
    for text in train_texts:
        tokens = re.findall(r"\b\w+\b", text.lower())
        if len(tokens) < 8:
            continue
        for idx in range(len(tokens) - 8 + 1):
            union_ngrams.add(" ".join(tokens[idx : idx + 8]))
    return text_hashes, union_ngrams


def phi_flags(text: str) -> dict[str, bool]:
    return {name: bool(re.search(pattern, text)) for name, pattern in PHI_PATTERNS.items()}


def lexical_overlap_flag(text: str, train_ngrams: set[str], threshold: float = 0.8) -> tuple[bool, float]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    if len(tokens) < 8 or not train_ngrams:
        return False, 0.0
    candidate = [" ".join(tokens[idx : idx + 8]) for idx in range(len(tokens) - 8 + 1)]
    if not candidate:
        return False, 0.0
    overlap = sum(1 for ngram in candidate if ngram in train_ngrams)
    ratio = overlap / len(candidate)
    return ratio >= threshold, ratio


def encode_texts(
    texts: list[str],
    model_name: str,
    requested_device: str,
    batch_size: int,
) -> tuple[np.ndarray, str]:
    resolved_device = resolve_embedding_device(requested_device)
    model = SentenceTransformer(model_name, device=resolved_device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32), resolved_device


def load_target_centroid(
    target_centroid_path: Path | None,
    anchor_df: pd.DataFrame,
    all_embeddings: np.ndarray,
    assignments_df: pd.DataFrame,
    target_cluster_ids: list[int],
) -> tuple[np.ndarray, float]:
    centroid_threshold = None
    if target_centroid_path and target_centroid_path.exists():
        if target_centroid_path.suffix.lower() == ".npy":
            centroid = np.load(target_centroid_path).astype(np.float32)
            if centroid.ndim > 1:
                centroid = centroid[0]
        else:
            obj = json.loads(target_centroid_path.read_text())
            centroid = np.asarray(obj["centroid"], dtype=np.float32)
            centroid_threshold = obj.get("distance_threshold")
        centroid = normalize_rows(np.asarray([centroid], dtype=np.float32))[0]
    else:
        centroids = build_centroids(all_embeddings, assignments_df, target_cluster_ids)
        if not centroids:
            raise ValueError("Could not build target cluster centroids from assignments")
        centroid = normalize_rows(np.mean(np.vstack(list(centroids.values())), axis=0, keepdims=True))[0]

    anchor_rows = anchor_df["dataset_local_row_id"].to_numpy(dtype=int)
    anchor_cos = np.clip(all_embeddings[anchor_rows] @ centroid, -1.0, 1.0)
    anchor_dist = 1.0 - anchor_cos
    return centroid, float(anchor_dist.mean()) if centroid_threshold is None else float(centroid_threshold)


def choose_centroid_distance_threshold(
    explicit_threshold: float | None,
    quantile: float,
    anchor_df: pd.DataFrame,
    all_embeddings: np.ndarray,
    centroid: np.ndarray,
) -> float:
    if explicit_threshold is not None:
        return float(explicit_threshold)
    anchor_rows = anchor_df["dataset_local_row_id"].to_numpy(dtype=int)
    distances = 1.0 - np.clip(all_embeddings[anchor_rows] @ centroid, -1.0, 1.0)
    return float(np.quantile(distances, quantile))


def build_generation_plans(
    n_candidates_per_anchor: int,
    seeds: list[int],
    temperatures: list[float],
    top_ps: list[float],
    top_ks: list[int],
) -> list[dict[str, Any]]:
    if not seeds:
        raise ValueError("At least one seed is required")
    temperatures = temperatures or [0.8]
    top_ps = top_ps or [0.9]
    top_ks = top_ks or [50]

    plans: list[dict[str, Any]] = []
    for idx in range(n_candidates_per_anchor):
        plans.append(
            {
                "candidate_index": idx,
                "seed": int(seeds[idx % len(seeds)] + idx // len(seeds)),
                "temperature": float(temperatures[idx % len(temperatures)]),
                "top_p": float(top_ps[idx % len(top_ps)]),
                "top_k": int(top_ks[idx % len(top_ks)]),
            }
        )
    return plans


def candidate_target_score(
    nearest_cluster_in_target: bool,
    target_centroid_cosine: float,
    source_cosine: float,
) -> float:
    return (10.0 if nearest_cluster_in_target else 0.0) + float(target_centroid_cosine) + (0.1 * float(source_cosine))


def group_flag_rate(df: pd.DataFrame, column: str) -> float:
    if len(df) == 0 or column not in df.columns:
        return math.nan
    return float(pd.to_numeric(df[column], errors="coerce").fillna(0).astype(float).mean())


def pairwise_diversity_summary(embeddings: np.ndarray) -> dict[str, float]:
    if embeddings.shape[0] < 2:
        return {"mean_pairwise_cosine": math.nan, "max_pairwise_cosine": math.nan, "min_pairwise_cosine": math.nan}
    sims = embeddings @ embeddings.T
    tri = sims[np.triu_indices(sims.shape[0], k=1)]
    return {
        "mean_pairwise_cosine": float(tri.mean()),
        "max_pairwise_cosine": float(tri.max()),
        "min_pairwise_cosine": float(tri.min()),
    }


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Closed-Loop Output-Space Enrichment",
        "",
        "## Status",
        f"- Method: `closed-loop output-space selection`",
        f"- Created at: `{summary['created_at']}`",
        f"- Total anchors: `{summary['total_anchors']}`",
        f"- Candidates per anchor: `{summary['n_candidates_per_anchor']}`",
        f"- Total candidates: `{summary['total_candidates_generated']}`",
        f"- Total accepted: `{summary['total_accepted']}`",
        f"- Acceptance rate: `{summary['acceptance_rate']:.4f}`",
        "",
        "## Target Metrics",
        f"- Accepted target-cluster hit rate: `{summary['accepted_target_cluster_rate']:.4f}`",
        f"- Accepted centroid-distance pass rate: `{summary['accepted_centroid_distance_pass_rate']:.4f}`",
        f"- Accepted mean source cosine: `{summary['accepted_mean_source_cosine']:.4f}`",
        f"- Rejected mean source cosine: `{summary['rejected_mean_source_cosine']:.4f}`",
        "",
        "## Privacy / Quality",
        f"- Accepted collapse rate: `{summary['accepted_collapse_rate']:.4f}`",
        f"- Rejected collapse rate: `{summary['rejected_collapse_rate']:.4f}`",
        f"- Accepted PHI-warning rate: `{summary['accepted_phi_warning_rate']:.4f}`",
        f"- Rejected PHI-warning rate: `{summary['rejected_phi_warning_rate']:.4f}`",
        f"- Train overlap screen: `{summary['train_overlap_screen_status']}`",
        "",
        "## Interpretation",
        "- This method controls the final output distribution by selecting notes whose re-embedded BGE vectors actually land in the target sparse region.",
        "- Report acceptance rates transparently to avoid cherry-picking concerns.",
        "- Results are privacy-risk-evaluated, not privacy-safe by assumption.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    anchor_manifest_path = Path(args.anchor_manifest_path).resolve()
    dataset_path = Path(args.dataset_path).resolve()
    cluster_assignments_path = Path(args.cluster_assignments_path).resolve()
    split_manifest_path = Path(args.split_manifest_path).resolve() if args.split_manifest_path else None
    target_basin_path = Path(args.target_basin_path).resolve() if args.target_basin_path else None
    target_centroid_path = Path(args.target_centroid_path).resolve() if args.target_centroid_path else None
    train_text_path = Path(args.train_text_path).resolve() if args.train_text_path else None

    preferred_join_cols = parse_csv_list(args.join_cols)
    target_cluster_ids = infer_target_cluster_ids(parse_int_list(args.target_cluster_ids), target_basin_path)

    anchor_raw_df = ensure_anchor_table(anchor_manifest_path)
    dataset, dataset_meta_df, source_embeddings = load_dataset(dataset_path)
    merged_meta_df = merge_metadata(
        dataset_meta_df,
        cluster_assignments_path,
        split_manifest_path,
        preferred_join_cols,
        args.source_split,
    )
    anchor_df, anchor_join_cols = merge_anchor_metadata(anchor_raw_df, merged_meta_df, preferred_join_cols)
    anchor_df = prepare_anchor_rows(anchor_df, target_cluster_ids)
    anchor_df = apply_anchor_shard(anchor_df, int(args.shard_id), int(args.num_shards))

    assignments_df = pd.read_csv(cluster_assignments_path)
    if "dataset_row_id" not in assignments_df.columns or "cluster_id" not in assignments_df.columns:
        raise KeyError("cluster_assignments_path must contain dataset_row_id and cluster_id columns")
    assignments_df["dataset_row_id"] = pd.to_numeric(assignments_df["dataset_row_id"], errors="raise").astype(int)
    assignments_df["cluster_id"] = pd.to_numeric(assignments_df["cluster_id"], errors="raise").astype(int)

    centroid, _ = load_target_centroid(
        target_centroid_path=target_centroid_path,
        anchor_df=anchor_df,
        all_embeddings=source_embeddings,
        assignments_df=assignments_df,
        target_cluster_ids=target_cluster_ids,
    )
    centroid_distance_threshold = choose_centroid_distance_threshold(
        explicit_threshold=args.target_centroid_distance_threshold,
        quantile=float(args.target_centroid_threshold_quantile),
        anchor_df=anchor_df,
        all_embeddings=source_embeddings,
        centroid=centroid,
    )

    cluster_centroids = build_centroids(
        real_embeddings=source_embeddings,
        assignments_df=assignments_df,
        cluster_ids=sorted(assignments_df["cluster_id"].unique().tolist()),
    )
    cluster_centroid_matrix = np.vstack([cluster_centroids[cid] for cid in sorted(cluster_centroids)])
    cluster_centroid_ids = np.asarray(sorted(cluster_centroids), dtype=int)

    train_texts, train_text_warning = maybe_load_train_texts(train_text_path, int(args.max_train_texts_for_overlap))
    train_text_hashes, train_ngrams = build_train_overlap_reference(train_texts) if train_texts else (set(), set())

    model, model_meta = load_generation_model(args.checkpoint_path, args.device)
    model.eval()
    from transformers import AutoTokenizer  # local import keeps startup lighter

    tokenizer = AutoTokenizer.from_pretrained(args.backbone_path)

    generation_plans = build_generation_plans(
        n_candidates_per_anchor=int(args.n_candidates_per_anchor),
        seeds=parse_int_list(args.seeds),
        temperatures=parse_float_list(args.temperature_values),
        top_ps=parse_float_list(args.top_p_values),
        top_ks=parse_int_list(args.top_k_values),
    )

    anchor_embeddings = source_embeddings[anchor_df["dataset_local_row_id"].to_numpy(dtype=int)]
    candidate_rows: list[dict[str, Any]] = []

    print(f"Loaded {len(anchor_df)} target-basin anchors for shard {args.shard_id + 1}/{args.num_shards}")
    print(f"Generating {len(generation_plans)} candidates per anchor ({len(anchor_df) * len(generation_plans)} total candidates on this shard)")

    for plan in generation_plans:
        set_random_seed(int(plan["seed"]))
        print(
            f"Generation plan {plan['candidate_index'] + 1}/{len(generation_plans)}: "
            f"seed={plan['seed']} temp={plan['temperature']} top_p={plan['top_p']} top_k={plan['top_k']}"
        )
        generated_notes: list[str] = []
        for start in range(0, len(anchor_embeddings), int(args.batch_size)):
            stop = min(start + int(args.batch_size), len(anchor_embeddings))
            batch_notes = batch_inference(
                model,
                tokenizer,
                anchor_embeddings[start:stop],
                args.device,
                task="clinic_note",
                repetition_penalty=float(args.repetition_penalty),
                temperature=float(plan["temperature"]),
                top_p=float(plan["top_p"]),
                top_k=int(plan["top_k"]),
                max_new_tokens=int(args.max_new_tokens),
                do_sample=True,
            )
            generated_notes.extend(batch_notes)

        if len(generated_notes) != len(anchor_df):
            raise ValueError("Generated note count does not match anchor count")

        for row_idx, note in enumerate(generated_notes):
            anchor_row = anchor_df.iloc[row_idx]
            candidate_id = f"{anchor_row['anchor_id']}_cand{int(plan['candidate_index']):02d}"
            qflags = quality_flags(note)
            phi_map = phi_flags(note)
            phi_warning = any(phi_map.values())
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "anchor_id": anchor_row["anchor_id"],
                    "candidate_index": int(plan["candidate_index"]),
                    "seed": int(plan["seed"]),
                    "temperature": float(plan["temperature"]),
                    "top_p": float(plan["top_p"]),
                    "top_k": int(plan["top_k"]),
                    "repetition_penalty": float(args.repetition_penalty),
                    "max_new_tokens": int(args.max_new_tokens),
                    "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
                    "backbone_path": str(Path(args.backbone_path).resolve()),
                    "checkpoint_format": model_meta["checkpoint_format"],
                    "dataset_path": str(dataset_path),
                    "source_split": args.source_split,
                    "dataset_row_id": int(anchor_row["dataset_row_id"]) if "dataset_row_id" in anchor_row and not pd.isna(anchor_row["dataset_row_id"]) else None,
                    "dataset_local_row_id": int(anchor_row["dataset_local_row_id"]),
                    "note_id": anchor_row.get("note_id"),
                    "subject_id": anchor_row.get("subject_id"),
                    "hadm_id": anchor_row.get("hadm_id"),
                    "patient_disjoint_from_train": None if "patient_disjoint_from_train" not in anchor_row else (None if pd.isna(anchor_row["patient_disjoint_from_train"]) else bool(anchor_row["patient_disjoint_from_train"])),
                    "hadm_disjoint_from_train": anchor_row.get("hadm_disjoint_from_train"),
                    "note_disjoint_from_train": anchor_row.get("note_disjoint_from_train"),
                    "source_cluster_id": int(anchor_row["cluster_id"]) if "cluster_id" in anchor_row and not pd.isna(anchor_row["cluster_id"]) else None,
                    "generated_text": note,
                    "text_hash": text_hash(note),
                    **qflags,
                    **{f"phi_{key}": value for key, value in phi_map.items()},
                    "phi_warning_flag": phi_warning,
                }
            )

    candidate_df = pd.DataFrame(candidate_rows)

    generated_embeddings, resolved_embedding_device = encode_texts(
        texts=candidate_df["generated_text"].fillna("").astype(str).tolist(),
        model_name=args.embedding_model_name,
        requested_device=args.embedding_device,
        batch_size=int(args.embedding_batch_size),
    )
    candidate_df["generated_embedding_row"] = np.arange(len(candidate_df), dtype=int)

    source_row_ids = candidate_df["dataset_local_row_id"].to_numpy(dtype=int)
    source_cos = np.sum(generated_embeddings * source_embeddings[source_row_ids], axis=1)
    target_centroid_cos = np.clip(generated_embeddings @ centroid, -1.0, 1.0)
    target_centroid_dist = 1.0 - target_centroid_cos

    centroid_sims = generated_embeddings @ cluster_centroid_matrix.T
    nearest_idx = np.argmax(centroid_sims, axis=1)
    nearest_cluster_ids = cluster_centroid_ids[nearest_idx]
    nearest_cluster_sims = centroid_sims[np.arange(len(candidate_df)), nearest_idx]

    candidate_df["source_synthetic_cosine"] = source_cos
    candidate_df["target_centroid_cosine"] = target_centroid_cos
    candidate_df["target_centroid_distance"] = target_centroid_dist
    candidate_df["nearest_cluster_id"] = nearest_cluster_ids
    candidate_df["nearest_cluster_cosine"] = nearest_cluster_sims
    candidate_df["nearest_cluster_in_target"] = candidate_df["nearest_cluster_id"].isin(target_cluster_ids)
    candidate_df["target_basin_membership_flag"] = candidate_df["nearest_cluster_in_target"]
    candidate_df["target_centroid_distance_pass"] = candidate_df["target_centroid_distance"] <= centroid_distance_threshold
    candidate_df["target_gate_pass"] = candidate_df["nearest_cluster_in_target"] | candidate_df["target_centroid_distance_pass"] | candidate_df["target_basin_membership_flag"]
    candidate_df["target_score"] = [
        candidate_target_score(in_target, target_cos, src_cos)
        for in_target, target_cos, src_cos in zip(
            candidate_df["nearest_cluster_in_target"],
            candidate_df["target_centroid_cosine"],
            candidate_df["source_synthetic_cosine"],
        )
    ]

    if train_texts:
        overlap_flags = []
        overlap_ratios = []
        duplicate_vs_train = []
        for text in candidate_df["generated_text"].fillna("").astype(str):
            duplicate_vs_train.append(text_hash(text) in train_text_hashes)
            overlap_flag, overlap_ratio = lexical_overlap_flag(text, train_ngrams)
            overlap_flags.append(overlap_flag)
            overlap_ratios.append(overlap_ratio)
        candidate_df["exact_duplicate_vs_train_flag"] = duplicate_vs_train
        candidate_df["high_ngram_overlap_with_train_flag"] = overlap_flags
        candidate_df["high_ngram_overlap_with_train_ratio"] = overlap_ratios
    else:
        candidate_df["exact_duplicate_vs_train_flag"] = False
        candidate_df["high_ngram_overlap_with_train_flag"] = False
        candidate_df["high_ngram_overlap_with_train_ratio"] = math.nan

    candidate_df["basic_quality_pass"] = (
        (~candidate_df["empty_output_flag"])
        & (~candidate_df["too_short_flag"])
        & (~candidate_df["repetition_or_collapse_flag"])
    )
    candidate_df["source_cosine_pass"] = candidate_df["source_synthetic_cosine"] >= float(args.min_source_cosine)
    candidate_df["privacy_pass"] = (
        (~candidate_df["phi_warning_flag"])
        & (~candidate_df["exact_duplicate_vs_train_flag"])
        & (~candidate_df["high_ngram_overlap_with_train_flag"])
    )
    candidate_df["preliminary_accept"] = (
        candidate_df["basic_quality_pass"]
        & candidate_df["source_cosine_pass"]
        & candidate_df["privacy_pass"]
        & candidate_df["target_gate_pass"]
    )
    candidate_df["rejection_reasons"] = ""

    for idx, row in candidate_df.iterrows():
        reasons = []
        if not bool(row["basic_quality_pass"]):
            if bool(row["empty_output_flag"]):
                reasons.append("empty_output")
            if bool(row["too_short_flag"]):
                reasons.append("too_short")
            if bool(row["repetition_or_collapse_flag"]):
                reasons.append("repetition_or_collapse")
        if not bool(row["source_cosine_pass"]):
            reasons.append("low_source_cosine")
        if not bool(row["privacy_pass"]):
            if bool(row["phi_warning_flag"]):
                reasons.append("phi_warning")
            if bool(row["exact_duplicate_vs_train_flag"]):
                reasons.append("exact_duplicate_vs_train")
            if bool(row["high_ngram_overlap_with_train_flag"]):
                reasons.append("high_ngram_overlap_with_train")
        if not bool(row["target_gate_pass"]):
            reasons.append("failed_target_gate")
        candidate_df.at[idx, "rejection_reasons"] = "|".join(reasons)

    accepted_indices: list[int] = []
    globally_seen_hashes: set[str] = set()
    globally_accepted_embeddings: list[np.ndarray] = []

    for anchor_id, group_df in candidate_df.groupby("anchor_id", sort=False):
        group_df = group_df.copy()
        prelim_df = group_df.loc[group_df["preliminary_accept"]].copy()
        if prelim_df.empty:
            continue

        if args.selection_strategy == "best_target_score":
            prelim_df = prelim_df.sort_values(
                ["target_score", "source_synthetic_cosine", "target_centroid_cosine"],
                ascending=[False, False, False],
            )
        else:
            prelim_df = prelim_df.sort_values(
                ["target_score", "source_synthetic_cosine"],
                ascending=[False, False],
            )

        anchor_selected: list[int] = []
        anchor_embs: list[np.ndarray] = []
        for idx in prelim_df.index.tolist():
            emb = generated_embeddings[int(candidate_df.at[idx, "generated_embedding_row"])]
            if candidate_df.at[idx, "text_hash"] in globally_seen_hashes:
                candidate_df.at[idx, "rejection_reasons"] = append_reason(candidate_df.at[idx, "rejection_reasons"], "duplicate_accepted_text")
                continue
            if any(float(np.dot(emb, prev)) >= float(args.candidate_near_duplicate_cosine) for prev in globally_accepted_embeddings):
                candidate_df.at[idx, "rejection_reasons"] = append_reason(candidate_df.at[idx, "rejection_reasons"], "global_near_duplicate")
                continue
            if any(float(np.dot(emb, prev)) >= float(args.anchor_diversity_cosine) for prev in anchor_embs):
                candidate_df.at[idx, "rejection_reasons"] = append_reason(candidate_df.at[idx, "rejection_reasons"], "anchor_diversity_reject")
                continue
            anchor_selected.append(idx)
            anchor_embs.append(emb)
            globally_accepted_embeddings.append(emb)
            globally_seen_hashes.add(candidate_df.at[idx, "text_hash"])
            if len(anchor_selected) >= int(args.accepted_per_anchor):
                break

        accepted_indices.extend(anchor_selected)

    candidate_df["accepted_flag"] = False
    candidate_df.loc[accepted_indices, "accepted_flag"] = True
    candidate_df.loc[candidate_df["accepted_flag"], "rejection_reasons"] = ""
    candidate_df.loc[~candidate_df["accepted_flag"] & candidate_df["preliminary_accept"] & (candidate_df["rejection_reasons"] == ""), "rejection_reasons"] = "not_selected_after_diversity"

    shard_suffix = "" if int(args.num_shards) == 1 else f"_shard{int(args.shard_id):02d}of{int(args.num_shards):02d}"
    candidate_manifest_path = output_dir / f"closed_loop_candidate_manifest{shard_suffix}.jsonl"
    accepted_manifest_path = output_dir / f"closed_loop_accepted_manifest{shard_suffix}.jsonl"
    rejected_manifest_path = output_dir / f"closed_loop_rejected_manifest{shard_suffix}.jsonl"
    summary_json_path = output_dir / f"closed_loop_enrichment_summary{shard_suffix}.json"
    summary_md_path = output_dir / f"closed_loop_enrichment_summary{shard_suffix}.md"

    candidate_rows = candidate_df.to_dict(orient="records")
    accepted_rows = candidate_df.loc[candidate_df["accepted_flag"]].to_dict(orient="records")
    rejected_rows = candidate_df.loc[~candidate_df["accepted_flag"]].to_dict(orient="records")
    save_jsonl(candidate_manifest_path, candidate_rows)
    save_jsonl(accepted_manifest_path, accepted_rows)
    save_jsonl(rejected_manifest_path, rejected_rows)

    accepted_df = candidate_df.loc[candidate_df["accepted_flag"]].copy()
    rejected_df = candidate_df.loc[~candidate_df["accepted_flag"]].copy()

    faithfulness_cols = [
        "candidate_id",
        "anchor_id",
        "accepted_flag",
        "source_synthetic_cosine",
        "target_centroid_cosine",
        "target_centroid_distance",
        "nearest_cluster_id",
        "nearest_cluster_in_target",
        "patient_disjoint_from_train",
        "generated_word_count",
        "repetition_or_collapse_flag",
        "phi_warning_flag",
        "rejection_reasons",
    ]
    candidate_df[faithfulness_cols].to_csv(output_dir / f"accepted_vs_rejected_faithfulness{shard_suffix}.csv", index=False)

    anchor_acceptance_rows = []
    for anchor_id, group_df in candidate_df.groupby("anchor_id", sort=False):
        anchor_acceptance_rows.append(
            {
                "anchor_id": anchor_id,
                "dataset_row_id": group_df["dataset_row_id"].iloc[0],
                "note_id": group_df["note_id"].iloc[0],
                "subject_id": group_df["subject_id"].iloc[0],
                "hadm_id": group_df["hadm_id"].iloc[0],
                "patient_disjoint_from_train": group_df["patient_disjoint_from_train"].iloc[0],
                "n_candidates": int(len(group_df)),
                "n_preliminary_pass": int(group_df["preliminary_accept"].sum()),
                "n_accepted": int(group_df["accepted_flag"].sum()),
                "acceptance_rate": float(group_df["accepted_flag"].mean()),
            }
        )
    pd.DataFrame(anchor_acceptance_rows).to_csv(output_dir / f"acceptance_rate_by_anchor{shard_suffix}.csv", index=False)

    if "patient_disjoint_from_train" in candidate_df.columns:
        patient_rows = []
        for key, group_df in candidate_df.groupby(candidate_df["patient_disjoint_from_train"].fillna("unknown"), sort=False):
            patient_rows.append(
                {
                    "patient_disjoint_from_train": key,
                    "n_candidates": int(len(group_df)),
                    "n_accepted": int(group_df["accepted_flag"].sum()),
                    "acceptance_rate": float(group_df["accepted_flag"].mean()),
                }
            )
        pd.DataFrame(patient_rows).to_csv(output_dir / f"acceptance_rate_by_patient_disjoint{shard_suffix}.csv", index=False)
    else:
        pd.DataFrame(
            [{"patient_disjoint_from_train": "unknown", "n_candidates": int(len(candidate_df)), "n_accepted": int(accepted_df.shape[0]), "acceptance_rate": float(candidate_df["accepted_flag"].mean())}]
        ).to_csv(output_dir / f"acceptance_rate_by_patient_disjoint{shard_suffix}.csv", index=False)

    coverage_rows = []
    if "patient_disjoint_from_train" in anchor_df.columns:
        anchor_patient_disjoint = anchor_df["patient_disjoint_from_train"].fillna(False).astype(bool)
    else:
        anchor_patient_disjoint = pd.Series(False, index=anchor_df.index)
    if "patient_disjoint_from_train" in accepted_df.columns:
        accepted_patient_disjoint = accepted_df["patient_disjoint_from_train"].fillna(False).astype(bool)
    else:
        accepted_patient_disjoint = pd.Series(False, index=accepted_df.index)

    analysis_groups = {
        "full": (
            pd.Series(True, index=anchor_df.index),
            pd.Series(True, index=accepted_df.index),
        ),
        "patient_disjoint": (
            anchor_patient_disjoint,
            accepted_patient_disjoint,
        ),
        "patient_overlap": (
            ~anchor_patient_disjoint,
            ~accepted_patient_disjoint,
        ),
    }

    for analysis_group, (anchor_mask, accepted_mask) in analysis_groups.items():
        anchor_subset = anchor_df.loc[anchor_mask].copy()
        accepted_subset = accepted_df.loc[accepted_mask].copy()

        anchor_counts = Counter(pd.to_numeric(anchor_subset["cluster_id"], errors="coerce").dropna().astype(int).tolist())
        accepted_counts = Counter(pd.to_numeric(accepted_subset["nearest_cluster_id"], errors="coerce").dropna().astype(int).tolist())
        all_clusters = sorted(set(anchor_counts) | set(accepted_counts) | set(target_cluster_ids))
        anchor_total = sum(anchor_counts.values()) or 1
        accepted_total = sum(accepted_counts.values()) or 1
        for cluster_id in all_clusters:
            coverage_rows.append(
                {
                    "analysis_group": analysis_group,
                    "cluster_id": int(cluster_id),
                    "is_target_cluster": bool(cluster_id in target_cluster_ids),
                    "anchor_count": int(anchor_counts.get(cluster_id, 0)),
                    "anchor_fraction": float(anchor_counts.get(cluster_id, 0) / anchor_total),
                    "accepted_count": int(accepted_counts.get(cluster_id, 0)),
                    "accepted_fraction": float(accepted_counts.get(cluster_id, 0) / accepted_total),
                }
            )
    pd.DataFrame(coverage_rows).to_csv(output_dir / f"target_region_coverage_before_after{shard_suffix}.csv", index=False)

    manual_review_df = accepted_df.sample(
        n=min(int(args.manual_review_sample_size), len(accepted_df)),
        random_state=42,
    ) if len(accepted_df) else accepted_df.head(0)
    review_cols = [
        "candidate_id",
        "anchor_id",
        "dataset_row_id",
        "note_id",
        "subject_id",
        "hadm_id",
        "patient_disjoint_from_train",
        "source_synthetic_cosine",
        "target_centroid_cosine",
        "nearest_cluster_id",
        "generated_word_count",
        "generated_text",
    ]
    manual_review_df[review_cols].to_csv(output_dir / f"accepted_note_manual_review_sample{shard_suffix}.csv", index=False)

    diversity = pairwise_diversity_summary(
        generated_embeddings[accepted_df["generated_embedding_row"].to_numpy(dtype=int)]
    ) if len(accepted_df) else {"mean_pairwise_cosine": math.nan, "max_pairwise_cosine": math.nan, "min_pairwise_cosine": math.nan}

    summary = {
        "created_at": now_iso(),
        "script_path": str(Path(__file__).resolve()),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "anchor_manifest_path": str(anchor_manifest_path),
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "backbone_path": str(Path(args.backbone_path).resolve()),
        "embedding_model_name": args.embedding_model_name,
        "embedding_device_resolved": resolved_embedding_device,
        "cluster_assignments_path": str(cluster_assignments_path),
        "split_manifest_path": str(split_manifest_path) if split_manifest_path else None,
        "target_cluster_ids": target_cluster_ids,
        "target_centroid_distance_threshold": float(centroid_distance_threshold),
        "n_candidates_per_anchor": int(args.n_candidates_per_anchor),
        "accepted_per_anchor": int(args.accepted_per_anchor),
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "total_anchors": int(len(anchor_df)),
        "total_candidates_generated": int(len(candidate_df)),
        "total_accepted": int(len(accepted_df)),
        "acceptance_rate": float(candidate_df["accepted_flag"].mean()) if len(candidate_df) else math.nan,
        "accepted_target_cluster_rate": float(accepted_df["nearest_cluster_in_target"].mean()) if len(accepted_df) else math.nan,
        "accepted_centroid_distance_pass_rate": float(accepted_df["target_centroid_distance_pass"].mean()) if len(accepted_df) else math.nan,
        "accepted_mean_source_cosine": float(accepted_df["source_synthetic_cosine"].mean()) if len(accepted_df) else math.nan,
        "rejected_mean_source_cosine": float(rejected_df["source_synthetic_cosine"].mean()) if len(rejected_df) else math.nan,
        "accepted_collapse_rate": group_flag_rate(accepted_df, "repetition_or_collapse_flag"),
        "rejected_collapse_rate": group_flag_rate(rejected_df, "repetition_or_collapse_flag"),
        "accepted_phi_warning_rate": group_flag_rate(accepted_df, "phi_warning_flag"),
        "rejected_phi_warning_rate": group_flag_rate(rejected_df, "phi_warning_flag"),
        "train_overlap_screen_status": "loaded" if train_texts and not train_text_warning else (train_text_warning or "skipped"),
        "privacy_warning_rates": {
            "accepted_exact_duplicate_vs_train_rate": group_flag_rate(accepted_df, "exact_duplicate_vs_train_flag"),
            "rejected_exact_duplicate_vs_train_rate": group_flag_rate(rejected_df, "exact_duplicate_vs_train_flag"),
            "accepted_high_ngram_overlap_rate": group_flag_rate(accepted_df, "high_ngram_overlap_with_train_flag"),
            "rejected_high_ngram_overlap_rate": group_flag_rate(rejected_df, "high_ngram_overlap_with_train_flag"),
        },
        "diversity_summary": diversity,
        "package_versions": package_versions(),
        "output_files": {
            "closed_loop_candidate_manifest": str(candidate_manifest_path),
            "closed_loop_accepted_manifest": str(accepted_manifest_path),
            "closed_loop_rejected_manifest": str(rejected_manifest_path),
            "closed_loop_enrichment_summary_json": str(summary_json_path),
            "closed_loop_enrichment_summary_md": str(summary_md_path),
            "accepted_vs_rejected_faithfulness": str(output_dir / f"accepted_vs_rejected_faithfulness{shard_suffix}.csv"),
            "target_region_coverage_before_after": str(output_dir / f"target_region_coverage_before_after{shard_suffix}.csv"),
            "acceptance_rate_by_anchor": str(output_dir / f"acceptance_rate_by_anchor{shard_suffix}.csv"),
            "acceptance_rate_by_patient_disjoint": str(output_dir / f"acceptance_rate_by_patient_disjoint{shard_suffix}.csv"),
            "accepted_note_manual_review_sample": str(output_dir / f"accepted_note_manual_review_sample{shard_suffix}.csv"),
        },
    }
    save_json(summary_json_path, summary)
    summary_md_path.write_text(markdown_summary(summary), encoding="utf-8")

    print("Saved closed-loop candidate manifest to:", candidate_manifest_path)
    print("Saved accepted manifest to:", accepted_manifest_path)
    print("Saved rejected manifest to:", rejected_manifest_path)
    print("Saved summary JSON to:", summary_json_path)
    print("Saved summary Markdown to:", summary_md_path)


def append_reason(existing: str, reason: str) -> str:
    reasons = [item for item in str(existing).split("|") if item]
    if reason not in reasons:
        reasons.append(reason)
    return "|".join(reasons)


if __name__ == "__main__":
    main()
