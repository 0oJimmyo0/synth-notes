#!/usr/bin/env python3
"""
Build a second-pass corrected HF dataset using a decode -> re-embed -> correct loop.

This Phase 2b pilot is intentionally minimal:
1. start from a first-pass shifted dataset
2. use the generated-note BGE embeddings from that pass
3. compare where the decoded notes landed versus a target neighborhood
4. apply one correction step in the same 1024-d embedding space
5. write a corrected HF dataset for a second decode pass
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a decode-reembed correction pilot dataset.")
    parser.add_argument("--source_dataset_path", required=True, help="Original real HF dataset, e.g. encoded_testing_filtered")
    parser.add_argument("--shifted_dataset_path", required=True, help="First-pass shifted HF dataset used for generation")
    parser.add_argument("--generated_manifest_path", required=True, help="First-pass generation manifest JSONL")
    parser.add_argument("--generated_embeddings_path", required=True, help="BGE re-embeddings of generated notes (.npy)")
    parser.add_argument("--factors_path", required=True, help="Factor table containing the target column")
    parser.add_argument("--target_column", required=True, help="Binary target column, e.g. cluster_target_29")
    parser.add_argument("--output_dir", required=True, help="Output directory for corrected dataset + metadata")
    parser.add_argument("--split_manifest_path", default=None, help="Optional filtered-aligned split manifest")
    parser.add_argument(
        "--join_cols",
        default="source_row_id,embedding_row_id,dataset_row_id,note_id,subject_id,hadm_id",
        help="Preferred join columns for metadata merges",
    )
    parser.add_argument("--source_split", default=None, help="Optional split filter such as test/dev/train")
    parser.add_argument(
        "--target_selection_query",
        default=None,
        help="Optional query for the real target pool. Defaults to `<target_column> == 1`.",
    )
    parser.add_argument(
        "--k_target_neighbors",
        type=int,
        default=8,
        help="Number of nearest target neighbors used to define the correction target",
    )
    parser.add_argument(
        "--neighbor_weight_mode",
        default="softmax_cosine",
        choices=["softmax_cosine", "inverse_distance", "uniform"],
        help="How to weight the local target neighbors inside the correction target mixture",
    )
    parser.add_argument(
        "--neighbor_temperature",
        type=float,
        default=0.02,
        help="Softmax temperature for neighbor weighting when using softmax_cosine",
    )
    parser.add_argument(
        "--correction_beta",
        type=float,
        default=0.5,
        help="Step size for adding the decode residual back to the input embedding",
    )
    parser.add_argument(
        "--max_correction_norm",
        type=float,
        default=0.20,
        help="Optional cap on the correction-step norm before final normalization",
    )
    parser.add_argument(
        "--normalize_after_correction",
        action="store_true",
        help="L2-normalize corrected embeddings after the correction step",
    )
    parser.add_argument(
        "--output_stem",
        default="decode_reembed_correction",
        help="Stem for metadata sidecar files written next to the saved HF dataset",
    )
    return parser


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    def _json_default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


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


def normalize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def maybe_int(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        return normalize_scalar(value)


def load_dataset_rows(dataset_path: Path) -> tuple[Dataset, pd.DataFrame, np.ndarray]:
    dataset = Dataset.load_from_disk(str(dataset_path))
    metadata_cols = [col for col in dataset.column_names if col not in {"input_ids", "domain_embeddings"}]
    if metadata_cols:
        base_df = dataset.select_columns(metadata_cols).to_pandas().reset_index(drop=True)
    else:
        base_df = pd.DataFrame(index=np.arange(len(dataset), dtype=int))
    if "dataset_row_id" not in base_df.columns:
        base_df.insert(0, "dataset_row_id", np.arange(len(dataset), dtype=int))

    embeddings = []
    for emb in dataset["domain_embeddings"]:
        if not isinstance(emb, list) or not emb:
            raise ValueError("Expected each dataset row to carry a non-empty domain_embeddings list.")
        first = emb[0]
        arr = np.asarray(first, dtype=np.float32)
        while arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise ValueError(f"Expected each domain embedding to resolve to 1D, got shape {arr.shape}")
        embeddings.append(arr)
    return dataset, base_df, normalize_rows(np.vstack(embeddings))


def choose_join_keys(frames: list[pd.DataFrame], preferred_keys: list[str]) -> list[str]:
    common_cols = set(frames[0].columns)
    for frame in frames[1:]:
        common_cols &= set(frame.columns)
    preferred_groups = [
        ["split", "dataset_row_id"],
        ["source_row_id"],
        ["embedding_row_id"],
        ["dataset_row_id"],
        ["note_id", "subject_id", "hadm_id"],
        ["note_id"],
        ["subject_id", "hadm_id"],
    ]
    for key in preferred_keys:
        if key in {"note_id", "subject_id", "hadm_id"}:
            continue
        if [key] not in preferred_groups:
            preferred_groups.append([key])
    for keys in preferred_groups:
        if all(key in common_cols for key in keys):
            return keys
    raise ValueError(f"Could not detect stable join keys. Shared columns were: {sorted(common_cols)}")


def normalize_join_cols(df: pd.DataFrame, join_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in join_cols:
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().all():
            out[col] = numeric.astype(int)
        else:
            out[col] = out[col].astype(str).str.strip()
    return out


def merge_metadata(
    base_df: pd.DataFrame,
    factors_path: Path,
    split_manifest_path: str | None,
    preferred_join_cols: list[str],
    source_split: str | None,
) -> pd.DataFrame:
    merged = base_df.copy()
    if split_manifest_path:
        split_df = pd.read_csv(split_manifest_path)
        join_cols = choose_join_keys([merged, split_df], preferred_join_cols)
        if join_cols == ["dataset_row_id"] and "split" in split_df.columns and source_split:
            split_df = split_df.loc[split_df["split"].astype(str) == str(source_split)].copy()
        merged = normalize_join_cols(merged, join_cols)
        split_df = normalize_join_cols(split_df, join_cols)
        split_df = split_df.drop_duplicates(subset=join_cols)
        merged = merged.merge(split_df, on=join_cols, how="left", validate="one_to_one", suffixes=("", "_split"))

    factors_df = pd.read_csv(factors_path)
    join_cols = choose_join_keys([merged, factors_df], preferred_join_cols)
    merged = normalize_join_cols(merged, join_cols)
    factors_df = normalize_join_cols(factors_df, join_cols)
    factors_df = factors_df.drop_duplicates(subset=join_cols)
    merged = merged.merge(factors_df, on=join_cols, how="left", validate="one_to_one", suffixes=("", "_factor"))
    return merged


def pairwise_topk_cosine(queries: np.ndarray, targets: np.ndarray, k: int, batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    all_scores = []
    all_indices = []
    for start in range(0, queries.shape[0], batch_size):
        stop = min(start + batch_size, queries.shape[0])
        sims = queries[start:stop] @ targets.T
        top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        all_scores.append(top_scores.astype(np.float32))
        all_indices.append(top_idx.astype(np.int32))
    return np.vstack(all_scores), np.vstack(all_indices)


def compute_neighbor_weights(cosine_similarities: np.ndarray, mode: str, temperature: float) -> np.ndarray:
    if mode == "uniform":
        weights = np.full_like(cosine_similarities, fill_value=1.0 / cosine_similarities.shape[0], dtype=np.float32)
    elif mode == "inverse_distance":
        distances = 1.0 - cosine_similarities
        inv = 1.0 / np.clip(distances, 1e-6, None)
        weights = inv / np.clip(inv.sum(), 1e-12, None)
        weights = weights.astype(np.float32)
    elif mode == "softmax_cosine":
        temp = max(float(temperature), 1e-6)
        shifted = cosine_similarities - float(cosine_similarities.max())
        logits = np.clip(shifted / temp, -80.0, 80.0)
        exps = np.exp(logits)
        weights = exps / np.clip(exps.sum(), 1e-12, None)
        weights = weights.astype(np.float32)
    else:
        raise ValueError(f"Unsupported neighbor weight mode: {mode}")
    return weights


def cap_norm(vec: np.ndarray, max_norm: float | None) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(vec))
    if max_norm is None or norm <= max_norm or norm <= 0.0:
        return vec.astype(np.float32), norm
    scaled = vec * (max_norm / norm)
    return scaled.astype(np.float32), float(np.linalg.norm(scaled))


def load_manifest(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True, dtype=False).reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()

    source_dataset_path = Path(args.source_dataset_path).resolve()
    shifted_dataset_path = Path(args.shifted_dataset_path).resolve()
    generated_manifest_path = Path(args.generated_manifest_path).resolve()
    generated_embeddings_path = Path(args.generated_embeddings_path).resolve()
    factors_path = Path(args.factors_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preferred_join_cols = parse_csv_list(args.join_cols)
    source_dataset, source_df, source_embeddings = load_dataset_rows(source_dataset_path)
    shifted_dataset, shifted_df, shifted_embeddings = load_dataset_rows(shifted_dataset_path)

    manifest_df = load_manifest(generated_manifest_path)
    generated_embeddings = np.load(generated_embeddings_path)

    if len(shifted_dataset) != len(manifest_df):
        raise ValueError(
            f"Shifted dataset row count {len(shifted_dataset)} does not match manifest row count {len(manifest_df)}"
        )
    if generated_embeddings.shape[0] != len(manifest_df):
        raise ValueError(
            f"Generated embedding row count {generated_embeddings.shape[0]} does not match manifest row count {len(manifest_df)}"
        )

    # Align on generation order. The current pipeline guarantees manifest order == dataset order.
    if "generation_index" in manifest_df.columns:
        manifest_df = manifest_df.sort_values("generation_index").reset_index(drop=True)

    if "dataset_row_id" not in manifest_df.columns or "dataset_row_id" not in shifted_df.columns:
        raise ValueError("Both manifest and shifted dataset must carry dataset_row_id for correction alignment.")
    shifted_ids = shifted_df["dataset_row_id"].astype(int).to_numpy()
    manifest_ids = manifest_df["dataset_row_id"].astype(int).to_numpy()
    if not np.array_equal(shifted_ids, manifest_ids):
        raise ValueError("Shifted dataset row order does not match manifest dataset_row_id order.")

    if args.source_split and "split" not in source_df.columns:
        source_df["split"] = args.source_split

    source_meta = merge_metadata(source_df, factors_path, args.split_manifest_path, preferred_join_cols, args.source_split)
    if args.source_split and "split" in source_meta.columns:
        source_meta = source_meta.loc[source_meta["split"] == args.source_split].copy()

    if args.target_column not in source_meta.columns:
        raise KeyError(f"Target column not found after merge: {args.target_column}")
    numeric = pd.to_numeric(source_meta[args.target_column], errors="coerce")
    source_meta = source_meta.loc[numeric.notna()].copy()
    source_meta[args.target_column] = numeric.loc[source_meta.index].astype(int)
    source_meta = source_meta.loc[source_meta[args.target_column].isin([0, 1])].copy()

    target_query = args.target_selection_query or f"{args.target_column} == 1"
    target_meta = source_meta.query(target_query, engine="python").copy()
    if target_meta.empty:
        raise ValueError("No real target rows found for the requested target selection.")

    target_row_ids = target_meta["dataset_row_id"].astype(int).to_numpy()
    target_embeddings = source_embeddings[target_row_ids]
    target_centroid = normalize_rows(target_embeddings.mean(axis=0, keepdims=True))[0]

    k = min(max(int(args.k_target_neighbors), 1), len(target_embeddings))
    gen_emb = normalize_rows(np.asarray(generated_embeddings, dtype=np.float32))
    top_scores, top_idx = pairwise_topk_cosine(gen_emb, target_embeddings, k=k)

    corrected_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for row_idx in range(len(shifted_dataset)):
        example = shifted_dataset[row_idx]
        shifted_embedding = shifted_embeddings[row_idx]
        generated_embedding = gen_emb[row_idx]
        source_row_id = int(shifted_ids[row_idx])
        source_embedding = source_embeddings[source_row_id]

        local_target_rows = target_row_ids[top_idx[row_idx]]
        local_target_scores = top_scores[row_idx]
        weights = compute_neighbor_weights(
            local_target_scores,
            mode=args.neighbor_weight_mode,
            temperature=args.neighbor_temperature,
        )
        local_target_mix = normalize_rows(np.sum(target_embeddings[top_idx[row_idx]] * weights[:, None], axis=0, keepdims=True))[0]

        residual = local_target_mix - generated_embedding
        correction_step, correction_norm = cap_norm(args.correction_beta * residual, args.max_correction_norm)
        corrected_embedding = shifted_embedding + correction_step
        if args.normalize_after_correction:
            corrected_embedding = normalize_rows(corrected_embedding.reshape(1, -1))[0]
        else:
            corrected_embedding = corrected_embedding.astype(np.float32)

        pre_target_mix_cos = float(np.dot(generated_embedding, local_target_mix))
        post_target_mix_cos = float(np.dot(corrected_embedding, local_target_mix) / max(np.linalg.norm(corrected_embedding), 1e-12))
        post_source_cos = float(np.dot(corrected_embedding, source_embedding) / max(np.linalg.norm(corrected_embedding), 1e-12))
        pre_source_cos = float(np.dot(generated_embedding, source_embedding))
        centroid_cos = float(np.dot(corrected_embedding, target_centroid) / max(np.linalg.norm(corrected_embedding), 1e-12))
        weight_entropy = float(-np.sum(weights * np.log(np.clip(weights, 1e-12, None))))

        row = dict(example)
        row["domain_embeddings"] = [corrected_embedding.astype(np.float32).tolist()]
        row["axis_label"] = f"{normalize_scalar(example.get('axis_label'))}__decode_reembed_correction"
        row["correction_mode"] = "decode_reembed_target_mix"
        row["correction_beta"] = float(args.correction_beta)
        row["max_correction_norm"] = float(args.max_correction_norm) if args.max_correction_norm is not None else None
        row["normalize_after_correction"] = bool(args.normalize_after_correction)
        row["correction_target_column"] = args.target_column
        row["correction_target_selection_query"] = target_query
        row["correction_target_neighbor_count"] = int(k)
        row["correction_neighbor_weight_mode"] = args.neighbor_weight_mode
        row["correction_neighbor_temperature"] = float(args.neighbor_temperature)
        row["correction_nearest_target_dataset_row_id"] = int(local_target_rows[0])
        row["correction_generated_to_target_mix_cosine"] = pre_target_mix_cos
        row["correction_corrected_to_target_mix_cosine"] = post_target_mix_cos
        row["correction_generated_to_source_cosine"] = pre_source_cos
        row["correction_corrected_to_source_cosine"] = post_source_cos
        row["correction_corrected_to_target_centroid_cosine"] = centroid_cos
        row["correction_step_norm"] = correction_norm
        row["correction_neighbor_weight_entropy"] = weight_entropy
        row["first_pass_generated_embedding_row_id"] = int(row_idx)
        row["first_pass_manifest_path"] = str(generated_manifest_path)
        row["first_pass_generated_embeddings_path"] = str(generated_embeddings_path)
        corrected_rows.append(row)

        manifest_rows.append(
            {
                "shifted_dataset_row_id": row_idx,
                "dataset_row_id": maybe_int(row.get("dataset_row_id")),
                "note_id": normalize_scalar(row.get("note_id")),
                "subject_id": maybe_int(row.get("subject_id")),
                "hadm_id": maybe_int(row.get("hadm_id")),
                "split": normalize_scalar(row.get("split")),
                "alpha": normalize_scalar(row.get("alpha")),
                "axis_label": normalize_scalar(row.get("axis_label")),
                "correction_mode": row["correction_mode"],
                "correction_beta": row["correction_beta"],
                "correction_step_norm": row["correction_step_norm"],
                "correction_generated_to_target_mix_cosine": row["correction_generated_to_target_mix_cosine"],
                "correction_corrected_to_target_mix_cosine": row["correction_corrected_to_target_mix_cosine"],
                "correction_corrected_to_source_cosine": row["correction_corrected_to_source_cosine"],
                "correction_corrected_to_target_centroid_cosine": row["correction_corrected_to_target_centroid_cosine"],
            }
        )

    corrected_dataset = Dataset.from_list(corrected_rows)
    corrected_dataset.save_to_disk(str(output_dir))

    manifest_csv = output_dir / f"{args.output_stem}_dataset_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)

    summary_df = pd.DataFrame(manifest_rows)
    summary_cols = []
    if "alpha" in summary_df.columns and summary_df["alpha"].notna().any():
        summary_cols.append("alpha")
    if not summary_cols:
        summary_cols.append("axis_label")
    correction_summary = (
        summary_df.groupby(summary_cols, dropna=False)
        .agg(
            n_rows=("shifted_dataset_row_id", "size"),
            mean_step_norm=("correction_step_norm", "mean"),
            mean_generated_to_target_mix_cosine=("correction_generated_to_target_mix_cosine", "mean"),
            mean_corrected_to_target_mix_cosine=("correction_corrected_to_target_mix_cosine", "mean"),
            mean_corrected_to_source_cosine=("correction_corrected_to_source_cosine", "mean"),
            mean_corrected_to_target_centroid_cosine=("correction_corrected_to_target_centroid_cosine", "mean"),
        )
        .reset_index()
    )
    correction_summary.to_csv(output_dir / f"{args.output_stem}_correction_summary.csv", index=False)

    run_metadata_path = output_dir / f"{args.output_stem}_run_metadata.json"
    save_json(
        run_metadata_path,
        {
            "created_at": now_iso(),
            "git_commit": get_git_commit(Path(__file__).resolve().parent),
            "script_path": str(Path(__file__).resolve()),
            "source_dataset_path": str(source_dataset_path),
            "shifted_dataset_path": str(shifted_dataset_path),
            "generated_manifest_path": str(generated_manifest_path),
            "generated_embeddings_path": str(generated_embeddings_path),
            "factors_path": str(factors_path),
            "split_manifest_path": str(Path(args.split_manifest_path).resolve()) if args.split_manifest_path else None,
            "output_dir": str(output_dir),
            "output_manifest_csv": str(manifest_csv),
            "output_correction_summary_csv": str(output_dir / f"{args.output_stem}_correction_summary.csv"),
            "target_column": args.target_column,
            "target_selection_query": target_query,
            "source_split": args.source_split,
            "k_target_neighbors": int(k),
            "neighbor_weight_mode": args.neighbor_weight_mode,
            "neighbor_temperature": float(args.neighbor_temperature),
            "correction_beta": float(args.correction_beta),
            "max_correction_norm": float(args.max_correction_norm) if args.max_correction_norm is not None else None,
            "normalize_after_correction": bool(args.normalize_after_correction),
            "n_rows": int(len(corrected_dataset)),
        },
    )

    print(f"Saved corrected dataset to: {output_dir}")
    print(f"Saved dataset manifest CSV to: {manifest_csv}")
    print(f"Saved correction summary CSV to: {output_dir / f'{args.output_stem}_correction_summary.csv'}")
    print(f"Saved run metadata to: {run_metadata_path}")


if __name__ == "__main__":
    main()
