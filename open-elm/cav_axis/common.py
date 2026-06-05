from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler


MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "null", "missing", "unknown"}


@dataclass
class EncodedTargets:
    matrix: np.ndarray
    manifest: pd.DataFrame
    factor_types: Dict[str, str]


def parse_csv_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def save_json(path: str | Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _normalize_join_cols(df: pd.DataFrame, join_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in join_cols:
        if col not in out.columns:
            raise KeyError(f"Required join column not found: {col}")
        out[col] = out[col].astype(str).str.strip()
    return out


def load_and_merge_tables(
    embeddings_path: str | Path,
    metadata_path: str | Path,
    factors_path: str | Path,
    join_cols: Sequence[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(embeddings_path)
    metadata_df = pd.read_csv(metadata_path)
    factors_df = pd.read_csv(factors_path)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}")
    if len(metadata_df) != embeddings.shape[0]:
        raise ValueError(
            f"Embeddings/metadata mismatch: {embeddings.shape[0]} rows vs {len(metadata_df)} metadata rows"
        )

    metadata_df = metadata_df.copy()
    metadata_df["embedding_row_id"] = np.arange(len(metadata_df))

    metadata_df = _normalize_join_cols(metadata_df, join_cols)
    factors_df = _normalize_join_cols(factors_df, join_cols)

    duplicate_mask = factors_df.duplicated(subset=list(join_cols), keep=False)
    if duplicate_mask.any():
        dup_count = int(duplicate_mask.sum())
        raise ValueError(
            f"Factor table has {dup_count} rows with duplicate join keys; "
            "deduplicate it before fitting the axis bank."
        )

    merged_df = metadata_df.merge(factors_df, on=list(join_cols), how="left", validate="many_to_one")
    return embeddings.astype(np.float64), merged_df


def clean_factor_values(df: pd.DataFrame, factor_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in factor_cols:
        if col not in out.columns:
            raise KeyError(f"Factor column not found: {col}")
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        as_str = out[col].astype(str).str.strip()
        out[col] = as_str.mask(as_str.str.lower().isin(MISSING_STRINGS))
    return out


def infer_factor_type(
    series: pd.Series,
    factor_name: str,
    continuous_overrides: Iterable[str],
    categorical_overrides: Iterable[str],
) -> str:
    if factor_name in set(continuous_overrides):
        return "continuous"
    if factor_name in set(categorical_overrides):
        return "categorical"

    numeric = pd.to_numeric(series, errors="coerce")
    non_null_numeric = numeric.dropna()
    if len(non_null_numeric) == len(series.dropna()):
        unique_values = sorted(pd.unique(non_null_numeric))
        if len(unique_values) > 8:
            return "continuous"
        if set(unique_values).issubset({0, 1}):
            return "categorical"
        return "continuous"
    return "categorical"


def encode_targets(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    continuous_overrides: Iterable[str] = (),
    categorical_overrides: Iterable[str] = (),
) -> EncodedTargets:
    matrices: List[np.ndarray] = []
    manifest_rows: List[Dict[str, object]] = []
    factor_types: Dict[str, str] = {}

    for factor in factor_cols:
        factor_type = infer_factor_type(
            df[factor], factor, continuous_overrides=continuous_overrides, categorical_overrides=categorical_overrides
        )
        factor_types[factor] = factor_type

        if factor_type == "continuous":
            numeric = pd.to_numeric(df[factor], errors="coerce").astype(float)
            mean = float(numeric.mean())
            std = float(numeric.std(ddof=0))
            if std == 0.0:
                std = 1.0
            values = ((numeric - mean) / std).to_numpy().reshape(-1, 1)
            matrices.append(values)
            manifest_rows.append(
                {
                    "factor": factor,
                    "target_column": factor,
                    "target_kind": "continuous",
                    "factor_type": "continuous",
                    "level": "",
                    "scale_mean": mean,
                    "scale_std": std,
                }
            )
            continue

        categories = df[factor].astype(str)
        levels = sorted(pd.unique(categories))
        for level in levels:
            column_name = f"{factor}::{level}"
            values = (categories == level).astype(float).to_numpy().reshape(-1, 1)
            matrices.append(values)
            manifest_rows.append(
                {
                    "factor": factor,
                    "target_column": column_name,
                    "target_kind": "binary",
                    "factor_type": "categorical",
                    "level": level,
                    "scale_mean": "",
                    "scale_std": "",
                }
            )

    if not matrices:
        raise ValueError("No target columns were created; check factor column selection.")

    matrix = np.concatenate(matrices, axis=1)
    manifest = pd.DataFrame(manifest_rows)
    return EncodedTargets(matrix=matrix, manifest=manifest, factor_types=factor_types)


def filter_complete_cases(
    embeddings: np.ndarray,
    merged_df: pd.DataFrame,
    factor_cols: Sequence[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    cleaned_df = clean_factor_values(merged_df, factor_cols)
    keep_mask = cleaned_df[list(factor_cols)].notna().all(axis=1).to_numpy()
    return embeddings[keep_mask], cleaned_df.loc[keep_mask].reset_index(drop=True)


def split_rows(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    group_col: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    row_indices = np.arange(len(df))

    if group_col and group_col in df.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        groups = df[group_col].astype(str).to_numpy()
        train_idx, test_idx = next(splitter.split(row_indices, groups=groups))
        return train_idx, test_idx

    train_idx, test_idx = train_test_split(
        row_indices, test_size=test_size, random_state=random_state, shuffle=True
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def fit_multioutput_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alphas: Sequence[float],
) -> Tuple[StandardScaler, RidgeCV, np.ndarray, np.ndarray]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_train_scaled = scaler.fit_transform(x_train)

    model = RidgeCV(alphas=np.asarray(alphas, dtype=float), fit_intercept=True)
    model.fit(x_train_scaled, y_train)

    coef_std = np.asarray(model.coef_, dtype=float)
    coef_raw = coef_std / scaler.scale_[None, :]
    intercept_raw = np.asarray(model.intercept_, dtype=float) - (scaler.mean_ / scaler.scale_) @ coef_std.T
    return scaler, model, coef_raw, intercept_raw


def compute_axis_bank(
    coefficient_matrix: np.ndarray,
    max_axes: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u, singular_values, vt = np.linalg.svd(coefficient_matrix, full_matrices=False)
    if max_axes is not None:
        limit = min(max_axes, vt.shape[0])
        u = u[:, :limit]
        singular_values = singular_values[:limit]
        vt = vt[:limit, :]
    axes = vt.T.copy()
    energy = singular_values ** 2
    energy_ratio = energy / energy.sum() if energy.sum() > 0 else np.zeros_like(energy)
    return axes, singular_values, energy_ratio, u


def reconstruct_coefficients(
    axes: np.ndarray,
    singular_values: np.ndarray,
    left_vectors: np.ndarray,
    k: int,
) -> np.ndarray:
    left_k = left_vectors[:, :k]
    singular_k = singular_values[:k]
    axes_k = axes[:, :k].T
    return (left_k * singular_k) @ axes_k


def score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for col_idx, row in manifest.reset_index(drop=True).iterrows():
        truth = y_true[:, col_idx]
        pred = y_pred[:, col_idx]

        if row["target_kind"] == "binary":
            unique_values = np.unique(truth)
            if len(unique_values) < 2:
                score = np.nan
            else:
                score = float(roc_auc_score(truth, pred))
            metric_name = "roc_auc"
        else:
            score = float(r2_score(truth, pred))
            metric_name = "r2"

        rows.append(
            {
                "factor": row["factor"],
                "target_column": row["target_column"],
                "target_kind": row["target_kind"],
                "metric": metric_name,
                "score": score,
            }
        )

    return pd.DataFrame(rows)


def summarize_metric_table(metrics_df: pd.DataFrame) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if metrics_df.empty:
        return summary

    binary_scores = metrics_df.loc[metrics_df["target_kind"] == "binary", "score"].dropna()
    continuous_scores = metrics_df.loc[metrics_df["target_kind"] == "continuous", "score"].dropna()

    if not binary_scores.empty:
        summary["mean_binary_roc_auc"] = float(binary_scores.mean())
    if not continuous_scores.empty:
        summary["mean_continuous_r2"] = float(continuous_scores.mean())

    clipped_scores = metrics_df["score"].dropna().clip(lower=0.0, upper=1.0)
    if not clipped_scores.empty:
        summary["mean_clipped_score"] = float(clipped_scores.mean())

    return summary


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def top_axis_targets(
    axis_loadings: np.ndarray,
    target_manifest: pd.DataFrame,
    top_n: int = 5,
) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for axis_idx in range(axis_loadings.shape[1]):
        values = axis_loadings[:, axis_idx]
        ranked = np.argsort(np.abs(values))[::-1][:top_n]
        targets = []
        for target_idx in ranked:
            row = target_manifest.iloc[target_idx]
            targets.append(
                {
                    "factor": row["factor"],
                    "target_column": row["target_column"],
                    "level": row["level"],
                    "loading": float(values[target_idx]),
                }
            )
        summaries.append({"axis": axis_idx, "top_targets": targets})
    return summaries
