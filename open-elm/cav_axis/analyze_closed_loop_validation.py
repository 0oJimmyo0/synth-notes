#!/usr/bin/env python3
"""Definitive validation analysis for a closed-loop enrichment pilot.

Produces: (A) same-anchor paired selected-vs-vanilla analysis, (B) an
equal-compute random-one-of-eight selector baseline, and (C) all-anchor
operational yield.  It uses existing manifests only; no generation is run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze closed-loop validation without new generation.")
    parser.add_argument("--accepted_manifest_path", required=True)
    parser.add_argument("--candidate_manifest_path", required=True)
    parser.add_argument("--vanilla_manifest_path", required=True)
    parser.add_argument("--target_cluster_ids", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--random_selector_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_ids(value: str) -> list[int]:
    return sorted({int(part.strip()) for part in value.split(",") if part.strip()})


def load(path: str) -> pd.DataFrame:
    p = Path(path).resolve()
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(p)
    return pd.read_json(p, lines=True)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).map({True: True, False: False, "True": True, "False": False, 1: True, 0: False}).fillna(False).astype(bool)


def attach_routes(df: pd.DataFrame, targets: list[int]) -> pd.DataFrame:
    out = df.copy()
    out["nearest_cluster_id"] = pd.to_numeric(out["nearest_cluster_id"], errors="coerce")
    out["exact_pooled_cluster_pass"] = out["nearest_cluster_id"].isin(targets)
    if "centroid_proximity_pass" in out:
        centroid = as_bool(out["centroid_proximity_pass"])
    else:
        centroid = as_bool(out["target_centroid_distance_pass"])
    out["centroid_proximity_pass"] = centroid
    out["gate_route"] = np.select(
        [out.exact_pooled_cluster_pass & centroid, out.exact_pooled_cluster_pass, centroid],
        ["exact_plus_centroid", "exact_only", "centroid_only"], default="neither"
    )
    out["clinical_rule_pass"] = (
        as_bool(out.get("basic_quality_pass", pd.Series(False, index=out.index)))
        & as_bool(out.get("clinical_quality_pass", pd.Series(False, index=out.index)))
        & as_bool(out.get("source_cosine_pass", pd.Series(False, index=out.index)))
        & as_bool(out.get("privacy_pass", pd.Series(False, index=out.index)))
    )
    return out


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value for discordant counts b/c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_ci(values: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    if not len(values):
        return math.nan, math.nan
    idx = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_analysis(accepted: pd.DataFrame, vanilla: pd.DataFrame, iterations: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for frame, name in [(accepted, "accepted"), (vanilla, "vanilla")]:
        if frame.anchor_id.duplicated().any():
            raise ValueError(f"{name} manifest must have one row per anchor")
    paired = accepted.merge(vanilla, on=["anchor_id", "dataset_row_id"], suffixes=("_accepted", "_vanilla"), validate="one_to_one")
    if paired.empty:
        raise ValueError("No same-anchor accepted/vanilla pairs")
    a = as_bool(paired["exact_pooled_cluster_pass_accepted"])
    v = as_bool(paired["exact_pooled_cluster_pass_vanilla"])
    risk_delta = a.astype(float).to_numpy() - v.astype(float).to_numpy()
    ci_low, ci_high = bootstrap_ci(risk_delta, iterations, rng)
    b = int((~a & v).sum())
    c = int((a & ~v).sum())
    metrics = [{
        "analysis_group": "full", "n_pairs": len(paired),
        "accepted_exact_pooled_rate": float(a.mean()), "vanilla_exact_pooled_rate": float(v.mean()),
        "paired_risk_difference": float(risk_delta.mean()), "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
        "discordant_accepted_false_vanilla_true": b, "discordant_accepted_true_vanilla_false": c,
        "exact_mcnemar_pvalue": exact_mcnemar_pvalue(b, c),
    }]
    if "patient_disjoint_from_train_accepted" in paired:
        pd_mask = as_bool(paired["patient_disjoint_from_train_accepted"])
        for label, mask in [("patient_disjoint", pd_mask), ("patient_overlap", ~pd_mask)]:
            sub = paired.loc[mask]
            if sub.empty:
                continue
            aa = as_bool(sub["exact_pooled_cluster_pass_accepted"])
            vv = as_bool(sub["exact_pooled_cluster_pass_vanilla"])
            deltas = aa.astype(float).to_numpy() - vv.astype(float).to_numpy()
            low, high = bootstrap_ci(deltas, iterations, rng)
            bb, cc = int((~aa & vv).sum()), int((aa & ~vv).sum())
            metrics.append({"analysis_group": label, "n_pairs": len(sub), "accepted_exact_pooled_rate": float(aa.mean()), "vanilla_exact_pooled_rate": float(vv.mean()), "paired_risk_difference": float(deltas.mean()), "bootstrap_ci_low": low, "bootstrap_ci_high": high, "discordant_accepted_false_vanilla_true": bb, "discordant_accepted_true_vanilla_false": cc, "exact_mcnemar_pvalue": exact_mcnemar_pvalue(bb, cc)})
    numeric_rows = []
    for metric in ["source_synthetic_cosine", "target_centroid_distance"]:
        left, right = f"{metric}_accepted", f"{metric}_vanilla"
        if left in paired and right in paired:
            delta = pd.to_numeric(paired[left], errors="coerce") - pd.to_numeric(paired[right], errors="coerce")
            numeric_rows.append({"metric": metric, "n_pairs": int(delta.notna().sum()), "accepted_mean": float(pd.to_numeric(paired[left], errors="coerce").mean()), "vanilla_mean": float(pd.to_numeric(paired[right], errors="coerce").mean()), "mean_delta_accepted_minus_vanilla": float(delta.mean())})
    return paired, pd.DataFrame(metrics), {"numeric_differences": numeric_rows}


def equal_compute(candidate: pd.DataFrame, accepted: pd.DataFrame, iterations: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = candidate.groupby("anchor_id").size()
    if counts.nunique() != 1:
        raise ValueError(f"Candidate counts differ by anchor: {counts.value_counts().to_dict()}")
    anchors = sorted(counts.index.tolist())
    groups = [candidate.loc[candidate.anchor_id == anchor].reset_index(drop=True) for anchor in anchors]
    selected = accepted.set_index("anchor_id")
    selected_rows = []
    for anchor in anchors:
        if anchor in selected.index:
            row = selected.loc[anchor]
            selected_rows.append({"anchor_id": anchor, "has_selected_note": True, "exact_pooled_cluster_pass": bool(row.exact_pooled_cluster_pass), "clinical_rule_pass": bool(row.clinical_rule_pass), "source_synthetic_cosine": float(row.source_synthetic_cosine), "target_centroid_distance": float(row.target_centroid_distance), "repetition_or_collapse_flag": bool(row.repetition_or_collapse_flag)})
        else:
            selected_rows.append({"anchor_id": anchor, "has_selected_note": False, "exact_pooled_cluster_pass": False, "clinical_rule_pass": False, "source_synthetic_cosine": math.nan, "target_centroid_distance": math.nan, "repetition_or_collapse_flag": False})
    selected_df = pd.DataFrame(selected_rows)
    selected_summary = {"condition": "selected_one_of_eight", "n_anchors": len(anchors), "n_candidates_per_anchor": int(counts.iloc[0]), "exact_pooled_rate": float(selected_df.exact_pooled_cluster_pass.mean()), "clinical_rule_pass_rate": float(selected_df.clinical_rule_pass.mean()), "usable_note_anchor_rate": float(selected_df.has_selected_note.mean()), "usable_exact_pooled_anchor_rate": float((selected_df.exact_pooled_cluster_pass & selected_df.clinical_rule_pass).mean()), "mean_source_cosine_selected_notes": float(selected_df.loc[selected_df.has_selected_note, "source_synthetic_cosine"].mean()), "mean_centroid_distance_selected_notes": float(selected_df.loc[selected_df.has_selected_note, "target_centroid_distance"].mean()), "repetition_rate_selected_notes": float(selected_df.loc[selected_df.has_selected_note, "repetition_or_collapse_flag"].mean())}
    # Vectorize random draws to avoid constructing 10,000 temporary DataFrames.
    n_candidates = int(counts.iloc[0])
    matrix = {
        "exact_pooled_rate": np.vstack([group.exact_pooled_cluster_pass.to_numpy(dtype=float) for group in groups]),
        "clinical_rule_pass_rate": np.vstack([group.clinical_rule_pass.to_numpy(dtype=float) for group in groups]),
        "mean_source_cosine": np.vstack([pd.to_numeric(group.source_synthetic_cosine, errors="coerce").to_numpy(dtype=float) for group in groups]),
        "mean_centroid_distance": np.vstack([pd.to_numeric(group.target_centroid_distance, errors="coerce").to_numpy(dtype=float) for group in groups]),
        "repetition_rate": np.vstack([as_bool(group.repetition_or_collapse_flag).to_numpy(dtype=float) for group in groups]),
    }
    draws = rng.integers(0, n_candidates, size=(iterations, len(anchors)))
    anchor_idx = np.arange(len(anchors))[None, :]
    simulation_data = {"iteration": np.arange(iterations, dtype=int)}
    for name, values in matrix.items():
        sampled = values[anchor_idx, draws]
        simulation_data[name] = np.nanmean(sampled, axis=1)
    simulation_data["usable_note_anchor_rate"] = simulation_data["clinical_rule_pass_rate"].copy()
    simulation_data["usable_exact_pooled_anchor_rate"] = np.mean(
        (matrix["exact_pooled_rate"][anchor_idx, draws] > 0)
        & (matrix["clinical_rule_pass_rate"][anchor_idx, draws] > 0),
        axis=1,
    )
    simulations = pd.DataFrame(simulation_data)
    random_summary = {"condition": "random_one_of_eight", "n_anchors": len(anchors), "n_candidates_per_anchor": int(counts.iloc[0]), **{f"mean_{col}": float(simulations[col].mean()) for col in simulations.columns if col != "iteration"}, **{f"ci95_low_{col}": float(simulations[col].quantile(.025)) for col in simulations.columns if col != "iteration"}, **{f"ci95_high_{col}": float(simulations[col].quantile(.975)) for col in simulations.columns if col != "iteration"}}
    return pd.DataFrame([selected_summary, random_summary]), simulations


def all_anchor_yield(candidate: pd.DataFrame, accepted: pd.DataFrame) -> pd.DataFrame:
    all_anchors = candidate[["anchor_id", "patient_disjoint_from_train"]].drop_duplicates("anchor_id").copy()
    accepted_routes = accepted[["anchor_id", "gate_route"]].drop_duplicates("anchor_id")
    out = all_anchors.merge(accepted_routes, on="anchor_id", how="left", validate="one_to_one")
    out["yield_category"] = out["gate_route"].fillna("no_accepted_note")
    rows = []
    for group, frame in [("full", out), ("patient_disjoint", out.loc[as_bool(out.patient_disjoint_from_train)]), ("patient_overlap", out.loc[~as_bool(out.patient_disjoint_from_train)])]:
        if frame.empty:
            continue
        counts = frame.yield_category.value_counts()
        for category in ["exact_plus_centroid", "exact_only", "centroid_only", "neither", "no_accepted_note"]:
            n = int(counts.get(category, 0))
            rows.append({"analysis_group": group, "yield_category": category, "n_anchors": n, "fraction_of_anchors": n / len(frame), "denominator_anchors": len(frame)})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    targets = parse_ids(args.target_cluster_ids); rng = np.random.default_rng(args.seed)
    accepted = attach_routes(load(args.accepted_manifest_path), targets)
    candidate = attach_routes(load(args.candidate_manifest_path), targets)
    vanilla = attach_routes(load(args.vanilla_manifest_path), targets)
    paired, paired_summary, paired_extra = paired_analysis(accepted, vanilla, args.bootstrap_iterations, rng)
    equal_summary, simulations = equal_compute(candidate, accepted, args.random_selector_iterations, rng)
    yield_df = all_anchor_yield(candidate, accepted)
    paired.to_csv(outdir / "same_anchor_paired_rows.csv", index=False)
    paired_summary.to_csv(outdir / "same_anchor_paired_summary.csv", index=False)
    equal_summary.to_csv(outdir / "equal_compute_selector_summary.csv", index=False)
    simulations.to_csv(outdir / "equal_compute_random_selector_simulations.csv", index=False)
    yield_df.to_csv(outdir / "all_anchor_end_to_end_yield.csv", index=False)
    report = {"target_cluster_ids": targets, "n_all_anchors": int(candidate.anchor_id.nunique()), "n_accepted_anchors": int(accepted.anchor_id.nunique()), "paired_analysis": paired_summary.to_dict(orient="records"), "paired_numeric_differences": paired_extra["numeric_differences"], "equal_compute": equal_summary.to_dict(orient="records"), "outputs": {"paired_rows": "same_anchor_paired_rows.csv", "paired_summary": "same_anchor_paired_summary.csv", "equal_compute_summary": "equal_compute_selector_summary.csv", "equal_compute_simulations": "equal_compute_random_selector_simulations.csv", "all_anchor_yield": "all_anchor_end_to_end_yield.csv"}}
    (outdir / "closed_loop_validation_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Closed-Loop Validation Analysis", "", f"- All anchors: `{report['n_all_anchors']}`", f"- Accepted anchors: `{report['n_accepted_anchors']}`", "- Primary endpoint: exact pooled-cluster pass (`nearest_cluster_id` in target cluster ids).", "- Equal-compute comparator: random one of the existing eight candidates per anchor."]
    (outdir / "closed_loop_validation_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved validation analysis to:", outdir)


if __name__ == "__main__":
    main()
