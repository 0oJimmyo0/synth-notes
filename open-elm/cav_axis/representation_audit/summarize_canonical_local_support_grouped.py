#!/usr/bin/env python3
"""Summarize repeated grouped local-support audits using prespecified gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--seeds", default="20260811,20260812,20260813,20260814,20260815")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rank_spearman_median_min", type=float, default=0.90)
    parser.add_argument("--rank_spearman_minimum", type=float, default=0.85)
    parser.add_argument("--sparse_jaccard_median_min", type=float, default=0.60)
    parser.add_argument("--anchor_sparse_frequency_min", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    root = Path(args.run_root).resolve()
    runs, diagnostics = [], []
    for seed in seeds:
        run_dir = root / f"seed_{seed}"
        summary = json.loads((run_dir / "canonical_local_support_stability_summary.json").read_text())
        if summary["split_seed"] != seed:
            raise ValueError(f"Seed mismatch in {run_dir}.")
        records = pd.read_json(run_dir / "canonical_dev_local_support.jsonl", lines=True)
        records["split_seed"] = seed
        runs.append(records)
        diagnostics.extend([{**row, "split_seed": seed} for row in summary["diagnostics"]])
    diagnostic_frame = pd.DataFrame(diagnostics)
    all_dev = diagnostic_frame.loc[diagnostic_frame.population.eq("all_dev")].copy()
    aggregate = all_dev.groupby("k", as_index=False).agg(
        n_seeds=("split_seed", "nunique"),
        median_rank_spearman=("rank_spearman_a_vs_b", "median"),
        minimum_rank_spearman=("rank_spearman_a_vs_b", "min"),
        median_sparse_decile_jaccard=("sparse_decile_jaccard_a_vs_b", "median"),
    )
    aggregate["within_seed_stability_pass"] = (
        aggregate.median_rank_spearman.ge(args.rank_spearman_median_min)
        & aggregate.minimum_rank_spearman.ge(args.rank_spearman_minimum)
        & aggregate.median_sparse_decile_jaccard.ge(args.sparse_jaccard_median_min)
    )
    combined = pd.concat(runs, ignore_index=True)
    identifier_columns = [column for column in ("source_index", "dataset_row_id", "note_id", "case_id", "subject_id", "patient_disjoint_from_train") if column in combined]
    frequency_rows = []
    for k in aggregate.k:
        value_column = f"mean_top_{k}_support"
        for seed, group in combined.groupby("split_seed"):
            n_sparse = max(1, int(np.ceil(len(group) * 0.10)))
            sparse_indices = set(group.nsmallest(n_sparse, value_column).source_index)
            frequency_rows.extend({"split_seed": seed, "source_index": index, "k": k, "sparse_in_seed": True} for index in sparse_indices)
    sparse = pd.DataFrame(frequency_rows)
    frequency = sparse.groupby(["source_index", "k"], as_index=False).agg(
        sparse_seed_count=("split_seed", "nunique")
    )
    frequency["sparse_frequency"] = frequency.sparse_seed_count / len(seeds)
    base = combined.sort_values("split_seed").drop_duplicates("source_index")[identifier_columns]
    frequency = frequency.merge(base, on="source_index", how="left", validate="many_to_one")
    frequency["anchor_sparse_frequency_pass"] = frequency.sparse_frequency.ge(args.anchor_sparse_frequency_min)
    adjacent_pass = {}
    stable_ks = set(aggregate.loc[aggregate.within_seed_stability_pass, "k"])
    ordered_ks = aggregate.k.tolist()
    for position, k in enumerate(ordered_ks):
        neighbors = set(ordered_ks[max(0, position - 1):position] + ordered_ks[position + 1:position + 2])
        adjacent_pass[k] = k in stable_ks and bool(neighbors & stable_ks)
    aggregate["adjacent_k_stability_pass"] = aggregate.k.map(adjacent_pass).fillna(False)
    aggregate["overall_support_gate_pass"] = aggregate.within_seed_stability_pass & aggregate.adjacent_k_stability_pass
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_frame.to_csv(output_dir / "per_seed_local_support_diagnostics.csv", index=False)
    aggregate.to_csv(output_dir / "local_support_stability_by_k.csv", index=False)
    frequency.to_csv(output_dir / "dev_sparse_anchor_frequency.csv", index=False)
    summary = {
        "scope": "real_train_reference_and_real_dev_only",
        "seeds": seeds,
        "decision_thresholds": {
            "median_rank_spearman_min": args.rank_spearman_median_min,
            "minimum_rank_spearman_min": args.rank_spearman_minimum,
            "median_sparse_decile_jaccard_min": args.sparse_jaccard_median_min,
            "anchor_sparse_frequency_min": args.anchor_sparse_frequency_min,
        },
        "per_k": aggregate.to_dict(orient="records"),
        "any_k_passes_repeated_support_gate": bool(aggregate.overall_support_gate_pass.any()),
        "security_note": "Outputs contain provenance IDs and derived support summaries only; no source-note text.",
    }
    (output_dir / "canonical_local_support_decision_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
