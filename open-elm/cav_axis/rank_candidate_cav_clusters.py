#!/usr/bin/env python3
"""
Rank candidate CAV target clusters with formal enrichment statistics.

This addresses the caution that raw subgroup fractions alone are not enough.
For each selected cluster and subgroup value, compare:
- in-cluster count
- out-of-cluster count
- Fisher exact odds ratio / p-value
- Benjamini-Hochberg adjusted q-value
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact


DEFAULT_SUBGROUP_METADATA_PATH = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/subgroup_metadata/subgroup_metadata_filtered.csv"
)
DEFAULT_CLUSTER_ASSIGNMENTS_PATH = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/coverage/real_all_filtered_precompute_with_subgroups/"
    "real_all_filtered_cluster_assignments.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/"
    "data_note_hadm_all/clinic_notes/1_task/cav_axis_inputs"
)
DEFAULT_CLUSTER_IDS = "11,20,25"
SUBGROUP_FIELDS = ["age_bin", "sex_gender", "race_ethnicity", "insurance", "admission_type", "service", "los_bin", "icu_flag"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank selected real-manifold clusters by subgroup enrichment.")
    parser.add_argument("--subgroup_metadata_path", default=DEFAULT_SUBGROUP_METADATA_PATH)
    parser.add_argument("--cluster_assignments_path", default=DEFAULT_CLUSTER_ASSIGNMENTS_PATH)
    parser.add_argument("--cluster_ids", default=DEFAULT_CLUSTER_IDS)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_stem", default="candidate_cluster_enrichment_11_20_25")
    parser.add_argument("--min_cluster_count", type=int, default=25)
    parser.add_argument("--min_subgroup_count", type=int, default=25)
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


def parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def detect_join_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    preferred = [
        ["source_row_id"],
        ["embedding_row_id"],
        ["dataset_row_id"],
        ["note_id", "subject_id", "hadm_id"],
        ["note_id"],
        ["subject_id", "hadm_id"],
    ]
    left_cols = set(left.columns)
    right_cols = set(right.columns)
    for keys in preferred:
        if all(key in left_cols and key in right_cols for key in keys):
            return keys
    raise ValueError("Could not find stable shared join keys.")


def bh_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    ranked = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * n
    running = 1.0
    for rank, (idx, pval) in enumerate(reversed(ranked), start=1):
        denom = n - rank + 1
        candidate = min(running, pval * n / denom)
        adjusted[idx] = candidate
        running = candidate
    return adjusted


def format_bool(value: object) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, bool):
        return str(value)
    return str(value)


def main() -> None:
    args = parse_args()
    cluster_ids = parse_csv_ints(args.cluster_ids)

    subgroup_df = pd.read_csv(args.subgroup_metadata_path)
    cluster_df = pd.read_csv(args.cluster_assignments_path)
    join_keys = detect_join_keys(subgroup_df, cluster_df)

    cluster_df = cluster_df[join_keys + ["cluster_id"]].drop_duplicates(subset=join_keys)
    merged = subgroup_df.merge(cluster_df, on=join_keys, how="inner", validate="one_to_one")

    rows: list[dict[str, object]] = []
    for cluster_id in cluster_ids:
        in_cluster = merged["cluster_id"].astype(int) == cluster_id
        cluster_n = int(in_cluster.sum())
        if cluster_n < args.min_cluster_count:
            continue

        for field in SUBGROUP_FIELDS:
            if field not in merged.columns:
                continue
            values = merged[field].map(format_bool if field == "icu_flag" else str)
            for subgroup_value, subgroup_mask in values.groupby(values):
                subgroup_n = int(subgroup_mask.shape[0])
                if subgroup_n < args.min_subgroup_count:
                    continue
                subgroup_sel = values == subgroup_value
                a = int((in_cluster & subgroup_sel).sum())
                b = int((in_cluster & ~subgroup_sel).sum())
                c = int((~in_cluster & subgroup_sel).sum())
                d = int((~in_cluster & ~subgroup_sel).sum())
                if a + c < args.min_subgroup_count:
                    continue

                odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
                cluster_fraction = a / max(cluster_n, 1)
                global_fraction = (a + c) / max(len(merged), 1)
                enrichment_ratio = cluster_fraction / global_fraction if global_fraction > 0 else math.inf

                rows.append(
                    {
                        "cluster_id": cluster_id,
                        "cluster_size": cluster_n,
                        "subgroup_name": field,
                        "subgroup_value": subgroup_value,
                        "in_cluster_with_value": a,
                        "in_cluster_without_value": b,
                        "out_cluster_with_value": c,
                        "out_cluster_without_value": d,
                        "cluster_fraction": cluster_fraction,
                        "global_fraction": global_fraction,
                        "enrichment_ratio": enrichment_ratio,
                        "odds_ratio": float(odds_ratio),
                        "p_value": float(p_value),
                    }
                )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError("No enrichment rows were produced. Check cluster ids and subgroup field coverage.")

    result_df["q_value"] = bh_adjust(result_df["p_value"].tolist())
    result_df = result_df.sort_values(
        ["cluster_id", "q_value", "enrichment_ratio", "in_cluster_with_value"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_stem}.csv"
    md_path = output_dir / f"{args.output_stem}.md"
    json_path = output_dir / f"{args.output_stem}.json"
    result_df.to_csv(csv_path, index=False)

    top_lines = []
    for cluster_id in cluster_ids:
        top = result_df.loc[result_df["cluster_id"] == cluster_id].head(8)
        top_lines.append(f"## Cluster {cluster_id}")
        if top.empty:
            top_lines.append("- No qualifying subgroup enrichments found.")
            continue
        for _, row in top.iterrows():
            top_lines.append(
                "- "
                + f"{row['subgroup_name']}={row['subgroup_value']}: "
                + f"cluster_fraction={row['cluster_fraction']:.3f}, "
                + f"global_fraction={row['global_fraction']:.3f}, "
                + f"enrichment_ratio={row['enrichment_ratio']:.2f}, "
                + f"odds_ratio={row['odds_ratio']:.2f}, "
                + f"q_value={row['q_value']:.3g}"
            )

    md_path.write_text("# Candidate Cluster Enrichment\n\n" + "\n".join(top_lines) + "\n", encoding="utf-8")
    payload = {
        "created_at": now_iso(),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "script_path": str(Path(__file__).resolve()),
        "cluster_ids": cluster_ids,
        "join_keys": join_keys,
        "n_rows": int(len(merged)),
        "n_tests": int(len(result_df)),
        "top_hits_by_cluster": {
            str(cluster_id): result_df.loc[result_df["cluster_id"] == cluster_id].head(5).to_dict(orient="records")
            for cluster_id in cluster_ids
        },
        "output_csv": str(csv_path.resolve()),
        "output_md": str(md_path.resolve()),
    }
    json_path.write_text(json.dumps(payload, indent=2))

    print("Saved enrichment table to:", csv_path)
    print("Saved enrichment report to:", md_path)
    print("Saved enrichment summary to:", json_path)


if __name__ == "__main__":
    main()
