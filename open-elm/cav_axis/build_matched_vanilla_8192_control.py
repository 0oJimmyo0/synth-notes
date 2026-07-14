#!/usr/bin/env python3
"""Build and evaluate a fair 8192-token matched vanilla control.

The closed-loop run selects generated notes after BGE re-embedding.  Its fair
comparison is therefore a *single-draw*, unselected vanilla decode from the
same source anchors with the same decoder settings.  This script has two
modes:

``build``
    Extract one, provenance-preserving anchor row per accepted closed-loop
    note and write the locked generation configuration for the control run.

``analyze``
    Join the completed one-draw vanilla candidate manifest back to the frozen
    accepted notes by anchor and write paired, anchor-level comparisons.

The script intentionally does not select, re-rank, or otherwise filter the
vanilla candidate.  That is what makes it a vanilla control rather than a
second closed-loop condition.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


QUALITY_FLAGS = [
    "target_gate_pass",
    "nearest_cluster_in_target",
    "target_centroid_distance_pass",
    "source_cosine_pass",
    "basic_quality_pass",
    "clinical_quality_pass",
    "structure_pass",
    "truncation_flag",
    "clinical_sanity_flag",
    "repetition_or_collapse_flag",
    "phi_warning_flag",
    "hit_max_new_tokens",
    "ended_with_eos",
]
NUMERIC_METRICS = [
    "source_synthetic_cosine",
    "target_centroid_cosine",
    "target_centroid_distance",
    "nearest_cluster_cosine",
    "generated_word_count",
    "generated_char_count",
    "generated_token_count",
    "required_section_group_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or analyze a matched 8192-token vanilla control.")
    parser.add_argument("--mode", choices=["build", "analyze"], required=True)
    parser.add_argument("--accepted_manifest_path", required=True, help="Frozen closed-loop accepted JSONL manifest")
    parser.add_argument("--output_dir", required=True, help="Control artifact directory")
    parser.add_argument("--vanilla_candidate_manifest_path", default=None, help="Completed one-draw vanilla candidate JSONL")
    parser.add_argument("--control_seed", type=int, default=42, help="Fixed seed for the one-draw vanilla control")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--clinic_note_prompt_mode", default="default", choices=["default", "discharge_structured"])
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(script_path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(script_path.parent.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"JSONL file is empty: {path}")
    return pd.read_json(path, lines=True, dtype=False).reset_index(drop=True)


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def coerce_anchor_ids(df: pd.DataFrame, label: str) -> pd.DataFrame:
    require_columns(df, ["anchor_id", "dataset_row_id"], label)
    out = df.copy()
    out["anchor_id"] = out["anchor_id"].astype(str)
    out["dataset_row_id"] = pd.to_numeric(out["dataset_row_id"], errors="raise").astype(int)
    return out


def bool_rate(series: pd.Series) -> float:
    if series.empty:
        return math.nan
    return float(series.fillna(False).astype(bool).mean())


def build_mode(args: argparse.Namespace, accepted_df: pd.DataFrame, output_dir: Path) -> None:
    accepted_df = coerce_anchor_ids(accepted_df, "accepted manifest")
    if accepted_df["anchor_id"].duplicated().any():
        duplicated = int(accepted_df["anchor_id"].duplicated().sum())
        raise ValueError(
            f"Accepted manifest has {duplicated} duplicate anchor_id values. "
            "This control is defined for one accepted note per anchor."
        )
    if accepted_df["dataset_row_id"].duplicated().any():
        duplicated = int(accepted_df["dataset_row_id"].duplicated().sum())
        raise ValueError(f"Accepted manifest has {duplicated} duplicate dataset_row_id values.")

    provenance_cols = [
        "anchor_id", "dataset_row_id", "dataset_local_row_id", "note_id", "subject_id", "hadm_id",
        "source_cluster_id", "patient_disjoint_from_train", "hadm_disjoint_from_train",
        "note_disjoint_from_train", "source_split", "checkpoint_path", "backbone_path",
        "candidate_id", "candidate_index", "seed", "temperature", "top_p", "top_k",
        "repetition_penalty", "max_new_tokens", "clinic_note_prompt_mode",
    ]
    available_cols = [col for col in provenance_cols if col in accepted_df.columns]
    anchor_df = accepted_df[available_cols].copy()
    anchor_df = anchor_df.rename(
        columns={
            "candidate_id": "closed_loop_accepted_candidate_id",
            "candidate_index": "closed_loop_accepted_candidate_index",
            "seed": "closed_loop_accepted_seed",
            "temperature": "closed_loop_accepted_temperature",
            "top_p": "closed_loop_accepted_top_p",
            "top_k": "closed_loop_accepted_top_k",
            "repetition_penalty": "closed_loop_accepted_repetition_penalty",
            "max_new_tokens": "closed_loop_accepted_max_new_tokens",
            "clinic_note_prompt_mode": "closed_loop_accepted_prompt_mode",
        }
    )
    anchor_df["control_condition"] = "matched_vanilla_8192_single_draw"
    anchor_df["control_seed"] = int(args.control_seed)
    anchor_df["control_temperature"] = float(args.temperature)
    anchor_df["control_top_p"] = float(args.top_p)
    anchor_df["control_top_k"] = int(args.top_k)
    anchor_df["control_repetition_penalty"] = float(args.repetition_penalty)
    anchor_df["control_max_new_tokens"] = int(args.max_new_tokens)
    anchor_df["control_clinic_note_prompt_mode"] = args.clinic_note_prompt_mode
    anchor_df = anchor_df.sort_values("anchor_id").reset_index(drop=True)

    anchor_path = output_dir / "matched_vanilla_8192_anchor_manifest.csv"
    anchor_df.to_csv(anchor_path, index=False)
    config = {
        "created_at": utc_now(),
        "script_path": str(Path(__file__).resolve()),
        "git_commit": git_commit(Path(__file__).resolve()),
        "accepted_manifest_path": str(Path(args.accepted_manifest_path).resolve()),
        "anchor_manifest_path": str(anchor_path.resolve()),
        "control_definition": (
            "One unselected vanilla ELM draw per frozen accepted anchor. "
            "The input anchor and decoder settings are matched; final BGE target-gating is not used for selection."
        ),
        "n_anchors": int(len(anchor_df)),
        "n_unique_dataset_row_ids": int(anchor_df["dataset_row_id"].nunique()),
        "generation_condition": "matched_vanilla_8192_single_draw",
        "generation_settings": {
            "n_candidates_per_anchor": 1,
            "accepted_per_anchor": 1,
            "seeds": str(args.control_seed),
            "temperature_values": str(args.temperature),
            "top_p_values": str(args.top_p),
            "top_k_values": str(args.top_k),
            "repetition_penalty": float(args.repetition_penalty),
            "max_new_tokens": int(args.max_new_tokens),
            "clinic_note_prompt_mode": args.clinic_note_prompt_mode,
        },
        "required_control_manifest_contract": {
            "one_row_per_anchor": True,
            "same_anchor_ids": True,
            "same_dataset_row_ids": True,
            "max_new_tokens": int(args.max_new_tokens),
            "clinic_note_prompt_mode": args.clinic_note_prompt_mode,
        },
    }
    config_path = output_dir / "matched_vanilla_8192_control_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved {len(anchor_df)} matched vanilla anchors to: {anchor_path}")
    print(f"Saved locked control configuration to: {config_path}")


def validate_control_contract(vanilla_df: pd.DataFrame, accepted_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    vanilla_df = coerce_anchor_ids(vanilla_df, "vanilla candidate manifest")
    accepted_df = coerce_anchor_ids(accepted_df, "accepted manifest")
    if vanilla_df["anchor_id"].duplicated().any():
        raise ValueError("Vanilla control must have exactly one candidate row per anchor.")
    expected = set(accepted_df["anchor_id"])
    observed = set(vanilla_df["anchor_id"])
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"Vanilla anchor set mismatch. missing={len(missing)} extra={len(extra)}")
    if "max_new_tokens" in vanilla_df.columns:
        values = pd.to_numeric(vanilla_df["max_new_tokens"], errors="raise").unique().tolist()
        if values != [int(args.max_new_tokens)]:
            raise ValueError(f"Vanilla max_new_tokens must be {args.max_new_tokens}, got {values}")
    if "clinic_note_prompt_mode" in vanilla_df.columns:
        values = sorted(vanilla_df["clinic_note_prompt_mode"].dropna().astype(str).unique().tolist())
        if values != [args.clinic_note_prompt_mode]:
            raise ValueError(f"Vanilla prompt mode must be {args.clinic_note_prompt_mode!r}, got {values}")
    if "dataset_row_id" in vanilla_df.columns:
        expected_map = accepted_df.set_index("anchor_id")["dataset_row_id"]
        observed_map = vanilla_df.set_index("anchor_id")["dataset_row_id"]
        if not observed_map.eq(expected_map).all():
            raise ValueError("Vanilla dataset_row_id values do not match their accepted anchors.")
    return vanilla_df


def analyze_mode(args: argparse.Namespace, accepted_df: pd.DataFrame, output_dir: Path) -> None:
    if not args.vanilla_candidate_manifest_path:
        raise ValueError("--vanilla_candidate_manifest_path is required for --mode analyze")
    vanilla_df = load_jsonl(Path(args.vanilla_candidate_manifest_path))
    accepted_df = coerce_anchor_ids(accepted_df, "accepted manifest")
    vanilla_df = validate_control_contract(vanilla_df, accepted_df, args)

    accepted_keep = ["anchor_id", "dataset_row_id", "candidate_id", "generated_text"] + [
        col for col in QUALITY_FLAGS + NUMERIC_METRICS + ["nearest_cluster_id", "patient_disjoint_from_train"]
        if col in accepted_df.columns
    ]
    vanilla_keep = ["anchor_id", "dataset_row_id", "candidate_id", "generated_text"] + [
        col for col in QUALITY_FLAGS + NUMERIC_METRICS + ["nearest_cluster_id", "patient_disjoint_from_train"]
        if col in vanilla_df.columns
    ]
    accepted = accepted_df.loc[:, list(dict.fromkeys(accepted_keep))].copy()
    vanilla = vanilla_df.loc[:, list(dict.fromkeys(vanilla_keep))].copy()
    paired = accepted.merge(vanilla, on=["anchor_id", "dataset_row_id"], how="inner", validate="one_to_one", suffixes=("_accepted", "_vanilla"))
    if len(paired) != len(accepted_df):
        raise ValueError("Paired table lost accepted anchors after the one-to-one merge.")

    for metric in NUMERIC_METRICS:
        left, right = f"{metric}_accepted", f"{metric}_vanilla"
        if left in paired.columns and right in paired.columns:
            paired[f"delta_{metric}_accepted_minus_vanilla"] = (
                pd.to_numeric(paired[left], errors="coerce") - pd.to_numeric(paired[right], errors="coerce")
            )
    for flag in QUALITY_FLAGS:
        left, right = f"{flag}_accepted", f"{flag}_vanilla"
        if left in paired.columns and right in paired.columns:
            paired[f"{flag}_accepted_only"] = paired[left].fillna(False).astype(bool) & ~paired[right].fillna(False).astype(bool)
            paired[f"{flag}_vanilla_only"] = ~paired[left].fillna(False).astype(bool) & paired[right].fillna(False).astype(bool)

    paired_path = output_dir / "paired_accepted_vs_matched_vanilla.csv"
    paired.to_csv(paired_path, index=False)

    rows: list[dict[str, Any]] = []
    for metric in NUMERIC_METRICS:
        accepted_col, vanilla_col = f"{metric}_accepted", f"{metric}_vanilla"
        delta_col = f"delta_{metric}_accepted_minus_vanilla"
        if accepted_col in paired.columns and vanilla_col in paired.columns:
            rows.append({
                "metric": metric,
                "metric_type": "numeric",
                "n_pairs": int(paired[[accepted_col, vanilla_col]].dropna().shape[0]),
                "accepted_mean": float(pd.to_numeric(paired[accepted_col], errors="coerce").mean()),
                "vanilla_mean": float(pd.to_numeric(paired[vanilla_col], errors="coerce").mean()),
                "mean_delta_accepted_minus_vanilla": float(paired[delta_col].mean()),
                "accepted_rate": math.nan,
                "vanilla_rate": math.nan,
                "rate_delta_accepted_minus_vanilla": math.nan,
            })
    for flag in QUALITY_FLAGS:
        accepted_col, vanilla_col = f"{flag}_accepted", f"{flag}_vanilla"
        if accepted_col in paired.columns and vanilla_col in paired.columns:
            accepted_rate = bool_rate(paired[accepted_col])
            vanilla_rate = bool_rate(paired[vanilla_col])
            rows.append({
                "metric": flag,
                "metric_type": "flag_rate",
                "n_pairs": int(len(paired)),
                "accepted_mean": math.nan,
                "vanilla_mean": math.nan,
                "mean_delta_accepted_minus_vanilla": math.nan,
                "accepted_rate": accepted_rate,
                "vanilla_rate": vanilla_rate,
                "rate_delta_accepted_minus_vanilla": accepted_rate - vanilla_rate,
            })
    summary_df = pd.DataFrame(rows)
    summary_path = output_dir / "paired_accepted_vs_matched_vanilla_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    transitions = []
    for condition in ["accepted", "vanilla"]:
        source_col = "nearest_cluster_id_accepted" if condition == "accepted" else "nearest_cluster_id_vanilla"
        if source_col not in paired.columns:
            continue
        grouped = paired.groupby(source_col, dropna=False).size().reset_index(name="n_notes")
        grouped = grouped.rename(columns={source_col: "output_cluster_id"})
        grouped["condition"] = condition
        grouped["fraction"] = grouped["n_notes"] / len(paired)
        transitions.append(grouped)
    landing_df = pd.concat(transitions, ignore_index=True) if transitions else pd.DataFrame()
    landing_path = output_dir / "matched_vanilla_vs_accepted_output_cluster_counts.csv"
    landing_df.to_csv(landing_path, index=False)

    result = {
        "created_at": utc_now(),
        "script_path": str(Path(__file__).resolve()),
        "git_commit": git_commit(Path(__file__).resolve()),
        "accepted_manifest_path": str(Path(args.accepted_manifest_path).resolve()),
        "vanilla_candidate_manifest_path": str(Path(args.vanilla_candidate_manifest_path).resolve()),
        "n_paired_anchors": int(len(paired)),
        "control_definition": "one unselected 8192-token vanilla draw per accepted closed-loop anchor",
        "fairness_contract": {
            "same_anchor_ids": True,
            "same_dataset_row_ids": True,
            "same_max_new_tokens": int(args.max_new_tokens),
            "same_prompt_mode": args.clinic_note_prompt_mode,
            "one_control_candidate_per_anchor": True,
        },
        "output_files": {
            "paired_table": str(paired_path),
            "metric_summary": str(summary_path),
            "output_cluster_counts": str(landing_path),
        },
        "interpretation_note": (
            "This compares selected closed-loop outputs with one unselected vanilla draw from the same anchors. "
            "It tests the benefit and cost of output-space selection, not a matched sampling-budget comparison."
        ),
    }
    json_path = output_dir / "paired_accepted_vs_matched_vanilla_summary.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_path = output_dir / "paired_accepted_vs_matched_vanilla_summary.md"
    md_path.write_text(
        "# Matched Vanilla Comparison\n\n"
        f"- Paired anchors: `{len(paired)}`\n"
        "- Control: one unselected vanilla draw per accepted anchor using the locked 8192-token default-prompt configuration.\n"
        "- Interpretation: this estimates the value/cost of final-output selection; it does not equate the candidate budget.\n"
        f"- Detailed paired rows: `{paired_path.name}`\n"
        f"- Metric summary: `{summary_path.name}`\n",
        encoding="utf-8",
    )
    print(f"Saved paired table to: {paired_path}")
    print(f"Saved paired metrics to: {summary_path}")
    print(f"Saved comparison summary to: {json_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_df = load_jsonl(Path(args.accepted_manifest_path))
    if args.mode == "build":
        build_mode(args, accepted_df, output_dir)
    else:
        analyze_mode(args, accepted_df, output_dir)


if __name__ == "__main__":
    main()
