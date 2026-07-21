#!/usr/bin/env python3
"""Freeze a deterministic, leakage-aware subset of a prompt-safe ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation_ledger_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_cases", type=int, default=8)
    parser.add_argument("--min_patient_disjoint", type=int, default=0,
                        help="Minimum patient-disjoint cases when the eligible ledger reserve permits it.")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--exclude_case_manifest_path", action="append", default=[], help="Prior pilot case manifest to exclude; repeat as needed.")
    return parser.parse_args()


def rank(case_id: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest(), 16)


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.generation_ledger_path).resolve().read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    required = {"case_id", "anchor_id", "dataset_row_id", "patient_disjoint_from_train"}
    if missing := required - set(frame.columns):
        raise KeyError(f"Generation ledger missing columns: {sorted(missing)}")
    if frame.case_id.duplicated().any() or len(frame) < args.n_cases:
        raise ValueError("Generation ledger lacks enough unique cases for the requested pilot.")
    if args.exclude_case_manifest_path:
        excluded_ids: set[str] = set()
        for path_string in args.exclude_case_manifest_path:
            excluded = pd.read_csv(Path(path_string).resolve())
            if "case_id" not in excluded.columns:
                raise KeyError("Excluded case manifest must contain case_id.")
            excluded_ids.update(excluded.case_id.astype(str))
        frame = frame.loc[~frame.case_id.astype(str).isin(excluded_ids)].copy()
        if len(frame) < args.n_cases:
            raise ValueError("Not enough untested cases remain for the requested pilot.")
    frame["patient_disjoint_from_train"] = frame.patient_disjoint_from_train.fillna(False).astype(bool)
    frame["stable_rank"] = frame.case_id.map(lambda value: rank(str(value), args.seed))
    pd_rows = frame.loc[frame.patient_disjoint_from_train].sort_values("stable_rank")
    overlap_rows = frame.loc[~frame.patient_disjoint_from_train].sort_values("stable_rank")
    proportional_pd_target = max(1, round(args.n_cases * len(pd_rows) / len(frame)))
    requested_pd_target = max(proportional_pd_target, int(args.min_patient_disjoint))
    n_pd = min(len(pd_rows), args.n_cases, requested_pd_target)
    selected = pd.concat([pd_rows.head(n_pd), overlap_rows.head(args.n_cases - n_pd)], ignore_index=True)
    if len(selected) != args.n_cases:
        selected = frame.sort_values("stable_rank").head(args.n_cases).copy()
    selected = selected[["case_id", "anchor_id", "dataset_row_id", "note_id", "patient_disjoint_from_train"]].sort_values("case_id")
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out / "pilot_case_manifest.csv", index=False)
    summary = {
        "n_cases": int(len(selected)), "patient_disjoint_count": int(selected.patient_disjoint_from_train.sum()),
        "min_patient_disjoint_requested": int(args.min_patient_disjoint),
        "seed": int(args.seed), "source_generation_ledger": str(Path(args.generation_ledger_path).resolve()),
        "excluded_case_manifests": [str(Path(path).resolve()) for path in args.exclude_case_manifest_path],
    }
    (out / "pilot_case_manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
