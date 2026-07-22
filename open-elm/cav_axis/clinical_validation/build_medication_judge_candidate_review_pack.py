#!/usr/bin/env python3
"""Create a blinded, restricted candidate-level medication-review pack.

This supports Phase 3a calibration. It samples existing candidates only and
does not alter generation, selection, or the locked held-out cluster-11 set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a blinded medication-judge calibration review pack.")
    parser.add_argument("--candidate_manifest_path", required=True)
    parser.add_argument("--ledger_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--exclude_manifest_path", help="Existing selected candidate manifest to exclude.")
    return parser.parse_args()


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def usable_ids(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame:
        return set()
    values = frame[column].dropna().astype(str).str.strip()
    return set(values[~values.str.lower().isin({"", "nan", "none"})])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_jsonl(Path(args.candidate_manifest_path).resolve()).fillna("")
    ledgers = read_jsonl(Path(args.ledger_path).resolve()).fillna("")
    required_candidates = {"generated_text", "case_id", "generation_ledger_sha256"}
    missing = required_candidates - set(candidates.columns)
    if missing:
        raise KeyError(f"Candidate manifest missing: {sorted(missing)}")
    if "generation_ledger_sha256" not in ledgers or "facts" not in ledgers:
        raise KeyError("Ledger JSONL requires generation_ledger_sha256 and facts.")
    if args.exclude_manifest_path:
        excluded_frame = read_jsonl(Path(args.exclude_manifest_path).resolve())
        before = len(candidates)
        keep = pd.Series(True, index=candidates.index)
        for column in ("raw_elm_candidate_id", "rescue_id"):
            excluded = usable_ids(excluded_frame, column)
            if column in candidates and excluded:
                keep &= ~candidates[column].fillna("").astype(str).isin(excluded)
        candidates = candidates[keep].copy()
        excluded_count = before - len(candidates)
    else:
        excluded_count = 0
    ledger_lookup = ledgers.drop_duplicates("generation_ledger_sha256", keep="last").set_index("generation_ledger_sha256")["facts"]
    candidates["verified_fact_ledger"] = candidates.generation_ledger_sha256.map(
        lambda key: json.dumps(ledger_lookup.get(key, []), ensure_ascii=True)
    )
    candidates = candidates[candidates.verified_fact_ledger != "[]"].copy()
    if len(candidates) < args.n_samples:
        raise ValueError(f"Only {len(candidates)} candidates available after exclusions; requested {args.n_samples}.")
    # Balance calibration cases across structural eligibility, geometry, and decode draw.
    for column in ("eligible_for_selection", "output_in_target_basin", "candidate_index"):
        if column not in candidates:
            candidates[column] = "unknown"
    candidates["sampling_stratum"] = candidates.apply(
        lambda row: f"eligible={row['eligible_for_selection']}|target={row['output_in_target_basin']}|draw={row['candidate_index']}", axis=1
    )
    shuffled = candidates.sample(frac=1.0, random_state=args.seed)
    pools = [group for _, group in shuffled.groupby("sampling_stratum", sort=True)]
    chosen = []
    cursor = 0
    while len(chosen) < args.n_samples and pools:
        pool = pools[cursor % len(pools)]
        if len(pool):
            chosen.append(pool.iloc[0])
            pools[cursor % len(pools)] = pool.iloc[1:]
        pools = [item for item in pools if len(item)]
        cursor += 1
    selected = pd.DataFrame(chosen).reset_index(drop=True)
    selected["blinded_output_id"] = [f"med_judge_dev_blind_{index:03d}" for index in range(1, len(selected) + 1)]
    review = pd.DataFrame({
        "blinded_output_id": selected.blinded_output_id,
        "case_id": selected.case_id,
        "patient_disjoint_from_train": selected.get("patient_disjoint_from_train", ""),
        "verified_fact_ledger": selected.verified_fact_ledger,
        "synthetic_note": selected.generated_text,
        "human_medication_error_yes_no": "",
        "human_error_types_pipe_delimited": "",
        "human_severe_medication_error_yes_no": "",
        "reviewer_notes": "",
    })
    key_columns = [column for column in (
        "blinded_output_id", "case_id", "raw_elm_candidate_id", "rescue_id", "anchor_id",
        "candidate_index", "seed", "eligible_for_selection", "output_in_target_basin", "sampling_stratum",
    ) if column in selected]
    review.to_csv(output_dir / "medication_judge_development_blinded_review.csv", index=False)
    selected[key_columns].to_csv(output_dir / "medication_judge_development_blinded_key.csv", index=False)
    summary = {
        "n_candidates_available": int(len(candidates)), "n_selected": int(len(selected)),
        "excluded_existing_selected_candidates": int(excluded_count), "seed": args.seed,
        "security_note": "Review CSV includes compact verified facts and synthetic notes. Keep it on approved project storage; do not use the key until blinded labels are final.",
    }
    (output_dir / "medication_judge_development_review_pack_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
