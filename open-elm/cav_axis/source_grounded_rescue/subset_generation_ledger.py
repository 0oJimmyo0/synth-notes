#!/usr/bin/env python3
"""Create a provenance-preserving prompt-safe ledger subset from frozen case IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subset a prompt-safe generation ledger by case manifest.")
    parser.add_argument("--generation_ledger_path", required=True)
    parser.add_argument("--case_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_manifest = pd.read_csv(Path(args.case_manifest_path).resolve())
    if "case_id" not in case_manifest.columns or case_manifest.case_id.duplicated().any():
        raise ValueError("case manifest must contain unique case_id values")
    requested_ids = set(case_manifest.case_id.astype(str))
    rows = [json.loads(line) for line in Path(args.generation_ledger_path).resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if str(row.get("case_id")) in requested_ids]
    selected_ids = {str(row["case_id"]) for row in selected}
    if selected_ids != requested_ids:
        raise ValueError(f"requested cases missing from source ledger: {sorted(requested_ids - selected_ids)}")
    if len(selected) != len(selected_ids):
        raise ValueError("source ledger has duplicate case_id rows")
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "generation_ledgers.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in sorted(selected, key=lambda item: str(item["case_id"])):
            handle.write(json.dumps(row) + "\n")
    summary = {
        "n_cases": len(selected),
        "n_generation_facts": int(sum(len(row.get("facts", [])) for row in selected)),
        "source_generation_ledger": str(Path(args.generation_ledger_path).resolve()),
        "case_manifest": str(Path(args.case_manifest_path).resolve()),
        "source_spans_in_output": False,
    }
    (output_dir / "generation_ledger_subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
