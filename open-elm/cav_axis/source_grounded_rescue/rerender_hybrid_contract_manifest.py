#!/usr/bin/env python3
"""Re-render deterministic transition sections without regenerating course prose."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_manifest_path", required=True)
    parser.add_argument("--contract_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_stem", default="hybrid_contract_rerendered")
    parser.add_argument("--rendering_revision", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_hybrid_contract_generation import render_note

    contracts = {str(row["case_id"]): row for row in read_jsonl(Path(args.contract_path).resolve())}
    rows = read_jsonl(Path(args.input_manifest_path).resolve())
    if not rows:
        raise ValueError("Input manifest is empty.")
    output = []
    for row in rows:
        case_id = str(row["case_id"])
        if case_id not in contracts:
            raise KeyError(f"No contract for generated case: {case_id}")
        contract = contracts[case_id]
        new_row = dict(row)
        new_row["generated_text"] = render_note(contract, str(row["hospital_course_text"]))
        new_row["contract_sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        new_row["rendering_revision"] = args.rendering_revision
        output.append(new_row)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{args.output_stem}_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "n_outputs": len(output),
        "n_cases": len({str(row["case_id"]) for row in output}),
        "course_prose_reused": True,
        "rendering_revision": args.rendering_revision,
        "security_note": "Output contains synthetic notes and remains on approved project storage.",
    }
    (output_dir / f"{args.output_stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
