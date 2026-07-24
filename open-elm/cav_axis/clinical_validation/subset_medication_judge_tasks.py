#!/usr/bin/env python3
"""Create a deterministic restricted task subset for judge runtime preflight."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Subset medication-judge tasks without exporting text.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--n_tasks", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing subset: {output_path}")
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.n_tasks < 1 or args.n_tasks > len(rows):
        raise ValueError(f"--n_tasks must be in [1, {len(rows)}]")
    indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.n_tasks))
    subset = []
    for index in indices:
        row = dict(rows[index])
        row.pop("human_labels", None)  # The model must never receive development labels.
        subset.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in subset:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps({
        "input_path": str(input_path), "output_path": str(output_path),
        "n_tasks": len(subset), "seed": args.seed,
        "human_labels_removed": True,
        "security_note": "Subset contains restricted compact ledger text and synthetic notes; keep on approved project storage.",
    }, indent=2))


if __name__ == "__main__":
    main()
