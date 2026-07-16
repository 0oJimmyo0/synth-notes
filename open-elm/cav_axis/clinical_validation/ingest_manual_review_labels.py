#!/usr/bin/env python3
"""Ingest completed review sheets without exporting source-note text.

This is a reporting utility, not a clinical-validity classifier. It preserves
row identifiers and reviewer labels while excluding free-text source notes,
synthetic notes, and reviewer comments from its derived outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TEXT_COLUMNS = {
    "source_real_note", "generated_text", "synthetic_note_a", "synthetic_note_b",
    "reviewer_notes", "detailed_evidence", "supporting_text",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize detailed manual-review labels securely.")
    parser.add_argument("--review_csv", action="append", required=True, help="One or more completed review CSV paths.")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def is_label_column(column: str) -> bool:
    lowered = column.lower()
    return any(token in lowered for token in (
        "flag", "failure", "severity", "pass_fail", "preserved", "unsupported",
        "omission", "mismatch", "contradiction", "score", "action", "stratum",
    ))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    label_frames, summary_rows = [], []
    for raw_path in args.review_csv:
        path = Path(raw_path).resolve()
        frame = pd.read_csv(path)
        frame.columns = [str(column).strip() for column in frame.columns]
        available = [column for column in frame.columns if column.lower() not in TEXT_COLUMNS]
        id_columns = [column for column in available if column.lower() in {"case_id", "review_slot", "blinded_note_id", "candidate_id", "anchor_id", "review_stratum"}]
        label_columns = [column for column in available if is_label_column(column)]
        retained = list(dict.fromkeys(id_columns + label_columns))
        safe_frame = frame[retained].copy()
        safe_frame.insert(0, "review_file", path.name)
        label_frames.append(safe_frame)

        for column in label_columns:
            values = frame[column].fillna("<missing>").astype(str).str.strip().replace("", "<blank>")
            for value, count in values.value_counts(dropna=False).items():
                summary_rows.append({
                    "review_file": path.name,
                    "field": column,
                    "value": value,
                    "count": int(count),
                    "fraction": float(count / len(frame)) if len(frame) else 0.0,
                })

    labels = pd.concat(label_frames, ignore_index=True, sort=False) if label_frames else pd.DataFrame()
    pd.DataFrame(summary_rows).to_csv(output_dir / "manual_review_label_value_counts.csv", index=False)
    labels.to_csv(output_dir / "manual_review_label_matrix.csv", index=False)
    summary = {
        "n_review_files": len(args.review_csv),
        "n_rows": int(len(labels)),
        "security_note": "Derived outputs exclude source text, synthetic text, and reviewer free-text comments.",
        "output_files": ["manual_review_label_value_counts.csv", "manual_review_label_matrix.csv"],
    }
    (output_dir / "manual_review_ingestion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
