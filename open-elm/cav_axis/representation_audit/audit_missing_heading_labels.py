#!/usr/bin/env python3
"""Audit heading labels for source fields missing under a frozen canonical spec.

This reads source pickles but writes only normalized heading labels and counts,
never source-derived clinical text.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from build_canonical_transition_split import enable_pickle_classes, extract_ordered_sections


KEYWORDS = {
    "principal_diagnosis": ("diagnos", "problem"),
    "hospital_course_events": ("hospital course", "course", "hospitalization"),
    "discharge_medications": ("medication", "meds", "prescription"),
    "disposition": ("disposition", "discharge location", "discharge destination"),
    "instructions": ("instruction", "discharge plan", "patient education"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_manifest_path", required=True)
    parser.add_argument("--source_split", required=True, choices=("train", "dev", "test"))
    parser.add_argument("--pickle_dir", required=True)
    parser.add_argument("--extraction_audit_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--progress_every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_pickle_classes()
    audit = pd.read_csv(Path(args.extraction_audit_csv).resolve()).fillna("")
    ready = audit["canonical_ready"].astype(str).str.strip().str.lower().eq("true")
    audit = audit.loc[~ready].copy()
    if audit.empty:
        raise ValueError("The extraction audit has no ineligible rows.")
    audit["dataset_row_id"] = pd.to_numeric(audit["dataset_row_id"], errors="raise").astype(int)
    missing_by_id = {
        int(row.dataset_row_id): set(filter(None, str(row.missing_required_fields).split("|")))
        for row in audit.itertuples(index=False)
    }
    split = pd.read_csv(Path(args.split_manifest_path).resolve())
    split["dataset_row_id"] = pd.to_numeric(split["dataset_row_id"], errors="raise").astype(int)
    split = split.loc[
        split["split"].astype(str).eq(args.source_split) & split["dataset_row_id"].isin(missing_by_id)
    ].drop_duplicates("dataset_row_id")
    rows_by_file: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in split.itertuples(index=False):
        rows_by_file[str(row.filename)].append(row._asdict())
    heading_counts = {field: Counter() for field in KEYWORDS}
    no_near_match = Counter()
    processed = 0
    for index, (filename, rows) in enumerate(rows_by_file.items(), start=1):
        if args.progress_every and (index == 1 or index % args.progress_every == 0 or index == len(rows_by_file)):
            print(f"Auditing source pickle {index}/{len(rows_by_file)}", flush=True)
        try:
            import pickle
            with (Path(args.pickle_dir).resolve() / filename).open("rb") as handle:
                patient = pickle.load(handle)
            notes = getattr(patient, "dsnotes", None)
        except Exception:
            continue
        if notes is None or getattr(notes, "empty", True):
            continue
        note_text = {str(item.get("note_id", "")): str(item.get("text", "")) for _, item in notes.iterrows()}
        for row in rows:
            fields = missing_by_id[int(row["dataset_row_id"])]
            text = note_text.get(str(row["note_id"]), "")
            if not text:
                continue
            headings = [heading for heading, _ in extract_ordered_sections(text)]
            processed += 1
            for field in fields:
                near = [heading for heading in headings if any(token in heading for token in KEYWORDS[field])]
                if near:
                    heading_counts[field].update(near)
                else:
                    no_near_match[field] += 1
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for field, counts in heading_counts.items():
        for heading, count in counts.most_common(args.top_n):
            rows.append({"missing_field": field, "heading_label": heading, "count": count})
    pd.DataFrame(rows).to_csv(output_dir / "missing_heading_label_inventory.csv", index=False)
    summary = {
        "source_split": args.source_split,
        "n_ineligible_rows_requested": len(audit),
        "n_ineligible_rows_with_source_loaded": processed,
        "no_near_match_count_by_field": dict(no_near_match),
        "security_note": "Outputs contain normalized heading labels and counts only; no source text is exported.",
    }
    (output_dir / "missing_heading_label_inventory_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
