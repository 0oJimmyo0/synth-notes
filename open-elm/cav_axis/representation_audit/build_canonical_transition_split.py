#!/usr/bin/env python3
"""Stream raw source notes into a frozen canonical transition-note representation.

Only source-derived canonical text is written. Inputs and outputs must remain on
approved project storage. Missing sections are reported, never inferred.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


HEADING = re.compile(r"^\s*([A-Za-z][A-Za-z /&-]{2,80}):\s*(.*)$")
MIMIC_MM_PATH = Path(__file__).resolve().parents[4] / "MIMIC-MM-Dataset-main"


def enable_pickle_classes() -> None:
    """Load the local class definitions required by the legacy pickle objects."""
    if not (MIMIC_MM_PATH / "minimal_API.py").exists():
        raise FileNotFoundError(f"minimal_API.py is required for source pickles: {MIMIC_MM_PATH}")
    if str(MIMIC_MM_PATH) not in sys.path:
        sys.path.insert(0, str(MIMIC_MM_PATH))
    import minimal_API  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_manifest_path", required=True)
    parser.add_argument("--source_split", required=True, choices=("train", "dev", "test"))
    parser.add_argument("--pickle_dir", required=True)
    parser.add_argument("--spec_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--row_id_manifest_path", default=None, help="Optional CSV/JSONL dataset_row_id subset for regression checks.")
    parser.add_argument("--max_rows", type=int, default=0, help="Deterministic row cap; 0 keeps the full split.")
    parser.add_argument("--progress_every", type=int, default=500)
    return parser.parse_args()


def clean_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().rstrip(":"))


def extract_ordered_sections(text: str) -> list[tuple[str, str]]:
    lines = str(text).replace("\r", "").splitlines(keepends=True)
    headings: list[tuple[str, int, str]] = []
    for index, line in enumerate(lines):
        match = HEADING.match(line.strip())
        if match:
            headings.append((clean_heading(match.group(1)), index, match.group(2).strip()))
    sections: list[tuple[str, str]] = []
    for position, (heading, line_index, inline) in enumerate(headings):
        next_line = headings[position + 1][1] if position + 1 < len(headings) else len(lines)
        body = "".join(lines[line_index + 1 : next_line]).strip()
        value = (inline + "\n" + body).strip() if inline else body
        if value:
            sections.append((heading, value))
    return sections


def first_match(
    sections: list[tuple[str, str]],
    aliases: list[str],
    excluded_headings: set[str] | None = None,
) -> tuple[str, str, int] | None:
    excluded_headings = excluded_headings or set()
    matches = []
    for alias in aliases:
        alias = clean_heading(alias)
        for heading, value in sections:
            if heading not in excluded_headings and (heading == alias or heading.startswith(alias)):
                matches.append((heading, value))
        if matches:
            return matches[0][0], matches[0][1], len(matches)
    return None


def source_rows_for_file(rows: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows.itertuples(index=False):
        payload = row._asdict()
        grouped.setdefault(str(payload["filename"]), []).append(payload)
    return grouped


def main() -> None:
    args = parse_args()
    enable_pickle_classes()
    spec_path = Path(args.spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    required = list(spec["required_sections"])
    optional = list(spec.get("optional_sections", []))
    aliases = {field: list(values) for field, values in spec["heading_aliases"].items()}
    display = {
        "principal_diagnosis": "Discharge Diagnosis",
        "hospital_course_events": "Brief Hospital Course",
        "discharge_medications": "Discharge Medications",
        "disposition": "Disposition",
        "instructions": "Discharge Instructions",
        "follow_up": "Follow-up",
    }
    if set(required + optional).difference(aliases) or set(required + optional).difference(display):
        raise KeyError("spec is missing aliases or display labels for canonical sections")
    spec_sha256 = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    split = pd.read_csv(Path(args.split_manifest_path).resolve())
    needed = {"dataset_row_id", "note_id", "filename", "split"}
    if missing := needed.difference(split.columns):
        raise KeyError(f"split manifest missing columns: {sorted(missing)}")
    split = split.loc[split["split"].astype(str).eq(args.source_split)].copy()
    split["dataset_row_id"] = pd.to_numeric(split["dataset_row_id"], errors="raise").astype(int)
    split = split.drop_duplicates("dataset_row_id").sort_values("dataset_row_id")
    if args.row_id_manifest_path:
        subset_path = Path(args.row_id_manifest_path).resolve()
        subset = pd.read_json(subset_path, lines=True) if subset_path.suffix == ".jsonl" else pd.read_csv(subset_path)
        if "dataset_row_id" not in subset.columns:
            raise KeyError("row-id manifest must include dataset_row_id")
        requested_ids = set(pd.to_numeric(subset["dataset_row_id"], errors="raise").astype(int))
        split = split.loc[split["dataset_row_id"].isin(requested_ids)].copy()
    if args.max_rows:
        split = split.head(args.max_rows).copy()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "canonical_transition_manifest.jsonl"
    audit_path = output_dir / "canonical_transition_extraction_audit.csv"
    counter = Counter()
    missing_counts = Counter()
    field_found = Counter()
    duplicate_counts = Counter()
    source_by_file = source_rows_for_file(split)
    audit_columns = [
        "dataset_row_id", "note_id", "source_split", "canonical_ready", "missing_required_fields",
        *[f"{field}_heading_found" for field in required + optional],
        *[f"{field}_duplicate_match_count" for field in required + optional],
    ]
    with manifest_path.open("w", encoding="utf-8") as manifest, audit_path.open("w", newline="", encoding="utf-8") as audit_file:
        audit_writer = csv.DictWriter(audit_file, fieldnames=audit_columns)
        audit_writer.writeheader()
        for file_index, (filename, rows) in enumerate(source_by_file.items(), start=1):
            if args.progress_every and (file_index == 1 or file_index % args.progress_every == 0 or file_index == len(source_by_file)):
                print(f"Processing source pickle {file_index}/{len(source_by_file)}", flush=True)
            path = Path(args.pickle_dir).resolve() / filename
            counter["pickle_files_seen"] += 1
            if not path.exists():
                counter["pickle_files_missing"] += 1
                continue
            try:
                with path.open("rb") as handle:
                    patient = pickle.load(handle)
                notes = getattr(patient, "dsnotes", None)
            except Exception:
                counter["pickle_files_unreadable"] += 1
                continue
            if notes is None or getattr(notes, "empty", True):
                counter["pickle_files_without_notes"] += 1
                continue
            by_note_id = {str(item.get("note_id", "")): str(item.get("text", "")).strip() for _, item in notes.iterrows()}
            for row in rows:
                counter["source_rows_seen"] += 1
                note_id = str(row["note_id"])
                text = by_note_id.get(note_id, "")
                if not text:
                    counter["source_rows_without_text"] += 1
                    continue
                sections = extract_ordered_sections(text)
                values, headings, duplicates, missing = {}, {}, {}, []
                used_headings: set[str] = set()
                for field in required + optional:
                    matched = first_match(sections, aliases[field], used_headings)
                    if matched is None:
                        if field in required:
                            missing.append(field)
                        duplicates[field] = 0
                        continue
                    heading, value, match_count = matched
                    values[field] = value
                    headings[field] = heading
                    used_headings.add(heading)
                    duplicates[field] = match_count
                    field_found[field] += 1
                    duplicate_counts[field] += max(0, match_count - 1)
                ready = not missing
                audit_row = {
                    "dataset_row_id": int(row["dataset_row_id"]),
                    "note_id": note_id,
                    "source_split": args.source_split,
                    "canonical_ready": ready,
                    "missing_required_fields": "|".join(missing),
                }
                for field in required + optional:
                    audit_row[f"{field}_heading_found"] = field in values
                    audit_row[f"{field}_duplicate_match_count"] = duplicates.get(field, 0)
                audit_writer.writerow(audit_row)
                if not ready:
                    counter["source_rows_ineligible"] += 1
                    for field in missing:
                        missing_counts[field] += 1
                    continue
                ordered = [field for field in spec["display_order"] if field in values]
                canonical_text = "\n\n".join(f"{display[field]}:\n{values[field]}" for field in ordered)
                record = {
                    "case_id": f"{args.source_split}_{int(row['dataset_row_id'])}",
                    "dataset_row_id": int(row["dataset_row_id"]),
                    "note_id": note_id,
                    "source_split": args.source_split,
                    "generated_text": canonical_text,
                    "canonical_ready": True,
                    "source_heading_by_field": headings,
                    "duplicate_match_count_by_field": duplicates,
                    "representation_id": spec["representation_id"],
                    "representation_spec_sha256": spec_sha256,
                }
                manifest.write(json.dumps(record) + "\n")
                counter["source_rows_ready"] += 1
    summary = {
        "representation_id": spec["representation_id"],
        "representation_spec_sha256": spec_sha256,
        "source_split": args.source_split,
        "row_id_manifest_path": str(Path(args.row_id_manifest_path).resolve()) if args.row_id_manifest_path else None,
        "n_requested_rows": int(len(split)),
        "n_source_rows_seen": int(counter["source_rows_seen"]),
        "n_ready": int(counter["source_rows_ready"]),
        "n_ineligible": int(counter["source_rows_ineligible"]),
        "n_without_text": int(counter["source_rows_without_text"]),
        "field_found_counts": dict(field_found),
        "missing_required_field_counts": dict(missing_counts),
        "duplicate_heading_excess_counts": dict(duplicate_counts),
        "security_note": "Canonical manifest contains source-derived text and must remain on approved project storage.",
    }
    (output_dir / "canonical_transition_extraction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
