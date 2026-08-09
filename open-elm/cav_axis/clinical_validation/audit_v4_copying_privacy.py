#!/usr/bin/env python3
"""Audit V4 source copying and patient-excluded train-text overlap.

All outputs are derived statistics only: source notes, train notes, and rendered
text remain on approved project storage and are never written by this script.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)*")
PHI_PATTERNS = [
    re.compile(r"\[\*\*.*?\*\*\]"),
    re.compile(r"\b\d{6,}\b"),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
]
HEADINGS = {
    "discharge diagnosis": "principal_diagnosis",
    "brief hospital course": "hospital_course_events",
    "discharge medications": "discharge_medications",
    "disposition": "disposition",
    "discharge disposition": "disposition",
    "follow up": "follow_up",
    "discharge instructions": "instructions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected_manifest_path", required=True)
    parser.add_argument("--source_reference_csv", required=True)
    parser.add_argument("--split_manifest_path", required=True)
    parser.add_argument("--pickle_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_span_tokens", type=int, default=5)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--progress_every_files", type=int, default=1000)
    return parser.parse_args()


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def ngrams(values: list[str], size: int) -> dict[tuple[str, ...], list[int]]:
    return {
        tuple(values[index : index + size]): [index]
        for index in range(max(0, len(values) - size + 1))
    }


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = defaultdict(list)
    current = "unlabeled"
    for line in str(text).splitlines():
        normalized = line.strip().rstrip(":").lower()
        if normalized in HEADINGS:
            current = HEADINGS[normalized]
            continue
        sections[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items() if "\n".join(lines).strip()}


def longest_span(query: list[str], reference: list[str], minimum: int) -> int:
    if len(query) < minimum or len(reference) < minimum:
        return 0
    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for pos in range(len(reference) - minimum + 1):
        index[tuple(reference[pos : pos + minimum])].append(pos)
    longest = 0
    for pos in range(len(query) - minimum + 1):
        for ref_pos in index.get(tuple(query[pos : pos + minimum]), []):
            length = minimum
            while pos + length < len(query) and ref_pos + length < len(reference) and query[pos + length] == reference[ref_pos + length]:
                length += 1
            longest = max(longest, length)
    return longest


def copy_metrics(query: list[str], reference: list[str], minimum: int) -> dict[str, float | int]:
    query_5 = set(ngrams(query, minimum))
    ref_5 = set(ngrams(reference, minimum))
    query_10 = set(ngrams(query, 10))
    ref_10 = set(ngrams(reference, 10))
    covered = set()
    for start in range(max(0, len(query) - minimum + 1)):
        if tuple(query[start : start + minimum]) in ref_5:
            covered.update(range(start, start + minimum))
    return {
        "source_token_copy_fraction": float(len(covered) / len(query)) if query else 0.0,
        "source_longest_exact_span": int(longest_span(query, reference, minimum)),
        "source_5gram_overlap_fraction": float(len(query_5 & ref_5) / len(query_5)) if query_5 else 0.0,
        "source_10gram_overlap_fraction": float(len(query_10 & ref_10) / len(query_10)) if query_10 else 0.0,
    }


def load_train_texts(rows: pd.DataFrame, pickle_dir: Path, progress_every: int):
    needed: dict[str, dict[str, tuple[int, str]]] = defaultdict(dict)
    for row in rows.itertuples(index=False):
        needed[str(row.filename)][str(row.note_id)] = (int(row.dataset_row_id), str(row.subject_id))
    total = len(needed)
    for file_no, (filename, note_map) in enumerate(needed.items(), start=1):
        if file_no == 1 or file_no % progress_every == 0 or file_no == total:
            print(f"Scanning train pickle {file_no}/{total}", flush=True)
        try:
            with (pickle_dir / filename).open("rb") as handle:
                patient = pickle.load(handle)
            notes = patient.dsnotes
        except Exception:
            continue
        for note in notes.itertuples(index=False):
            note_id = str(getattr(note, "note_id", ""))
            if note_id in note_map:
                row_id, subject_id = note_map[note_id]
                yield row_id, subject_id, str(getattr(note, "text", ""))


def main() -> None:
    args = parse_args()
    minimum = int(args.min_span_tokens)
    if minimum < 2:
        raise ValueError("--min_span_tokens must be at least 2")
    selected = [json.loads(line) for line in Path(args.selected_manifest_path).read_text().splitlines() if line.strip()]
    source = pd.read_csv(Path(args.source_reference_csv), dtype=str).fillna("")
    split = pd.read_csv(Path(args.split_manifest_path), dtype=str).fillna("")
    if {"case_id", "source_real_note"}.difference(source.columns):
        raise KeyError("source reference needs case_id and source_real_note")
    required_split = {"split", "dataset_row_id", "subject_id", "filename", "note_id"}
    if required_split.difference(split.columns):
        raise KeyError(f"split manifest missing columns: {sorted(required_split.difference(split.columns))}")
    source_by_case = source.drop_duplicates("case_id").set_index("case_id").source_real_note.to_dict()
    selected_ids = {str(row["case_id"]) for row in selected}
    if len(selected_ids) != len(selected):
        raise ValueError("selected manifest has duplicate case IDs")
    missing_source = selected_ids.difference(source_by_case)
    if missing_source:
        raise ValueError(f"source reference missing selected cases: {sorted(missing_source)}")
    subject_by_row = split.drop_duplicates("dataset_row_id").set_index("dataset_row_id").subject_id.to_dict()

    records = []
    candidate_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    candidate_10_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    candidate_tokens: list[list[str]] = []
    candidate_subjects: list[str] = []
    candidate_10_counts: list[Counter] = []
    for note_no, item in enumerate(selected):
        case_id = str(item["case_id"])
        row_id = str(item["dataset_row_id"])
        if row_id not in subject_by_row:
            raise ValueError(f"selected dataset_row_id missing from split manifest: {row_id}")
        for section, text in split_sections(item["generated_text"]).items():
            query = tokens(text)
            source_metrics = copy_metrics(query, tokens(source_by_case[case_id]), minimum)
            record = {"case_id": case_id, "dataset_row_id": row_id, "subject_id": subject_by_row[row_id], "section": section, **source_metrics}
            record["phi_like_pattern_count"] = sum(len(pattern.findall(text)) for pattern in PHI_PATTERNS)
            records.append(record)
            index = len(candidate_tokens)
            candidate_tokens.append(query)
            candidate_subjects.append(subject_by_row[row_id])
            candidate_10_counts.append(Counter())
            for gram in ngrams(query, minimum):
                candidate_index[gram].append(index)
            for gram in ngrams(query, 10):
                candidate_10_index[gram].append(index)

    # Scan each real train note once. Matching is patient-excluded so same-subject
    # longitudinal notes cannot be mistaken for unrelated training memorization.
    best_train_span = [0] * len(candidate_tokens)
    matched_5grams: list[set[tuple[str, ...]]] = [set() for _ in candidate_tokens]
    matched_10grams: list[set[tuple[str, ...]]] = [set() for _ in candidate_tokens]
    train_rows = split.loc[split.split.eq(args.train_split)].copy()
    for _, train_subject, train_text in load_train_texts(train_rows, Path(args.pickle_dir), args.progress_every_files):
        train_values = tokens(train_text)
        seen: set[tuple[int, tuple[str, ...]]] = set()
        for pos in range(max(0, len(train_values) - minimum + 1)):
            gram = tuple(train_values[pos : pos + minimum])
            for candidate_no in candidate_index.get(gram, []):
                if train_subject == candidate_subjects[candidate_no] or (candidate_no, gram) in seen:
                    continue
                seen.add((candidate_no, gram))
                matched_5grams[candidate_no].add(gram)
                best_train_span[candidate_no] = max(
                    best_train_span[candidate_no], longest_span(candidate_tokens[candidate_no], train_values, minimum)
                )
        for pos in range(max(0, len(train_values) - 9)):
            gram = tuple(train_values[pos : pos + 10])
            for candidate_no in candidate_10_index.get(gram, []):
                if train_subject == candidate_subjects[candidate_no]:
                    continue
                matched_10grams[candidate_no].add(gram)
                candidate_10_counts[candidate_no][gram] += 1

    for index, record in enumerate(records):
        total_5 = len(ngrams(candidate_tokens[index], minimum))
        total_10 = len(ngrams(candidate_tokens[index], 10))
        train_5_fraction = float(len(matched_5grams[index]) / total_5) if total_5 else 0.0
        train_fraction = float(len(matched_10grams[index]) / total_10) if total_10 else 0.0
        source_fraction = float(record["source_10gram_overlap_fraction"])
        record.update({
            "unrelated_train_longest_exact_span": int(best_train_span[index]),
            "unrelated_train_5gram_overlap_fraction": train_5_fraction,
            "unrelated_train_10gram_overlap_fraction": train_fraction,
            "rare_train_10gram_count": int(sum(count <= 1 for count in candidate_10_counts[index].values())),
            "excess_unrelated_10gram_overlap_fraction": train_fraction - source_fraction,
        })
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_section = pd.DataFrame(records).sort_values(["case_id", "section"])
    per_section.to_csv(output_dir / "v4_copying_privacy_per_section.csv", index=False)
    per_note = per_section.groupby(["case_id", "dataset_row_id", "subject_id"], as_index=False).agg(
        section_count=("section", "size"),
        source_token_copy_fraction=("source_token_copy_fraction", "mean"),
        source_longest_exact_span=("source_longest_exact_span", "max"),
        unrelated_train_longest_exact_span=("unrelated_train_longest_exact_span", "max"),
        unrelated_train_5gram_overlap_fraction=("unrelated_train_5gram_overlap_fraction", "max"),
        unrelated_train_10gram_overlap_fraction=("unrelated_train_10gram_overlap_fraction", "max"),
        excess_unrelated_10gram_overlap_fraction=("excess_unrelated_10gram_overlap_fraction", "max"),
        rare_train_10gram_count=("rare_train_10gram_count", "sum"),
        phi_like_pattern_count=("phi_like_pattern_count", "sum"),
    )
    per_note.to_csv(output_dir / "v4_copying_privacy_per_note.csv", index=False)
    summary = {
        "n_notes": int(len(per_note)), "n_sections": int(len(per_section)),
        "train_reference": "all train notes, excluding every same-subject train note per synthetic note",
        "min_span_tokens": minimum,
        "max_source_token_copy_fraction": float(per_note.source_token_copy_fraction.max()),
        "max_unrelated_train_longest_exact_span": int(per_note.unrelated_train_longest_exact_span.max()),
        "n_notes_with_phi_like_pattern_hits": int(per_note.phi_like_pattern_count.gt(0).sum()),
        "security_note": "Outputs contain provenance IDs and derived overlap statistics only; no note text is exported.",
    }
    (output_dir / "v4_copying_privacy_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
