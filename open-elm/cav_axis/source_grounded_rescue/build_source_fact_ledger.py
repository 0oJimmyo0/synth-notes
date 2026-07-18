#!/usr/bin/env python3
"""Build provisional, reviewable fact ledgers from held-out source notes.

The extractor only captures text under recognizable discharge-summary headings.
It never infers missing facts; every extracted fact is marked ``pending`` for
manual verification before any source-grounded generation is allowed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
from closed_loop_train_text_privacy_screen import infer_pickle_dir, load_note_texts_for_rows


FIELD_ALIASES = {
    "admission_reason": ("chief complaint", "reason for admission", "history of present illness"),
    "principal_diagnosis": ("discharge diagnosis", "discharge diagnoses", "principal diagnosis"),
    "secondary_diagnoses": ("secondary diagnosis", "secondary diagnoses", "diagnoses"),
    "procedures_this_admission": ("major surgical or invasive procedure", "procedures", "procedure"),
    "complications": ("complications",),
    "hospital_course_events": ("brief hospital course", "hospital course"),
    "important_results": ("pertinent results", "significant results", "laboratory data"),
    "discharge_medications": ("discharge medications", "medications on discharge", "discharge medication"),
    "disposition": ("discharge disposition", "disposition"),
    "follow_up": ("followup instructions", "follow up instructions", "follow-up instructions", "follow up"),
    "instructions": ("discharge instructions", "instructions"),
}
RECOVERY_PATTERNS = {
    "follow_up": r"follow[- ]?up(?: instructions)?|appointments?|post[- ]discharge care",
    "instructions": r"discharge instructions|patient instructions|instructions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build provisional source-fact ledgers for held-out notes.")
    parser.add_argument("--anchor_manifest_path", required=True, help="CSV or JSONL with dataset_row_id for selected held-out anchors.")
    parser.add_argument("--dataset_path", required=True, help="Encoded dataset path used to infer approved pickle storage.")
    parser.add_argument("--split_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_split", default="test")
    parser.add_argument("--pickle_dir", default=None)
    parser.add_argument("--max_cases", type=int, default=0, help="Optional deterministic cap; 0 keeps all provided anchors.")
    parser.add_argument("--max_fact_chars", type=int, default=2000)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True) if path.suffix.lower() == ".jsonl" else pd.read_csv(path)


def clean_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().rstrip(":"))


def extract_sections(text: str) -> dict[str, tuple[str, int, int]]:
    lines = str(text).replace("\r", "").splitlines(keepends=True)
    offsets, cursor = [], 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    headings: list[tuple[str, int, int, str]] = []
    for index, line in enumerate(lines):
        match = re.match(r"^\s*([A-Za-z][A-Za-z /&-]{2,80}):\s*(.*)$", line.strip())
        if match:
            headings.append((clean_heading(match.group(1)), index, offsets[index], match.group(2).strip()))
    sections: dict[str, tuple[str, int, int]] = {}
    for heading_index, (heading, line_index, start_offset, inline) in enumerate(headings):
        next_offset = headings[heading_index + 1][2] if heading_index + 1 < len(headings) else len(text)
        body_start = offsets[line_index] + len(lines[line_index])
        body = (inline + "\n" + text[body_start:next_offset]).strip() if inline else text[body_start:next_offset].strip()
        if body:
            sections[heading] = (body, start_offset, next_offset)
    return sections


def find_section(sections: dict[str, tuple[str, int, int]], aliases: tuple[str, ...]) -> tuple[str, str, int, int] | None:
    for alias in aliases:
        normalized = clean_heading(alias)
        for heading, payload in sections.items():
            if heading == normalized or heading.startswith(normalized):
                return heading, payload[0], payload[1], payload[2]
    return None


def is_placeholder(value: str) -> bool:
    normalized = re.sub(r"[\s_\-.:/]+", "", value).lower()
    return not normalized or normalized in {"na", "none", "unknown", "notavailable"}


def recover_best_heading(text: str, pattern: str) -> tuple[str, int, int] | None:
    """Return the longest non-placeholder matching section for later review."""
    heading = re.compile(rf"(?im)^\s*(?:{pattern})\s*:?\s*(.*)$")
    candidates: list[tuple[str, int, int]] = []
    for match in heading.finditer(text):
        start = match.start(1)
        kept: list[str] = []
        for line in text[start:].splitlines()[:8]:
            if kept and re.match(r"^\s*[A-Z][A-Z /&-]{2,80}:?\s*$", line.strip()):
                break
            kept.append(line)
        value = "\n".join(kept).strip()
        if not is_placeholder(value):
            candidates.append((value, start, start + len(value)))
    return max(candidates, key=lambda item: len(item[0].split()), default=None)


def add_fact(facts: list[dict[str, object]], case_id: str, field: str, value: str, section: str, start: int | None, end: int | None, max_chars: int) -> None:
    compact = re.sub(r"\s+", " ", value).strip()[:max_chars]
    if not compact:
        return
    facts.append({
        "fact_id": f"{case_id}_{field}_{len(facts)+1:02d}",
        "field": field,
        "value": compact,
        "source_section": section,
        "supporting_text": compact,
        "source_char_start": start,
        "source_char_end": end,
        "extraction_confidence": 1.0,
        "manual_verification_status": "pending",
        "manual_verified_value": None,
    })


def main() -> None:
    args = parse_args()
    anchor_path = Path(args.anchor_manifest_path).resolve()
    dataset_path = Path(args.dataset_path).resolve()
    split_path = Path(args.split_manifest_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    anchors = read_table(anchor_path)
    if "dataset_row_id" not in anchors.columns:
        raise KeyError("anchor manifest must contain dataset_row_id")
    anchors["dataset_row_id"] = pd.to_numeric(anchors["dataset_row_id"], errors="raise").astype(int)
    anchors = anchors.drop_duplicates("dataset_row_id").sort_values("dataset_row_id").reset_index(drop=True)
    if args.max_cases:
        anchors = anchors.head(args.max_cases).copy()

    split = pd.read_csv(split_path)
    split["dataset_row_id"] = pd.to_numeric(split["dataset_row_id"], errors="raise").astype(int)
    split = split.loc[split["split"].astype(str) == str(args.source_split)].copy()
    required = ["dataset_row_id", "note_id", "filename"]
    missing = [column for column in required if column not in split.columns]
    if missing:
        raise KeyError(f"split manifest is missing required provenance columns: {missing}")
    split = split.drop_duplicates("dataset_row_id", keep="first")
    anchors_for_source = anchors.drop(columns=["note_id", "filename"], errors="ignore")
    sources = anchors_for_source.merge(split[required], on="dataset_row_id", how="left", validate="one_to_one")
    if sources[["note_id", "filename"]].isna().any().any():
        raise ValueError("some anchors have no source-note provenance in the requested split")

    explicit_pickle_dir = Path(args.pickle_dir).resolve() if args.pickle_dir else None
    pickle_dir = infer_pickle_dir(dataset_path, explicit_pickle_dir=explicit_pickle_dir)
    if pickle_dir is None:
        raise FileNotFoundError("could not resolve approved pickle_ds_note_hadm_all directory")
    texts = load_note_texts_for_rows(sources, pickle_dir)

    ledgers, review_rows, reference_rows = [], [], []
    for ordinal, row in enumerate(sources.itertuples(index=False), start=1):
        row_id = int(row.dataset_row_id)
        text = texts.get(row_id, "")
        if not text:
            continue
        case_id = f"ledger_{ordinal:03d}"
        sections = extract_sections(text)
        facts: list[dict[str, object]] = []
        sex_match = re.search(r"\bSex:\s*([A-Za-z]+)", text, flags=re.IGNORECASE)
        if sex_match:
            add_fact(facts, case_id, "demographics.sex", sex_match.group(1), "header", sex_match.start(1), sex_match.end(1), args.max_fact_chars)
        for field, aliases in FIELD_ALIASES.items():
            found = find_section(sections, aliases)
            if field in RECOVERY_PATTERNS and (found is None or is_placeholder(found[1])):
                recovered = recover_best_heading(text, RECOVERY_PATTERNS[field])
                if recovered is not None:
                    value, start, end = recovered
                    add_fact(facts, case_id, field, value, "full_note_heading_recovery", start, end, args.max_fact_chars)
                    continue
            if found is not None:
                heading, value, start, end = found
                add_fact(facts, case_id, field, value, heading, start, end, args.max_fact_chars)
        ledger = {
            "case_id": case_id,
            "source_provenance": {
                "dataset_row_id": row_id,
                "note_id": str(row.note_id),
                "source_split": str(args.source_split),
                "anchor_id": str(getattr(row, "anchor_id", "")),
            },
            "facts": facts,
            "security_note": "Contains source-derived spans; retain only on approved MIMIC-IV project storage.",
        }
        ledgers.append(ledger)
        reference_rows.append({
            "case_id": case_id,
            "dataset_row_id": row_id,
            "note_id": str(row.note_id),
            "source_real_note": text,
        })
        for fact in facts:
            review_rows.append({"case_id": case_id, "dataset_row_id": row_id, "note_id": str(row.note_id), **fact})

    with (output_dir / "source_fact_ledgers.jsonl").open("w", encoding="utf-8") as handle:
        for ledger in ledgers:
            handle.write(json.dumps(ledger) + "\n")
    pd.DataFrame(review_rows).to_csv(output_dir / "source_fact_ledger_manual_verification.csv", index=False)
    pd.DataFrame(reference_rows).to_csv(output_dir / "source_fact_ledger_source_reference_RESTRICTED.csv", index=False)
    summary = {
        "n_requested_anchors": int(len(anchors)),
        "n_ledgers_written": int(len(ledgers)),
        "n_provisional_facts": int(len(review_rows)),
        "all_facts_require_manual_verification": True,
        "pickle_dir": str(pickle_dir),
        "security_note": "Ledger outputs and the restricted source-reference CSV contain source-derived text and must remain on approved project storage.",
    }
    (output_dir / "source_fact_ledger_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
