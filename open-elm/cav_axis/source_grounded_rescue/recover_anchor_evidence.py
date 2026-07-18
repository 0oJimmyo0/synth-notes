#!/usr/bin/env python3
"""Run a restricted, provenance-aware recovery audit for incomplete fact ledgers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


FIELD_PATTERNS = {
    "follow_up": r"follow[- ]?up(?: instructions)?|appointments?|post[- ]discharge care",
    "instructions": r"discharge instructions|instructions|patient instructions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_reference_csv", required=True)
    parser.add_argument("--anchor_manifest_path", required=True)
    parser.add_argument("--admissions_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def classify(value: str, field: str) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    normalized = re.sub(r"[\s_\-.:/]+", "", compact).lower()
    if not normalized or normalized in {"na", "none", "unknown", "notavailable"}:
        return "placeholder_only"
    if field in {"follow_up", "instructions"}:
        return "substantive" if len(compact.split()) >= 4 else "fragmentary"
    if len(compact.split()) < 8:
        return "fragmentary"
    return "substantive"


def recover_text_field(text: str, pattern: str, field: str) -> tuple[str | None, int | None, int | None]:
    # Prefer the strongest matching heading because notes may contain an early placeholder.
    heading = re.compile(rf"(?im)^\s*(?:{pattern})\s*:?\s*(.*)$")
    candidates: list[tuple[str, int, int]] = []
    for match in heading.finditer(text):
        start = match.start(1)
        tail = text[start:]
        kept = []
        for line in tail.splitlines()[:8]:
            if kept and re.match(r"^\s*[A-Z][A-Z /&-]{2,80}:?\s*$", line.strip()):
                break
            kept.append(line)
        value = "\n".join(kept).strip()
        candidates.append((value, start, start + len(value)))
    if not candidates:
        return None, None, None
    rank = {"substantive": 2, "fragmentary": 1, "placeholder_only": 0}
    return max(candidates, key=lambda item: (rank[classify(item[0], field)], len(item[0].split())))


def main() -> None:
    args = parse_args()
    references = pd.read_csv(Path(args.source_reference_csv).resolve())
    anchors = pd.read_csv(Path(args.anchor_manifest_path).resolve())
    required_reference = {"case_id", "dataset_row_id", "source_real_note"}
    required_anchor = {"dataset_row_id", "hadm_id"}
    if missing := required_reference - set(references.columns):
        raise KeyError(f"source reference missing columns: {sorted(missing)}")
    if missing := required_anchor - set(anchors.columns):
        raise KeyError(f"anchor manifest missing columns: {sorted(missing)}")
    frame = references.merge(anchors[["dataset_row_id", "hadm_id"]], on="dataset_row_id", how="left", validate="one_to_one")
    if frame.hadm_id.isna().any():
        raise ValueError("Some source references could not be linked to hadm_id.")
    admissions = pd.read_csv(Path(args.admissions_path).resolve(), usecols=["hadm_id", "discharge_location", "hospital_expire_flag"])
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="raise").astype(int)
    admissions = admissions.drop_duplicates("hadm_id")
    frame["hadm_id"] = pd.to_numeric(frame["hadm_id"], errors="raise").astype(int)
    frame = frame.merge(admissions, on="hadm_id", how="left", validate="many_to_one")

    rows: list[dict[str, object]] = []
    for record in frame.itertuples(index=False):
        text = str(record.source_real_note)
        for field, pattern in FIELD_PATTERNS.items():
            value, start, end = recover_text_field(text, pattern, field)
            rows.append({
                "case_id": record.case_id, "dataset_row_id": int(record.dataset_row_id), "hadm_id": int(record.hadm_id),
                "field": field, "recovery_route": "full_note_heading_search", "evidence_strength": classify(value or "", field),
                "recovered_value_RESTRICTED": value, "source_char_start": start, "source_char_end": end,
            })
        disposition = record.discharge_location
        if pd.notna(disposition):
            rows.append({
                "case_id": record.case_id, "dataset_row_id": int(record.dataset_row_id), "hadm_id": int(record.hadm_id),
                "field": "disposition", "recovery_route": "admissions.discharge_location", "evidence_strength": "structured_candidate",
                "recovered_value_RESTRICTED": str(disposition), "source_char_start": None, "source_char_end": None,
            })
    result = pd.DataFrame(rows)
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "recovered_anchor_evidence_RESTRICTED.csv", index=False)
    summary = result.groupby(["field", "recovery_route", "evidence_strength"], as_index=False).size().rename(columns={"size": "n_cases"})
    summary.to_csv(out / "recovered_anchor_evidence_summary.csv", index=False)
    (out / "recovered_anchor_evidence_summary.json").write_text(json.dumps({"n_cases": int(frame.case_id.nunique()), "summary": summary.to_dict(orient="records"), "security_note": "Restricted evidence file contains source-derived text and stays on approved storage."}, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
