#!/usr/bin/env python3
"""Audit ledger-to-output medication reconciliation against completed reviews.

This is a conservative lexical diagnostic, not an automatic clinical verifier.
It detects exact medication-name coverage failures and possible extra medication
mentions, then reports agreement with blinded human labels before any future
use as a rejection gate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


MEDICATION_HEADER = re.compile(
    r"(?im)^\s*\*{0,2}(?:discharge\s+)?medications?\*{0,2}\s*:\s*"
)
NEXT_HEADER = re.compile(r"(?im)^\s*\*{0,2}[a-z][a-z /-]{1,60}\*{0,2}\s*:")
DOSE_NAME = re.compile(
    r"\b([a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*){0,2})\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?)\b",
    flags=re.IGNORECASE,
)
FREQUENCY = re.compile(r"\b(?:once|twice|daily|every|at bedtime|as needed|prn)\b", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review_csv", action="append", required=True, help="Completed blinded-review CSV; repeat for multiple regions.")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).lower()).strip(" .;,:-")


def ledger_facts(raw: object) -> list[dict]:
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError("verified_fact_ledger must contain valid JSON.") from exc


def medication_terms(value: str) -> set[str]:
    value = normalize(value)
    terms = {normalize(match.group(1)) for match in DOSE_NAME.finditer(value)}
    for segment in re.split(r"[;\n]", value):
        segment = normalize(segment)
        if not segment:
            continue
        before_frequency = FREQUENCY.split(segment, maxsplit=1)[0].strip(" ,")
        before_dose = re.split(r"\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?)\b", before_frequency, maxsplit=1)[0]
        fallback = normalize(before_dose)
        if fallback and len(fallback) <= 48 and re.fullmatch(r"[a-z][a-z0-9 -]*", fallback):
            terms.add(fallback)
    return {term for term in terms if len(term) >= 3}


def extract_medication_section(note: str) -> str:
    match = MEDICATION_HEADER.search(str(note))
    if not match:
        return ""
    remaining = str(note)[match.end():]
    next_match = NEXT_HEADER.search(remaining)
    return remaining[:next_match.start()] if next_match else remaining


def main() -> None:
    args = parse_args()
    frames = []
    for review_path in args.review_csv:
        frame = pd.read_csv(Path(review_path).resolve())
        required = {
            "blinded_output_id", "verified_fact_ledger", "synthetic_note",
            "discharge_medications_supported_yes_no", "unsupported_major_claim_yes_no",
            "critical_omission_yes_no", "overall_clinical_usability_pass_fail",
        }
        if missing := required.difference(frame.columns):
            raise KeyError(f"{review_path} missing required columns: {sorted(missing)}")
        frame["review_source"] = str(Path(review_path).resolve())
        frames.append(frame)
    review = pd.concat(frames, ignore_index=True)
    review["review_output_id"] = review["review_source"].astype(str) + "::" + review["blinded_output_id"].astype(str)
    if review.review_output_id.duplicated().any():
        raise ValueError("Repeated review inputs contain duplicate review-source/blinded-ID pairs.")

    expected_by_row = []
    global_medication_vocabulary: set[str] = set()
    for raw_ledger in review.verified_fact_ledger:
        terms = set()
        for fact in ledger_facts(raw_ledger):
            if str(fact.get("field")) == "discharge_medications":
                terms.update(medication_terms(str(fact.get("value", ""))))
        expected_by_row.append(terms)
        global_medication_vocabulary.update(terms)

    rows = []
    for row, expected in zip(review.itertuples(index=False), expected_by_row):
        medication_section = extract_medication_section(row.synthetic_note)
        observed = {
            term for term in global_medication_vocabulary
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", medication_section, flags=re.IGNORECASE)
        }
        missing = sorted(expected.difference(observed))
        possible_extra = sorted(observed.difference(expected))
        medication_failure = str(row.discharge_medications_supported_yes_no).strip().lower() == "no"
        lexical_flag = bool(missing or possible_extra)
        rows.append({
            "review_output_id": row.review_output_id,
            "blinded_output_id": row.blinded_output_id,
            "case_id": getattr(row, "case_id", ""),
            "review_source": row.review_source,
            "manual_medication_failure": medication_failure,
            "manual_unsupported_major_claim": str(row.unsupported_major_claim_yes_no).strip().lower() == "yes",
            "manual_critical_omission": str(row.critical_omission_yes_no).strip().lower() == "yes",
            "manual_pass": str(row.overall_clinical_usability_pass_fail).strip().lower() == "pass",
            "expected_medication_terms": "|".join(sorted(expected)),
            "observed_global_medication_terms": "|".join(sorted(observed)),
            "missing_expected_medication_terms": "|".join(missing),
            "possible_extra_medication_terms": "|".join(possible_extra),
            "lexical_medication_reconciliation_flag": lexical_flag,
            "medication_section_present": bool(medication_section.strip()),
        })
    audit = pd.DataFrame(rows)
    positive = audit.manual_medication_failure
    flagged = audit.lexical_medication_reconciliation_flag
    summary = {
        "n_reviewed_outputs": int(len(audit)),
        "n_manual_medication_failures": int(positive.sum()),
        "n_lexically_flagged": int(flagged.sum()),
        "sensitivity_for_manual_medication_failure": float((flagged & positive).sum() / positive.sum()) if positive.any() else None,
        "flag_rate_among_manual_medication_passes": float((flagged & ~positive).sum() / (~positive).sum()) if (~positive).any() else None,
        "limitation": "Lexical terms are a conservative lower-bound detector. Possible extra terms are restricted to the reviewed-ledger medication vocabulary and require human or stronger semantic validation.",
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "medication_reconciliation_row_audit.csv", index=False)
    (out / "medication_reconciliation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
