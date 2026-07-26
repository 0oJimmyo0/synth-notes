#!/usr/bin/env python3
"""Pre-screen a held-out embedding region for source-grounded note eligibility.

This script is deliberately a *triage* tool.  It never creates generation
facts, never imputes unsupported information, and never exposes source-note
text.  It identifies which under-covered real-note regions are worth the
expensive manual-ledger review required before complete-note generation.
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
from build_source_fact_ledger import FIELD_ALIASES, extract_sections, find_section


REQUIRED_FIELDS = (
    "principal_diagnosis",
    "hospital_course_events",
    "discharge_medications",
    "disposition",
    "follow_up",
    "instructions",
)
RECOVERY_PATTERNS = {
    "follow_up": r"follow[- ]?up(?: instructions)?|appointments?|post[- ]discharge care",
    "instructions": r"discharge instructions|patient instructions|instructions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_cluster_assignments_path", required=True)
    parser.add_argument("--dataset_path", required=True, help="Used only to resolve approved pickle storage.")
    parser.add_argument("--split_manifest_path", required=True)
    parser.add_argument("--target_cluster_ids", required=True, help="Comma-separated held-out cluster IDs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_split", default="test")
    parser.add_argument("--admissions_path", default=None, help="Optional MIMIC admissions.csv for disposition corroboration.")
    parser.add_argument("--pickle_dir", default=None)
    parser.add_argument(
        "--optional_fields",
        default="",
        help="Comma-separated fields permitted to be absent, for example follow_up.",
    )
    parser.add_argument("--max_rows", type=int, default=0, help="Optional deterministic cap for smoke tests; 0 keeps all candidates.")
    return parser.parse_args()


def classify_text(value: str | None, field: str) -> str:
    compact = re.sub(r"\s+", " ", str(value or "")).strip()
    normalized = re.sub(r"[\s_\-.:/]+", "", compact).lower()
    if not normalized or normalized in {"na", "none", "unknown", "notavailable"}:
        return "absent_or_placeholder"
    # Diagnoses and medication/instruction lists are routinely short or
    # bullet-formatted. This is a high-recall review screen, not a fact gate.
    if field == "principal_diagnosis":
        return "substantive" if len(compact.split()) >= 1 else "fragmentary"
    if field == "discharge_medications":
        reference_only = {"see medications", "see medication list", "medications per reconciliation"}
        return "fragmentary" if normalized in {re.sub(r"[\s_\-.:/]+", "", item) for item in reference_only} else "substantive"
    if field in {"follow_up", "instructions"}:
        return "substantive" if len(compact.split()) >= 4 else "fragmentary"
    if len(compact.split()) < 8:
        return "fragmentary"
    return "substantive"


def recover_field(text: str, pattern: str, field: str) -> str | None:
    heading = re.compile(rf"(?im)^\s*(?:{pattern})\s*:?\s*(.*)$")
    candidates: list[str] = []
    for match in heading.finditer(text):
        tail = text[match.start(1):]
        lines: list[str] = []
        for line in tail.splitlines()[:8]:
            if lines and re.match(r"^\s*[A-Z][A-Z /&-]{2,80}:?\s*$", line.strip()):
                break
            lines.append(line)
        candidates.append("\n".join(lines).strip())
    if not candidates:
        return None
    rank = {"substantive": 2, "fragmentary": 1, "absent_or_placeholder": 0}
    return max(candidates, key=lambda value: (rank[classify_text(value, field)], len(value.split())))


def field_evidence(text: str, field: str, has_structured_disposition: bool) -> tuple[str, str]:
    sections = extract_sections(text)
    found = find_section(sections, FIELD_ALIASES[field])
    if found is not None:
        _, value, _, _ = found
        strength = classify_text(value, field)
        if strength == "substantive":
            return strength, "source_note_section"
    if field in RECOVERY_PATTERNS:
        recovered = recover_field(text, RECOVERY_PATTERNS[field], field)
        if classify_text(recovered, field) == "substantive":
            return "substantive", "full_note_heading_recovery"
    if field == "disposition" and has_structured_disposition:
        return "structured_candidate", "admissions.discharge_location"
    return (classify_text(found[1], field) if found is not None else "absent_or_placeholder"), "source_note_section" if found is not None else "none"


def main() -> None:
    args = parse_args()
    target_ids = {int(value) for value in args.target_cluster_ids.split(",") if value.strip()}
    if not target_ids:
        raise ValueError("--target_cluster_ids must contain at least one ID")
    optional_fields = {value.strip() for value in args.optional_fields.split(",") if value.strip()}
    unknown_optional = optional_fields.difference(REQUIRED_FIELDS)
    if unknown_optional:
        raise ValueError(f"--optional_fields contains unknown fields: {sorted(unknown_optional)}")
    active_required_fields = set(REQUIRED_FIELDS).difference(optional_fields)
    assignments = pd.read_csv(Path(args.real_cluster_assignments_path).resolve())
    required = {"dataset_row_id", "cluster_id", "note_id", "hadm_id", "patient_disjoint_from_train"}
    if missing := required - set(assignments.columns):
        raise KeyError(f"cluster assignments missing columns: {sorted(missing)}")
    assignments["dataset_row_id"] = pd.to_numeric(assignments["dataset_row_id"], errors="raise").astype(int)
    # Full-real coverage assignments contain every split, while source-note
    # provenance below is intentionally limited to one held-out split.
    if "split" in assignments.columns:
        assignments = assignments.loc[assignments["split"].astype(str) == str(args.source_split)].copy()
    candidates = assignments.loc[assignments["cluster_id"].isin(target_ids)].drop_duplicates("dataset_row_id").sort_values("dataset_row_id").copy()
    if args.max_rows:
        candidates = candidates.head(args.max_rows).copy()
    if candidates.empty:
        raise ValueError("No rows match --target_cluster_ids.")

    split = pd.read_csv(Path(args.split_manifest_path).resolve())
    split["dataset_row_id"] = pd.to_numeric(split["dataset_row_id"], errors="raise").astype(int)
    split = split.loc[split["split"].astype(str) == str(args.source_split), ["dataset_row_id", "note_id", "filename"]].drop_duplicates("dataset_row_id")
    sources = candidates.drop(columns=["note_id", "filename"], errors="ignore").merge(split, on="dataset_row_id", how="left", validate="one_to_one")
    if sources[["note_id", "filename"]].isna().any().any():
        raise ValueError("Some selected cluster rows are absent from the requested split manifest.")

    structured_hadm_ids: set[int] = set()
    if args.admissions_path:
        admissions = pd.read_csv(Path(args.admissions_path).resolve(), usecols=["hadm_id", "discharge_location"])
        admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce")
        structured_hadm_ids = set(admissions.loc[admissions["hadm_id"].notna() & admissions["discharge_location"].notna(), "hadm_id"].astype(int))

    explicit_pickle_dir = Path(args.pickle_dir).resolve() if args.pickle_dir else None
    pickle_dir = infer_pickle_dir(Path(args.dataset_path).resolve(), explicit_pickle_dir=explicit_pickle_dir)
    if pickle_dir is None:
        raise FileNotFoundError("Could not resolve approved pickle_ds_note_hadm_all directory.")
    texts = load_note_texts_for_rows(sources, pickle_dir)

    rows: list[dict[str, object]] = []
    for record in sources.itertuples(index=False):
        row_id = int(record.dataset_row_id)
        text = texts.get(row_id, "")
        evidence: dict[str, tuple[str, str]] = {
            field: field_evidence(text, field, int(record.hadm_id) in structured_hadm_ids) for field in REQUIRED_FIELDS
        }
        substantive = {field for field, (strength, _) in evidence.items() if strength == "substantive"}
        disposition_supported = evidence["disposition"][0] in {"substantive", "structured_candidate"}
        complete_review_candidate = active_required_fields - {"disposition"} <= substantive and disposition_supported
        partial_review_candidate = {"principal_diagnosis", "hospital_course_events", "discharge_medications"} <= substantive and disposition_supported
        tier = "tier1_complete_review_candidate" if complete_review_candidate else ("tier2_partial_document_candidate" if partial_review_candidate else "tier3_insufficient_source_evidence")
        row = {
            "dataset_row_id": row_id, "note_id": str(record.note_id), "hadm_id": int(record.hadm_id),
            "cluster_id": int(record.cluster_id), "patient_disjoint_from_train": bool(record.patient_disjoint_from_train),
            "eligibility_tier": tier,
            "manual_ledger_verification_required": True,
        }
        for field, (strength, route) in evidence.items():
            row[f"{field}_evidence_strength"] = strength
            row[f"{field}_evidence_route"] = route
        rows.append(row)

    result = pd.DataFrame(rows)
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "region_source_eligibility_all_candidates.csv", index=False)
    for tier, stem in {
        "tier1_complete_review_candidate": "complete_note_review_candidate_manifest.csv",
        "tier2_partial_document_candidate": "partial_document_review_candidate_manifest.csv",
        "tier3_insufficient_source_evidence": "ineligible_anchor_manifest.csv",
    }.items():
        result.loc[result["eligibility_tier"] == tier].to_csv(out / stem, index=False)
    summary = result.groupby("eligibility_tier", as_index=False).agg(
        n_rows=("dataset_row_id", "size"), patient_disjoint_count=("patient_disjoint_from_train", "sum")
    )
    for field in REQUIRED_FIELDS:
        summary[f"{field}_substantive_rate"] = summary["eligibility_tier"].map(
            result.groupby("eligibility_tier")[f"{field}_evidence_strength"].apply(lambda values: float((values == "substantive").mean()))
        )
    summary.to_csv(out / "region_source_eligibility_summary.csv", index=False)
    report = {
        "target_cluster_ids": sorted(target_ids), "source_split": str(args.source_split), "n_candidates": int(len(result)),
        "tier_counts": {key: int(value) for key, value in result["eligibility_tier"].value_counts().items()},
        "required_fields": sorted(active_required_fields), "optional_fields": sorted(optional_fields),
        "manual_ledger_verification_required_for_tier1": True,
        "security_note": "Outputs contain provenance IDs and evidence classes only; no source-note text is exported.",
    }
    (out / "region_source_eligibility_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
