#!/usr/bin/env python3
"""Generate only hospital-course prose and deterministically assemble transition notes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

from run_source_grounded_rescue import generate, load_model


RENDERED_FIELDS = (
    ("principal_diagnosis", "Discharge Diagnosis"),
    ("discharge_medications", "Discharge Medications"),
    ("disposition", "Disposition"),
    ("instructions", "Discharge Instructions"),
    ("follow_up", "Follow-up"),
)

# Transition facts are rendered from the reviewed contract. The free-form
# hospital course must not restate them, where it could introduce a conflict.
COURSE_CONSTRAINT_PATTERNS = {
    "gendered_pronoun": re.compile(r"(?i)\b(?:he|she|him|her|his|hers)\b"),
    "disposition_or_transfer": re.compile(
        r"(?i)\b(?:discharg(?:e|ed|ing)|transfer(?:red|ring)?|facility|rehabilitation|rehab|placement|home)\b"
    ),
}
TERMINAL_OUTCOME_PATTERN = re.compile(r"(?i)\b(?:resolved|resolution|healed|cured|cleared)\b")
POSTOPERATIVE_DAY_PATTERN = re.compile(
    r"(?i)\b(?:post[-\s]?operative|post[-\s]?op)\s+day\s*(\d+)\b|\bpod\s*(\d+)\b"
)
UNSUPPORTED_COURSE_ASSERTION_PATTERNS = {
    "unsupported_negative_evaluation": re.compile(r"(?i)\b(?:did not|was not)\s+(?:assess(?:ed)?|evaluate(?:d)?)\b"),
    "unsupported_treatment_relationship": re.compile(r"(?i)\btreated\s+for\b"),
    "unsupported_procedure_outcome": re.compile(r"(?i)\b(?:tolerated|uncomplicated|without complications?)\b"),
}
TRANSITION_SENTENCE_PATTERN = re.compile(
    r"(?i)\b(?:discharg(?:e|ed|ing)|transfer(?:red|ring)?|facility|rehabilitation|rehab|placement|home)\b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract_path", required=True)
    parser.add_argument("--generation_ledger_path", required=True, help="Frozen source ledger used only for anchor provenance.")
    parser.add_argument("--backbone_path", required=True)
    parser.add_argument("--model_condition", choices=["untouched_backbone", "checkpoint_8215"], required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--case_ids", default="", help="Optional comma-separated smoke-test case IDs.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--n_candidates_per_case", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_course_attempts", type=int, default=3)
    parser.add_argument("--output_stem", default="hybrid_contract")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"empty JSONL: {path}")
    return records


def remove_transition_sentences(value: str) -> str:
    """Keep the model course inpatient-only; the contract renders transition facts."""
    sentences = re.split(r"(?<=[.!?])\s+", value.strip())
    return " ".join(sentence for sentence in sentences if not TRANSITION_SENTENCE_PATTERN.search(sentence)).strip()


def course_facts(contract: dict) -> list[dict[str, str]]:
    allowed = {"principal_diagnosis", "hospital_course_events", "procedures_this_admission", "complications"}
    facts = []
    for fact in contract["facts"]:
        if str(fact["field"]) not in allowed or str(fact["status"]) == "explicit_none":
            continue
        value = remove_transition_sentences(str(fact["generation_value"]))
        if value:
            facts.append({"fact_id": str(fact["fact_id"]), "field": str(fact["field"]), "value": value})
    return facts


def build_course_prompt(facts: list[dict[str, str]]) -> str:
    return (
        "Write only the Brief Hospital Course section of a synthetic discharge transition note.\n"
        "Use only the verified facts below. Do not write a diagnosis, medication list, disposition, "
        "follow-up, instructions, new laboratory values, or unsupported causal/action relationship. "
        "Do not state active discharge medication plans, discharge destination, transfer, placement, "
        "facility, rehabilitation, or home status. Refer to the patient only as 'the patient'; never use "
        "gendered pronouns. Do not claim a condition resolved, healed, cured, or cleared unless that exact "
        "outcome is present in the verified facts. If a detail is not supplied, omit it.\n\n"
        "VERIFIED HOSPITAL-COURSE FACTS:\n" + json.dumps(facts, indent=2, ensure_ascii=True)
    )


def course_constraint_reasons(course: str, facts: list[dict[str, str]]) -> list[str]:
    reasons = [name for name, pattern in COURSE_CONSTRAINT_PATTERNS.items() if pattern.search(course)]
    source_text = " ".join(fact["value"] for fact in facts).lower()
    if any(token.group(0).lower() not in source_text for token in TERMINAL_OUTCOME_PATTERN.finditer(course)):
        reasons.append("unsupported_terminal_outcome")
    source_postoperative_days = {
        next(day for day in match.groups() if day is not None)
        for match in POSTOPERATIVE_DAY_PATTERN.finditer(source_text)
    }
    generated_postoperative_days = {
        next(day for day in match.groups() if day is not None)
        for match in POSTOPERATIVE_DAY_PATTERN.finditer(course)
    }
    if not generated_postoperative_days.issubset(source_postoperative_days):
        reasons.append("unsupported_postoperative_day")
    for name, pattern in UNSUPPORTED_COURSE_ASSERTION_PATTERNS.items():
        if any(match.group(0).lower() not in source_text for match in pattern.finditer(course)):
            reasons.append(name)
    return reasons


def render_note(contract: dict, course: str) -> str:
    sections = []
    facts = contract["facts"]
    for field, heading in RENDERED_FIELDS:
        values = [
            str(fact["generation_value"]).strip()
            for fact in facts
            if str(fact["field"]) == field and str(fact["status"]) in {"required", "optional"}
            and str(fact["generation_value"]).strip()
        ]
        if values:
            sections.append(f"{heading}:\n" + "\n".join(f"- {value}" for value in values))
    course = re.sub(r"(?is)^\s*brief\s+hospital\s+course\s*:\s*", "", course).strip()
    sections.insert(1 if sections else 0, "Brief Hospital Course:\n" + course)
    return "\n\n".join(sections).strip()


def main() -> None:
    args = parse_args()
    selected_case_ids = {item.strip() for item in args.case_ids.split(",") if item.strip()}
    contracts = {str(row["case_id"]): row for row in load_jsonl(Path(args.contract_path).resolve())}
    ledgers = {str(row["case_id"]): row for row in load_jsonl(Path(args.generation_ledger_path).resolve())}
    case_ids = sorted(set(contracts).intersection(ledgers))
    if selected_case_ids:
        missing = selected_case_ids.difference(case_ids)
        if missing:
            raise ValueError(f"requested cases missing from contract/ledger intersection: {sorted(missing)}")
        case_ids = sorted(selected_case_ids)
    case_ids = [case_id for case_id in case_ids if contracts[case_id].get("ready_for_hybrid_generation")]
    if not case_ids:
        raise ValueError("no source-complete cases are ready for hybrid generation")
    if args.max_course_attempts < 1:
        raise ValueError("--max_course_attempts must be at least 1")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_path)
    model, model_metadata = load_model(args)
    model.eval()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{args.output_stem}_manifest.jsonl"
    rows = []
    with manifest_path.open("w", encoding="utf-8") as handle:
        for case_id in case_ids:
            contract, ledger = contracts[case_id], ledgers[case_id]
            facts = course_facts(contract)
            if not facts:
                raise ValueError(f"{case_id} has no verified hospital-course facts")
            prompt = build_course_prompt(facts)
            for candidate_index in range(args.n_candidates_per_case):
                course, metadata, reasons, attempt = "", {}, ["generation_not_attempted"], 0
                for attempt in range(1, args.max_course_attempts + 1):
                    course, metadata = generate(
                        model, tokenizer, prompt, args.max_new_tokens, args.do_sample,
                        args.temperature, args.top_p,
                        args.seed + candidate_index * args.max_course_attempts + attempt - 1,
                    )
                    reasons = course_constraint_reasons(course, facts)
                    if not reasons:
                        break
                record = {
                    "rescue_id": f"{case_id}__{args.model_condition}__hybrid__cand{candidate_index:02d}",
                    "case_id": case_id,
                    "anchor_id": ledger.get("anchor_id"),
                    "dataset_row_id": ledger.get("dataset_row_id"),
                    "note_id": ledger.get("note_id"),
                    "review_stratum": ledger.get("review_stratum"),
                    "patient_disjoint_from_train": ledger.get("patient_disjoint_from_train"),
                    "arm": "hybrid_contract", "document_type": "discharge_transition_note",
                    "candidate_index": candidate_index, "contract_version": contract["contract_version"],
                    "contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
                    "generation_ledger_sha256": ledger.get("generation_ledger_sha256"),
                    "hospital_course_text": course, "generated_text": render_note(contract, course),
                    "course_constraint_pass": not reasons,
                    "course_constraint_rejection_reasons": reasons,
                    "course_generation_attempt": attempt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), **model_metadata, **metadata,
                }
                handle.write(json.dumps(record) + "\n")
                rows.append(record)
                print(json.dumps({key: record[key] for key in ["rescue_id", "empty_output", "hit_max_new_tokens", "ended_with_eos"]}), flush=True)
    summary = {
        "n_cases": len(case_ids), "n_outputs": len(rows), "n_candidates_per_case": args.n_candidates_per_case,
        "max_new_tokens": args.max_new_tokens, "model_condition": args.model_condition,
        "safety_boundary": "Only hospital-course prose is generated; high-risk transition sections are rendered deterministically from the reviewed contract.",
    }
    (output_dir / f"{args.output_stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
