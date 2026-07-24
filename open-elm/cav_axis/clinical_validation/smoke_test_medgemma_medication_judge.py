#!/usr/bin/env python3
"""Run fabricated-only local MedGemma medication-judge smoke tests.

This intentionally contains no MIMIC-derived facts, notes, or identifiers.
It validates the local offline chat-template, generation, JSON parsing, and
basic expected-direction behavior before any restricted judge task is used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FABRICATED_CASES = [
    {
        "case_id": "fabricated_supported_regimen",
        "ledger": "Verified discharge medications: acetaminophen 500 mg by mouth every 6 hours as needed.",
        "note": "Discharge medications: acetaminophen 500 mg by mouth every 6 hours as needed.",
        "expected_final_reject": False,
    },
    {
        "case_id": "fabricated_omission",
        "ledger": "Verified discharge medications: metformin 500 mg by mouth twice daily; lisinopril 10 mg by mouth daily.",
        "note": "Discharge medications: metformin 500 mg by mouth twice daily.",
        "expected_final_reject": True,
    },
    {
        "case_id": "fabricated_action_contradiction",
        "ledger": "Verified medication action: stop warfarin at discharge because of active bleeding.",
        "note": "Instructions: continue warfarin at discharge.",
        "expected_final_reject": True,
    },
    {
        "case_id": "fabricated_unsupported_addition",
        "ledger": "Verified discharge medications: amoxicillin 500 mg by mouth three times daily for 5 days.",
        "note": "Discharge medications: amoxicillin 500 mg by mouth three times daily for 5 days; morphine 30 mg by mouth every 4 hours.",
        "expected_final_reject": True,
    },
    {
        "case_id": "fabricated_no_medications",
        "ledger": "Verified discharge medication plan: no discharge medications are prescribed.",
        "note": "Discharge medications: none.",
        "expected_final_reject": False,
    },
]


SYSTEM_PROMPT = """You are a medication-reconciliation safety checker. Use only the supplied fabricated ledger and fabricated note. Do not invent facts. Return one JSON object only, with no Markdown or explanation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fabricated-only MedGemma medication-judge smoke test.")
    parser.add_argument("--model_path", required=True, help="Local MedGemma snapshot only.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def build_prompt(case: dict[str, object]) -> str:
    return (
        "Task: compare the fabricated medication ledger with the fabricated note.\n"
        "Return exactly one JSON object and then stop. Do not use Markdown, code fences, analysis, "
        "or a thought section. JSON booleans must be unquoted true or false.\n"
        "Required JSON keys: medication_reconciliation_pass (boolean), final_reject (boolean), "
        "error_types (array containing omission, unsupported_addition, action_contradiction, or none), "
        "and evidence (array of short strings grounded in the supplied text).\n"
        f"Fabricated ledger: {case['ledger']}\n"
        f"Fabricated note: {case['note']}\n"
        "Set final_reject=true for a clinically material omission, unsupported medication addition, "
        "or action contradiction. Otherwise set it to false."
    )


def extract_json(text: str) -> tuple[dict[str, object] | None, str | None]:
    # Prefer the first fenced object if the model ignores the no-Markdown instruction.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start < 0:
        return None, "no_json_object"
    try:
        parsed, end = json.JSONDecoder().raw_decode(candidate[start:])
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc.msg}"
    if not isinstance(parsed, dict):
        return None, "json_not_object"
    required = {"medication_reconciliation_pass", "final_reject", "error_types", "evidence"}
    missing = sorted(required.difference(parsed))
    if missing:
        return parsed, f"missing_keys:{','.join(missing)}"
    if not isinstance(parsed["final_reject"], bool) or not isinstance(parsed["medication_reconciliation_pass"], bool):
        return parsed, "non_boolean_decision"
    return parsed, None


def main() -> None:
    args = parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise EnvironmentError("Require HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 for this local-only smoke test.")

    model_path = Path(args.model_path).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model path does not exist: {model_path}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    input_device = next(model.parameters()).device

    rows: list[dict[str, object]] = []
    for case in FABRICATED_CASES:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(case)},
        ]
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(input_device)
        prompt_length = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                # Gemma ends its chat turn with token 106, not just tokenizer EOS=1.
                eos_token_id=model.generation_config.eos_token_id,
            )
        completion_ids = output_ids[0, prompt_length:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        parsed, parse_error = extract_json(completion)
        predicted_reject = parsed.get("final_reject") if parsed else None
        rows.append(
            {
                "case_id": case["case_id"],
                "expected_final_reject": case["expected_final_reject"],
                "predicted_final_reject": predicted_reject,
                "matches_expected_direction": predicted_reject == case["expected_final_reject"],
                "parse_error": parse_error,
                "generated_token_count": int(completion_ids.shape[0]),
                "hit_max_new_tokens": int(completion_ids.shape[0]) >= args.max_new_tokens - 1,
                "raw_completion": completion,
                "parsed_output": parsed,
            }
        )

    (output_dir / "fabricated_medication_judge_smoke_rows.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    summary = {
        "scope": "fabricated_only_no_mimic_or_project_clinical_content",
        "model_path": str(model_path),
        "n_cases": len(rows),
        "valid_json_rate": sum(row["parse_error"] is None for row in rows) / len(rows),
        "expected_direction_accuracy": sum(row["matches_expected_direction"] for row in rows) / len(rows),
        "cap_hit_rate": sum(row["hit_max_new_tokens"] for row in rows) / len(rows),
        "offline_only": True,
        "rows_path": str(output_dir / "fabricated_medication_judge_smoke_rows.json"),
    }
    (output_dir / "fabricated_medication_judge_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
