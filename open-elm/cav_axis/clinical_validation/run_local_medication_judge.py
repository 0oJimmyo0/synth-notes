#!/usr/bin/env python3
"""Run an approved local instruction model as an evidence-first medication judge.

This intentionally does not download a model or call an external service. The
model path must be an approved local instruction-following checkpoint distinct
from the ELM generation checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_judge_json import validate as validate_judge_payload


SCHEMA_PATH = Path(__file__).with_name("medication_judge_schema.json")
PROMPT_PATH = Path(__file__).with_name("medication_judge_prompt_v1.txt")
DEFAULT_MEDICATION_EVIDENCE_FIELDS = ("discharge_medications", "instructions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local ledger-grounded medication reconciliation judge.")
    parser.add_argument("--task_path", required=True)
    parser.add_argument("--model_path", required=True, help="Approved local instruction model path; never an external API ID.")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--schema_path", default=str(SCHEMA_PATH), help="Versioned local output schema.")
    parser.add_argument("--prompt_path", default=str(PROMPT_PATH), help="Versioned local prompt template with {schema}, {ledger}, and {note} placeholders.")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_chat_template", action="store_true", help="Use the model's installed chat template, required for MedGemma.")
    parser.add_argument("--system_prompt", default="", help="Optional local system instruction used only with --use_chat_template.")
    parser.add_argument("--bf16", action="store_true", help="Load in BF16; use for the verified MedGemma deployment.")
    parser.add_argument("--do_sample", action="store_true", help="Enable low-temperature repeated stability checks.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--medication_only_evidence", action="store_true", help="Pass only discharge-medication and medication-action evidence from a JSON ledger.")
    parser.add_argument(
        "--medication_evidence_fields",
        default=",".join(DEFAULT_MEDICATION_EVIDENCE_FIELDS),
        help=(
            "Comma-separated verified ledger fields exposed when --medication_only_evidence is set. "
            "The v3.1 default remains discharge_medications,instructions; v3.2 adds hospital_course_events."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Number of same-repeat requests decoded together.")
    parser.add_argument("--max_generation_seconds", type=float, default=0.0, help="Per-batch generation time limit; 0 disables it.")
    parser.add_argument(
        "--progress_every",
        type=int,
        default=10,
        help="Emit one progress line every N batches (0 disables progress output).",
    )
    return parser.parse_args()


def parse_evidence_fields(raw_fields: str) -> set[str]:
    fields = {field.strip() for field in raw_fields.split(",") if field.strip()}
    if not fields:
        raise ValueError("--medication_evidence_fields must name at least one ledger field.")
    return fields


def medication_evidence(ledger_text: str, evidence_fields: set[str]) -> str:
    """Retain only ledger fields relevant to medication reconciliation."""
    try:
        ledger = json.loads(ledger_text)
    except json.JSONDecodeError as exc:
        raise ValueError("--medication_only_evidence requires a JSON verified_fact_ledger.") from exc
    if not isinstance(ledger, list):
        raise ValueError("--medication_only_evidence requires a list-valued verified_fact_ledger.")
    relevant = [
        item for item in ledger
        if isinstance(item, dict) and item.get("field") in evidence_fields
    ]
    if not relevant:
        raise ValueError("No requested medication evidence fields were found in the verified ledger.")
    return json.dumps(relevant, ensure_ascii=True, separators=(",", ":"))


def prompt_for(
    task: dict[str, object],
    schema: dict[str, object],
    template: str,
    medication_only: bool,
    evidence_fields: set[str],
) -> str:
    ledger = str(task["verified_fact_ledger"])
    if medication_only:
        ledger = medication_evidence(ledger, evidence_fields)
    return template.format(
        schema=json.dumps(schema, ensure_ascii=True),
        ledger=ledger,
        note=task["synthetic_note"],
    )


def extract_json(text: str) -> tuple[dict[str, object] | None, str | None]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start < 0:
        return None, "no_json_object"
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate[start:])
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc.msg}"
    return value if isinstance(value, dict) else None, None if isinstance(value, dict) else "json_not_object"


def render_prompt(tokenizer: AutoTokenizer, prompt: str, args: argparse.Namespace) -> str:
    if args.use_chat_template:
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt


def completion_without_padding(token_ids: torch.Tensor, eos_ids: set[int], pad_id: int | None) -> tuple[torch.Tensor, bool]:
    for index, token_id in enumerate(token_ids.tolist()):
        if token_id in eos_ids:
            return token_ids[: index + 1], True
        if pad_id is not None and token_id == pad_id:
            return token_ids[:index], False
    return token_ids, False


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")
    if args.max_generation_seconds < 0:
        raise ValueError("--max_generation_seconds must be nonnegative")
    if args.progress_every < 0:
        raise ValueError("--progress_every must be nonnegative")
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Approved local model path does not exist: {model_path}")
    task_path = Path(args.task_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = json.loads(Path(args.schema_path).resolve().read_text(encoding="utf-8"))
    schema.pop("$schema", None)
    schema.pop("title", None)
    prompt_path = Path(args.prompt_path).resolve()
    template = prompt_path.read_text(encoding="utf-8")
    required_placeholders = {"{schema}", "{ledger}", "{note}"}
    missing_placeholders = [item for item in required_placeholders if item not in template]
    if missing_placeholders:
        raise ValueError(f"Prompt template is missing placeholders: {missing_placeholders}")
    evidence_fields = parse_evidence_fields(args.medication_evidence_fields)
    tasks = [json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True,
        dtype=torch.bfloat16 if args.bf16 else "auto",
        device_map="auto" if args.device == "cuda" else None,
    )
    if args.device != "cuda":
        model.to(args.device)
    model.eval()
    input_device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    eos_value = model.generation_config.eos_token_id
    eos_ids = set(eos_value if isinstance(eos_value, list) else [eos_value])
    run_manifest_path = output_path.with_suffix(output_path.suffix + ".run.json")
    run_manifest = {
        "task_count": len(tasks),
        "repeats_requested": args.repeats,
        "expected_output_rows": len(tasks) * args.repeats,
        "output_path": str(output_path),
        "completed": False,
    }
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    with output_path.open("w", encoding="utf-8") as handle:
        for repeat_index in range(args.repeats):
            seed = args.seed + repeat_index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            for offset in range(0, len(tasks), args.batch_size):
                task_batch = tasks[offset : offset + args.batch_size]
                rendered = [
                    render_prompt(
                        tokenizer,
                        prompt_for(task, schema, template, args.medication_only_evidence, evidence_fields),
                        args,
                    )
                    for task in task_batch
                ]
                encoded = tokenizer(rendered, padding=True, return_tensors="pt")
                encoded = {key: value.to(input_device) for key, value in encoded.items()}
                prompt_length = int(encoded["input_ids"].shape[1])
                generation_kwargs: dict[str, object] = {
                    "do_sample": args.do_sample,
                    "max_new_tokens": args.max_new_tokens,
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                    "eos_token_id": eos_value,
                }
                if args.do_sample:
                    generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
                if args.max_generation_seconds:
                    generation_kwargs["max_time"] = args.max_generation_seconds
                with torch.inference_mode():
                    generated = model.generate(**encoded, **generation_kwargs)
                for batch_index, task in enumerate(task_batch):
                    completion_ids, ended_with_eos = completion_without_padding(
                        generated[batch_index, prompt_length:], eos_ids, tokenizer.pad_token_id
                    )
                    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                    judge_output, parse_error = extract_json(completion)
                    schema_errors = validate_judge_payload(judge_output, schema) if judge_output is not None else [parse_error or "output_not_object"]
                    token_count = int(completion_ids.shape[0])
                    hit_cap = token_count >= args.max_new_tokens - 1
                    likely_time_limited = bool(args.max_generation_seconds and not ended_with_eos and not hit_cap)
                    handle.write(json.dumps({
                        "task_id": task["task_id"], "blinded_output_id": task["blinded_output_id"],
                        "repeat_index": repeat_index, "seed": seed, "judge_output": judge_output,
                        "parse_error": parse_error, "model_path": str(model_path), "prompt_path": str(prompt_path),
                        "schema_valid": not schema_errors,
                        "schema_validation_errors": schema_errors,
                        "used_chat_template": args.use_chat_template, "do_sample": args.do_sample,
                        "medication_only_evidence": args.medication_only_evidence,
                        "medication_evidence_fields": sorted(evidence_fields) if args.medication_only_evidence else None,
                        "temperature": args.temperature if args.do_sample else None,
                        "generated_token_count": token_count,
                        "hit_max_new_tokens": hit_cap,
                        "ended_with_eos": ended_with_eos,
                        "likely_time_limited": likely_time_limited,
                        "generation_time_limit_seconds": args.max_generation_seconds or None,
                        "batch_size": len(task_batch),
                        "raw_completion": completion,
                    }, ensure_ascii=True) + "\n")
                handle.flush()
                batch_number = offset // args.batch_size + 1
                total_batches = (len(tasks) + args.batch_size - 1) // args.batch_size
                if args.progress_every and (batch_number % args.progress_every == 0 or batch_number == total_batches):
                    print(
                        f"completed repeat={repeat_index + 1}/{args.repeats} tasks={offset + 1}-{offset + len(task_batch)}/{len(tasks)} "
                        f"batch_size={len(task_batch)}",
                        flush=True,
                    )
    run_manifest["completed"] = True
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    main()
