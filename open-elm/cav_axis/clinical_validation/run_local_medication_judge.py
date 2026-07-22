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
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCHEMA_PATH = Path(__file__).with_name("medication_judge_schema.json")
PROMPT_PATH = Path(__file__).with_name("medication_judge_prompt_v1.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local ledger-grounded medication reconciliation judge.")
    parser.add_argument("--task_path", required=True)
    parser.add_argument("--model_path", required=True, help="Approved local instruction model path; never an external API ID.")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--prompt_path", default=str(PROMPT_PATH), help="Versioned local prompt template with {schema}, {ledger}, and {note} placeholders.")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def prompt_for(task: dict[str, object], schema: dict[str, object], template: str) -> str:
    return template.format(
        schema=json.dumps(schema, ensure_ascii=True),
        ledger=task["verified_fact_ledger"],
        note=task["synthetic_note"],
    )


def extract_json(text: str) -> tuple[dict[str, object] | None, str | None]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None, "no_json_object"
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc.msg}"
    return value if isinstance(value, dict) else None, None if isinstance(value, dict) else "json_not_object"


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Approved local model path does not exist: {model_path}")
    task_path = Path(args.task_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema.pop("$schema", None)
    schema.pop("title", None)
    prompt_path = Path(args.prompt_path).resolve()
    template = prompt_path.read_text(encoding="utf-8")
    required_placeholders = {"{schema}", "{ledger}", "{note}"}
    missing_placeholders = [item for item in required_placeholders if item not in template]
    if missing_placeholders:
        raise ValueError(f"Prompt template is missing placeholders: {missing_placeholders}")
    tasks = [json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype="auto", device_map="auto" if args.device == "cuda" else None,
    )
    if args.device != "cuda":
        model.to(args.device)
    model.eval()
    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            for repeat_index in range(args.repeats):
                seed = args.seed + repeat_index
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                encoded = tokenizer(prompt_for(task, schema, template), return_tensors="pt")
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded, do_sample=False, max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    )
                completion = tokenizer.decode(generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
                judge_output, parse_error = extract_json(completion)
                handle.write(json.dumps({
                    "task_id": task["task_id"], "blinded_output_id": task["blinded_output_id"],
                    "repeat_index": repeat_index, "seed": seed, "judge_output": judge_output,
                    "parse_error": parse_error, "model_path": str(model_path), "prompt_path": str(prompt_path),
                }, ensure_ascii=True) + "\n")
    print(json.dumps({"tasks": len(tasks), "repeats": args.repeats, "output_path": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
