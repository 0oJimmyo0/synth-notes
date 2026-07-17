#!/usr/bin/env python3
"""Generate deterministic fact-conditioned correction and fact-only notes locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

OPEN_ELM_DIR = Path(__file__).resolve().parents[2]
if str(OPEN_ELM_DIR) not in sys.path:
    sys.path.insert(0, str(OPEN_ELM_DIR))

from generate_synthetic_notes import load_generation_model
from src.model import LlamaForEmbeddingLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local source-grounded rescue generation.")
    parser.add_argument("--generation_ledger_path", required=True)
    parser.add_argument("--raw_elm_manifest_path", required=True)
    parser.add_argument("--backbone_path", required=True)
    parser.add_argument("--model_condition", choices=["untouched_backbone", "checkpoint_8215"], required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--arms", default="correction,fact_only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--n_candidates_per_case", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do_sample", action="store_true", help="Sample candidates instead of deterministic decoding.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--output_stem", default="source_grounded_rescue")
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def load_ledgers(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("generation ledger is empty")
    for row in rows:
        missing = {"case_id", "anchor_id", "facts", "generation_ledger_sha256"}.difference(row)
        if missing:
            raise KeyError(f"generation ledger entry missing: {sorted(missing)}")
        for fact in row["facts"]:
            if set(fact) != {"fact_id", "field", "value"}:
                raise ValueError("generation facts must contain only fact_id, field, and value")
    return rows


def load_model(args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, object]]:
    if args.model_condition == "untouched_backbone":
        model = LlamaForEmbeddingLM.from_pretrained(
            args.backbone_path, torch_dtype=torch.bfloat16, device_map=args.device, low_cpu_mem_usage=True
        )
        return model, {"model_condition": args.model_condition, "checkpoint_path": None}
    if not args.checkpoint_path:
        raise ValueError("checkpoint_8215 condition requires --checkpoint_path")
    model, metadata = load_generation_model(args.checkpoint_path, args.device)
    metadata["model_condition"] = args.model_condition
    metadata["checkpoint_path"] = str(Path(args.checkpoint_path).resolve())
    return model, metadata


def build_prompt(facts: list[dict[str, str]], arm: str, raw_draft: str | None) -> str:
    ledger_json = json.dumps(facts, indent=2, ensure_ascii=True)
    common = (
        "Create a concise synthetic discharge summary from the verified fact ledger below.\n"
        "The ledger is the only factual authority. Do not invent diagnoses, procedures,\n"
        "complications, medications, doses, routes, laboratory values, demographics,\n"
        "disposition, follow-up, or dates. If a detail is absent from the ledger, omit it.\n"
        "For discharge medications, use only ledger medications, list each medication at most once,\n"
        "and never state that the same medication is both continued and discontinued.\n"
        "Use substantive sections only when supported: Discharge Diagnosis, Brief Hospital\n"
        "Course, Major Procedure, Discharge Medications, Disposition, Follow-up, and\n"
        "Instructions. Do not reproduce long source wording verbatim.\n\n"
        "VERIFIED FACT LEDGER:\n"
    )
    if arm == "fact_only":
        return common + ledger_json
    if arm != "correction" or raw_draft is None:
        raise ValueError(f"unsupported arm={arm}")
    correction = (
        "\n\nRAW ELM DRAFT (UNTRUSTED):\n"
        "The draft may describe a different patient. Retain no claim from it unless the\n"
        "verified ledger explicitly supports that claim. Remove unsupported content.\n\n"
    )
    return common + correction + raw_draft + "\n\n" + ledger_json


def generate(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
) -> tuple[str, dict[str, object]]:
    device = next(model.parameters()).device
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    eos_ids = model.config.eos_token_id
    eos_set = {int(eos_ids)} if isinstance(eos_ids, int) else {int(value) for value in eos_ids}
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generation_kwargs: dict[str, object] = {
        "attention_mask": torch.ones_like(input_ids, device=device),
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_ids,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})
    with torch.inference_mode():
        output = model.generate(input_ids=input_ids, **generation_kwargs)[0]
    generated = output[input_ids.shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    last = int(generated[-1].item()) if len(generated) else None
    return text, {
        "prompt_token_count": int(input_ids.shape[1]),
        "generated_token_count": int(len(generated)),
        "max_new_tokens": int(max_new_tokens),
        "hit_max_new_tokens": int(len(generated)) >= max_new_tokens - 1,
        "ended_with_eos": last in eos_set if last is not None else False,
        "last_generated_token_id": last,
        "empty_output": not bool(text),
        "seed": int(seed),
        "do_sample": bool(do_sample),
        "temperature": float(temperature) if do_sample else None,
        "top_p": float(top_p) if do_sample else None,
    }


def main() -> None:
    args = parse_args()
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    if set(arms).difference({"correction", "fact_only"}):
        raise ValueError("--arms may contain only correction,fact_only")
    ledgers = load_ledgers(Path(args.generation_ledger_path).resolve())
    if args.n_candidates_per_case < 1:
        raise ValueError("--n_candidates_per_case must be at least one")
    raw = pd.read_json(Path(args.raw_elm_manifest_path).resolve(), lines=True)
    if raw.anchor_id.duplicated().any():
        raise ValueError("raw ELM manifest must contain one selected note per anchor")
    raw_by_anchor = raw.set_index("anchor_id").to_dict(orient="index")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_path)
    model, model_metadata = load_model(args)
    model.eval()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = str(args.output_stem).strip()
    if not output_stem:
        raise ValueError("--output_stem must not be empty")
    manifest_path = output_dir / f"{output_stem}_manifest.jsonl"
    rows = []
    with manifest_path.open("a" if args.append else "w", encoding="utf-8") as handle:
        for ledger in ledgers:
            anchor_id = str(ledger["anchor_id"])
            raw_row = raw_by_anchor.get(anchor_id)
            if raw_row is None:
                raise ValueError(f"no raw ELM note found for anchor_id={anchor_id}")
            for arm in arms:
                raw_draft = str(raw_row["generated_text"]) if arm == "correction" else None
                prompt = build_prompt(ledger["facts"], arm, raw_draft)
                for candidate_index in range(args.n_candidates_per_case):
                    candidate_seed = int(args.seed + candidate_index)
                    text, metadata = generate(
                        model, tokenizer, prompt, int(args.max_new_tokens), args.do_sample,
                        float(args.temperature), float(args.top_p), candidate_seed,
                    )
                    rescue_id = f"{ledger['case_id']}__{args.model_condition}__{arm}__cand{candidate_index:02d}"
                    record = {
                        "rescue_id": rescue_id,
                        "case_id": ledger["case_id"],
                        "source_review_case_id": ledger.get("source_review_case_id"),
                        "anchor_id": anchor_id,
                        "dataset_row_id": ledger.get("dataset_row_id"),
                        "note_id": ledger.get("note_id"),
                        "review_stratum": ledger.get("review_stratum"),
                        "patient_disjoint_from_train": ledger.get("patient_disjoint_from_train"),
                        "arm": arm,
                        "candidate_index": int(candidate_index),
                        **model_metadata,
                        "generation_ledger_sha256": ledger["generation_ledger_sha256"],
                        "n_generation_facts": len(ledger["facts"]),
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "raw_elm_candidate_id": raw_row.get("candidate_id"),
                        "generated_text": text,
                        **metadata,
                    }
                    handle.write(json.dumps(record) + "\n")
                    rows.append(record)
                    print(json.dumps({key: record[key] for key in ["rescue_id", "generated_token_count", "hit_max_new_tokens", "ended_with_eos", "empty_output"]}), flush=True)
    summary = {
        "n_outputs": len(rows), "n_cases": len(ledgers), "arms": arms,
        "model_condition": args.model_condition, "max_new_tokens": int(args.max_new_tokens),
        "n_candidates_per_case": int(args.n_candidates_per_case), "do_sample": bool(args.do_sample),
    }
    (output_dir / f"{output_stem}_{args.model_condition}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
