#!/usr/bin/env python3
"""Nonclinical smoke test for ordinary text generation through ELM model code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

OPEN_ELM_DIR = Path(__file__).resolve().parents[2]
if str(OPEN_ELM_DIR) not in sys.path:
    sys.path.insert(0, str(OPEN_ELM_DIR))

from generate_synthetic_notes import load_generation_model
from src.model import LlamaForEmbeddingLM


PROMPT = 'Return exactly this phrase and nothing else: "text generation path works."'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test ordinary text generation without clinical data.")
    parser.add_argument("--backbone_path", required=True)
    parser.add_argument("--condition", required=True, choices=["untouched_backbone", "checkpoint_8215"])
    parser.add_argument("--checkpoint_path", default=None, required=False)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, str]]:
    if args.condition == "untouched_backbone":
        model = LlamaForEmbeddingLM.from_pretrained(
            args.backbone_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            low_cpu_mem_usage=True,
        )
        return model, {"condition": args.condition, "checkpoint_path": None}
    if not args.checkpoint_path:
        raise ValueError("--checkpoint_path is required for condition=checkpoint_8215")
    model, metadata = load_generation_model(args.checkpoint_path, args.device)
    metadata.update({"condition": args.condition, "checkpoint_path": str(Path(args.checkpoint_path).resolve())})
    return model, metadata


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for this smoke test")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_path)
    model, metadata = load_model(args)
    model.eval()
    device = next(model.parameters()).device
    messages = [{"role": "user", "content": PROMPT}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    eos_ids = model.config.eos_token_id
    eos_id_set = {int(eos_ids)} if isinstance(eos_ids, int) else {int(value) for value in eos_ids}
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=int(args.max_new_tokens),
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0, input_ids.shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    last_token = int(generated_ids[-1].item()) if len(generated_ids) else None
    result = {
        **metadata,
        "prompt": PROMPT,
        "generated_text": text,
        "generated_token_count": int(len(generated_ids)),
        "max_new_tokens": int(args.max_new_tokens),
        "empty_output": not bool(text),
        "contains_expected_phrase": "text generation path works" in text.lower(),
        "prompt_echo_prefix": text.lower().startswith(PROMPT.lower()),
        "hit_max_new_tokens": int(len(generated_ids)) >= int(args.max_new_tokens) - 1,
        "ended_with_eos": last_token in eos_id_set if last_token is not None else False,
        "last_generated_token_id": last_token,
        "device": str(device),
    }
    output = Path(args.output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "generated_text"}, indent=2))


if __name__ == "__main__":
    main()
