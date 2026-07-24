#!/usr/bin/env python3
"""Verify offline local BF16 MedGemma loading before restricted-data inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify an offline local MedGemma snapshot.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--load_bf16", action="store_true", help="Load full weights with BF16 and device_map=auto.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    required = ("config.json", "tokenizer_config.json")
    missing = [name for name in required if not (model_path / name).is_file()]
    weights = sorted(model_path.glob("*.safetensors")) + sorted(model_path.glob("pytorch_model*.bin"))
    if missing:
        raise FileNotFoundError(f"Missing required model files in {model_path}: {missing}")
    if not weights:
        raise FileNotFoundError(f"No model weight files found in {model_path}")
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if "gemma" not in str(config.model_type).lower():
        raise RuntimeError(f"Expected a Gemma-family model; found model_type={config.model_type!r}")
    report = {
        "model_path": str(model_path),
        "model_type": config.model_type,
        "architectures": getattr(config, "architectures", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "vocab_size": len(tokenizer),
        "weight_file_count": len(weights),
        "weight_bytes": sum(path.stat().st_size for path in weights),
        "offline_load": True,
        "bf16_load_requested": bool(args.load_bf16),
    }
    if args.load_bf16:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the BF16 loading check.")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, dtype=torch.bfloat16, device_map="auto"
        )
        report["bf16_load_succeeded"] = True
        report["device_map"] = str(getattr(model, "hf_device_map", {}))
        report["cuda_device_count"] = torch.cuda.device_count()
        report["cuda_max_memory_allocated_bytes"] = {
            str(index): int(torch.cuda.max_memory_allocated(index)) for index in range(torch.cuda.device_count())
        }
        del model
        torch.cuda.empty_cache()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
