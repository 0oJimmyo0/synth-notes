#!/usr/bin/env python3
"""
Minimal adapter-only calibration training for ELM.

This script:
- loads a calibration HF dataset carrying shifted embeddings
- freezes the base decoder
- trains only `adapter.*`
- optimizes LM loss plus a small adapter anchoring loss

The goal is to keep the repo on the existing ELM path while testing whether a
small adapter recalibration helps shifted embeddings decode more faithfully.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

# Prevent DeepSpeed from being imported in this lightweight path.
os.environ["DS_SKIP_CUDA_CHECK"] = "1"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "0"


class BlockDeepSpeedImport:
    def find_spec(self, name, path, target=None):
        if name == "deepspeed" or name.startswith("deepspeed."):
            return None
        return None


sys.meta_path.insert(0, BlockDeepSpeedImport())

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

THIS_DIR = Path(__file__).resolve().parent
OPEN_ELM_DIR = THIS_DIR.parent.parent
if str(OPEN_ELM_DIR) not in sys.path:
    sys.path.insert(0, str(OPEN_ELM_DIR))

from src.model import LlamaForEmbeddingLM
from src.utils import batch_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a lightweight adapter-only calibration model.")
    parser.add_argument("--train_dataset_path", required=True, help="Calibration HF dataset for training")
    parser.add_argument("--eval_dataset_path", required=True, help="Calibration HF dataset for validation")
    parser.add_argument("--checkpoint_path", required=True, help="Base ELM checkpoint to calibrate from")
    parser.add_argument("--backbone_model_path", required=True, help="Backbone/tokenizer path, e.g. open-elm/initial_elm_model")
    parser.add_argument("--output_dir", required=True, help="Directory to save the calibrated checkpoint")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Adapter calibration learning rate")
    parser.add_argument("--max_steps", type=int, default=1500, help="Maximum optimization steps")
    parser.add_argument("--save_steps", type=int, default=250, help="Checkpoint save frequency")
    parser.add_argument("--eval_steps", type=int, default=250, help="Evaluation frequency")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging frequency")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--anchor_loss_weight", type=float, default=0.05, help="Weight on adapter anchoring loss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validation_probe_size", type=int, default=64, help="Rows to decode+reembed after training (0 disables)")
    parser.add_argument("--validation_probe_batch_size", type=int, default=4, help="Batch size for the decode probe")
    parser.add_argument("--validation_probe_max_new_tokens", type=int, default=1024, help="Max new tokens for the decode probe")
    parser.add_argument("--embedding_model_name", default="BAAI/bge-large-en-v1.5", help="Embedding model for validation probe")
    parser.add_argument("--embedding_device", default="cuda", help="Device for validation probe embedding model")
    return parser


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_local_rank() == 0


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    def _json_default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


def get_git_commit(script_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(script_dir.parent.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "transformers": transformers.__version__,
    }
    try:
        import datasets as datasets_lib

        versions["datasets"] = datasets_lib.__version__
    except Exception:
        pass
    try:
        import sentence_transformers

        versions["sentence_transformers"] = sentence_transformers.__version__
    except Exception:
        pass
    return versions


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return arr / norms


def collate_calibration_batch(examples: list[dict[str, Any]], max_seq_length: int | None = None) -> dict[str, Any]:
    input_ids = []
    labels = []
    source_domain_embeddings = []
    target_prototype_embeddings = []
    shifted_domain_embeddings = []
    row_metadata = []

    max_length_in_batch = max(len(example["input_ids"]) for example in examples) - 1
    max_length = min(max_length_in_batch, max_seq_length) if max_seq_length is not None else max_length_in_batch
    pad_token_id = 128009

    for example in examples:
        gen_tok_pos = example["input_ids"].index(128003)
        ids_without_gen_token = example["input_ids"][:gen_tok_pos] + example["input_ids"][gen_tok_pos + 1 :]

        if max_seq_length is not None and len(ids_without_gen_token) > max_seq_length:
            prompt_length = gen_tok_pos
            available_target_length = max_seq_length - prompt_length
            if available_target_length > 0:
                ids_without_gen_token = ids_without_gen_token[: prompt_length + available_target_length]
            else:
                ids_without_gen_token = ids_without_gen_token[:max_seq_length]
                gen_tok_pos = min(gen_tok_pos, max_seq_length)

        input_ids_padded = torch.full((max_length,), pad_token_id, dtype=torch.long)
        actual_length = min(len(ids_without_gen_token), max_length)
        input_ids_padded[:actual_length] = torch.tensor(ids_without_gen_token[:actual_length], dtype=torch.long)
        input_ids.append(input_ids_padded)

        labels_padded = torch.full((max_length,), -100, dtype=torch.long)
        target_start = min(gen_tok_pos, actual_length)
        target_end = min(actual_length, len(ids_without_gen_token))
        if target_start < target_end:
            labels_padded[target_start:target_end] = input_ids_padded[target_start:target_end]
        labels.append(labels_padded)

        shifted_domain_embeddings.append(torch.tensor(example["domain_embeddings"][0], dtype=torch.float32))
        source_domain_embeddings.append(torch.tensor(example["source_domain_embeddings"], dtype=torch.float32))
        target_prototype_embeddings.append(torch.tensor(example["target_prototype_embedding"], dtype=torch.float32))
        row_metadata.append(
            {
                "dataset_row_id": example.get("dataset_row_id"),
                "source_row_id": example.get("source_row_id"),
                "note_id": example.get("note_id"),
                "subject_id": example.get("subject_id"),
                "hadm_id": example.get("hadm_id"),
                "split": example.get("split"),
                "alpha": example.get("alpha"),
                "axis_label": example.get("axis_label"),
            }
        )

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "domain_embeddings": shifted_domain_embeddings,
        "source_domain_embeddings": torch.stack(source_domain_embeddings),
        "target_prototype_embedding": torch.stack(target_prototype_embeddings),
        "row_metadata": row_metadata,
    }


class AdapterCalibrationTrainer(Trainer):
    def __init__(self, *args, anchor_loss_weight: float, reference_adapter: torch.nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_loss_weight = float(anchor_loss_weight)
        self.reference_adapter = reference_adapter.eval()
        for param in self.reference_adapter.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        source_domain_embeddings = inputs.pop("source_domain_embeddings", None)
        target_prototype_embedding = inputs.pop("target_prototype_embedding", None)
        row_metadata = inputs.pop("row_metadata", None)
        domain_embeddings = inputs["domain_embeddings"]

        outputs = model(**inputs)
        lm_loss = outputs.loss

        domain_batch = torch.stack(domain_embeddings).to(dtype=torch.float32)
        base_model = model.module if hasattr(model, "module") else model
        adapter_dtype = next(base_model.adapter.parameters()).dtype
        current_adapter = base_model.adapter(domain_batch.to(dtype=adapter_dtype)).float()
        with torch.no_grad():
            reference_adapter = self.reference_adapter(domain_batch.to(device=next(self.reference_adapter.parameters()).device)).float()

        anchor_loss = F.mse_loss(current_adapter, reference_adapter)
        total_loss = lm_loss + (self.anchor_loss_weight * anchor_loss)

        self.log(
            {
                "lm_loss": float(lm_loss.detach().cpu()),
                "anchor_loss": float(anchor_loss.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
            }
        )

        if source_domain_embeddings is not None:
            inputs["source_domain_embeddings"] = source_domain_embeddings
        if target_prototype_embedding is not None:
            inputs["target_prototype_embedding"] = target_prototype_embedding
        if row_metadata is not None:
            inputs["row_metadata"] = row_metadata

        return (total_loss, outputs) if return_outputs else total_loss


class MemoryClearCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return
        clear_frequency = 25 if state.global_step < max(args.max_steps * 0.8, 1) else 10
        if state.global_step % clear_frequency == 0:
            torch.cuda.empty_cache()


def freeze_all_but_adapter(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("adapter.")


def summarize_trainable_parameters(model: torch.nn.Module) -> dict[str, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return {"trainable": int(trainable), "total": int(total)}


def run_validation_probe(
    model: LlamaForEmbeddingLM,
    tokenizer: AutoTokenizer,
    eval_dataset: Dataset,
    sample_size: int,
    batch_size: int,
    max_new_tokens: int,
    embedding_model_name: str,
    embedding_device: str,
) -> dict[str, Any]:
    if sample_size <= 0 or len(eval_dataset) == 0:
        return {"enabled": False}
    if not is_main_process():
        return {"enabled": False, "skipped_nonzero_rank": True}

    sample_n = min(sample_size, len(eval_dataset))
    subset = eval_dataset.select(range(sample_n))

    shifted_embeddings = []
    source_embeddings = []
    target_embeddings = []
    for row in subset:
        shifted_embeddings.append(np.asarray(row["domain_embeddings"][0], dtype=np.float32))
        source_embeddings.append(np.asarray(row["source_domain_embeddings"], dtype=np.float32))
        target_embeddings.append(np.asarray(row["target_prototype_embedding"], dtype=np.float32))

    generated_texts: list[str] = []
    infer_model = model.module if hasattr(model, "module") else model
    infer_model.eval()
    generation_device = f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu"
    for start in range(0, sample_n, batch_size):
        batch_embeddings = shifted_embeddings[start : start + batch_size]
        generated_texts.extend(
            batch_inference(
                model=infer_model,
                tokenizer=tokenizer,
                embeddings=batch_embeddings,
                device=generation_device,
                task="clinic_note",
                repetition_penalty=1.2,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        )

    embedder = SentenceTransformer(embedding_model_name, device=embedding_device)
    generated_vecs = embedder.encode(
        generated_texts,
        batch_size=max(batch_size, 1),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    generated_vecs = normalize_rows(generated_vecs)
    source_vecs = normalize_rows(np.stack(source_embeddings, axis=0))
    target_vecs = normalize_rows(np.stack(target_embeddings, axis=0))

    target_cos = np.sum(generated_vecs * target_vecs, axis=1)
    source_cos = np.sum(generated_vecs * source_vecs, axis=1)

    return {
        "enabled": True,
        "sample_size": int(sample_n),
        "mean_target_prototype_cosine": float(np.mean(target_cos)),
        "median_target_prototype_cosine": float(np.median(target_cos)),
        "mean_source_cosine": float(np.mean(source_cos)),
        "median_source_cosine": float(np.median(source_cos)),
        "mean_generated_word_count": float(np.mean([len(text.split()) for text in generated_texts])),
        "nonempty_rate": float(np.mean([bool(text.strip()) for text in generated_texts])),
    }


def main() -> None:
    args = build_parser().parse_args()
    set_random_seed(args.seed)

    local_rank = get_local_rank()
    world_size = get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = Dataset.load_from_disk(args.train_dataset_path)
    eval_dataset = Dataset.load_from_disk(args.eval_dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone_model_path)
    model = LlamaForEmbeddingLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.train()
    freeze_all_but_adapter(model)

    reference_device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    reference_adapter = copy.deepcopy(model.adapter).to(device=reference_device, dtype=torch.float32)
    trainable_summary = summarize_trainable_parameters(model)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(output_dir / "logs"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        save_total_limit=2,
        save_strategy="steps",
        evaluation_strategy="steps",
        logging_strategy="steps",
        report_to=[],
        optim="adamw_torch",
        load_best_model_at_end=False,
        ddp_find_unused_parameters=True if world_size > 1 else None,
    )

    collate_fn = partial(collate_calibration_batch, max_seq_length=args.max_seq_length)
    trainer = AdapterCalibrationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        anchor_loss_weight=args.anchor_loss_weight,
        reference_adapter=reference_adapter,
    )
    trainer.add_callback(MemoryClearCallback())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer.train()
    trainer.save_model(str(output_dir))
    if is_main_process():
        tokenizer.save_pretrained(str(output_dir))

    probe_summary = run_validation_probe(
        model=trainer.model.eval(),
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        sample_size=args.validation_probe_size,
        batch_size=args.validation_probe_batch_size,
        max_new_tokens=args.validation_probe_max_new_tokens,
        embedding_model_name=args.embedding_model_name,
        embedding_device=args.embedding_device,
    )

    run_metadata = {
        "created_at": now_iso(),
        "git_commit": get_git_commit(Path(__file__).resolve().parent),
        "script_path": str(Path(__file__).resolve()),
        "train_dataset_path": str(Path(args.train_dataset_path).resolve()),
        "eval_dataset_path": str(Path(args.eval_dataset_path).resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "backbone_model_path": str(Path(args.backbone_model_path).resolve()),
        "output_dir": str(output_dir),
        "loss": {
            "lm_loss": "teacher-forced next-token cross entropy on shifted embeddings",
            "anchor_loss": "MSE(adapter(shifted_embedding), frozen_adapter(shifted_embedding))",
            "anchor_loss_weight": float(args.anchor_loss_weight),
        },
        "trainable_parameters": trainable_summary,
        "training_args": {
            "batch_size": int(args.batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "learning_rate": float(args.learning_rate),
            "max_steps": int(args.max_steps),
            "save_steps": int(args.save_steps),
            "eval_steps": int(args.eval_steps),
            "max_seq_length": int(args.max_seq_length),
            "seed": int(args.seed),
        },
        "distributed": {
            "world_size": int(world_size),
            "local_rank": int(local_rank),
        },
        "validation_probe": probe_summary,
        "package_versions": get_package_versions(),
    }
    if is_main_process():
        save_json(output_dir / "adapter_calibration_run_metadata.json", run_metadata)

    if is_main_process():
        print("=" * 60)
        print("Adapter calibration complete")
        print("=" * 60)
        print(f"Train rows: {len(train_dataset):,}")
        print(f"Eval rows: {len(eval_dataset):,}")
        print(f"Trainable params: {trainable_summary['trainable']:,} / {trainable_summary['total']:,}")
        print(f"World size: {world_size}")
        print(f"Saved calibrated checkpoint to: {output_dir}")
        if probe_summary.get("enabled"):
            print(f"Validation probe median target cosine: {probe_summary['median_target_prototype_cosine']:.6f}")
            print(f"Validation probe median source cosine: {probe_summary['median_source_cosine']:.6f}")


if __name__ == "__main__":
    main()
