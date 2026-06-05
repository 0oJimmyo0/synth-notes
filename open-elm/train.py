import argparse
import os
import sys

# Prevent DeepSpeed from being imported (causes CUDA_HOME errors)
# Set environment variables before any imports that might trigger DeepSpeed
os.environ['DS_SKIP_CUDA_CHECK'] = '1'
os.environ['ACCELERATE_USE_DEEPSPEED'] = '0'

# Install import hook to prevent DeepSpeed import
# This prevents accelerate from trying to import DeepSpeed
class BlockDeepSpeedImport:
    def find_spec(self, name, path, target=None):
        if name == 'deepspeed' or name.startswith('deepspeed.'):
            # Return None to prevent import
            return None
        return None

# Add the import hook before any other imports
sys.meta_path.insert(0, BlockDeepSpeedImport())

from src.model import LlamaForEmbeddingLM
from src.utils import collate_function_dynamic_padding
from datasets import Dataset
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from functools import partial
import torch

def main():
    parser = argparse.ArgumentParser(description="Train a embedding language model.")
    parser.add_argument("--datahome", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--eval_steps", type=int, default=100, help="Steps between evaluations")
    parser.add_argument("--save_steps", type=float, default=0.1, help="Steps between saving model checkpoints (if < 1, treated as fraction of total steps; if >= 1, treated as absolute step count)")
    parser.add_argument("--checkpoint_path", type=str, default="initial_elm_model", help="Path to initial model checkpoint")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length (reduce to save memory)")
    parser.add_argument("--truncated", action="store_true", default=False, help="Use truncated datasets (encoded_training, encoded_dev). If False, use full datasets (encoded_training_full, encoded_dev_full)")
    parser.add_argument("--filtered", action="store_true", default=False, help="Use filtered datasets (encoded_training_filtered, encoded_dev_filtered)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Determine dataset suffix
    if args.filtered:
        dataset_suffix = "_filtered"
    elif args.truncated:
        dataset_suffix = ""
    else:
        dataset_suffix = "_full"
    
    # Determine dataset paths and type
    training_path = args.datahome + f"encoded_training{dataset_suffix}"
    dev_path = args.datahome + f"encoded_dev{dataset_suffix}"
    
    if args.filtered:
        dataset_type = "FILTERED (sequences > 8191 removed)"
    elif args.truncated:
        dataset_type = "truncated"
    else:
        dataset_type = "full/untruncated"
    
    print("="*60)
    print("Dataset Configuration")
    print("="*60)
    print(f"Dataset type: {dataset_type}")
    print(f"Dataset suffix: '{dataset_suffix}'")
    print(f"Training dataset path: {training_path}")
    print(f"Dev dataset path: {dev_path}")
    print("="*60)
    print("")
    
    # Load datasets
    print("Loading datasets...")
    training_dataset = Dataset.load_from_disk(training_path)
    dev_dataset = Dataset.load_from_disk(dev_path)
    
    print(f"✓ Loaded training dataset: {len(training_dataset):,} samples")
    print(f"✓ Loaded dev dataset: {len(dev_dataset):,} samples")
    print("")

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    elm = LlamaForEmbeddingLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    # Define PEFT config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj"],
        modules_to_save=["adapter"],
    )

    elm_lora = get_peft_model(elm, peft_config)
    print(elm_lora.print_trainable_parameters())

    # Enable gradient checkpointing to save memory
    if hasattr(elm_lora.model, 'gradient_checkpointing_enable'):
        elm_lora.model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled to save GPU memory")

    # Calculate training steps
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    if effective_batch_size == 0:
        raise ValueError("Effective batch size cannot be zero (batch_size × gradient_accumulation_steps)")
    
    num_training_steps = (args.num_train_epochs * len(training_dataset)) // effective_batch_size
    if num_training_steps == 0:
        raise ValueError(f"Number of training steps is zero. Check dataset size ({len(training_dataset)}) and batch configuration.")

    # Handle save_steps: if < 1, treat as fraction; if >= 1, treat as absolute steps
    if args.save_steps < 1.0:
        # Fraction of total steps (e.g., 0.1 = save every 10% of training)
        save_steps_int = max(1, int(num_training_steps * args.save_steps))
    else:
        # Absolute step count
        save_steps_int = max(1, int(args.save_steps))  # Ensure at least 1

    # Calculate number of checkpoints that will be created (before cleanup)
    num_checkpoints_created = (num_training_steps // save_steps_int) + 1  # +1 for final checkpoint
    max_checkpoints_kept = 3  # From save_total_limit
    
    print("="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Total training steps: {num_training_steps:,}")
    print(f"Effective batch size: {effective_batch_size} (batch_size={args.batch_size} × gradient_accumulation={args.gradient_accumulation_steps})")
    save_percentage = (save_steps_int / num_training_steps * 100) if num_training_steps > 0 else 0
    print(f"Save checkpoints every: {save_steps_int:,} steps ({save_percentage:.1f}% of training)")
    print(f"Total checkpoints that will be created: ~{num_checkpoints_created:,}")
    print(f"Maximum checkpoints kept: {max_checkpoints_kept} (older checkpoints are automatically deleted)")
    print(f"Estimated checkpoint size: ~30GB each")
    print(f"Maximum disk space for checkpoints: ~{max_checkpoints_kept * 30}GB ({max_checkpoints_kept} × 30GB)")
    print("="*60)
    print("")
    
    # Safety warning if too many checkpoints would be created
    if num_checkpoints_created > 100:
        print("⚠️  WARNING: Many checkpoints will be created during training.")
        print(f"   Consider increasing --save_steps to reduce checkpoint frequency.")
        print(f"   Current: {save_steps_int} steps (creates ~{num_checkpoints_created} checkpoints)")
        print(f"   Suggested: --save_steps {max(100, num_training_steps // 20)} (would create ~20 checkpoints)")
        print("")

    # Training arguments
    # CRITICAL: save_total_limit=3 ensures only the last 3 checkpoints are kept
    # This prevents disk space issues. Each checkpoint is ~30GB, so max ~90GB total.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.output_dir + "/logs",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,
        save_steps=save_steps_int,
        max_steps=num_training_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.eval_steps,
        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        dataloader_pin_memory=False,  # Disable pin_memory to save memory
        dataloader_num_workers=0,     # Reduce workers to save memory
        # Use 8-bit optimizer to reduce memory usage (requires bitsandbytes)
        optim="adamw_bnb_8bit" if torch.cuda.is_available() else "adamw_torch",  # 8-bit AdamW optimizer
        save_total_limit=3,  # SAFETY: Keep only the last 3 checkpoints to save disk space (each checkpoint is ~30GB)
        save_strategy="steps",  # Explicitly set to save based on steps
        load_best_model_at_end=False,  # Don't load best model (saves space)
    )

    # Create a collate function with max_seq_length bound
    # CRITICAL: This ensures sequences are truncated to max_seq_length to prevent OOM
    collate_fn = partial(collate_function_dynamic_padding, max_seq_length=args.max_seq_length)

    # Initialize trainer
    trainer = SFTTrainer(
        elm_lora,
        train_dataset=training_dataset,
        eval_dataset=dev_dataset,
        peft_config=peft_config,
        args=training_args,
        data_collator=collate_fn,
        max_seq_length=args.max_seq_length,
    )

    # Add callback to periodically clear cache to reduce fragmentation
    class MemoryClearCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            # Adaptive clearing frequency: more frequent as we approach the end
            # Early steps: every 25 steps, Later steps (>80%): every 10 steps
            total_steps = args.max_steps if hasattr(args, 'max_steps') and args.max_steps else 1000
            progress = state.global_step / total_steps if total_steps > 0 else 0
            
            if progress > 0.8:
                # Final 20% of training: clear every 10 steps
                clear_frequency = 10
            elif progress > 0.5:
                # Middle 30%: clear every 15 steps
                clear_frequency = 15
            else:
                # Early training: clear every 25 steps
                clear_frequency = 25
            
            if state.global_step % clear_frequency == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Log memory usage occasionally for monitoring
                if state.global_step % 100 == 0:
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                        print(f"Step {state.global_step}: GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    trainer.add_callback(MemoryClearCallback())

    # Clear cache before training to reduce memory fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache before training")

    # Resume from checkpoint if provided
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    else:
        resume_from_checkpoint = None

    # Start training
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

if __name__ == "__main__":
    main()