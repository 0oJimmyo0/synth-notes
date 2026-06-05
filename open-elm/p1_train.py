import argparse
import os
import shutil
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
from src.utils import count_trainable_parameters, collate_function_dynamic_padding
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import torch
from functools import partial

def main():
    parser = argparse.ArgumentParser(description="Train a Phase 1 of the embedding language model.")
    parser.add_argument("--datahome", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--eval_steps", type=int, default=100, help="Steps between evaluations")
    parser.add_argument("--save_steps", type=float, default=0.5, help="Steps between saving model checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="initial_elm_model", help="Path to initial model checkpoint")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length (reduce to save memory)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # load datasets
    training_dataset = Dataset.load_from_disk(args.datahome+"encoded_training")
    dev_dataset = Dataset.load_from_disk(args.datahome+"encoded_dev")

    # load pre-trained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    elm = LlamaForEmbeddingLM.from_pretrained(
        args.checkpoint_path, 
        torch_dtype=torch.bfloat16,
        device_map=device)

    print("All parameters that require grad:{}".format(count_trainable_parameters(elm)))

    # freeze everything but the adapter
    for param in elm.model.parameters():
        param.requires_grad = False
    for param in elm.lm_head.parameters():
        param.requires_grad = False
    for param in elm.adapter.parameters():
        param.requires_grad = True

    print("After freezing, now the number of parameters that require grad:{}".format(count_trainable_parameters(elm)))
    
    # Enable gradient checkpointing to save memory
    if hasattr(elm.model, 'gradient_checkpointing_enable'):
        elm.model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled to save GPU memory")
    
    # Calculate training steps
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    num_training_steps = (args.num_train_epochs * len(training_dataset)) // effective_batch_size

    sft_config = SFTConfig(
        output_dir = args.output_dir,
        logging_dir= args.output_dir + "/logs",
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=int(args.num_train_epochs),  # Ensure integer (max_steps will override this)
        save_steps = args.save_steps, 
        max_steps = num_training_steps, 
        eval_steps=args.eval_steps,
        logging_steps = args.eval_steps,
        remove_unused_columns=False,
        max_seq_length=args.max_seq_length,
        bf16=True,
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        dataloader_pin_memory=False,  # Disable pin_memory to save memory
        dataloader_num_workers=0,     # Reduce workers to save memory
        # Use 8-bit optimizer to reduce memory usage (requires bitsandbytes)
        # This can save 2-4GB of GPU memory
        optim="adamw_bnb_8bit" if torch.cuda.is_available() else "adamw_torch",  # 8-bit AdamW optimizer
        save_total_limit=3,  # Keep only the last 3 checkpoints to save disk space (each checkpoint is ~30GB)
    )

    # Create a collate function with max_seq_length bound
    # CRITICAL: This ensures sequences are truncated to max_seq_length to prevent OOM
    collate_fn = partial(collate_function_dynamic_padding, max_seq_length=args.max_seq_length)
    
    trainer = SFTTrainer(
        elm,
        train_dataset=training_dataset,
        eval_dataset=dev_dataset,
        args=sft_config,
        data_collator=collate_fn
    )
    
    # Add callback to periodically clear cache to reduce fragmentation
    # More frequent clearing as training progresses to prevent OOM
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
    
    # When resuming, skip loading optimizer state to save memory (critical for final steps)
    # This will restart optimizer from scratch but preserves model weights
    optimizer_state_backup = None
    scheduler_backup = None
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        print("Note: Optimizer state will be skipped to save memory for final steps")
        
        # Temporarily move optimizer state files to skip loading them
        optimizer_file = os.path.join(resume_from_checkpoint, "optimizer.pt")
        scheduler_file = os.path.join(resume_from_checkpoint, "scheduler.pt")
        
        # Backup and remove optimizer files to prevent loading
        if os.path.exists(optimizer_file):
            optimizer_state_backup = optimizer_file + ".backup"
            shutil.move(optimizer_file, optimizer_state_backup)
            print(f"Moved optimizer state to {optimizer_state_backup} to save memory")
        
        if os.path.exists(scheduler_file):
            scheduler_backup = scheduler_file + ".backup"
            shutil.move(scheduler_file, scheduler_backup)
            print(f"Moved scheduler state to {scheduler_backup} to save memory")
        
        # Clear cache before resuming
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except Exception as e:
        # Restore optimizer files if training fails
        if optimizer_state_backup and os.path.exists(optimizer_state_backup):
            original_file = optimizer_state_backup.replace(".backup", "")
            shutil.move(optimizer_state_backup, original_file)
            print(f"Restored optimizer state file after error: {original_file}")
        if scheduler_backup and os.path.exists(scheduler_backup):
            original_file = scheduler_backup.replace(".backup", "")
            shutil.move(scheduler_backup, original_file)
            print(f"Restored scheduler state file after error: {original_file}")
        raise e
    
    # Restore optimizer state files after training (for future reference)
    if optimizer_state_backup and os.path.exists(optimizer_state_backup):
        original_file = optimizer_state_backup.replace(".backup", "")
        shutil.move(optimizer_state_backup, original_file)
        print(f"Restored optimizer state file: {original_file}")
    if scheduler_backup and os.path.exists(scheduler_backup):
        original_file = scheduler_backup.replace(".backup", "")
        shutil.move(scheduler_backup, original_file)
        print(f"Restored scheduler state file: {original_file}")

if __name__ == "__main__":
    main()