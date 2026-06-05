#!/usr/bin/env python3
"""
Simple script to generate synthetic clinic notes from embeddings using your trained ELM model.

Usage:
    python generate_synthetic_notes.py \
        --checkpoint_path /path/to/checkpoint-1746 \
        --embeddings_file embeddings.npy \
        --output_file synthetic_notes.txt

Or with a HuggingFace dataset:
    python generate_synthetic_notes.py \
        --checkpoint_path /path/to/checkpoint-1746 \
        --dataset_path /path/to/encoded_testing_filtered \
        --output_file synthetic_notes.txt
"""

from src.model import LlamaForEmbeddingLM
from src.utils import batch_inference

import argparse
import os
import torch
import numpy as np
import random
from transformers import AutoTokenizer
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic clinic notes from embeddings using trained ELM model'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to the trained checkpoint directory (e.g., checkpoint-1746)'
    )
    parser.add_argument(
        '--backbone_model_path',
        type=str,
        default='initial_elm_model',
        help='Path to the backbone model (default: initial_elm_model)'
    )
    parser.add_argument(
        '--embeddings_file',
        type=str,
        default=None,
        help='Path to .npy file containing embeddings (shape: [N, 1024])'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help='Path to HuggingFace dataset directory (one dataset). Use this or --dataset_paths.'
    )
    parser.add_argument(
        '--dataset_paths',
        type=str,
        default=None,
        help='Comma-separated paths to HuggingFace datasets to generate from (more notes). E.g. path/to/test,path/to/dev,path/to/train'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='synthetic_notes_improved.txt',
        help='Output file to save generated notes (default: synthetic_notes_improved.txt)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for generation (default: 8)'
    )
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.2,
        help='Repetition penalty for generation (default: 1.2, higher = less repetition)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8, higher = more creative/diverse, None = greedy)'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Nucleus sampling parameter (default: 0.9, controls diversity, None = disabled)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Top-k sampling parameter (default: 50, limits vocabulary choices, None = disabled)'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=2048,
        help='Maximum number of new tokens to generate (default: 2048, increase for longer notes)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu, default: cuda)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to generate (None = all, default: None)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.embeddings_file and not args.dataset_path and not args.dataset_paths:
        raise ValueError("Must provide one of: --embeddings_file, --dataset_path, or --dataset_paths")
    
    if args.embeddings_file and (args.dataset_path or args.dataset_paths):
        raise ValueError("Cannot combine --embeddings_file with --dataset_path or --dataset_paths")
    if args.dataset_path and args.dataset_paths:
        raise ValueError("Use either --dataset_path or --dataset_paths, not both")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")

    # Set RNG seeds for reproducible-yet-configurable sampling runs.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("="*60)
    print("Loading Model")
    print("="*60)
    print(f"Backbone model: {args.backbone_model_path}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Seed: {args.seed}")
    print("")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_model_path)
    
    model = LlamaForEmbeddingLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    model.eval()
    print("✓ Model loaded successfully")
    print("")
    
    # Load embeddings
    print("="*60)
    print("Loading Embeddings")
    print("="*60)
    
    embeddings = []
    
    if args.embeddings_file:
        print(f"Loading embeddings from: {args.embeddings_file}")
        emb_array = np.load(args.embeddings_file)
        
        # Handle different shapes
        if emb_array.ndim == 1:
            embeddings = [emb_array]
        elif emb_array.ndim == 2:
            embeddings = [emb_array[i] for i in range(emb_array.shape[0])]
        else:
            raise ValueError(f"Unexpected embedding shape: {emb_array.shape}. Expected [N, 1024] or [1024]")
        
        print(f"✓ Loaded {len(embeddings)} embeddings")
    
    elif args.dataset_path or args.dataset_paths:
        paths = [args.dataset_path] if args.dataset_path else [p.strip() for p in args.dataset_paths.split(",") if p.strip()]
        print(f"Loading dataset(s) from {len(paths)} path(s)...")
        embeddings = []
        for dpath in paths:
            if not os.path.exists(dpath):
                print(f"  Warning: skip (not found): {dpath}")
                continue
            dataset = Dataset.load_from_disk(dpath)
            if args.max_samples and len(embeddings) + len(dataset) > args.max_samples:
                take = args.max_samples - len(embeddings)
                dataset = dataset.select(range(min(take, len(dataset))))
            print(f"  {dpath}: {len(dataset)} rows")
            for idx, example in enumerate(dataset):
                if "domain_embeddings" in example:
                    emb = example["domain_embeddings"]
                    if isinstance(emb, list) and len(emb) > 0:
                        if isinstance(emb[0], torch.Tensor):
                            emb_array = emb[0].cpu().numpy()
                        elif isinstance(emb[0], np.ndarray):
                            emb_array = emb[0]
                        else:
                            emb_array = np.array(emb[0])
                        embeddings.append(emb_array)
                if args.max_samples and len(embeddings) >= args.max_samples:
                    break
            if args.max_samples and len(embeddings) >= args.max_samples:
                break
        print(f"✓ Extracted {len(embeddings)} embeddings")
    
    if len(embeddings) == 0:
        raise ValueError("No embeddings loaded!")
    
    print(f"Embedding dimension: {len(embeddings[0])}")
    print("")
    
    # Generate clinic notes
    print("="*60)
    print("Generating Synthetic Clinic Notes (Improved)")
    print("="*60)
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Generation parameters:")
    print(f"  Repetition penalty: {args.repetition_penalty}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print("")
    
    all_generated_notes = []
    
    # Open file for incremental writing (so we can recover partial results if job times out)
    print("="*60)
    print("Saving Results (Incremental)")
    print("="*60)
    print(f"Output file: {args.output_file}")
    print("")
    
    # Remove existing file if it exists (start fresh)
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    
    note_counter = 0
    
    for i in range(0, len(embeddings), args.batch_size):
        batch_end = min(i + args.batch_size, len(embeddings))
        batch_embs = embeddings[i:batch_end]
        
        # Generate notes with improved parameters
        generated_notes = batch_inference(
            model,
            tokenizer,
            batch_embs,
            args.device,
            task="clinic_note",
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            do_sample=True  # Enable sampling for better variety
        )
        
        all_generated_notes.extend(generated_notes)
        
        # Write batch incrementally to file (so we can recover if job times out)
        with open(args.output_file, 'a', encoding='utf-8') as f:
            for note in generated_notes:
                note_counter += 1
                f.write(f"=== Note {note_counter} ===\n")
                f.write(note)
                f.write("\n\n")
                f.flush()  # Ensure data is written to disk immediately
        
        # Progress update
        print(f"Generated {batch_end}/{len(embeddings)} notes... (saved to file)", end='\r')
    
    print(f"\n✓ Generated and saved {len(all_generated_notes)} synthetic clinic notes")
    print(f"✓ Final file: {args.output_file}")
    print("")
    
    # Print statistics
    note_lengths = [len(note.split()) for note in all_generated_notes]
    print("="*60)
    print("Generation Statistics")
    print("="*60)
    print(f"Total notes: {len(all_generated_notes)}")
    print(f"Average words per note: {sum(note_lengths)/len(note_lengths):.0f}")
    print(f"Min words: {min(note_lengths)}")
    print(f"Max words: {max(note_lengths)}")
    print("")
    
    # Print sample
    print("="*60)
    print("Sample Generated Note (First Note)")
    print("="*60)
    if all_generated_notes:
        sample = all_generated_notes[0]
        if len(sample) > 500:
            print(sample[:500] + "...")
        else:
            print(sample)
        print("")
    
    print("="*60)
    print("Generation Complete!")
    print("="*60)
    print(f"Total notes generated: {len(all_generated_notes)}")
    print(f"Output saved to: {args.output_file}")
    print(f"Compare with previous output: synthetic_notes_from_test.txt")
    print("="*60)

if __name__ == "__main__":
    main()


