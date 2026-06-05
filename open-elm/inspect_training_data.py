#!/usr/bin/env python3
"""
Inspect the structure of training data to understand how embeddings and text are used.

This script shows:
1. What's in each training example
2. How the input_ids are structured
3. Where the embedding token and generation token are
4. How the model uses both embedding and text during training
"""

import argparse
from datasets import Dataset
from transformers import AutoTokenizer
import torch
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description='Inspect training data structure'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to HuggingFace dataset directory'
    )
    parser.add_argument(
        '--sample_idx',
        type=int,
        default=0,
        help='Index of sample to inspect (default: 0)'
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default='initial_elm_model',
        help='Path to tokenizer (default: initial_elm_model)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Inspecting Training Data Structure")
    print("="*70)
    print(f"Dataset: {args.dataset_path}")
    print(f"Sample index: {args.sample_idx}")
    print("")
    
    # Load dataset
    print("Loading dataset...")
    dataset = Dataset.load_from_disk(args.dataset_path)
    print(f"✓ Loaded dataset with {len(dataset)} samples")
    print("")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print("✓ Loaded tokenizer")
    print("")
    
    # Get sample
    if args.sample_idx >= len(dataset):
        print(f"ERROR: Sample index {args.sample_idx} is out of range (dataset has {len(dataset)} samples)")
        return
    
    example = dataset[args.sample_idx]
    
    print("="*70)
    print("Sample Structure")
    print("="*70)
    print(f"Keys in example: {list(example.keys())}")
    print("")
    
    # Inspect input_ids
    print("="*70)
    print("1. INPUT_IDS (Full Conversation: Prompt + Ground Truth Text)")
    print("="*70)
    input_ids = example["input_ids"]
    
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    elif isinstance(input_ids, list):
        pass
    else:
        input_ids = list(input_ids)
    
    print(f"Length: {len(input_ids)} tokens")
    print("")
    
    # Find special tokens
    EMB_TOKEN = 128002  # <|reserved_special_token_0|>
    GEN_TOKEN = 128003  # <|reserved_special_token_1|>
    
    emb_positions = [i for i, tok in enumerate(input_ids) if tok == EMB_TOKEN]
    gen_positions = [i for i, tok in enumerate(input_ids) if tok == GEN_TOKEN]
    
    print(f"Embedding token ({EMB_TOKEN}) positions: {emb_positions}")
    print(f"Generation token ({GEN_TOKEN}) positions: {gen_positions}")
    print("")
    
    # Decode different parts
    if gen_positions:
        gen_pos = gen_positions[0]
        
        # Part 1: Prompt (before generation token)
        prompt_ids = input_ids[:gen_pos]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        
        # Part 2: Ground truth text (after generation token)
        text_ids = input_ids[gen_pos+1:]
        text_text = tokenizer.decode(text_ids, skip_special_tokens=True)
        
        print("─" * 70)
        print("PROMPT (what model sees as input):")
        print("─" * 70)
        print(prompt_text)
        print("")
        print("─" * 70)
        print("GROUND TRUTH TEXT (what model learns to predict):")
        print("─" * 70)
        print(text_text)
        print("")
        
        # Show token breakdown
        print("─" * 70)
        print("Token Breakdown:")
        print("─" * 70)
        print(f"Tokens 0-{gen_pos-1}: Prompt ({len(prompt_ids)} tokens)")
        print(f"Token {gen_pos}: Generation token (marks where generation starts)")
        print(f"Tokens {gen_pos+1}-{len(input_ids)-1}: Ground truth text ({len(text_ids)} tokens)")
        print("")
        print("During training:")
        print("  - Loss is computed ONLY on tokens after generation token")
        print("  - Tokens before generation token are ignored (label = -100)")
        print("")
    else:
        print("⚠️  Warning: No generation token found in input_ids")
        print("Full decoded text:")
        print(tokenizer.decode(input_ids, skip_special_tokens=True))
        print("")
    
    # Inspect domain_embeddings
    print("="*70)
    print("2. DOMAIN_EMBEDDINGS (Conditioning Input)")
    print("="*70)
    if "domain_embeddings" in example:
        embs = example["domain_embeddings"]
        print(f"Type: {type(embs)}")
        print(f"Length: {len(embs)}")
        
        if len(embs) > 0:
            emb = embs[0]
            print(f"Embedding type (raw): {type(emb)}")
            
            # Debug: show structure if it's nested
            if isinstance(emb, list):
                print(f"Embedding is a list with {len(emb)} elements")
                if len(emb) > 0:
                    print(f"First element type: {type(emb[0])}")
                    if isinstance(emb[0], (list, np.ndarray, torch.Tensor)):
                        print(f"First element shape/length: {getattr(emb[0], 'shape', len(emb[0]) if hasattr(emb[0], '__len__') else 'N/A')}")
            
            # Convert to numpy array, handling different input types
            if isinstance(emb, torch.Tensor):
                emb_array = emb.cpu().numpy()
            elif isinstance(emb, np.ndarray):
                emb_array = emb
            elif isinstance(emb, list):
                # Handle nested lists (e.g., list of lists) or flat list
                emb_array = np.array(emb, dtype=np.float32)
                # If it's a nested list, flatten it
                if emb_array.ndim > 1:
                    print(f"⚠️  Note: Embedding is nested, flattening from shape {emb_array.shape}")
                    emb_array = emb_array.flatten()
            else:
                # Try to convert to numpy array
                try:
                    emb_array = np.array(emb, dtype=np.float32)
                except Exception as e:
                    print(f"⚠️  Warning: Could not convert embedding to numpy array: {e}")
                    print(f"Embedding value (first 10 elements): {emb[:10] if hasattr(emb, '__getitem__') else emb}")
                    return
            
            print(f"Embedding shape: {emb_array.shape}")
            print(f"Embedding dtype: {emb_array.dtype}")
            print(f"Embedding range: [{emb_array.min():.4f}, {emb_array.max():.4f}]")
            print(f"Embedding mean: {emb_array.mean():.4f}")
            print("")
            print("This embedding:")
            print("  - Represents the semantic content of the clinic note")
            print("  - Gets inserted at the embedding token position during forward pass")
            print("  - Conditions the model on what to generate")
        else:
            print("⚠️  Warning: domain_embeddings list is empty")
    else:
        print("⚠️  Warning: No domain_embeddings found in example")
    print("")
    
    # Show how they work together
    print("="*70)
    print("3. HOW THEY WORK TOGETHER DURING TRAINING")
    print("="*70)
    print("")
    print("Step 1: Model receives input_ids and domain_embeddings")
    print("  └─> input_ids contains: [prompt, <emb_token>, <gen_token>, text]")
    print("  └─> domain_embeddings contains: [1024-dim vector]")
    print("")
    print("Step 2: Model processes input_ids token by token")
    print("  └─> When it encounters <emb_token> (128002):")
    print("      └─> Replaces it with the actual embedding vector (via adapter)")
    print("      └─> This embedding 'conditions' the model")
    print("")
    print("Step 3: Model continues processing")
    print("  └─> Processes <gen_token> (128003)")
    print("  └─> Predicts next tokens (the clinic note text)")
    print("")
    print("Step 4: Loss calculation")
    print("  └─> Compares predictions with ground truth text")
    print("  └─> Only computes loss on tokens after <gen_token>")
    print("  └─> Tokens before <gen_token> are ignored (label = -100)")
    print("")
    print("Result: Model learns 'Given this embedding, generate this text'")
    print("")
    
    # Show inference difference
    print("="*70)
    print("4. DIFFERENCE DURING INFERENCE")
    print("="*70)
    print("")
    print("During inference, we only provide:")
    print("  └─> input_ids: [prompt, <emb_token>, <gen_token>]  (NO text!)")
    print("  └─> domain_embeddings: [1024-dim vector]")
    print("")
    print("Model then:")
    print("  └─> Processes prompt and embedding (same as training)")
    print("  └─> Generates tokens one by one (autoregressive)")
    print("  └─> Stops when it generates end-of-text token")
    print("")
    print("Output: Generated clinic note text")
    print("")
    
    print("="*70)
    print("Inspection Complete!")
    print("="*70)

if __name__ == "__main__":
    main()

