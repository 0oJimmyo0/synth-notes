#!/usr/bin/env python3
"""
Initialize ELM model from Llama 3.1
This script creates the initial_elm_model directory that is needed for training.

Usage:
    python initialize_model.py [--output_dir ./initial_elm_model]
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.model import initialize_embedding_model_from_causal_lm, LlamaForEmbeddingLM
from transformers import LlamaForCausalLM, AutoTokenizer
import torch

def main():
    parser = argparse.ArgumentParser(description="Initialize ELM model from Llama 3.1")
    parser.add_argument(
        "--llama_model_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to Llama model (HuggingFace ID or local path)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./initial_elm_model",
        help="Directory to save the initialized ELM model"
    )
    parser.add_argument(
        "--dim_embed_domain",
        type=int,
        default=1024,
        help="Dimension of domain embeddings (should match your embedding dimension)"
    )
    parser.add_argument(
        "--dim_adapter_hidden",
        type=int,
        default=2048,
        help="Hidden dimension of the adapter"
    )
    parser.add_argument(
        "--validate_weights",
        action="store_true",
        default=True,
        help="Validate that ELM model correctly inherited weights from Llama (requires loading original model)"
    )
    parser.add_argument(
        "--skip_weight_validation",
        action="store_true",
        help="Skip weight validation to save memory"
    )
    args = parser.parse_args()
    
    # Handle skip flag
    if args.skip_weight_validation:
        args.validate_weights = False

    print("="*60)
    print("Initializing ELM Model")
    print("="*60)
    print(f"Llama model: {args.llama_model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Embedding dimension: {args.dim_embed_domain}")
    print(f"Adapter hidden dimension: {args.dim_adapter_hidden}")
    print("="*60)
    print()

    # Check if output directory exists
    if os.path.exists(args.output_dir):
        response = input(f"Output directory {args.output_dir} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        print(f"Removing existing directory: {args.output_dir}")
        import shutil
        shutil.rmtree(args.output_dir)

    print("Step 1: Initializing embedding model from pretrained Llama...")
    try:
        model = initialize_embedding_model_from_causal_lm(
            args.llama_model_path,
            dim_embed_domain=args.dim_embed_domain,
            dim_adapter_hidden=args.dim_adapter_hidden
        )
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        print("\nMake sure you have:")
        print("1. Access to the Llama model (authenticated with HuggingFace)")
        print("2. Sufficient GPU memory (8B model requires ~16GB)")
        raise

    print("\nStep 2: Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.llama_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        raise

    print(f"\nStep 3: Saving model and tokenizer to {args.output_dir}...")
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("✓ Model and tokenizer saved successfully")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        raise

    print("\nStep 4: Validating saved model...")
    try:
        loaded_model = LlamaForEmbeddingLM.from_pretrained(
            args.output_dir,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise

    # Step 5: Weight comparison validation (from notebook Cell 6-7)
    if args.validate_weights:
        print("\nStep 5: Validating weights match original Llama model...")
        print("(This loads the original Llama model for comparison)")
        try:
            llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Compare embedding weights
            print("  Comparing embedding weights...")
            embed_match = torch.allclose(
                loaded_model.model.embed_tokens.weight,
                llama_model.model.embed_tokens.weight
            )
            if embed_match:
                print("  ✓ Embedding weights match")
            else:
                print("  ✗ WARNING: Embedding weights do not match!")
            
            # Compare layer weights (sample from first and middle layers)
            print("  Comparing attention layer weights...")
            layer0_q_match = torch.allclose(
                loaded_model.model.layers[0].self_attn.q_proj.weight,
                llama_model.model.layers[0].self_attn.q_proj.weight
            )
            layer0_k_match = torch.allclose(
                loaded_model.model.layers[0].self_attn.k_proj.weight,
                llama_model.model.layers[0].self_attn.k_proj.weight
            )
            
            # Get middle layer index
            num_layers = len(loaded_model.model.layers)
            mid_layer_idx = num_layers // 2
            layer_mid_q_match = torch.allclose(
                loaded_model.model.layers[mid_layer_idx].self_attn.q_proj.weight,
                llama_model.model.layers[mid_layer_idx].self_attn.q_proj.weight
            )
            layer_mid_k_match = torch.allclose(
                loaded_model.model.layers[mid_layer_idx].self_attn.k_proj.weight,
                llama_model.model.layers[mid_layer_idx].self_attn.k_proj.weight
            )
            
            if layer0_q_match and layer0_k_match and layer_mid_q_match and layer_mid_k_match:
                print("  ✓ Attention layer weights match")
            else:
                print("  ✗ WARNING: Some attention layer weights do not match!")
            
            # Overall validation
            all_match = embed_match and layer0_q_match and layer0_k_match and layer_mid_q_match and layer_mid_k_match
            if all_match:
                print("  ✓ All weight validations passed!")
            else:
                print("  ⚠ WARNING: Some weights do not match. This may indicate an issue.")
            
            # Clean up original model to free memory
            del llama_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  ⚠ Weight validation failed: {e}")
            print("  (This is non-critical - model may still be valid)")
    else:
        print("\nStep 5: Skipping weight validation (use --validate_weights to enable)")

    print("\n" + "="*60)
    print("✓ ELM model initialization completed!")
    print(f"Model saved to: {args.output_dir}")
    print("="*60)
    print("\nYou can now use this model for training:")
    print(f"  --checkpoint_path {args.output_dir}")

if __name__ == "__main__":
    main()


