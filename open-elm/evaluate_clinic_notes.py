"""
Semantic consistency verification for the ELM (Embedding Language Model).

This script checks that the ELM reliably maps embeddings → text that preserves
semantics: for each test embedding E, we generate text T with the ELM, then
re-encode T with an external embedding model to get E'. We report cosine
similarity(E, E'). High similarity indicates the model works as expected
(embedding space is preserved in the generated text).
"""
from src.model import LlamaForEmbeddingLM
from src.utils import pairwise_cosine_similarity, batch_inference

import argparse
import os

import torch
import numpy as np

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import Dataset

# Default base dir for data and checkpoints (scratch)
_DEFAULT_BASE = "/gpfs/radev/scratch/xu_hua/shared/data/synthnote/mimiciv/3.1"
_DEFAULT_DATA = os.path.join(_DEFAULT_BASE, "data/clinic_notes/1_task")
_DEFAULT_CHECKPOINT = os.path.join(_DEFAULT_DATA, "elm_training_outputs/filtered_training")

def main():
    parser = argparse.ArgumentParser(
        description='ELM semantic consistency verification: compare input embeddings with re-encoded generated text (cosine similarity).'
    )
    parser.add_argument('--backbone_model_path', type=str, default="initial_elm_model",
                        help='Path to the backbone model')
    parser.add_argument('--checkpoint_path', type=str, default=_DEFAULT_CHECKPOINT,
                        help='Path to the trained checkpoint (directory containing adapter weights)')
    parser.add_argument('--embedding_model_path', type=str, default="BAAI/bge-large-en-v1.5",
                        help='Path to the embedding model used to re-encode generated text (should match training embedding model if possible)')
    parser.add_argument('--test_data_path', type=str,
                        default=os.path.join(_DEFAULT_DATA, "encoded_testing_filtered"),
                        help='Path to the test dataset (HuggingFace dataset directory)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--repetition_penalty', type=float, default=1.2,
                        help='Value for repetition penalty (use > 1.0 to reduce repetition)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda or cpu)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (None = all)')
    
    args = parser.parse_args()
    
    # Load embedding model for evaluation
    print(f"Loading embedding model from {args.embedding_model_path}")
    embedding_model = SentenceTransformer(args.embedding_model_path)

    # Load tokenizer & ELM model
    print(f"Loading backbone model from {args.backbone_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_model_path)

    elm = LlamaForEmbeddingLM.from_pretrained(
        args.backbone_model_path, 
        torch_dtype=torch.bfloat16,
        device_map=args.device)

    # Load trained checkpoint (adapter weights)
    print(f"Loading trained checkpoint from {args.checkpoint_path}")
    # The checkpoint should contain the adapter weights
    # Since we trained with SFTTrainer, the checkpoint contains the full model state
    checkpoint_model = LlamaForEmbeddingLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    
    # Use the checkpoint model for inference
    trained_elm = checkpoint_model
    trained_elm.eval()  # Set to evaluation mode

    # Load test dataset
    print(f"Loading test data from {args.test_data_path}")
    if not os.path.exists(args.test_data_path):
        raise FileNotFoundError(f"Test dataset not found at {args.test_data_path}")
    
    test_dataset = Dataset.load_from_disk(args.test_data_path)
    print(f"Loaded {len(test_dataset)} test samples")
    
    # Limit samples if specified
    if args.max_samples is not None:
        test_dataset = test_dataset.select(range(min(args.max_samples, len(test_dataset))))
        print(f"Evaluating on {len(test_dataset)} samples (limited from original)")
    
    # Extract embeddings and ground truth texts from test dataset
    # The dataset structure has: input_ids, domain_embeddings, labels
    # - domain_embeddings: list with one embedding (torch.Tensor or numpy array)
    # - labels: token IDs for ground truth text (with -100 for ignored tokens)
    print("\nExtracting embeddings and ground truth texts from test dataset...")
    test_embeddings = []
    ground_truth_notes = []
    
    for idx, example in enumerate(test_dataset):
        # Extract domain embeddings
        if "domain_embeddings" in example:
            emb = example["domain_embeddings"]
            if isinstance(emb, list) and len(emb) > 0:
                # Convert to numpy array
                if isinstance(emb[0], torch.Tensor):
                    emb_array = emb[0].cpu().numpy()
                elif isinstance(emb[0], np.ndarray):
                    emb_array = emb[0]
                else:
                    emb_array = np.array(emb[0])
                test_embeddings.append(emb_array)
            else:
                print(f"Warning: Sample {idx} has unexpected embedding format, skipping")
                continue
        else:
            print(f"Warning: Sample {idx} has no domain_embeddings, skipping")
            continue
        
        # Extract ground truth text from labels
        if "labels" in example:
            labels = example["labels"]
            # Handle both list and tensor formats
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().tolist()
            # Filter out -100 (ignored tokens) and decode
            valid_labels = [int(token_id) for token_id in labels if token_id != -100]
            if valid_labels:
                ground_truth_text = tokenizer.decode(valid_labels, skip_special_tokens=True)
                ground_truth_notes.append(ground_truth_text)
            else:
                ground_truth_notes.append("")
        else:
            ground_truth_notes.append("")
        
        # Progress indicator
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(test_dataset)} samples...")
    
    print(f"Extracted {len(test_embeddings)} embeddings and {len(ground_truth_notes)} ground truth notes")
    
    # Semantic consistency: Emb → generated text → re-encode → compare to Emb
    print("\n" + "="*60)
    print("Semantic Consistency Verification: Embedding → Clinic Note → Re-encode")
    print("(Compare input embedding vs embedding of generated text; higher = more reliable ELM)")
    print("="*60)
    
    list_of_cos = []
    number_of_cases = len(test_embeddings)
    
    print(f"Evaluating {number_of_cases} samples in batches of {args.batch_size}...")
    
    for i in range(0, number_of_cases, args.batch_size):
        batch_end = min(i + args.batch_size, number_of_cases)
        batch_embs = test_embeddings[i:batch_end]
        
        # Generate clinic notes from embeddings
        decoded_outputs = batch_inference(
            trained_elm, 
            tokenizer, 
            batch_embs, 
            args.device, 
            task="clinic_note", 
            repetition_penalty=args.repetition_penalty
        )
        
        # Encode generated notes back to embeddings
        decoded_embs = embedding_model.encode(decoded_outputs, show_progress_bar=False)
        
        # Compare original embeddings with generated text embeddings
        coss = pairwise_cosine_similarity(batch_embs, decoded_embs)
        list_of_cos += coss
        
        # Print progress
        if (i // args.batch_size + 1) % 10 == 0:
            current_mean = np.mean(list_of_cos)
            print(f"  Processed {batch_end}/{number_of_cases} samples, Current mean cosine similarity: {current_mean:.4f}")
    
    # Print final results
    mean_cos = float(np.mean(list_of_cos))
    print("\n" + "="*60)
    print("Semantic Consistency Results (input emb vs generated-text emb)")
    print("="*60)
    print(f"Total samples: {len(list_of_cos)}")
    print(f"Mean cosine similarity: {mean_cos:.4f}")
    print(f"Std cosine similarity: {np.std(list_of_cos):.4f}")
    print(f"Min cosine similarity: {np.min(list_of_cos):.4f}")
    print(f"Max cosine similarity: {np.max(list_of_cos):.4f}")
    print(f"Median cosine similarity: {np.median(list_of_cos):.4f}")
    print("="*60)
    if mean_cos >= 0.7:
        print("Interpretation: High consistency — ELM output aligns well with input semantics.")
    elif mean_cos >= 0.5:
        print("Interpretation: Moderate consistency — some semantic preservation; consider tuning.")
    else:
        print("Interpretation: Low consistency — generated text may not reflect input embedding well.")
    
    # Optional: Compare with ground truth notes (if available)
    if ground_truth_notes and any(gt for gt in ground_truth_notes):
        print("\n" + "="*60)
        print("Additional Analysis: Generated vs Ground Truth")
        print("="*60)
        
        # Re-encode ground truth notes
        valid_gt_notes = [gt for gt in ground_truth_notes if gt and len(gt.strip()) > 0]
        if len(valid_gt_notes) == len(list_of_cos):
            gt_embs = embedding_model.encode(valid_gt_notes, show_progress_bar=False)
            gt_coss = pairwise_cosine_similarity(test_embeddings[:len(valid_gt_notes)], gt_embs)
            
            print(f"Ground truth vs original embeddings:")
            print(f"  Mean cosine similarity: {np.mean(gt_coss):.4f}")
            print(f"  Std cosine similarity: {np.std(gt_coss):.4f}")
            print(f"\nGenerated vs Ground Truth:")
            # Compare generated text embeddings with ground truth text embeddings
            generated_embs = embedding_model.encode(
                [decoded_outputs[j] for j in range(len(decoded_outputs)) if j < len(valid_gt_notes)],
                show_progress_bar=False
            )
            gen_gt_coss = pairwise_cosine_similarity(gt_embs[:len(generated_embs)], generated_embs)
            print(f"  Mean cosine similarity: {np.mean(gen_gt_coss):.4f}")
            print(f"  Std cosine similarity: {np.std(gen_gt_coss):.4f}")
        else:
            print(f"Warning: Mismatch in number of ground truth notes ({len(valid_gt_notes)}) vs samples ({len(list_of_cos)})")
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()

