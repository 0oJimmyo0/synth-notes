#!/usr/bin/env python3
"""
Script to filter out sequences with length > 8191 from the dataset.
This helps remove extremely long sequences that may cause issues during training.
"""

import argparse
import os
from datasets import Dataset

def filter_long_sequences(dataset_path, max_length=7148, output_path=None):
    """
    Filter out sequences with actual training length > max_length.
    
    Args:
        dataset_path: Path to the HuggingFace dataset directory
        max_length: Maximum allowed sequence length (default: 8191)
        output_path: Path to save filtered dataset (if None, appends '_filtered' to dataset_path)
    """
    print(f"Loading dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    # Load the dataset
    dataset = Dataset.load_from_disk(dataset_path)
    print(f"Loaded dataset with {len(dataset):,} samples")
    
    # Determine output path
    if output_path is None:
        output_path = dataset_path + "_filtered"
    
    print(f"\nFiltering sequences with length > {max_length}...")
    print("="*60)
    
    # Calculate actual sequence lengths and filter
    gen_token_id = 128003  # Generation token ID
    valid_indices = []
    removed_count = 0
    removed_lengths = []
    skipped_count = 0
    skipped_reasons = {"no_input_ids": 0, "unexpected_format": 0, "no_gen_token": 0}
    
    for idx, example in enumerate(dataset):
        if "input_ids" not in example:
            skipped_count += 1
            skipped_reasons["no_input_ids"] += 1
            # Skip examples without input_ids - they can't be used for training anyway
            if skipped_count <= 10:  # Only print first 10 warnings
                print(f"Warning: Sample {idx} has no input_ids field, skipping (will be excluded from filtered dataset)")
            continue
        
        input_ids = example["input_ids"]
        
        # Handle different formats - convert to list
        if isinstance(input_ids, list):
            input_ids_list = input_ids
        elif hasattr(input_ids, 'tolist'):
            input_ids_list = input_ids.tolist()
        elif hasattr(input_ids, 'numpy'):
            input_ids_list = input_ids.numpy().tolist()
        else:
            skipped_count += 1
            skipped_reasons["unexpected_format"] += 1
            if skipped_count <= 10:  # Only print first 10 warnings
                print(f"Warning: Sample {idx} has unexpected input_ids format, skipping")
            continue
        
        # Find generation token position
        try:
            gen_tok_pos = input_ids_list.index(gen_token_id)
        except ValueError:
            skipped_count += 1
            skipped_reasons["no_gen_token"] += 1
            if skipped_count <= 10:  # Only print first 10 warnings
                print(f"Warning: Sample {idx} has no generation token (128003), skipping")
            continue
        
        # Remove generation token to get actual training sequence
        # This matches what collate_function_dynamic_padding does
        ids_without_gen = input_ids_list[:gen_tok_pos] + input_ids_list[gen_tok_pos+1:]
        actual_len = len(ids_without_gen)
        
        # Keep if within limit
        if actual_len <= max_length:
            valid_indices.append(idx)
        else:
            removed_count += 1
            removed_lengths.append(actual_len)
            # Print details for first few removed sequences
            if removed_count <= 5:
                print(f"  Removing sample {idx}: actual_len={actual_len:,} (raw_len={len(input_ids_list):,}, gen_pos={gen_tok_pos})")
        
        # Progress indicator
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1:,}/{len(dataset):,} samples... "
                  f"(kept: {len(valid_indices):,}, removed: {removed_count:,}, skipped: {skipped_count:,})")
    
    print(f"\nFinished processing all {len(dataset):,} samples")
    print(f"  Kept: {len(valid_indices):,} samples ({100*len(valid_indices)/len(dataset):.2f}%)")
    print(f"  Removed: {removed_count:,} samples ({100*removed_count/len(dataset):.2f}%)")
    print(f"  Skipped: {skipped_count:,} samples ({100*skipped_count/len(dataset):.2f}%)")
    if skipped_count > 0:
        print(f"\nSkipped reasons:")
        for reason, count in skipped_reasons.items():
            if count > 0:
                print(f"  {reason}: {count:,}")
    
    # Verify we processed all examples
    total_processed = len(valid_indices) + removed_count + skipped_count
    if total_processed != len(dataset):
        print(f"\n⚠️  WARNING: Total processed ({total_processed:,}) doesn't match dataset size ({len(dataset):,})!")
        print(f"  Missing: {len(dataset) - total_processed:,} examples")
    
    if removed_lengths:
        import numpy as np
        print(f"\nRemoved sequence length statistics:")
        print(f"  Min: {min(removed_lengths):,}")
        print(f"  Max: {max(removed_lengths):,}")
        print(f"  Mean: {np.mean(removed_lengths):.1f}")
        print(f"  Median: {np.median(removed_lengths):.1f}")
    
    # Additional verification: count all sequences > max_length using numpy
    if len(valid_indices) + removed_count > 0:
        import numpy as np
        all_lengths = []
        for idx in list(valid_indices)[:1000] + [i for i in range(len(dataset)) if i not in valid_indices and i < 1000]:
            if idx < len(dataset):
                example = dataset[idx]
                if "input_ids" in example:
                    input_ids = example["input_ids"]
                    if isinstance(input_ids, list):
                        input_ids_list = input_ids
                    elif hasattr(input_ids, 'tolist'):
                        input_ids_list = input_ids.tolist()
                    elif hasattr(input_ids, 'numpy'):
                        input_ids_list = input_ids.numpy().tolist()
                    else:
                        continue
                    try:
                        gen_tok_pos = input_ids_list.index(gen_token_id)
                        ids_without_gen = input_ids_list[:gen_tok_pos] + input_ids_list[gen_tok_pos+1:]
                        all_lengths.append(len(ids_without_gen))
                    except (ValueError, IndexError):
                        continue
        
        if all_lengths:
            all_lengths = np.array(all_lengths)
            count_above = np.sum(all_lengths > max_length)
            count_at_least = np.sum(all_lengths >= max_length)
            print(f"\nVerification (sample of {len(all_lengths):,} examples):")
            print(f"  Sequences > {max_length}: {count_above:,}")
            print(f"  Sequences >= {max_length}: {count_at_least:,}")
    
    # Create filtered dataset
    print(f"\nCreating filtered dataset...")
    filtered_dataset = dataset.select(valid_indices)
    
    # Save filtered dataset
    print(f"Saving filtered dataset to: {output_path}")
    filtered_dataset.save_to_disk(output_path)
    
    print(f"\n✓ Successfully saved filtered dataset with {len(filtered_dataset):,} samples")
    print(f"  Original: {len(dataset):,} samples")
    print(f"  Filtered: {len(filtered_dataset):,} samples")
    print(f"  Removed: {removed_count:,} samples ({100*removed_count/len(dataset):.2f}%)")
    
    return filtered_dataset

def main():
    parser = argparse.ArgumentParser(
        description='Filter out sequences with length > 8191 from dataset'
    )
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to a specific HuggingFace dataset directory (if None, filters all datasets in base_dir)')
    parser.add_argument('--base_dir', type=str,
                        default='/gpfs/radev/scratch/xu_hua/shared/data/synthnote/mimiciv/3.1/data/clinic_notes/1_task',
                        help='Base directory containing datasets (used if dataset_path is None)')
    parser.add_argument('--dataset_suffix', type=str, default='_full',
                        help='Suffix for dataset names (e.g., "_full" for encoded_training_full)')
    parser.add_argument('--max_length', type=int, default=8191,
                        help='Maximum allowed sequence length (default: 8191)')
    parser.add_argument('--output_suffix', type=str, default='_filtered',
                        help='Suffix to append to output dataset names (default: "_filtered")')
    parser.add_argument('--filter_all', action='store_true',
                        help='Filter all datasets (training, dev, test) in base_dir')
    
    args = parser.parse_args()
    
    if args.dataset_path:
        # Filter a specific dataset
        output_path = args.dataset_path + args.output_suffix
        filter_long_sequences(args.dataset_path, args.max_length, output_path)
    elif args.filter_all:
        # Filter all datasets in base_dir
        # Base names without suffix
        base_names = ["encoded_training", "encoded_dev", "encoded_testing"]
        
        for base_name in base_names:
            # Input dataset with suffix (e.g., encoded_training_full)
            input_dataset_name = f"{base_name}{args.dataset_suffix}"
            dataset_path = os.path.join(args.base_dir, input_dataset_name)
            
            if os.path.exists(dataset_path):
                # Output dataset: remove suffix, add _filtered (e.g., encoded_training_filtered)
                output_dataset_name = f"{base_name}_filtered"
                output_path = os.path.join(args.base_dir, output_dataset_name)
                
                print(f"\n{'='*60}")
                print(f"Filtering dataset: {input_dataset_name}")
                print(f"Output will be saved to: {output_dataset_name}")
                print(f"{'='*60}")
                filter_long_sequences(dataset_path, args.max_length, output_path)
            else:
                print(f"\nSkipping {input_dataset_name} (not found at {dataset_path})")
    else:
        # Default: filter training dataset
        dataset_path = os.path.join(args.base_dir, f"encoded_training{args.dataset_suffix}")
        output_path = dataset_path + args.output_suffix
        filter_long_sequences(dataset_path, args.max_length, output_path)

if __name__ == "__main__":
    main()

