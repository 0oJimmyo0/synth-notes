#!/usr/bin/env python3
"""
Script to analyze sequence lengths in the training dataset.
This helps verify that MAX_SEQ_LENGTH is set appropriately.
"""

import argparse
import numpy as np
from datasets import Dataset
import os

def analyze_sequence_lengths(dataset_path, max_samples=None):
    """
    Analyze sequence lengths in a HuggingFace dataset.
    
    Args:
        dataset_path: Path to the HuggingFace dataset directory
        max_samples: Maximum number of samples to analyze (None = all)
    """
    print(f"Loading dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    # Load the dataset
    dataset = Dataset.load_from_disk(dataset_path)
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Limit samples if specified
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Analyzing {len(dataset)} samples (limited from original)")
    
    print("\n" + "="*60)
    print("Analyzing sequence lengths...")
    print("="*60)
    
    # Extract sequence lengths from input_ids
    # IMPORTANT: We need to measure the actual training sequence length
    # The raw input_ids includes: prompt + generation_token (128003) + target
    # During training, the collate function removes the generation token
    # So actual length = prompt_length + target_length (after removing gen token)
    
    raw_sequence_lengths = []  # Raw input_ids length (for reference)
    actual_sequence_lengths = []  # After removing generation token (what training uses)
    prompt_lengths = []  # Length of prompt part
    target_lengths = []  # Length of target part
    valid_samples = 0
    gen_token_id = 128003  # Generation token ID
    
    for idx, example in enumerate(dataset):
        if "input_ids" in example:
            input_ids = example["input_ids"]
            
            # Handle different formats - convert to list
            if isinstance(input_ids, list):
                input_ids_list = input_ids
            elif hasattr(input_ids, 'tolist'):
                input_ids_list = input_ids.tolist()
            elif hasattr(input_ids, 'numpy'):
                input_ids_list = input_ids.numpy().tolist()
            else:
                print(f"Warning: Sample {idx} has unexpected input_ids format, skipping")
                continue
            
            raw_len = len(input_ids_list)
            raw_sequence_lengths.append(raw_len)
            
            # Find generation token position
            try:
                gen_tok_pos = input_ids_list.index(gen_token_id)
            except ValueError:
                print(f"Warning: Sample {idx} has no generation token (128003), skipping")
                continue
            
            # Remove generation token to get actual training sequence
            # This matches what collate_function_dynamic_padding does
            ids_without_gen = input_ids_list[:gen_tok_pos] + input_ids_list[gen_tok_pos+1:]
            actual_len = len(ids_without_gen)
            
            prompt_len = gen_tok_pos  # Everything before gen token is prompt
            target_len = actual_len - prompt_len  # Everything after gen token is target
            
            actual_sequence_lengths.append(actual_len)
            prompt_lengths.append(prompt_len)
            target_lengths.append(target_len)
            valid_samples += 1
            
            # Progress indicator
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1}/{len(dataset)} samples...")
        else:
            print(f"Warning: Sample {idx} has no input_ids field")
    
    if not actual_sequence_lengths:
        print("ERROR: No valid sequences found!")
        return
    
    raw_sequence_lengths = np.array(raw_sequence_lengths)
    actual_sequence_lengths = np.array(actual_sequence_lengths)
    prompt_lengths = np.array(prompt_lengths)
    target_lengths = np.array(target_lengths)
    
    print(f"\nValid samples analyzed: {valid_samples}")
    print("\n" + "="*60)
    print("RAW Sequence Length Statistics (includes generation token)")
    print("="*60)
    print(f"Min: {np.min(raw_sequence_lengths)}, Max: {np.max(raw_sequence_lengths)}")
    print(f"Mean: {np.mean(raw_sequence_lengths):.2f}, Median: {np.median(raw_sequence_lengths):.2f}")
    
    print("\n" + "="*60)
    print("ACTUAL Training Sequence Length Statistics")
    print("(After removing generation token - what training actually uses)")
    print("="*60)
    print(f"Total samples: {len(actual_sequence_lengths)}")
    print(f"Min sequence length: {np.min(actual_sequence_lengths)}")
    print(f"Max sequence length: {np.max(actual_sequence_lengths)}")
    print(f"Mean sequence length: {np.mean(actual_sequence_lengths):.2f}")
    print(f"Median sequence length: {np.median(actual_sequence_lengths):.2f}")
    print(f"Std sequence length: {np.std(actual_sequence_lengths):.2f}")
    print("\nPercentiles:")
    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9, 100]
    for p in percentiles:
        value = np.percentile(actual_sequence_lengths, p)
        print(f"  {p:5.1f}th percentile: {value:.1f}")
    
    print("\n" + "="*60)
    print("Prompt and Target Length Statistics")
    print("="*60)
    print(f"Prompt length - Min: {np.min(prompt_lengths)}, Max: {np.max(prompt_lengths)}")
    print(f"Prompt length - Mean: {np.mean(prompt_lengths):.2f}, Median: {np.median(prompt_lengths):.2f}")
    print(f"Target length - Min: {np.min(target_lengths)}, Max: {np.max(target_lengths)}")
    print(f"Target length - Mean: {np.mean(target_lengths):.2f}, Median: {np.median(target_lengths):.2f}")
    
    print("\n" + "="*60)
    print("Analysis for MAX_SEQ_LENGTH=256")
    print("="*60)
    
    # Count sequences that exceed 256 (actual training length)
    exceeds_256 = np.sum(actual_sequence_lengths > 256)
    exceeds_256_pct = (exceeds_256 / len(actual_sequence_lengths)) * 100
    
    print(f"Sequences > 256: {exceeds_256} ({exceeds_256_pct:.2f}%)")
    print(f"Sequences <= 256: {len(actual_sequence_lengths) - exceeds_256} ({100 - exceeds_256_pct:.2f}%)")
    
    if exceeds_256 > 0:
        print(f"\n⚠️  WARNING: {exceeds_256} sequences exceed MAX_SEQ_LENGTH=256")
        print(f"   These will be truncated during training.")
        print(f"   Max observed length: {np.max(actual_sequence_lengths)}")
        print(f"   Consider increasing MAX_SEQ_LENGTH if this is significant.")
    else:
        print(f"\n✅ All sequences fit within MAX_SEQ_LENGTH=256")
    
    # Show distribution
    print("\n" + "="*60)
    print("Actual Sequence Length Distribution (bins)")
    print("="*60)
    
    # Determine max length first (needed for binning logic)
    max_len = int(np.max(actual_sequence_lengths))
    
    if max_len > 2048:
        print(f"Note: Maximum sequence length is {max_len}, so bins extend beyond 2048 to capture all sequences.")
    
    # Base bins up to 2048
    base_bins = [0, 64, 128, 192, 256, 320, 384, 512, 1024, 2048]
    base_labels = ['0-64', '65-128', '129-192', '193-256', '257-320', '321-384', '385-512', '513-1024', '1025-2048']
    
    # Create additional bins for sequences > 2048 if needed
    
    if max_len > 2048:
        extended_bins = []
        extended_labels = []
        
        # First, add the interval [2048, 4000) explicitly
        extended_bins.append(4000)
        extended_labels.append('2048-3999')
        
        # Create granular bins for 4000-8191 range
        # 4000-5000, 5000-6000, 6000-6500, 6500-7000, 7000-8000, 8000-8191
        granular_start = 4000
        granular_end = 8192  # We'll handle 8191 as the last bin
        
        current = granular_start
        while current < granular_end:
            # Special handling for 6000-7000 range: split into 500-token bins
            if current == 6000:
                # Create 6000-6500 and 6500-7000 bins
                next_bin = 6500
                extended_bins.append(next_bin)
                extended_labels.append(f'{current}-{next_bin}')
                current = next_bin
                # Continue with 6500-7000
                next_bin = 7000
                extended_bins.append(next_bin)
                extended_labels.append(f'{current}-{next_bin}')
                current = next_bin
                continue
            
            # Default: 1000-token intervals for other ranges
            next_bin = min(current + 1000, granular_end)
            if next_bin == granular_end:
                # Last bin in granular range: 8000-8191 (since 8192 is the next power-of-2 boundary)
                extended_bins.append(next_bin)
                extended_labels.append(f'{current}-{next_bin-1}')
            else:
                extended_bins.append(next_bin)
                # Label shows the range boundaries (e.g., "4000-5000" for [4000, 5000))
                # Note: This is [current, next_bin), so values are current <= x < next_bin
                extended_labels.append(f'{current}-{next_bin}')
            current = next_bin
        
        # Continue with powers of 2 starting from 8192 if needed
        if max_len >= 8192:
            # The granular section ends with 8192 as the last boundary
            # extended_bins[-1] == 8192
            current = 8192
            while current <= max_len:
                next_bin = current * 2
                if next_bin > max_len:
                    # This is the last bin needed (e.g., 16384+)
                    # Only add current if it's not already the last boundary
                    if extended_bins[-1] != current:
                        extended_bins.append(current)
                    extended_labels.append(f'{current}+')
                    extended_bins.append(float('inf'))
                    break
                else:
                    # Add intermediate bin (e.g., 8192-16383)
                    # current (8192) is already in extended_bins, so just add next_bin
                    extended_bins.append(next_bin)
                    extended_labels.append(f'{current}-{next_bin-1}')
                    current = next_bin
        else:
            # No sequences >= 8192, just close the last bin
            extended_bins.append(float('inf'))
        
        # Combine base and extended bins
        # base_bins ends with 2048, extended_bins starts with 4000
        bins = base_bins + extended_bins
        bin_labels = base_labels + extended_labels
    else:
        # No sequences > 2048, use base bins only
        bins = base_bins + [float('inf')]
        bin_labels = base_labels + ['>2048']
    
    # Safety check: ensure bins and labels match
    expected_intervals = len(bins) - 1
    num_labels = len(bin_labels)
    
    if expected_intervals != num_labels:
        print(f"WARNING: Mismatch - {expected_intervals} intervals but {num_labels} labels")
        print(f"  This may cause an IndexError. Using minimum to proceed...")
        num_intervals = min(expected_intervals, num_labels)
    else:
        num_intervals = expected_intervals
    
    # Direct verification: count sequences > 16384
    direct_count_above_16384 = np.sum(actual_sequence_lengths > 16384)
    direct_count_at_least_16384 = np.sum(actual_sequence_lengths >= 16384)
    
    for i in range(num_intervals):
        count = np.sum((actual_sequence_lengths > bins[i]) & (actual_sequence_lengths <= bins[i+1]))
        pct = (count / len(actual_sequence_lengths)) * 100
        
        if i < num_labels:
            label = bin_labels[i]
            print(f"  {label:15s}: {count:6d} ({pct:5.2f}%)")
            # Verify "16384+" bin
            if '16384' in label and '+' in label:
                print(f"    [VERIFICATION] Direct count > 16384: {direct_count_above_16384}, bin count: {count}")
                if count != direct_count_above_16384:
                    print(f"    ⚠️  MISMATCH DETECTED! Bin counting may be incorrect.")
        else:
            # Fallback if label is missing
            bin_end = bins[i+1] if bins[i+1] != float('inf') else 'inf'
            print(f"  [{int(bins[i])}-{bin_end}): {count:6d} ({pct:5.2f}%)")
    
    # Show some examples of long sequences
    if exceeds_256 > 0:
        print("\n" + "="*60)
        print("Examples of sequences exceeding 256:")
        print("="*60)
        long_indices = np.where(actual_sequence_lengths > 256)[0]
        for i, idx in enumerate(long_indices[:10]):  # Show first 10
            print(f"  Sample {idx}: actual_length={actual_sequence_lengths[idx]}, "
                  f"prompt={prompt_lengths[idx]}, target={target_lengths[idx]}")
        if len(long_indices) > 10:
            print(f"  ... and {len(long_indices) - 10} more")
    
    print("\n" + "="*60)
    print("Recommendations")
    print("="*60)
    
    max_len = np.max(actual_sequence_lengths)
    p95_len = np.percentile(actual_sequence_lengths, 95)
    p99_len = np.percentile(actual_sequence_lengths, 99)
    
    if max_len <= 256:
        print("✅ MAX_SEQ_LENGTH=256 is sufficient (covers all sequences)")
    elif p99_len <= 256:
        print(f"✅ MAX_SEQ_LENGTH=256 covers 99% of sequences")
        print(f"   Only {exceeds_256} sequences ({exceeds_256_pct:.2f}%) will be truncated")
    elif p95_len <= 256:
        print(f"⚠️  MAX_SEQ_LENGTH=256 covers 95% of sequences")
        print(f"   {exceeds_256} sequences ({exceeds_256_pct:.2f}%) will be truncated")
        print(f"   Consider MAX_SEQ_LENGTH={int(p99_len)} to cover 99% of sequences")
    else:
        print(f"⚠️  MAX_SEQ_LENGTH=256 may be too restrictive")
        print(f"   {exceeds_256} sequences ({exceeds_256_pct:.2f}%) will be truncated")
        print(f"   Consider MAX_SEQ_LENGTH={int(p95_len)} to cover 95% of sequences")
        print(f"   Or MAX_SEQ_LENGTH={int(p99_len)} to cover 99% of sequences")
    
    if max_len > 256:
        print(f"\n   Maximum observed length: {max_len}")
        print(f"   To cover all sequences, you would need MAX_SEQ_LENGTH={max_len}")
        print(f"   (This may cause OOM - use with caution!)")
    
    # Simulate what happens with MAX_SEQ_LENGTH=256
    print("\n" + "="*60)
    print("Simulation: What happens with MAX_SEQ_LENGTH=256 during training")
    print("="*60)
    
    # Simulate the collate function behavior
    truncated_lengths = []
    for i in range(len(actual_sequence_lengths)):
        actual_len = actual_sequence_lengths[i]
        prompt_len = prompt_lengths[i]
        if actual_len > 256:
            # Truncate: keep prompt, truncate target
            available_target = max(0, 256 - prompt_len)
            truncated_len = prompt_len + available_target
        else:
            truncated_len = actual_len
        truncated_lengths.append(truncated_len)
    
    truncated_lengths = np.array(truncated_lengths)
    avg_truncated_target = np.mean([max(0, 256 - p) if a > 256 else (a - p) 
                                     for a, p in zip(actual_sequence_lengths, prompt_lengths)])
    avg_original_target = np.mean(target_lengths)
    target_loss_pct = ((avg_original_target - avg_truncated_target) / avg_original_target) * 100
    
    print(f"Average original target length: {avg_original_target:.1f} tokens")
    print(f"Average truncated target length (with MAX_SEQ_LENGTH=256): {avg_truncated_target:.1f} tokens")
    print(f"Average target loss: {target_loss_pct:.1f}%")
    print(f"\n⚠️  With MAX_SEQ_LENGTH=256, you will lose {target_loss_pct:.1f}% of target text on average")
    
    # Final verification summary
    print("\n" + "="*60)
    print("Verification: Direct Counts (for sequences > 16384)")
    print("="*60)
    direct_above_16384 = np.sum(actual_sequence_lengths > 16384)
    direct_at_least_16384 = np.sum(actual_sequence_lengths >= 16384)
    print(f"Sequences > 16384: {direct_above_16384:,}")
    print(f"Sequences >= 16384: {direct_at_least_16384:,}")
    if direct_above_16384 > 0:
        above_indices = np.where(actual_sequence_lengths > 16384)[0]
        print(f"Indices with length > 16384: {above_indices[:10].tolist()}")
        print(f"Lengths: {actual_sequence_lengths[above_indices[:10]].tolist()}")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Analyze sequence lengths in training dataset')
    parser.add_argument('--dataset_path', type=str,
                        default='/gpfs/radev/scratch/xu_hua/shared/data/synthnote/mimiciv/3.1/data/clinic_notes/1_task/encoded_training_filtered',
                        help='Path to the HuggingFace dataset directory')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to analyze (None = all, useful for quick check)')
    
    args = parser.parse_args()
    
    analyze_sequence_lengths(args.dataset_path, args.max_samples)

if __name__ == "__main__":
    main()

