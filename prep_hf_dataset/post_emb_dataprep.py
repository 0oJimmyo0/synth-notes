import numpy as np
import pandas as pd
import pickle
import os
import sys
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
import torch

# Add path to minimal_API so pickle files can be loaded
MIMIC_MM_PATH = "/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/MIMIC-MM-Dataset-main"
if MIMIC_MM_PATH not in sys.path:
    sys.path.insert(0, MIMIC_MM_PATH)

try:
    import minimal_API
except ImportError as e:
    print(f"Warning: Could not import minimal_API: {e}")

# Default paths (can be overridden by CLI args)
DEFAULT_BASE_DIR = "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1"
DEFAULT_EMBEDDING_SUBDIR = "embeddings-BAAI-bge-large-en-v1.5"
DEFAULT_PICKLE_SUBDIR = "pickle_ds_note_hadm_all"
DEFAULT_OUTPUT_BASE_DIR = "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data/clinic_notes/2_task"

# MODEL_PATH: Can be either:
#   1. HuggingFace model ID (requires authentication for gated models): 'meta-llama/Llama-3.1-8B-Instruct'
#   2. Local path to downloaded model: '/path/to/local/llama-3.1-8b-instruct'
# For gated models, you need to:
#   - Request access at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
#   - Run: huggingface-cli login
#   - Or download the model locally and use the local path
DEFAULT_MODEL_PATH = 'meta-llama/Llama-3.1-8B-Instruct'
DEFAULT_OUTPUT_SUFFIX = "_full"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_DEV_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_RANDOM_SEED = 42

# Special tokens
emb_token = '<|reserved_special_token_0|>'
gen_token = '<|reserved_special_token_1|>'

# Sequence length configuration
# PRESERVE FULL SEQUENCES: No truncation or chunking during dataset preparation
# Truncation will be handled dynamically by the collate function during training
MAX_TARGET_TOKENS = 100000  # Very high limit to preserve full sequences (truncation happens during training)
MAX_SEQ_LENGTH = 100000     # Not used for truncation, just for reference
USE_CHUNKING = False        # Disable chunking - preserve full sequences
CHUNK_OVERLAP = 0           # Not used when USE_CHUNKING=False

tokenizer = None


def parse_args():
    parser = argparse.ArgumentParser(description="Build train/dev/test HuggingFace datasets from note embeddings")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR,
                        help="Base directory for a dataset version (e.g., mimiciv/3.1 or mimiciii/1.4)")
    parser.add_argument("--embedding_subdir", default=DEFAULT_EMBEDDING_SUBDIR,
                        help="Subdirectory under base_dir that contains sentence_embeddings.npy")
    parser.add_argument("--pickle_subdir", default=DEFAULT_PICKLE_SUBDIR,
                        help="Subdirectory under base_dir that contains patient pickle files")
    parser.add_argument("--embedding_dir", default=None,
                        help="Optional absolute path to embedding directory (overrides --base_dir/--embedding_subdir)")
    parser.add_argument("--pickle_dir", default=None,
                        help="Optional absolute path to pickle directory (overrides --base_dir/--pickle_subdir)")
    parser.add_argument("--output_base_dir", default=DEFAULT_OUTPUT_BASE_DIR,
                        help="Directory to save encoded_training/encoded_dev/encoded_testing datasets")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH,
                        help="Tokenizer model path or HuggingFace model id")
    parser.add_argument("--output_suffix", default=DEFAULT_OUTPUT_SUFFIX,
                        help="Suffix for dataset folders (e.g., _full, _mimic3, or empty string)")
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO,
                        help="Train split ratio")
    parser.add_argument("--dev_ratio", type=float, default=DEFAULT_DEV_RATIO,
                        help="Dev split ratio")
    parser.add_argument("--test_ratio", type=float, default=DEFAULT_TEST_RATIO,
                        help="Test split ratio")
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help="Random seed for split reproducibility")
    return parser.parse_args()


def initialize_tokenizer(model_path):
    global tokenizer
    print(f"Loading tokenizer from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("\n" + "=" * 60)
        print("SOLUTION: Llama 3.1 is a gated model. You need to:")
        print("1. Request access at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        print("2. Authenticate with HuggingFace:")
        print("   huggingface-cli login")
        print("3. Or use a local model path if you have it downloaded")
        print("=" * 60)
        raise
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

def load_embeddings_and_metadata(embedding_dir):
    """Load embeddings and metadata from the BAAI embedding directory."""
    print(f"Loading embeddings from {embedding_dir}...")
    
    embeddings_path = os.path.join(embedding_dir, 'sentence_embeddings.npy')
    metadata_path = os.path.join(embedding_dir, 'sentence_embeddings_metadata.csv')
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    embeddings = np.load(embeddings_path)
    metadata_df = pd.read_csv(metadata_path)
    
    print(f"Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
    print(f"Loaded {len(metadata_df)} metadata records")
    
    # Ensure embeddings and metadata are aligned
    assert embeddings.shape[0] == len(metadata_df), \
        f"Mismatch: {embeddings.shape[0]} embeddings vs {len(metadata_df)} metadata records"
    
    return embeddings, metadata_df

def load_full_clinic_notes(metadata_df, pickle_dir):
    """
    Load full clinic note texts from pickle files based on metadata.
    
    Args:
        metadata_df: DataFrame with columns: filename, note_id, subject_id, hadm_id, etc.
        pickle_dir: Directory containing pickle files
    
    Returns:
        List of full clinic note texts in the same order as metadata_df
    """
    print(f"Loading full clinic note texts from {pickle_dir}...")
    clinic_notes = []
    
    # Group by filename to minimize file I/O
    filename_groups = metadata_df.groupby('filename')
    
    for filename, group in filename_groups:
        filepath = os.path.join(pickle_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            # Add empty strings for missing files
            clinic_notes.extend([''] * len(group))
            continue
        
        try:
            with open(filepath, 'rb') as f:
                patient_obj = pickle.load(f)
            
            # Extract discharge summary notes
            if hasattr(patient_obj, 'dsnotes') and patient_obj.dsnotes is not None:
                if not patient_obj.dsnotes.empty:
                    # Create a mapping from note_id to text
                    note_dict = {}
                    for _, note in patient_obj.dsnotes.iterrows():
                        note_id = str(note.get('note_id', ''))
                        if pd.notna(note.get('text', None)) and str(note.get('text', '')).strip():
                            text = str(note['text']).strip()
                            if len(text) > 50:  # Only include substantial texts
                                note_dict[note_id] = text
                    
                    # Match notes from metadata
                    for _, row in group.iterrows():
                        note_id = str(row.get('note_id', ''))
                        if note_id in note_dict:
                            clinic_notes.append(note_dict[note_id])
                        else:
                            print(f"Warning: Note ID {note_id} not found in {filename}")
                            clinic_notes.append('')
                else:
                    clinic_notes.extend([''] * len(group))
            else:
                clinic_notes.extend([''] * len(group))
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            clinic_notes.extend([''] * len(group))
    
    # Filter out empty notes and corresponding embeddings/metadata
    valid_indices = [i for i, note in enumerate(clinic_notes) if note and len(note.strip()) > 50]
    
    print(f"Loaded {len(clinic_notes)} clinic notes")
    print(f"Valid notes (length > 50): {len(valid_indices)}")
    
    return clinic_notes, valid_indices

def split_data(embeddings, clinic_notes, metadata_df, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Split data into train/dev/test sets.
    
    Args:
        embeddings: numpy array of embeddings
        clinic_notes: list of clinic note texts
        metadata_df: DataFrame with metadata
        train_ratio, dev_ratio, test_ratio: split ratios (should sum to 1.0)
        random_seed: random seed for reproducibility
    
    Returns:
        Tuple of (train, dev, test) splits, each containing (embeddings, notes, metadata)
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    print(f"Splitting data: train={train_ratio}, dev={dev_ratio}, test={test_ratio}")
    
    # First split: train vs (dev + test)
    train_emb, temp_emb, train_notes, temp_notes, train_meta, temp_meta = train_test_split(
        embeddings, clinic_notes, metadata_df,
        test_size=(dev_ratio + test_ratio),
        random_state=random_seed,
        shuffle=True
    )
    
    # Second split: dev vs test
    dev_size = dev_ratio / (dev_ratio + test_ratio)
    dev_emb, test_emb, dev_notes, test_notes, dev_meta, test_meta = train_test_split(
        temp_emb, temp_notes, temp_meta,
        test_size=(1 - dev_size),
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"Train: {len(train_notes)} examples")
    print(f"Dev: {len(dev_notes)} examples")
    print(f"Test: {len(test_notes)} examples")
    
    return (
        (train_emb, train_notes, train_meta),
        (dev_emb, dev_notes, dev_meta),
        (test_emb, test_notes, test_meta)
    )

def embedding2clinicnote_generator(emb, tgts, max_target_tokens=MAX_TARGET_TOKENS, use_chunking=USE_CHUNKING, chunk_overlap=CHUNK_OVERLAP, stats=None):
    """
    This generator generates outputs of clinic note generation task.
    
    If use_chunking=True: Splits long notes into chunks (uses all data)
    If use_chunking=False: Truncates to first max_target_tokens (discards rest)

    emb is a 2D array: (number of notes, size of embeddings)
    tgts is a list of clinic note texts
    max_target_tokens: Maximum tokens per chunk for target text
    use_chunking: If True, split long notes into chunks. If False, truncate.
    chunk_overlap: Number of overlapping tokens between chunks
    stats: Optional dict to collect statistics (mutated in place)
    """
    if stats is None:
        stats = {
            'total': 0,
            'truncated': 0,
            'chunked': 0,
            'skipped_empty': 0,
            'original_lengths': [],
            'chunk_lengths': [],
            'total_chunks': 0
        }
    
    for i, tgt in enumerate(tgts):
        stats['total'] += 1
        
        # Skip empty targets
        if not tgt or len(tgt.strip()) < 50:
            stats['skipped_empty'] += 1
            continue
        
        # Tokenize target text
        target_tokens = tokenizer.encode(tgt, add_special_tokens=False)
        original_target_length = len(target_tokens)
        stats['original_lengths'].append(original_target_length)
        
        if use_chunking:
            # CHUNKING MODE: Split into multiple chunks
            # Each chunk becomes a separate training example with the same embedding
            chunks = []
            start_idx = 0
            
            while start_idx < len(target_tokens):
                # Calculate end index for this chunk
                end_idx = min(start_idx + max_target_tokens, len(target_tokens))
                chunk_tokens = target_tokens[start_idx:end_idx]
                
                # Decode chunk back to text
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
                stats['chunk_lengths'].append(len(chunk_tokens))
                
                # Move to next chunk (with overlap if specified)
                if end_idx >= len(target_tokens):
                    break
                start_idx = end_idx - chunk_overlap
            
            stats['total_chunks'] += len(chunks)
            if len(chunks) > 1:
                stats['chunked'] += 1
            
            # Yield one example per chunk (all with same embedding)
            for chunk_text in chunks:
                chat = [
                    {
                        "role": "user",
                        "content": f"Provide the text of the clinic note {emb_token}"
                    },
                    {
                        "role": "assistant",
                        "content": gen_token + chunk_text
                    },
                ]

                # Apply chat template and convert to tensor
                input_ids = tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt"
                )
                # Remove batch dimension if present
                if input_ids.dim() > 1:
                    input_ids = input_ids[0]
                
                # Verify the final sequence length
                final_length = len(input_ids)
                if final_length > MAX_SEQ_LENGTH:
                    print(f"WARNING: Sample {i} chunk still exceeds MAX_SEQ_LENGTH: {final_length} > {MAX_SEQ_LENGTH}")
                
                yield {
                    "input_ids": input_ids,
                    "domain_embeddings": [torch.Tensor(emb[i])]
                }
        else:
            # NO TRUNCATION MODE: Preserve full sequences
            # Truncation will be handled dynamically by collate function during training
            full_tgt = tgt  # Use full target text without truncation
            
            target_length = len(target_tokens)
            stats['chunk_lengths'].append(target_length)
                
            chat = [
                {
                    "role": "user",
                    "content": f"Provide the text of the clinic note {emb_token}"
                },
                {
                    "role": "assistant",
                    "content": gen_token + full_tgt
                },
            ]

            # Apply chat template and convert to tensor
            input_ids = tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )
            # Remove batch dimension if present
            if input_ids.dim() > 1:
                input_ids = input_ids[0]
            
            # Note: Sequence length may exceed MAX_SEQ_LENGTH, but that's OK
            # The collate function will handle truncation during training
            final_length = len(input_ids)
            if final_length > 1000:  # Only warn for very long sequences
                print(f"INFO: Sample {i} has length {final_length} tokens (will be truncated during training if needed)")
            
            yield {
                "input_ids": input_ids,
                "domain_embeddings": [torch.Tensor(emb[i])]
            }

# Main execution
if __name__ == "__main__":
    args = parse_args()

    # Resolve effective paths
    embedding_dir = args.embedding_dir or os.path.join(args.base_dir, args.embedding_subdir)
    pickle_dir = args.pickle_dir or os.path.join(args.base_dir, args.pickle_subdir)
    output_base_dir = args.output_base_dir
    output_suffix = args.output_suffix

    # Initialize tokenizer
    initialize_tokenizer(args.model_path)

    # Load embeddings and metadata
    embeddings, metadata_df = load_embeddings_and_metadata(embedding_dir)
    
    # Load full clinic note texts
    clinic_notes, valid_indices = load_full_clinic_notes(metadata_df, pickle_dir)
    
    # Filter to valid indices
    if valid_indices:
        embeddings = embeddings[valid_indices]
        clinic_notes = [clinic_notes[i] for i in valid_indices]
        metadata_df = metadata_df.iloc[valid_indices].reset_index(drop=True)
    
    # Split data
    train_data, dev_data, test_data = split_data(
        embeddings, clinic_notes, metadata_df,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    
    train_embeddings, train_notes, train_meta = train_data
    dev_embeddings, dev_notes, dev_meta = dev_data
    test_embeddings, test_notes, test_meta = test_data
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Create training dataset
    print("\nCreating training dataset...")
    if USE_CHUNKING:
        print(f"CHUNKING MODE: Splitting long notes into chunks of {MAX_TARGET_TOKENS} tokens")
        print(f"Each chunk will fit within MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}")
        if CHUNK_OVERLAP > 0:
            print(f"Using {CHUNK_OVERLAP} token overlap between chunks")
    else:
        print(f"FULL SEQUENCE MODE: Preserving full sequences without truncation")
        print(f"Truncation will be handled dynamically by collate function during training (max_seq_length=256)")
    train_stats = {'total': 0, 'truncated': 0, 'chunked': 0, 'skipped_empty': 0, 'original_lengths': [], 'chunk_lengths': [], 'total_chunks': 0}
    embedding2clinic_train_dataset = Dataset.from_generator(
        lambda: embedding2clinicnote_generator(train_embeddings, train_notes, max_target_tokens=MAX_TARGET_TOKENS, use_chunking=USE_CHUNKING, chunk_overlap=CHUNK_OVERLAP, stats=train_stats)
    )
    print(f"We got {len(embedding2clinic_train_dataset)} clinic-note training examples.")
    train_output_path = os.path.join(output_base_dir, f"encoded_training{output_suffix}")
    print(f"Saving training dataset to: {train_output_path}")
    embedding2clinic_train_dataset.save_to_disk(train_output_path)
    
    # Print training statistics
    if train_stats['total'] > 0:
        import numpy as np
        print(f"\n📊 Training Dataset Statistics:")
        print(f"   Total original notes processed: {train_stats['total']}")
        print(f"   Skipped (empty): {train_stats['skipped_empty']}")
        if USE_CHUNKING:
            print(f"   Notes split into chunks: {train_stats['chunked']} ({train_stats['chunked']/train_stats['total']*100:.1f}%)")
            print(f"   Total chunks created: {train_stats['total_chunks']}")
            print(f"   Average chunks per note: {train_stats['total_chunks']/train_stats['total']:.2f}")
        else:
            print(f"   Truncated: {train_stats['truncated']} ({train_stats['truncated']/train_stats['total']*100:.1f}%)")
        if train_stats['original_lengths']:
            print(f"   Original target length - Mean: {np.mean(train_stats['original_lengths']):.1f}, "
                  f"Median: {np.median(train_stats['original_lengths']):.1f}, "
                  f"Max: {np.max(train_stats['original_lengths'])}")
        if train_stats['chunk_lengths']:
            print(f"   Chunk length - Mean: {np.mean(train_stats['chunk_lengths']):.1f}, "
                  f"Median: {np.median(train_stats['chunk_lengths']):.1f}, "
                  f"Max: {np.max(train_stats['chunk_lengths'])}")
    
    # Create dev dataset
    print("\nCreating dev dataset...")
    if USE_CHUNKING:
        print(f"CHUNKING MODE: Splitting long notes into chunks of {MAX_TARGET_TOKENS} tokens")
    else:
        print(f"FULL SEQUENCE MODE: Preserving full sequences without truncation")
    dev_stats = {'total': 0, 'truncated': 0, 'chunked': 0, 'skipped_empty': 0, 'original_lengths': [], 'chunk_lengths': [], 'total_chunks': 0}
    embedding2clinic_dev_dataset = Dataset.from_generator(
        lambda: embedding2clinicnote_generator(dev_embeddings, dev_notes, max_target_tokens=MAX_TARGET_TOKENS, use_chunking=USE_CHUNKING, chunk_overlap=CHUNK_OVERLAP, stats=dev_stats)
    )
    print(f"We got {len(embedding2clinic_dev_dataset)} clinic-note dev examples.")
    dev_output_path = os.path.join(output_base_dir, f"encoded_dev{output_suffix}")
    print(f"Saving dev dataset to: {dev_output_path}")
    embedding2clinic_dev_dataset.save_to_disk(dev_output_path)
    
    # Print dev statistics
    if dev_stats['total'] > 0:
        import numpy as np
        print(f"\n📊 Dev Dataset Statistics:")
        print(f"   Total original notes processed: {dev_stats['total']}")
        print(f"   Skipped (empty): {dev_stats['skipped_empty']}")
        if USE_CHUNKING:
            print(f"   Notes split into chunks: {dev_stats['chunked']} ({dev_stats['chunked']/dev_stats['total']*100:.1f}%)")
            print(f"   Total chunks created: {dev_stats['total_chunks']}")
        else:
            print(f"   Truncated: {dev_stats['truncated']} ({dev_stats['truncated']/dev_stats['total']*100:.1f}%)")
        if dev_stats['original_lengths']:
            print(f"   Original target length - Mean: {np.mean(dev_stats['original_lengths']):.1f}, "
                  f"Median: {np.median(dev_stats['original_lengths']):.1f}, "
                  f"Max: {np.max(dev_stats['original_lengths'])}")
        if dev_stats['chunk_lengths']:
            print(f"   Chunk length - Mean: {np.mean(dev_stats['chunk_lengths']):.1f}, "
                  f"Median: {np.median(dev_stats['chunk_lengths']):.1f}, "
                  f"Max: {np.max(dev_stats['chunk_lengths'])}")
    
    # Create test dataset
    print("\nCreating test dataset...")
    if USE_CHUNKING:
        print(f"CHUNKING MODE: Splitting long notes into chunks of {MAX_TARGET_TOKENS} tokens")
    else:
        print(f"FULL SEQUENCE MODE: Preserving full sequences without truncation")
    test_stats = {'total': 0, 'truncated': 0, 'chunked': 0, 'skipped_empty': 0, 'original_lengths': [], 'chunk_lengths': [], 'total_chunks': 0}
    embedding2clinic_test_dataset = Dataset.from_generator(
        lambda: embedding2clinicnote_generator(test_embeddings, test_notes, max_target_tokens=MAX_TARGET_TOKENS, use_chunking=USE_CHUNKING, chunk_overlap=CHUNK_OVERLAP, stats=test_stats)
    )
    print(f"We got {len(embedding2clinic_test_dataset)} clinic-note test examples.")
    test_output_path = os.path.join(output_base_dir, f"encoded_testing{output_suffix}")
    print(f"Saving test dataset to: {test_output_path}")
    embedding2clinic_test_dataset.save_to_disk(test_output_path)
    
    # Print test statistics
    if test_stats['total'] > 0:
        import numpy as np
        print(f"\n📊 Test Dataset Statistics:")
        print(f"   Total original notes processed: {test_stats['total']}")
        print(f"   Skipped (empty): {test_stats['skipped_empty']}")
        if USE_CHUNKING:
            print(f"   Notes split into chunks: {test_stats['chunked']} ({test_stats['chunked']/test_stats['total']*100:.1f}%)")
            print(f"   Total chunks created: {test_stats['total_chunks']}")
        else:
            print(f"   Truncated: {test_stats['truncated']} ({test_stats['truncated']/test_stats['total']*100:.1f}%)")
        if test_stats['original_lengths']:
            print(f"   Original target length - Mean: {np.mean(test_stats['original_lengths']):.1f}, "
                  f"Median: {np.median(test_stats['original_lengths']):.1f}, "
                  f"Max: {np.max(test_stats['original_lengths'])}")
        if test_stats['chunk_lengths']:
            print(f"   Chunk length - Mean: {np.mean(test_stats['chunk_lengths']):.1f}, "
                  f"Median: {np.median(test_stats['chunk_lengths']):.1f}, "
                  f"Max: {np.max(test_stats['chunk_lengths'])}")
    
    print("\n✅ Dataset preparation complete!")
    print(f"Output saved to: {output_base_dir}")
    if output_suffix:
        print(f"Dataset folders: encoded_training{output_suffix}, encoded_dev{output_suffix}, encoded_testing{output_suffix}")
    else:
        print(f"Dataset folders: encoded_training, encoded_dev, encoded_testing")
