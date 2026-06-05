#!/usr/bin/env python3
"""
Sentence Transformer Embedding Generator for Patient Discharge Notes

This script generates high-quality embeddings for patient discharge summary notes using
sentence transformers. It processes pickle files efficiently with batch processing.

Usage:
    python generate_sentence_embeddings.py --input_dir /path/to/pickle/files --output_dir /path/to/embeddings
"""

import pickle
import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import time
from typing import List, Dict, Tuple
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add path to minimal_API so pickle files can be loaded
MIMIC_MM_PATH = "/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/MIMIC-MM-Dataset-main"
if MIMIC_MM_PATH not in sys.path:
    sys.path.insert(0, MIMIC_MM_PATH)

# Verify minimal_API can be imported (needed for pickle loading)
try:
    import minimal_API
    logger.info(f"Successfully imported minimal_API from {MIMIC_MM_PATH}")
except ImportError as e:
    logger.warning(f"Could not import minimal_API: {e}")
    logger.warning(f"Make sure minimal_API.py is in {MIMIC_MM_PATH}")

# Sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.error("sentence-transformers not installed. Please install it:")
    logger.error("  pip install sentence-transformers")
    exit(1)


class SentenceTransformerEmbedder:
    """Generate embeddings using sentence transformers for patient discharge summary notes."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', device='auto', batch_size=32):
        """
        Initialize the sentence transformer embedder.
        
        Args:
            model_name: Name of the sentence transformer model to use
                       Options: 'all-MiniLM-L6-v2' (fast, 384 dim),
                                'all-mpnet-base-v2' (high quality, 768 dim),
                                'clinicalbert' (medical domain, if available)
            device: Device to use ('auto', 'cpu', 'cuda')
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        # Load the model
        logger.info(f"Loading sentence transformer model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.info("Trying to use 'all-MiniLM-L6-v2' as fallback...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.model_name = 'all-MiniLM-L6-v2'
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        
        # Progress tracking
        self.processed_files = 0
        self.total_texts = 0
        self.start_time = time.time()
    
    def load_patient_data(self, pickle_dir: str, max_files: int = None) -> Tuple[List[str], List[Dict]]:
        """
        Load all patient pickle files and extract discharge summary texts.
        
        Args:
            pickle_dir: Directory containing pickle files
            max_files: Maximum number of files to process (for testing)
        
        Returns:
            texts: List of discharge summary texts
            metadata: List of metadata dictionaries for each text
        """
        texts = []
        metadata = []
        
        pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pkl')]
        
        if max_files:
            pickle_files = pickle_files[:max_files]
            logger.info(f"Processing {len(pickle_files)} files (limited by max_files={max_files})")
        else:
            logger.info(f"Found {len(pickle_files)} pickle files")
        
        for filename in tqdm(pickle_files, desc="Loading patient data"):
            filepath = os.path.join(pickle_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    patient_obj = pickle.load(f)
                
                # Extract discharge summary notes
                if hasattr(patient_obj, 'dsnotes') and patient_obj.dsnotes is not None:
                    if not patient_obj.dsnotes.empty:
                        for _, note in patient_obj.dsnotes.iterrows():
                            if pd.notna(note.get('text', None)) and str(note.get('text', '')).strip():
                                text = str(note['text']).strip()
                                # Only include substantial texts (at least 50 characters)
                                if len(text) > 50:
                                    texts.append(text)
                                    metadata.append({
                                        'filename': filename,
                                        'note_id': str(note.get('note_id', 'N/A')),
                                        'subject_id': str(note.get('subject_id', 'N/A')),
                                        'hadm_id': str(note.get('hadm_id', 'N/A')),
                                        'note_type': str(note.get('note_type', 'DS')),
                                        'charttime': str(note.get('charttime', 'N/A')),
                                        'text_length': len(text),
                                        'text_preview': text[:200] + '...' if len(text) > 200 else text
                                    })
                
                self.processed_files += 1
                
            except Exception as e:
                logger.warning(f"Error processing {filename}: {e}")
                self.processed_files += 1
                continue
        
        self.total_texts = len(texts)
        logger.info(f"Loaded {len(texts)} discharge summary texts from {self.processed_files} files")
        return texts, metadata
    
    def generate_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate sentence transformer embeddings.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        logger.info(f"Using batch size: {self.batch_size}")
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for better similarity computation
            device=self.device
        )
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, metadata: List[Dict], output_dir: str):
        """Save embeddings and metadata to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings as numpy array
        embeddings_path = output_dir / 'sentence_embeddings.npy'
        np.save(embeddings_path, embeddings)
        logger.info(f"Saved embeddings to {embeddings_path}")
        logger.info(f"Embeddings file size: {embeddings_path.stat().st_size / (1024**2):.2f} MB")
        
        # Save metadata as CSV
        metadata_df = pd.DataFrame(metadata)
        metadata_path = output_dir / 'sentence_embeddings_metadata.csv'
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Save embedding info
        num_texts = embeddings.shape[0]
        total_time = time.time() - self.start_time
        info = {
            'model_name': self.model_name,
            'embedding_dim': embeddings.shape[1],
            'num_texts': num_texts,
            'device': self.device,
            'normalized': True,
            'processed_files': self.processed_files,
            'total_time_seconds': total_time,
            'average_time_per_text': total_time / num_texts if num_texts > 0 else 0
        }
        
        info_path = output_dir / 'sentence_embeddings_info.txt'
        with open(info_path, 'w') as f:
            for key, value in info.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Saved embedding info to {info_path}")
        
        return embeddings_path, metadata_path, info_path


def find_similar_notes(embeddings: np.ndarray, metadata: pd.DataFrame, query_idx: int, top_k: int = 5):
    """
    Find similar notes using cosine similarity.
    
    Args:
        embeddings: Array of embeddings
        metadata: DataFrame of metadata
        query_idx: Index of the query note
        top_k: Number of similar notes to return
        
    Returns:
        DataFrame with similar notes and their similarity scores
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    query_embedding = embeddings[query_idx:query_idx+1]
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top-k similar notes (excluding the query itself)
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    results = []
    for idx in similar_indices:
        results.append({
            'index': idx,
            'similarity_score': similarities[idx],
            'subject_id': metadata.iloc[idx]['subject_id'],
            'hadm_id': metadata.iloc[idx]['hadm_id'],
            'text_preview': metadata.iloc[idx].get('text_preview', 'N/A')
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Generate sentence transformer embeddings for patient notes')
    parser.add_argument('--input_dir', required=True,
                       help='Directory containing pickle files')
    parser.add_argument('--output_dir', required=True, 
                       help='Directory to save embeddings')
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                       help='Sentence transformer model to use (default: all-MiniLM-L6-v2)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for computation (default: auto)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for embedding generation (default: 32)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    # Check if sentence-transformers is available
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("sentence-transformers is not installed!")
        logger.error("Please install it with: pip install sentence-transformers")
        return
    
    # Create embedder
    embedder = SentenceTransformerEmbedder(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Load data
    logger.info("Loading patient data...")
    texts, metadata = embedder.load_patient_data(args.input_dir, max_files=args.max_files)
    
    if not texts:
        logger.error("No texts found! Check your input directory.")
        return
    
    # Generate embeddings
    logger.info("Generating sentence transformer embeddings...")
    embeddings = embedder.generate_embeddings(texts, show_progress=True)
    
    # Save results
    logger.info("Saving embeddings...")
    embedder.save_embeddings(embeddings, metadata, args.output_dir)
    
    # Final statistics
    total_time = time.time() - embedder.start_time
    logger.info("=" * 60)
    logger.info("Embedding generation completed!")
    logger.info(f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
    logger.info(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Average processing rate: {embedder.processed_files/total_time:.2f} files/second")
    logger.info(f"Average time per text: {total_time/len(texts):.4f} seconds")
    logger.info("=" * 60)
    
    # Example: Find similar notes
    if len(texts) > 1:
        logger.info("\nExample: Finding similar notes to the first note...")
        metadata_df = pd.DataFrame(metadata)
        similar_notes = find_similar_notes(embeddings, metadata_df, query_idx=0, top_k=3)
        logger.info("\nMost similar notes:")
        for _, row in similar_notes.iterrows():
            logger.info(f"  Score: {row['similarity_score']:.4f}, Subject ID: {row['subject_id']}, HADM ID: {row['hadm_id']}")


if __name__ == "__main__":
    main()

