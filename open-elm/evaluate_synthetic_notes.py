#!/usr/bin/env python3
"""
Evaluation Framework for Synthetic Clinical Notes

This script implements a comprehensive evaluation plan:
1. Choose 2 specific tasks well-adapted for discharge notes
2. Test 3-4 baseline models on real notes (baseline performance)
3. Test same models on synthetic notes (synthetic performance)
4. Train models with real+synthetic data and compare with real-only training

Tasks:
- Task 1: Named Entity Recognition (NER) - Information extraction, no labels needed
- Task 2: Readmission Prediction - Binary classification, requires labels from MIMIC-IV

Models:
- Baseline 1: Logistic Regression with TF-IDF
- Baseline 2: Random Forest with TF-IDF
- Baseline 3: Bio_ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT)
- Baseline 4: XGBoost with TF-IDF (for classification tasks)
"""

import argparse
import os
import json
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, accuracy_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Bio_ClinicalBERT models will be skipped.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not available. XGBoost models will be skipped.")


def parse_notes_file(filepath: str) -> List[str]:
    """Parse notes from file (assumes format: === Note N ===\n<note text>\n\n)."""
    notes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'=== Note \d+ ===\n(.*?)(?=\n=== Note |\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        note = match.strip()
        if note and len(note) > 100:
            notes.append(note)
    
    return notes


def load_mimic_discharge_notes(discharge_csv_path: str, max_notes: Optional[int] = None) -> List[str]:
    """Load discharge notes from MIMIC-IV discharge.csv file."""
    print(f"Loading MIMIC-IV discharge notes from: {discharge_csv_path}")
    
    # Read CSV in chunks if it's large
    chunks = []
    chunk_size = 10000
    
    for chunk in pd.read_csv(discharge_csv_path, chunksize=chunk_size, dtype=str):
        if 'text' in chunk.columns:
            notes = chunk['text'].dropna().tolist()
            chunks.extend(notes)
        elif 'note_text' in chunk.columns:
            notes = chunk['note_text'].dropna().tolist()
            chunks.extend(notes)
        else:
            # Try to find text column
            text_cols = [col for col in chunk.columns if 'text' in col.lower() or 'note' in col.lower()]
            if text_cols:
                notes = chunk[text_cols[0]].dropna().tolist()
                chunks.extend(notes)
        
        if max_notes and len(chunks) >= max_notes:
            chunks = chunks[:max_notes]
            break
    
    print(f"✓ Loaded {len(chunks)} discharge notes")
    return chunks


def load_readmission_labels(admissions_csv_path: str, discharge_csv_path: str) -> Dict[str, bool]:
    """
    Load readmission labels from MIMIC-IV admissions table.
    Returns a dictionary mapping note_id or hadm_id to 30-day readmission status.
    """
    print(f"Loading readmission labels from: {admissions_csv_path}")
    
    # Read admissions table
    admissions_df = pd.read_csv(admissions_csv_path, dtype=str)
    
    # Read discharge notes to get hadm_id mapping
    discharge_df = pd.read_csv(discharge_csv_path, dtype=str, nrows=1000)  # Sample to check structure
    
    # This is a simplified version - you may need to adjust based on actual MIMIC-IV structure
    # The actual implementation depends on how notes are linked to admissions
    labels = {}
    
    print("⚠️  Note: Readmission label loading needs to be customized based on MIMIC-IV schema")
    print("    This requires linking discharge notes to admissions table via hadm_id")
    
    return labels


# ============================================================================
# TASK 1: Named Entity Recognition (NER)
# ============================================================================

def extract_entities_rule_based(note: str) -> Dict[str, List[str]]:
    """Extract medical entities using rule-based patterns."""
    entities = {
        'diagnoses': [],
        'medications': [],
        'procedures': [],
        'lab_values': []
    }
    
    # Diagnosis patterns
    diagnosis_patterns = [
        r'\b(hypertension|HTN)\b',
        r'\b(diabetes|DM|diabetes mellitus)\b',
        r'\b(pneumonia)\b',
        r'\b(sepsis)\b',
        r'\b(stroke|CVA)\b',
        r'\b(MI|myocardial infarction|heart attack)\b',
        r'\b(heart failure|CHF|congestive heart failure)\b',
        r'\b(CKD|chronic kidney disease)\b',
        r'\b(COPD|chronic obstructive pulmonary disease)\b',
    ]
    
    # Medication patterns
    medication_patterns = [
        r'\b(aspirin|ASA)\b',
        r'\b(insulin)\b',
        r'\b(warfarin|coumadin)\b',
        r'\b(metformin)\b',
        r'\b(lisinopril)\b',
        r'\b(atorvastatin|lipitor)\b',
        r'\b(amlodipine|norvasc)\b',
    ]
    
    # Procedure patterns
    procedure_patterns = [
        r'\b(surgery|surgical procedure)\b',
        r'\b(endoscopy)\b',
        r'\b(catheter|catheterization)\b',
        r'\b(intubation)\b',
        r'\b(dialysis)\b',
        r'\b(biopsy)\b',
    ]
    
    # Lab value patterns
    lab_patterns = [
        r'\b(WBC|white blood cell)\b',
        r'\b(HCT|hematocrit)\b',
        r'\b(BUN|blood urea nitrogen)\b',
        r'\b(Cr|creatinine)\b',
        r'\b(Na|sodium)\b',
        r'\b(K|potassium)\b',
        r'\b(Cl|chloride)\b',
        r'\b(glucose)\b',
        r'\b(hemoglobin|Hgb)\b',
        r'\b(platelet)\b',
    ]
    
    for pattern in diagnosis_patterns:
        matches = re.findall(pattern, note, re.IGNORECASE)
        entities['diagnoses'].extend(matches)
    
    for pattern in medication_patterns:
        matches = re.findall(pattern, note, re.IGNORECASE)
        entities['medications'].extend(matches)
    
    for pattern in procedure_patterns:
        matches = re.findall(pattern, note, re.IGNORECASE)
        entities['procedures'].extend(matches)
    
    for pattern in lab_patterns:
        matches = re.findall(pattern, note, re.IGNORECASE)
        entities['lab_values'].extend(matches)
    
    # Remove duplicates while preserving order
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))
    
    return entities


def evaluate_ner_task(real_notes: List[str], synthetic_notes: List[str], 
                     model_name: str = "rule_based") -> Dict:
    """
    Evaluate NER task on real and synthetic notes.
    Returns metrics comparing entity extraction performance.
    """
    print(f"\n{'='*70}")
    print(f"TASK 1: Named Entity Recognition - Model: {model_name}")
    print(f"{'='*70}")
    
    # Extract entities from real notes
    print("Extracting entities from real notes...")
    real_entities = [extract_entities_rule_based(note) for note in real_notes]
    
    # Extract entities from synthetic notes
    print("Extracting entities from synthetic notes...")
    synthetic_entities = [extract_entities_rule_based(note) for note in synthetic_notes]
    
    # Calculate statistics
    def calculate_stats(entities_list: List[Dict]) -> Dict:
        stats = {
            'avg_diagnoses': np.mean([len(e['diagnoses']) for e in entities_list]),
            'avg_medications': np.mean([len(e['medications']) for e in entities_list]),
            'avg_procedures': np.mean([len(e['procedures']) for e in entities_list]),
            'avg_lab_values': np.mean([len(e['lab_values']) for e in entities_list]),
            'total_entities': np.mean([sum(len(v) for v in e.values()) for e in entities_list]),
            'notes_with_diagnoses': sum(1 for e in entities_list if e['diagnoses']),
            'notes_with_medications': sum(1 for e in entities_list if e['medications']),
            'notes_with_procedures': sum(1 for e in entities_list if e['procedures']),
            'notes_with_labs': sum(1 for e in entities_list if e['lab_values']),
        }
        return stats
    
    real_stats = calculate_stats(real_entities)
    synthetic_stats = calculate_stats(synthetic_entities)
    
    # Calculate ratios
    ratios = {}
    for key in real_stats:
        if real_stats[key] > 0:
            ratios[key] = synthetic_stats[key] / real_stats[key]
        else:
            ratios[key] = 0.0
    
    results = {
        'model': model_name,
        'real_stats': real_stats,
        'synthetic_stats': synthetic_stats,
        'ratios': ratios,
        'real_notes_count': len(real_notes),
        'synthetic_notes_count': len(synthetic_notes)
    }
    
    # Print results
    print(f"\nReal Notes Statistics:")
    print(f"  Avg diagnoses per note: {real_stats['avg_diagnoses']:.2f}")
    print(f"  Avg medications per note: {real_stats['avg_medications']:.2f}")
    print(f"  Avg procedures per note: {real_stats['avg_procedures']:.2f}")
    print(f"  Avg lab values per note: {real_stats['avg_lab_values']:.2f}")
    print(f"  Total entities per note: {real_stats['total_entities']:.2f}")
    
    print(f"\nSynthetic Notes Statistics:")
    print(f"  Avg diagnoses per note: {synthetic_stats['avg_diagnoses']:.2f}")
    print(f"  Avg medications per note: {synthetic_stats['avg_medications']:.2f}")
    print(f"  Avg procedures per note: {synthetic_stats['avg_procedures']:.2f}")
    print(f"  Avg lab values per note: {synthetic_stats['avg_lab_values']:.2f}")
    print(f"  Total entities per note: {synthetic_stats['total_entities']:.2f}")
    
    print(f"\nRatios (Synthetic / Real):")
    print(f"  Diagnoses: {ratios['avg_diagnoses']:.2%}")
    print(f"  Medications: {ratios['avg_medications']:.2%}")
    print(f"  Procedures: {ratios['avg_procedures']:.2%}")
    print(f"  Lab values: {ratios['avg_lab_values']:.2%}")
    print(f"  Total entities: {ratios['total_entities']:.2%}")
    
    return results


# ============================================================================
# TASK 2: Readmission Prediction
# ============================================================================

def prepare_readmission_data(notes: List[str], labels: Optional[List[int]] = None) -> Tuple:
    """
    Prepare data for readmission prediction.
    If labels are None, creates dummy labels for demonstration.
    """
    if labels is None:
        # Create dummy labels (50% readmission rate for demonstration)
        labels = np.random.randint(0, 2, size=len(notes)).tolist()
        print("⚠️  Using dummy labels for demonstration. Provide real labels from MIMIC-IV for actual evaluation.")
    
    return notes, labels


def train_baseline_models(X_train: List[str], y_train: List[int], 
                         X_test: List[str], y_test: List[int],
                         model_types: List[str] = ['lr', 'rf', 'xgb']) -> Dict:
    """
    Train baseline models for readmission prediction.
    
    Args:
        X_train: Training notes
        y_train: Training labels
        X_test: Test notes
        y_test: Test labels
        model_types: List of model types to train ['lr', 'rf', 'xgb', 'bert']
    
    Returns:
        Dictionary with model performance metrics
    """
    results = {}
    
    # TF-IDF vectorization
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Logistic Regression
    if 'lr' in model_types:
        print("\nTraining Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train_tfidf, y_train)
        y_pred = lr_model.predict(X_test_tfidf)
        y_pred_proba = lr_model.predict_proba(X_test_tfidf)[:, 1]
        
        results['logistic_regression'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.0
        }
        print(f"  Accuracy: {results['logistic_regression']['accuracy']:.4f}")
        print(f"  F1: {results['logistic_regression']['f1']:.4f}")
        print(f"  AUC-ROC: {results['logistic_regression']['auc_roc']:.4f}")
    
    # Random Forest
    if 'rf' in model_types:
        print("\nTraining Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_tfidf, y_train)
        y_pred = rf_model.predict(X_test_tfidf)
        y_pred_proba = rf_model.predict_proba(X_test_tfidf)[:, 1]
        
        results['random_forest'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.0
        }
        print(f"  Accuracy: {results['random_forest']['accuracy']:.4f}")
        print(f"  F1: {results['random_forest']['f1']:.4f}")
        print(f"  AUC-ROC: {results['random_forest']['auc_roc']:.4f}")
    
    # XGBoost
    if 'xgb' in model_types and XGBOOST_AVAILABLE:
        print("\nTraining XGBoost...")
        xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        xgb_model.fit(X_train_tfidf, y_train)
        y_pred = xgb_model.predict(X_test_tfidf)
        y_pred_proba = xgb_model.predict_proba(X_test_tfidf)[:, 1]
        
        results['xgboost'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.0
        }
        print(f"  Accuracy: {results['xgboost']['accuracy']:.4f}")
        print(f"  F1: {results['xgboost']['f1']:.4f}")
        print(f"  AUC-ROC: {results['xgboost']['auc_roc']:.4f}")
    
    return results


def evaluate_readmission_task(real_notes: List[str], synthetic_notes: List[str],
                             real_labels: Optional[List[int]] = None,
                             synthetic_labels: Optional[List[int]] = None,
                             test_size: float = 0.2) -> Dict:
    """
    Evaluate readmission prediction task.
    
    Evaluation plan:
    1. Train models on real notes (train set)
    2. Test on real notes (test set) -> baseline performance
    3. Test on synthetic notes (same test set indices) -> synthetic performance
    4. Train on real+synthetic notes -> compare with real-only
    """
    print(f"\n{'='*70}")
    print(f"TASK 2: Readmission Prediction")
    print(f"{'='*70}")
    
    # Prepare data
    real_notes, real_labels = prepare_readmission_data(real_notes, real_labels)
    synthetic_notes, synthetic_labels = prepare_readmission_data(synthetic_notes, synthetic_labels)
    
    # Ensure same size for fair comparison
    min_size = min(len(real_notes), len(synthetic_notes))
    real_notes = real_notes[:min_size]
    synthetic_notes = synthetic_notes[:min_size]
    real_labels = real_labels[:min_size]
    synthetic_labels = synthetic_labels[:min_size]
    
    print(f"Using {min_size} notes for evaluation")
    
    # Split real notes into train/test
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        real_notes, real_labels, test_size=test_size, random_state=42, stratify=real_labels
    )
    
    # Use same indices for synthetic notes
    train_indices, test_indices = train_test_split(
        range(len(synthetic_notes)), test_size=test_size, random_state=42, stratify=synthetic_labels
    )
    X_train_synthetic = [synthetic_notes[i] for i in train_indices]
    X_test_synthetic = [synthetic_notes[i] for i in test_indices]
    y_train_synthetic = [synthetic_labels[i] for i in train_indices]
    y_test_synthetic = [synthetic_labels[i] for i in test_indices]
    
    # Model types to train
    model_types = ['lr', 'rf']
    if XGBOOST_AVAILABLE:
        model_types.append('xgb')
    
    # 1. Train on real notes, test on real notes (baseline)
    print("\n" + "-"*70)
    print("1. Training on REAL notes, testing on REAL notes (BASELINE)")
    print("-"*70)
    baseline_results = train_baseline_models(
        X_train_real, y_train_real, X_test_real, y_test_real, model_types
    )
    
    # 2. Test trained models on synthetic notes
    print("\n" + "-"*70)
    print("2. Testing trained models on SYNTHETIC notes")
    print("-"*70)
    synthetic_results = train_baseline_models(
        X_train_real, y_train_real, X_test_synthetic, y_test_synthetic, model_types
    )
    
    # 3. Train on real+synthetic, test on real notes
    print("\n" + "-"*70)
    print("3. Training on REAL+SYNTHETIC notes, testing on REAL notes")
    print("-"*70)
    X_train_combined = X_train_real + X_train_synthetic
    y_train_combined = y_train_real + y_train_synthetic
    combined_results = train_baseline_models(
        X_train_combined, y_train_combined, X_test_real, y_test_real, model_types
    )
    
    # Compile results
    results = {
        'baseline_real': baseline_results,
        'synthetic_test': synthetic_results,
        'combined_training': combined_results,
        'train_size_real': len(X_train_real),
        'train_size_combined': len(X_train_combined),
        'test_size': len(X_test_real)
    }
    
    # Print comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"\n{'Model':<20} {'Metric':<15} {'Real→Real':<12} {'Real→Synth':<12} {'Real+Synth→Real':<15}")
    print("-"*70)
    
    for model_name in baseline_results.keys():
        for metric in ['accuracy', 'f1', 'auc_roc']:
            baseline_val = baseline_results[model_name].get(metric, 0)
            synthetic_val = synthetic_results[model_name].get(metric, 0)
            combined_val = combined_results[model_name].get(metric, 0)
            print(f"{model_name:<20} {metric:<15} {baseline_val:<12.4f} {synthetic_val:<12.4f} {combined_val:<15.4f}")
    
    return results


# ============================================================================
# Main Evaluation Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate synthetic clinical notes using downstream tasks'
    )
    parser.add_argument(
        '--real_notes_path',
        type=str,
        help='Path to real discharge notes (CSV file or text file)'
    )
    parser.add_argument(
        '--synthetic_notes_path',
        type=str,
        required=True,
        help='Path to synthetic notes file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--max_notes',
        type=int,
        default=None,
        help='Maximum number of notes to evaluate (for faster processing)'
    )
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default=['ner', 'readmission'],
        choices=['ner', 'readmission'],
        help='Tasks to evaluate'
    )
    parser.add_argument(
        '--admissions_csv',
        type=str,
        help='Path to MIMIC-IV admissions.csv for readmission labels'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("SYNTHETIC CLINICAL NOTES EVALUATION")
    print("="*70)
    
    # Load synthetic notes
    print(f"\nLoading synthetic notes from: {args.synthetic_notes_path}")
    synthetic_notes = parse_notes_file(args.synthetic_notes_path)
    if args.max_notes:
        synthetic_notes = synthetic_notes[:args.max_notes]
    print(f"✓ Loaded {len(synthetic_notes)} synthetic notes")
    
    # Load real notes
    real_notes = []
    if args.real_notes_path:
        if args.real_notes_path.endswith('.csv'):
            real_notes = load_mimic_discharge_notes(args.real_notes_path, args.max_notes)
        else:
            real_notes = parse_notes_file(args.real_notes_path)
            if args.max_notes:
                real_notes = real_notes[:args.max_notes]
        print(f"✓ Loaded {len(real_notes)} real notes")
    else:
        print("⚠️  No real notes provided. Some evaluations may be limited.")
    
    # Ensure same size for fair comparison
    if real_notes and synthetic_notes:
        min_size = min(len(real_notes), len(synthetic_notes))
        real_notes = real_notes[:min_size]
        synthetic_notes = synthetic_notes[:min_size]
        print(f"Using {min_size} notes for each set (matched size)")
    
    # Run evaluations
    all_results = {}
    
    # Task 1: NER
    if 'ner' in args.tasks:
        ner_results = evaluate_ner_task(real_notes, synthetic_notes)
        all_results['ner'] = ner_results
    
    # Task 2: Readmission Prediction
    if 'readmission' in args.tasks:
        readmission_labels = None
        if args.admissions_csv and args.real_notes_path:
            readmission_labels = load_readmission_labels(args.admissions_csv, args.real_notes_path)
        
        readmission_results = evaluate_readmission_task(
            real_notes, synthetic_notes,
            real_labels=readmission_labels,
            synthetic_labels=None
        )
        all_results['readmission'] = readmission_results
    
    # Save results
    output_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Evaluation complete! Results saved to: {output_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

