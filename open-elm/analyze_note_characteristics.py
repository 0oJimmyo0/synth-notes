#!/usr/bin/env python3
"""
Analyze characteristics of synthetic and real clinic notes to understand:
1. What type of notes they are (discharge summaries)
2. What sections they contain
3. What information is available
4. How to preprocess them for downstream tasks
5. What tasks are appropriate

This will help design appropriate downstream evaluation tasks.
"""

import argparse
import re
import os
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import json

def parse_notes_file(filepath: str) -> List[str]:
    """Parse notes from file (assumes format: === Note N ===\n<note text>\n\n)."""
    notes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract notes using regex
    pattern = r'=== Note \d+ ===\n(.*?)(?=\n=== Note |\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        note = match.strip()
        if note and len(note) > 100:  # Filter very short notes
            notes.append(note)
    
    return notes


def extract_sections(note: str) -> Dict[str, str]:
    """Extract structured sections from a discharge summary note."""
    sections = {}
    
    # Common section headers in discharge summaries
    section_patterns = {
        'header': r'(Name:|Unit No:|Admission Date:|Discharge Date:|Date of Birth:|Sex:|Service:|Allergies:|Attending:)',
        'chief_complaint': r'Chief Complaint:\s*\n(.*?)(?=\n[A-Z]|\n\n|$)',
        'history_present_illness': r'History of Present Illness:\s*\n(.*?)(?=\n[A-Z][a-z]+ [A-Z]|\nPast Medical|\nPhysical|\n\n\n|$)',
        'past_medical_history': r'Past Medical History:\s*\n(.*?)(?=\nPAST SURGERY|\nSocial|\nPhysical|\n\n\n|$)',
        'past_surgery': r'PAST SURGERY:\s*\n(.*?)(?=\nSocial|\nPhysical|\n[A-Z]|\n\n\n|$)',
        'social_history': r'Social History:\s*\n(.*?)(?=\nFamily|\nPhysical|\n[A-Z]|\n\n\n|$)',
        'family_history': r'Family History:\s*\n(.*?)(?=\nPhysical|\n[A-Z]|\n\n\n|$)',
        'physical_exam': r'Physical Exam:\s*\n(.*?)(?=\n[A-Z][A-Z]+:|$)',
        'medications': r'Medications|Discharge Medications:\s*\n(.*?)(?=\n[A-Z]|\n\n\n|$)',
        'diagnosis': r'Discharge Diagnosis|Diagnosis:\s*\n(.*?)(?=\n[A-Z]|\n\n\n|$)',
        'procedures': r'Major Surgical or Invasive Procedure:\s*\n(.*?)(?=\n[A-Z]|\n\n\n|$)',
    }
    
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, note, re.IGNORECASE | re.DOTALL)
        if match:
            # Some patterns may have multiple groups, get the last one (the content)
            if match.lastindex and match.lastindex > 0:
                sections[section_name] = match.group(match.lastindex).strip()
            else:
                sections[section_name] = match.group(0).strip()
        else:
            sections[section_name] = None
    
    return sections


def analyze_note_structure(notes: List[str]) -> Dict:
    """Analyze the structure and content of notes."""
    stats = {
        'total_notes': len(notes),
        'note_lengths': [],
        'word_counts': [],
        'sections_present': defaultdict(int),
        'section_lengths': defaultdict(list),
        'common_phrases': Counter(),
        'medical_terms': Counter(),
        'deidentified_indicators': 0,
        'has_dates': 0,
        'has_labs': 0,
        'has_medications': 0,
    }
    
    # Medical term patterns
    medical_patterns = {
        'diagnosis': r'\b(hypertension|diabetes|pneumonia|sepsis|stroke|MI|heart failure|CKD|COPD)\b',
        'medications': r'\b(aspirin|insulin|warfarin|metformin|lisinopril|atorvastatin|amlodipine)\b',
        'lab_values': r'\b(WBC|HCT|BUN|Cr|Na|K|Cl|glucose|hemoglobin|platelet)\b',
        'vitals': r'\b(HR|BP|RR|O2|temperature|fever|hypotension|tachycardia)\b',
        'procedures': r'\b(surgery|endoscopy|catheter|intubation|dialysis|biopsy)\b',
    }
    
    for note in notes:
        # Basic stats
        note_len = len(note)
        word_count = len(note.split())
        stats['note_lengths'].append(note_len)
        stats['word_counts'].append(word_count)
        
        # Extract sections
        sections = extract_sections(note)
        for section_name, section_content in sections.items():
            if section_content:
                stats['sections_present'][section_name] += 1
                stats['section_lengths'][section_name].append(len(section_content))
        
        # Check for de-identification
        if '___' in note or '[name]' in note.lower() or '[**' in note:
            stats['deidentified_indicators'] += 1
        
        # Check for dates
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', note):
            stats['has_dates'] += 1
        
        # Check for medical content
        note_lower = note.lower()
        for category, pattern in medical_patterns.items():
            matches = re.findall(pattern, note_lower, re.IGNORECASE)
            if matches:
                if category == 'lab_values':
                    stats['has_labs'] += 1
                elif category == 'medications':
                    stats['has_medications'] += 1
                stats['medical_terms'][category] += len(matches)
    
    return stats


def compare_note_sets(real_stats: Dict, synthetic_stats: Dict) -> Dict:
    """Compare statistics between real and synthetic notes."""
    comparison = {}
    
    # Compare basic stats
    comparison['note_count'] = {
        'real': real_stats['total_notes'],
        'synthetic': synthetic_stats['total_notes'],
        'ratio': synthetic_stats['total_notes'] / real_stats['total_notes'] if real_stats['total_notes'] > 0 else 0
    }
    
    # Compare lengths
    if real_stats['word_counts'] and synthetic_stats['word_counts']:
        comparison['avg_word_count'] = {
            'real': sum(real_stats['word_counts']) / len(real_stats['word_counts']),
            'synthetic': sum(synthetic_stats['word_counts']) / len(synthetic_stats['word_counts']),
        }
        comparison['avg_word_count']['ratio'] = (
            comparison['avg_word_count']['synthetic'] / comparison['avg_word_count']['real']
            if comparison['avg_word_count']['real'] > 0 else 0
        )
    
    # Compare sections
    comparison['sections'] = {}
    all_sections = set(real_stats['sections_present'].keys()) | set(synthetic_stats['sections_present'].keys())
    for section in all_sections:
        real_count = real_stats['sections_present'].get(section, 0)
        synth_count = synthetic_stats['sections_present'].get(section, 0)
        comparison['sections'][section] = {
            'real': real_count,
            'synthetic': synth_count,
            'real_pct': (real_count / real_stats['total_notes'] * 100) if real_stats['total_notes'] > 0 else 0,
            'synthetic_pct': (synth_count / synthetic_stats['total_notes'] * 100) if synthetic_stats['total_notes'] > 0 else 0,
        }
    
    # Compare medical content
    comparison['medical_content'] = {}
    for category in ['diagnosis', 'medications', 'lab_values', 'vitals', 'procedures']:
        real_count = real_stats['medical_terms'].get(category, 0)
        synth_count = synthetic_stats['medical_terms'].get(category, 0)
        comparison['medical_content'][category] = {
            'real': real_count,
            'synthetic': synth_count,
            'ratio': synth_count / real_count if real_count > 0 else 0
        }
    
    # De-identification
    comparison['deidentification'] = {
        'real': {
            'count': real_stats['deidentified_indicators'],
            'pct': (real_stats['deidentified_indicators'] / real_stats['total_notes'] * 100) if real_stats['total_notes'] > 0 else 0
        },
        'synthetic': {
            'count': synthetic_stats['deidentified_indicators'],
            'pct': (synthetic_stats['deidentified_indicators'] / synthetic_stats['total_notes'] * 100) if synthetic_stats['total_notes'] > 0 else 0
        }
    }
    
    return comparison


def suggest_downstream_tasks(comparison: Dict, real_stats: Dict, synthetic_stats: Dict) -> List[Dict]:
    """Suggest appropriate downstream tasks based on note characteristics."""
    tasks = []
    
    # Task 1: Section Extraction / Information Extraction
    sections_available = comparison.get('sections', {}) if comparison else {}
    if sections_available or synthetic_stats.get('sections_present'):
        num_sections = len(sections_available) if sections_available else len(synthetic_stats.get('sections_present', {}))
        tasks.append({
            'task': 'Section Extraction',
            'description': 'Extract structured sections (Chief Complaint, HPI, PMH, etc.) from notes',
            'feasibility': 'high',
            'reason': f"Notes contain {num_sections} distinct sections",
            'preprocessing': 'Parse section headers, extract text between sections',
            'evaluation': 'Exact match, F1 score for section boundaries'
        })
    
    # Task 2: Named Entity Recognition
    diagnosis_count = 0
    if comparison and 'medical_content' in comparison:
        diagnosis_count = comparison['medical_content'].get('diagnosis', {}).get('real', 0)
    elif synthetic_stats.get('medical_terms', {}).get('diagnosis', 0) > 0:
        diagnosis_count = synthetic_stats['medical_terms']['diagnosis']
    
    if diagnosis_count > 0:
        tasks.append({
            'task': 'Named Entity Recognition (NER)',
            'description': 'Extract medical entities: diagnoses, medications, procedures, lab values',
            'feasibility': 'high',
            'reason': 'Notes contain rich medical terminology',
            'preprocessing': 'Direct use - no preprocessing needed',
            'evaluation': 'Entity-level F1, Precision, Recall'
        })
    
    # Task 3: Medication Extraction
    med_count = 0
    if comparison and 'medical_content' in comparison:
        med_count = comparison['medical_content'].get('medications', {}).get('real', 0)
    elif synthetic_stats.get('has_medications', 0) > 0:
        med_count = synthetic_stats['has_medications']
    
    if med_count > 0:
        tasks.append({
            'task': 'Medication Extraction',
            'description': 'Extract medications, dosages, frequencies from notes',
            'feasibility': 'high',
            'reason': f"Medications found in {synthetic_stats['has_medications']} synthetic notes",
            'preprocessing': 'May need to handle abbreviations and dosage patterns',
            'evaluation': 'Precision, Recall, F1 for medication mentions'
        })
    
    # Task 4: Clinical Coding (ICD-10)
    diagnosis_section_count = 0
    if sections_available:
        diagnosis_section_count = sections_available.get('diagnosis', {}).get('real', 0)
    elif synthetic_stats.get('sections_present', {}).get('diagnosis', 0) > 0:
        diagnosis_section_count = synthetic_stats['sections_present']['diagnosis']
    
    if diagnosis_section_count > 0:
        tasks.append({
            'task': 'Clinical Coding (ICD-10)',
            'description': 'Predict ICD-10 diagnosis codes from discharge summaries',
            'feasibility': 'medium',
            'reason': 'Discharge summaries contain diagnosis information',
            'preprocessing': 'Extract diagnosis section, may need labels from MIMIC-IV',
            'evaluation': 'Macro F1, Micro F1, Exact match accuracy',
            'note': 'Requires labels from MIMIC-IV diagnoses_icd table'
        })
    
    # Task 5: Readmission Prediction
    tasks.append({
        'task': 'Readmission Prediction',
        'description': 'Predict 30-day hospital readmission from discharge summary',
        'feasibility': 'medium',
        'reason': 'Discharge summaries contain information relevant to readmission risk',
        'preprocessing': 'Extract features from full note or specific sections (HPI, PMH)',
        'evaluation': 'AUC-ROC, F1, Precision, Recall',
        'note': 'Requires labels from MIMIC-IV admissions table (need to link readmissions)'
    })
    
    # Task 6: Length of Stay Prediction
    tasks.append({
        'task': 'Length of Stay Prediction',
        'description': 'Predict hospital length of stay from discharge summary',
        'feasibility': 'medium',
        'reason': 'Discharge summaries contain clinical information that correlates with LOS',
        'preprocessing': 'Extract features from note, may need to calculate LOS from admission/discharge dates',
        'evaluation': 'MAE, RMSE, R²',
        'note': 'Requires labels from MIMIC-IV admissions table'
    })
    
    # Task 7: Mortality Prediction
    tasks.append({
        'task': 'Mortality Prediction',
        'description': 'Predict in-hospital mortality from discharge summary',
        'feasibility': 'medium',
        'reason': 'Discharge summaries contain severity indicators',
        'preprocessing': 'Extract features from note, especially HPI and physical exam sections',
        'evaluation': 'AUC-ROC, F1, Precision, Recall',
        'note': 'Requires labels from MIMIC-IV admissions table (deathtime field)'
    })
    
    # Task 8: Chief Complaint Classification
    cc_count = 0
    cc_pct = 0
    if sections_available:
        cc_info = sections_available.get('chief_complaint', {})
        cc_count = cc_info.get('real', 0)
        cc_pct = cc_info.get('real_pct', 0)
    elif synthetic_stats.get('sections_present', {}).get('chief_complaint', 0) > 0:
        cc_count = synthetic_stats['sections_present']['chief_complaint']
        cc_pct = (cc_count / synthetic_stats['total_notes'] * 100) if synthetic_stats['total_notes'] > 0 else 0
    
    if cc_count > 0:
        tasks.append({
            'task': 'Chief Complaint Classification',
            'description': 'Classify chief complaint into categories (e.g., chest pain, abdominal pain, etc.)',
            'feasibility': 'high',
            'reason': f"Chief complaint section present in {cc_pct:.1f}% of notes",
            'preprocessing': 'Extract chief complaint section, create categories',
            'evaluation': 'Accuracy, F1 (multi-class)'
        })
    
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description='Analyze characteristics of synthetic and real clinic notes'
    )
    parser.add_argument(
        '--real_notes_path',
        type=str,
        help='Path to file containing real clinical notes (optional, for comparison)'
    )
    parser.add_argument(
        '--synthetic_notes_path',
        type=str,
        required=True,
        help='Path to file containing synthetic clinical notes'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./note_analysis',
        help='Directory to save analysis results'
    )
    parser.add_argument(
        '--max_notes',
        type=int,
        default=None,
        help='Maximum number of notes to analyze (for faster processing)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("Analyzing Clinic Note Characteristics")
    print("="*70)
    print()
    
    # Load synthetic notes
    print(f"Loading synthetic notes from: {args.synthetic_notes_path}")
    synthetic_notes = parse_notes_file(args.synthetic_notes_path)
    if args.max_notes:
        synthetic_notes = synthetic_notes[:args.max_notes]
    print(f"✓ Loaded {len(synthetic_notes)} synthetic notes")
    print()
    
    # Analyze synthetic notes
    print("Analyzing synthetic notes...")
    synthetic_stats = analyze_note_structure(synthetic_notes)
    print("✓ Analysis complete")
    print()
    
    # Load and analyze real notes if provided
    real_stats = None
    if args.real_notes_path and os.path.exists(args.real_notes_path):
        print(f"Loading real notes from: {args.real_notes_path}")
        real_notes = parse_notes_file(args.real_notes_path)
        if args.max_notes:
            real_notes = real_notes[:min(args.max_notes, len(real_notes))]
        print(f"✓ Loaded {len(real_notes)} real notes")
        print()
        
        print("Analyzing real notes...")
        real_stats = analyze_note_structure(real_notes)
        print("✓ Analysis complete")
        print()
        
        # Compare
        print("Comparing real vs synthetic notes...")
        comparison = compare_note_sets(real_stats, synthetic_stats)
        print("✓ Comparison complete")
        print()
    else:
        print("No real notes provided - analyzing synthetic notes only")
        comparison = None
        print()
    
    # Print summary
    print("="*70)
    print("Summary Statistics")
    print("="*70)
    print(f"\nSynthetic Notes:")
    print(f"  Total: {synthetic_stats['total_notes']}")
    if synthetic_stats['word_counts']:
        avg_words = sum(synthetic_stats['word_counts']) / len(synthetic_stats['word_counts'])
        print(f"  Average words: {avg_words:.0f}")
        print(f"  Min words: {min(synthetic_stats['word_counts'])}")
        print(f"  Max words: {max(synthetic_stats['word_counts'])}")
    print(f"  Sections found: {len(synthetic_stats['sections_present'])}")
    print(f"  Notes with medications: {synthetic_stats['has_medications']} ({synthetic_stats['has_medications']/synthetic_stats['total_notes']*100:.1f}%)")
    print(f"  Notes with lab values: {synthetic_stats['has_labs']} ({synthetic_stats['has_labs']/synthetic_stats['total_notes']*100:.1f}%)")
    
    if real_stats:
        print(f"\nReal Notes:")
        print(f"  Total: {real_stats['total_notes']}")
        if real_stats['word_counts']:
            avg_words = sum(real_stats['word_counts']) / len(real_stats['word_counts'])
            print(f"  Average words: {avg_words:.0f}")
        print(f"  Sections found: {len(real_stats['sections_present'])}")
        print(f"  Notes with medications: {real_stats['has_medications']} ({real_stats['has_medications']/real_stats['total_notes']*100:.1f}%)")
        print(f"  Notes with lab values: {real_stats['has_labs']} ({real_stats['has_labs']/real_stats['total_notes']*100:.1f}%)")
    
    # Suggest downstream tasks
    print("\n" + "="*70)
    print("Suggested Downstream Tasks")
    print("="*70)
    
    if comparison:
        tasks = suggest_downstream_tasks(comparison, real_stats, synthetic_stats)
    else:
        # Still suggest tasks based on synthetic notes only
        tasks = suggest_downstream_tasks({}, {}, synthetic_stats)
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{i}. {task['task']}")
        print(f"   Description: {task['description']}")
        print(f"   Feasibility: {task['feasibility']}")
        print(f"   Reason: {task['reason']}")
        print(f"   Preprocessing: {task['preprocessing']}")
        print(f"   Evaluation: {task['evaluation']}")
        if 'note' in task:
            print(f"   Note: {task['note']}")
    
    # Save results
    results = {
        'synthetic_stats': {
            'total_notes': synthetic_stats['total_notes'],
            'avg_word_count': sum(synthetic_stats['word_counts']) / len(synthetic_stats['word_counts']) if synthetic_stats['word_counts'] else 0,
            'sections_present': dict(synthetic_stats['sections_present']),
            'medical_terms': dict(synthetic_stats['medical_terms']),
        },
        'suggested_tasks': tasks
    }
    
    if comparison:
        results['comparison'] = comparison
        results['real_stats'] = {
            'total_notes': real_stats['total_notes'],
            'avg_word_count': sum(real_stats['word_counts']) / len(real_stats['word_counts']) if real_stats['word_counts'] else 0,
            'sections_present': dict(real_stats['sections_present']),
            'medical_terms': dict(real_stats['medical_terms']),
        }
    
    results_file = os.path.join(args.output_dir, 'note_analysis.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

