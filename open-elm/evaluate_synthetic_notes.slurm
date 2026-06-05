#!/bin/bash
#SBATCH --job-name=eval_synth_notes
#SBATCH --output=log/evaluate_synthetic_notes_%j.out
#SBATCH --error=log/evaluate_synthetic_notes_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Evaluation of Synthetic Clinical Notes
# This script evaluates synthetic notes using downstream tasks:
# 1. Named Entity Recognition (NER)
# 2. Readmission Prediction

echo "=========================================="
echo "Synthetic Notes Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Set up paths
SCRIPT_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm"
cd "$SCRIPT_DIR"

# Paths to data
SYNTHETIC_NOTES="/gpfs/radev/scratch/xu_hua/shared/data/synthnote/mimiciv/3.1/data/clinic_notes/1_task/synthetic_notes/synthetic_notes_improved.txt"
REAL_NOTES="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv"
ADMISSIONS_CSV="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/mimiciv/3.1/hosp/admissions.csv"

# Output directory
OUTPUT_DIR="./evaluation_results"
MAX_NOTES=5000  # Adjust based on available time

echo "Configuration:"
echo "  Synthetic notes: $SYNTHETIC_NOTES"
echo "  Real notes: $REAL_NOTES"
echo "  Admissions CSV: $ADMISSIONS_CSV"
echo "  Output directory: $OUTPUT_DIR"
echo "  Max notes: $MAX_NOTES"
echo ""

# Activate conda environment
echo "Activating conda environment..."
module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate elm

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import sklearn; import numpy; import pandas; print('✓ Basic packages available')" || exit 1

# Run evaluation
echo ""
echo "=========================================="
echo "Running Evaluation"
echo "=========================================="
echo ""

python "${SCRIPT_DIR}/evaluate_synthetic_notes.py" \
    --real_notes_path "${REAL_NOTES}" \
    --synthetic_notes_path "${SYNTHETIC_NOTES}" \
    --admissions_csv "${ADMISSIONS_CSV}" \
    --tasks ner readmission \
    --max_notes "${MAX_NOTES}" \
    --output_dir "${OUTPUT_DIR}"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: Evaluation failed with exit code $EXIT_CODE"
    echo "Time: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Evaluation completed successfully!"
echo "Job ID: $SLURM_JOB_ID"
echo "End Time: $(date)"
echo "Output directory: ${OUTPUT_DIR}"
echo "Check results: ${OUTPUT_DIR}/evaluation_results.json"
echo "=========================================="

# Show summary if results file exists
if [ -f "${OUTPUT_DIR}/evaluation_results.json" ]; then
    echo ""
    echo "Results file created. Use the following to view summary:"
    echo "  python -m json.tool ${OUTPUT_DIR}/evaluation_results.json | head -50"
fi

