#!/bin/bash
#SBATCH --job-name=elm_generate
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=44:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu

# Use the canonical repository path so Slurm spool copies do not break relative assets.
SCRIPT_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm"

# Create log directory
LOG_DIR="${SCRIPT_DIR}/log"
mkdir -p "${LOG_DIR}"

# Redirect output to log files
exec > >(tee "${LOG_DIR}/generate_synthetic_notes_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/generate_synthetic_notes_${SLURM_JOB_ID}.err" >&2)

# Change to script directory
cd "${SCRIPT_DIR}"

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo ""

# Configuration
# Default to the new note/HADM-aligned MIMIC-IV pipeline under shared datasets.
BASE_DIR="${BASE_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1}"
DATAHOME="${DATAHOME:-${BASE_DIR}/data_note_hadm_all/clinic_notes/1_task/}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DATAHOME}elm_training_outputs/filtered_training/checkpoint-8215}"
BACKBONE_MODEL="${SCRIPT_DIR}/initial_elm_model"
OUTPUT_DIR="${OUTPUT_DIR:-${DATAHOME}synthetic_notes}"

# Check if checkpoint exists
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT_PATH}"
    echo "Please check the checkpoint path. Available checkpoints:"
    ls -d "${DATAHOME}elm_training_outputs/filtered_training/checkpoint-"* 2>/dev/null || echo "  None found"
    exit 1
fi

# Check if backbone model exists
if [ ! -d "${BACKBONE_MODEL}" ]; then
    echo "ERROR: Backbone model not found at ${BACKBONE_MODEL}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Load modules and activate conda environment
echo "Loading modules and setting up environment..."
module load miniconda

# Initialize conda properly
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Activate conda environment
conda activate elm

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'elm'"
    exit 1
fi

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Prevent DeepSpeed from being imported
if pip show deepspeed &>/dev/null; then
    pip uninstall -y deepspeed 2>&1 | grep -v "not installed" || true
fi

export DS_SKIP_CUDA_CHECK=1
export ACCELERATE_USE_DEEPSPEED=0

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# Print configuration
echo "=========================================="
echo "Generation Configuration"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Backbone model: ${BACKBONE_MODEL}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Default: Generate from held-out test dataset (~33k notes in the new note/HADM pipeline)
# To get MORE synthetic notes, choose one of the options below (uncomment and set OUTPUT_FILE accordingly).
#
# Option A — Dev only (~33k notes): diverse, no train-set overlap
#   DATASET_PATH="${DATAHOME}encoded_dev_filtered"
#   OUTPUT_FILE="${OUTPUT_DIR}/synthetic_notes_dev_vanilla_seed42.txt"
#
# Option B — Test + Dev (~66k notes): combine two splits
#   DATASET_PATHS="${DATAHOME}encoded_testing_filtered,${DATAHOME}encoded_dev_filtered"
#   OUTPUT_FILE="${OUTPUT_DIR}/synthetic_notes_test_dev_vanilla_seed42.txt"
#   (use --dataset_paths "${DATASET_PATHS}" in the python call below instead of --dataset_path)
#
# Option C — Test + Dev + Train (~331k notes): max volume; train-set synthetic may be very similar to real notes
#   DATASET_PATHS="${DATAHOME}encoded_testing_filtered,${DATAHOME}encoded_dev_filtered,${DATAHOME}encoded_training_filtered"
#   OUTPUT_FILE="${OUTPUT_DIR}/synthetic_notes_all_splits_vanilla_seed42.txt"
#   (use --dataset_paths "${DATASET_PATHS}" in the python call below instead of --dataset_path)
#
# Baseline default: single held-out split only
DATASET_PATH="${DATASET_PATH:-${DATAHOME}encoded_testing_filtered}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/synthetic_notes_test_vanilla_seed42.txt}"
MANIFEST_FILE="${MANIFEST_FILE:-${OUTPUT_DIR}/synthetic_notes_test_vanilla_seed42_manifest.jsonl}"
DATASET_PATHS="${DATASET_PATHS:-}"   # Keep empty for single split; set only for Option B/C
SPLIT_NAME="${SPLIT_NAME:-test}"
GENERATION_CONDITION="${GENERATION_CONDITION:-vanilla}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-${DATAHOME}leakage_audit/split_manifest_note_level.csv}"
SEED="${SEED:-42}"

# Check dataset exists (use first path if using DATASET_PATHS)
if [ -n "${DATASET_PATHS}" ]; then
    FIRST_PATH="${DATASET_PATHS%%,*}"
    DATASET_PATH="${FIRST_PATH}"
fi
if [ ! -d "${DATASET_PATH}" ]; then
    echo "WARNING: Dataset not found at ${DATASET_PATH}"
    echo "Available datasets:"
    ls -d "${DATAHOME}encoded_"* 2>/dev/null | head -5 || echo "  None found"
    echo ""
    echo "You can modify this script to use a different dataset or embeddings file."
    exit 1
fi

echo "Generating synthetic notes from: ${DATASET_PATHS:-$DATASET_PATH}"
echo "Output will be saved to: ${OUTPUT_FILE}"
echo "Manifest will be saved to: ${MANIFEST_FILE}"
echo "Sampling seed: ${SEED}"
echo "Data root: ${DATAHOME}"
echo ""

# Run generation: use --dataset_paths if DATASET_PATHS is set, else --dataset_path
if [ -n "${DATASET_PATHS}" ]; then
python "${SCRIPT_DIR}/generate_synthetic_notes.py" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --backbone_model_path "${BACKBONE_MODEL}" \
    --dataset_paths "${DATASET_PATHS}" \
    --output_file "${OUTPUT_FILE}" \
    --manifest_output "${MANIFEST_FILE}" \
    --generation_condition "${GENERATION_CONDITION}" \
    --split_manifest_path "${SPLIT_MANIFEST_PATH}" \
    --batch_size 8 \
    --repetition_penalty 1.2 \
    --temperature 0.8 \
    --top_p 0.9 \
    --top_k 50 \
    --max_new_tokens 2048 \
    --device cuda \
    --seed "${SEED}"
else
python "${SCRIPT_DIR}/generate_synthetic_notes.py" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --backbone_model_path "${BACKBONE_MODEL}" \
    --dataset_path "${DATASET_PATH}" \
    --output_file "${OUTPUT_FILE}" \
    --manifest_output "${MANIFEST_FILE}" \
    --generation_condition "${GENERATION_CONDITION}" \
    --split "${SPLIT_NAME}" \
    --split_manifest_path "${SPLIT_MANIFEST_PATH}" \
    --batch_size 8 \
    --repetition_penalty 1.2 \
    --temperature 0.8 \
    --top_p 0.9 \
    --top_k 50 \
    --max_new_tokens 2048 \
    --device cuda \
    --seed "${SEED}"
fi

GEN_EXIT_CODE=$?

if [ $GEN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: Generation failed with exit code $GEN_EXIT_CODE"
    echo "Time: $(date)"
    echo "=========================================="
    exit $GEN_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Generation completed successfully!"
echo "Job ID: $SLURM_JOB_ID"
echo "End Time: $(date)"
echo "Output file: ${OUTPUT_FILE}"
echo "Manifest file: ${MANIFEST_FILE}"
echo "Check logs: ${LOG_DIR}/generate_synthetic_notes_${SLURM_JOB_ID}.out"
echo "=========================================="

# Show file size
if [ -f "${OUTPUT_FILE}" ]; then
    FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
    LINE_COUNT=$(wc -l < "${OUTPUT_FILE}")
    echo ""
    echo "Output file statistics:"
    echo "  Size: ${FILE_SIZE}"
    echo "  Lines: ${LINE_COUNT}"
    echo ""
    echo "Preview (first 20 lines):"
    head -20 "${OUTPUT_FILE}"
fi
