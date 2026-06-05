#!/bin/bash
#SBATCH --job-name=elm_train_clinic
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=47:00:00          
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu

# Get script directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Create log directory
LOG_DIR="${SCRIPT_DIR}/log"
mkdir -p "${LOG_DIR}"

# Redirect output to log files
exec > >(tee "${LOG_DIR}/train_clinic_notes_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/train_clinic_notes_${SLURM_JOB_ID}.err" >&2)

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
BASE_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1 
DATAHOME="/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/"
INITIAL_MODEL="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/initial_elm_model"
OUTPUT_DIR="/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/elm_training_outputs"

# Check if filtered datasets exist
if [ ! -d "${DATAHOME}encoded_training_filtered" ]; then
    echo "ERROR: Filtered training dataset not found at ${DATAHOME}encoded_training_filtered"
    echo "Please run filter_long_sequences.py first to create the filtered datasets."
    exit 1
fi

if [ ! -d "${DATAHOME}encoded_dev_filtered" ]; then
    echo "ERROR: Filtered dev dataset not found at ${DATAHOME}encoded_dev_filtered"
    exit 1
fi

# Check if initial model exists
if [ ! -d "${INITIAL_MODEL}" ]; then
    echo "ERROR: Initial ELM model not found at ${INITIAL_MODEL}"
    echo "You need to create it first by running:"
    echo "  python initialize_model.py --output_dir ${INITIAL_MODEL}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check available disk space before starting (safety check)
echo "Checking available disk space..."
AVAILABLE_SPACE=$(df -h "${OUTPUT_DIR}" | tail -1 | awk '{print $4}')
echo "Available space in output directory: ${AVAILABLE_SPACE}"
echo "Estimated checkpoint storage needed: ~90GB (3 checkpoints × 30GB each)"
echo ""

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
echo "Training Configuration"
echo "=========================================="
echo "Data directory: ${DATAHOME}"
echo "Using filtered datasets: encoded_training_filtered, encoded_dev_filtered"
echo "Initial model: ${INITIAL_MODEL}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Using default settings from train.py"
echo "=========================================="
echo ""

# Run training with filtered data using default settings
python "${SCRIPT_DIR}/train.py" \
    --datahome "${DATAHOME}" \
    --output_dir "${OUTPUT_DIR}/filtered_training" \
    --checkpoint_path "${INITIAL_MODEL}" \
    --filtered

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    echo "Time: $(date)"
    echo "=========================================="
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Training completed successfully!"
echo "Job ID: $SLURM_JOB_ID"
echo "End Time: $(date)"
echo "Output directory: ${OUTPUT_DIR}/filtered_training"
echo "Check logs: ${LOG_DIR}/train_clinic_notes_${SLURM_JOB_ID}.out"
echo "=========================================="
