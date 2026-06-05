#!/bin/bash
#SBATCH --job-name=elm_init
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=4:00:00          # 4 hours should be enough for initialization
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G               # Increased to 128G - loading 8B model requires ~40-50GB+ RAM
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --partition=gpu          # Adjust to your cluster's GPU partition name

# Get the directory where the script is located (use SLURM_SUBMIT_DIR if available)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    # Fallback: try to get script directory from the script path
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Create log directory
LOG_DIR="${SCRIPT_DIR}/log"
mkdir -p "${LOG_DIR}"

# Redirect all output to log files
exec > >(tee "${LOG_DIR}/initialize_model_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/initialize_model_${SLURM_JOB_ID}.err" >&2)

# Change to script directory
cd "${SCRIPT_DIR}"

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Script Directory: ${SCRIPT_DIR}"
echo "Log Directory: ${LOG_DIR}"
echo "=========================================="
echo ""

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# Configuration
INITIAL_MODEL="${SCRIPT_DIR}/initial_elm_model"
LLAMA_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"  # Or use local path if downloaded
DIM_EMBED_DOMAIN=1024  # Should match your embedding dimension (BAAI-bge-large-en-v1.5 = 1024)
DIM_ADAPTER_HIDDEN=2048

# Check if model already exists
if [ -d "${INITIAL_MODEL}" ]; then
    echo "WARNING: Initial ELM model already exists at ${INITIAL_MODEL}"
    echo "Skipping initialization. If you want to recreate it, delete the directory first:"
    echo "  rm -rf ${INITIAL_MODEL}"
    echo "Then resubmit this job."
    echo ""
    echo "Exiting without error (model already exists)."
    exit 0
fi

# Load modules and activate conda environment
echo "Loading modules and setting up environment..."
module load miniconda

# Initialize conda properly
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Activate conda environment
conda activate elm

# Verify conda activation
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'elm'"
    echo "Make sure the environment exists: conda env list"
    exit 1
fi

# Verify environment
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# Print configuration
echo "=========================================="
echo "ELM Model Initialization Configuration"
echo "=========================================="
echo "Llama model: ${LLAMA_MODEL_PATH}"
echo "Output directory: ${INITIAL_MODEL}"
echo "Embedding dimension: ${DIM_EMBED_DOMAIN}"
echo "Adapter hidden dimension: ${DIM_ADAPTER_HIDDEN}"
echo "=========================================="
echo ""

# Run initialization
echo "Starting model initialization..."
echo "Time: $(date)"
echo ""

python "${SCRIPT_DIR}/initialize_model.py" \
    --llama_model_path "${LLAMA_MODEL_PATH}" \
    --output_dir "${INITIAL_MODEL}" \
    --dim_embed_domain ${DIM_EMBED_DOMAIN} \
    --dim_adapter_hidden ${DIM_ADAPTER_HIDDEN} \
    --skip_weight_validation  # Skip to save time and memory

INIT_EXIT_CODE=$?

if [ $INIT_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: Model initialization failed with exit code $INIT_EXIT_CODE"
    echo "Time: $(date)"
    echo "=========================================="
    exit $INIT_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Model initialization completed successfully!"
echo "Time: $(date)"
echo "Model saved to: ${INITIAL_MODEL}"
echo "=========================================="
echo ""

# Verify the model was created
if [ -d "${INITIAL_MODEL}" ]; then
    echo "Verifying model files..."
    ls -lh "${INITIAL_MODEL}" | head -10
    echo ""
    echo "✓ Model directory created successfully"
else
    echo "⚠ WARNING: Model directory not found after initialization"
fi

# Print final GPU usage
echo ""
echo "Final GPU Usage:"
nvidia-smi
echo ""

# Print completion information
echo "=========================================="
echo "Next steps:"
echo "1. Verify the model was created: ls -lh ${INITIAL_MODEL}"
echo "2. Submit training job: sbatch train_clinic_notes.slurm"
echo "3. Check logs in: ${LOG_DIR}/initialize_model_${SLURM_JOB_ID}.out"
echo "=========================================="

