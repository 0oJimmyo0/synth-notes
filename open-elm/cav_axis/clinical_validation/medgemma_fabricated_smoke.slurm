#!/bin/bash
#SBATCH --job-name=medgemma_smoke
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/clinical_validation/logs/medgemma_smoke_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/clinical_validation/logs/medgemma_smoke_%j.err

set -euo pipefail
umask 007

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
conda activate medgemma_judge

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

REPO_ROOT=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the local MedGemma snapshot before sbatch.}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR on approved project storage before sbatch.}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}

nvidia-smi
python "$REPO_ROOT/open-elm/cav_axis/clinical_validation/smoke_test_medgemma_medication_judge.py" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_new_tokens "$MAX_NEW_TOKENS"
