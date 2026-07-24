#!/bin/bash
#SBATCH --job-name=medgemma_bf16_verify
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/clinical_validation/logs/medgemma_bf16_verify_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/clinical_validation/logs/medgemma_bf16_verify_%j.err

set -euo pipefail
umask 007

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
# Slurm can inherit an interactive Conda prompt state without its shell hooks.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
conda activate medgemma_judge

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

REPO_ROOT=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the local MedGemma snapshot before sbatch.}
VERIFY_OUTPUT=${VERIFY_OUTPUT:-$MODEL_PATH/MEDGEMMA_INSTALL_VERIFICATION_BF16.json}

nvidia-smi
python "$REPO_ROOT/open-elm/cav_axis/clinical_validation/verify_medgemma_install.py" \
  --model_path "$MODEL_PATH" \
  --output_path "$VERIFY_OUTPUT" \
  --load_bf16
