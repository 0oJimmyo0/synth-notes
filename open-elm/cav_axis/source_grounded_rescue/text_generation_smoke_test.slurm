#!/bin/bash
#SBATCH --job-name=elm_text_smoke
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/source_grounded_rescue/text_smoke_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/source_grounded_rescue/text_smoke_%j.err
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

OPEN_ELM_DIR=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm
SCRIPT_PATH=$OPEN_ELM_DIR/cav_axis/source_grounded_rescue/test_text_generation_path.py
BACKBONE_PATH=${BACKBONE_PATH:-$OPEN_ELM_DIR/initial_elm_model}
DATAHOME=${DATAHOME:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-$DATAHOME/elm_training_outputs/filtered_training/checkpoint-8215}
OUTPUT_DIR=${OUTPUT_DIR:-$DATAHOME/source_grounded_rescue/text_generation_smoke_test}

mkdir -p "$OUTPUT_DIR" "$OPEN_ELM_DIR/log/source_grounded_rescue"
for required in "$SCRIPT_PATH" "$BACKBONE_PATH" "$CHECKPOINT_PATH"; do
  if [ ! -e "$required" ]; then
    echo "ERROR: required path does not exist: $required" >&2
    exit 1
  fi
done

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID: $SLURM_JOB_ID"
nvidia-smi
cd "$OPEN_ELM_DIR"

python -u "$SCRIPT_PATH" \
  --backbone_path "$BACKBONE_PATH" \
  --condition untouched_backbone \
  --output_path "$OUTPUT_DIR/untouched_backbone.json" \
  --device cuda \
  --max_new_tokens 32

python -u "$SCRIPT_PATH" \
  --backbone_path "$BACKBONE_PATH" \
  --condition checkpoint_8215 \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --output_path "$OUTPUT_DIR/checkpoint_8215.json" \
  --device cuda \
  --max_new_tokens 32
