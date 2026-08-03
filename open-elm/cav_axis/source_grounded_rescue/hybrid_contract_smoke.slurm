#!/bin/bash
#SBATCH --job-name=hybrid_contract_smoke
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/source_grounded_rescue/logs/hybrid_contract_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/source_grounded_rescue/logs/hybrid_contract_%j.err
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

OPEN_ELM_DIR=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm
SCRIPT_PATH=$OPEN_ELM_DIR/cav_axis/source_grounded_rescue/run_hybrid_contract_generation.py
DATAHOME=${DATAHOME:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task}
BACKBONE_PATH=${BACKBONE_PATH:-$OPEN_ELM_DIR/initial_elm_model}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-$DATAHOME/elm_training_outputs/filtered_training/checkpoint-8215}
: "${CONTRACT_PATH:?Set CONTRACT_PATH to transition_note_contract_v1.jsonl}"
: "${GENERATION_LEDGER_PATH:?Set GENERATION_LEDGER_PATH to the frozen prompt-safe ledger}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
CASE_IDS=${CASE_IDS:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1536}
N_CANDIDATES_PER_CASE=${N_CANDIDATES_PER_CASE:-2}
SEED=${SEED:-42}
DO_SAMPLE=${DO_SAMPLE:-1}
MAX_COURSE_ATTEMPTS=${MAX_COURSE_ATTEMPTS:-3}

for required in "$SCRIPT_PATH" "$BACKBONE_PATH" "$CHECKPOINT_PATH" "$CONTRACT_PATH" "$GENERATION_LEDGER_PATH"; do
  [[ -e "$required" ]] || { echo "ERROR: required path does not exist: $required" >&2; exit 1; }
done

mkdir -p "$OUTPUT_DIR" "$OPEN_ELM_DIR/cav_axis/source_grounded_rescue/logs"
module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID: $SLURM_JOB_ID"
echo "Contract: $CONTRACT_PATH"
echo "Ledger: $GENERATION_LEDGER_PATH"
echo "Cases: ${CASE_IDS:-all source-complete contract/ledger cases}"
nvidia-smi
cd "$OPEN_ELM_DIR"

cmd=(python -u "$SCRIPT_PATH"
  --contract_path "$CONTRACT_PATH"
  --generation_ledger_path "$GENERATION_LEDGER_PATH"
  --backbone_path "$BACKBONE_PATH"
  --model_condition checkpoint_8215
  --checkpoint_path "$CHECKPOINT_PATH"
  --output_dir "$OUTPUT_DIR"
  --output_stem hybrid_contract_smoke
  --device cuda
  --max_new_tokens "$MAX_NEW_TOKENS"
  --n_candidates_per_case "$N_CANDIDATES_PER_CASE"
  --seed "$SEED"
  --max_course_attempts "$MAX_COURSE_ATTEMPTS")
[[ -n "$CASE_IDS" ]] && cmd+=(--case_ids "$CASE_IDS")
[[ "$DO_SAMPLE" == "1" ]] && cmd+=(--do_sample)
"${cmd[@]}"
