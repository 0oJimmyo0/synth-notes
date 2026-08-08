#!/bin/bash
# Run two independent hybrid-generation workers, one per allocated GPU.
# Each worker writes a separate manifest so no concurrent process touches a file.
#SBATCH --job-name=hybrid_contract_2a40
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/source_grounded_rescue/logs/hybrid_contract_2a40_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/source_grounded_rescue/logs/hybrid_contract_2a40_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
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
: "${GENERATION_LEDGER_PATH:?Set GENERATION_LEDGER_PATH to generation_ledgers.jsonl}"
: "${CASE_IDS_A:?Set CASE_IDS_A to comma-separated case IDs for GPU 0}"
: "${CASE_IDS_B:?Set CASE_IDS_B to comma-separated case IDs for GPU 1}"
: "${OUTPUT_DIR_A:?Set OUTPUT_DIR_A for GPU 0}"
: "${OUTPUT_DIR_B:?Set OUTPUT_DIR_B for GPU 1}"
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1536}
N_CANDIDATES_PER_CASE=${N_CANDIDATES_PER_CASE:-4}
SEED=${SEED:-20260814}
DO_SAMPLE=${DO_SAMPLE:-1}
MAX_COURSE_ATTEMPTS=${MAX_COURSE_ATTEMPTS:-3}

for required in "$SCRIPT_PATH" "$BACKBONE_PATH" "$CHECKPOINT_PATH" "$CONTRACT_PATH" "$GENERATION_LEDGER_PATH"; do
  [[ -e "$required" ]] || { echo "ERROR: required path does not exist: $required" >&2; exit 1; }
done

mkdir -p "$OUTPUT_DIR_A" "$OUTPUT_DIR_B" "$OPEN_ELM_DIR/cav_axis/source_grounded_rescue/logs"
module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID: $SLURM_JOB_ID"
echo "GPU 0 cases: $(tr ',' '\n' <<< "$CASE_IDS_A" | wc -l)"
echo "GPU 1 cases: $(tr ',' '\n' <<< "$CASE_IDS_B" | wc -l)"
nvidia-smi
cd "$OPEN_ELM_DIR"

run_worker() {
  local gpu_id=$1
  local case_ids=$2
  local output_dir=$3
  local worker_name=$4

  CUDA_VISIBLE_DEVICES="$gpu_id" python -u "$SCRIPT_PATH" \
    --contract_path "$CONTRACT_PATH" \
    --generation_ledger_path "$GENERATION_LEDGER_PATH" \
    --backbone_path "$BACKBONE_PATH" \
    --model_condition checkpoint_8215 \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_dir "$output_dir" \
    --output_stem hybrid_contract_smoke \
    --device cuda \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --n_candidates_per_case "$N_CANDIDATES_PER_CASE" \
    --seed "$SEED" \
    --max_course_attempts "$MAX_COURSE_ATTEMPTS" \
    --case_ids "$case_ids" \
    $([[ "$DO_SAMPLE" == "1" ]] && printf '%s' '--do_sample') \
    >"$output_dir/${worker_name}.out" 2>"$output_dir/${worker_name}.err"
}

run_worker 0 "$CASE_IDS_A" "$OUTPUT_DIR_A" worker_gpu0 &
pid_a=$!
run_worker 1 "$CASE_IDS_B" "$OUTPUT_DIR_B" worker_gpu1 &
pid_b=$!

set +e
wait "$pid_a"
status_a=$?
wait "$pid_b"
status_b=$?
set -e

if [[ "$status_a" -ne 0 || "$status_b" -ne 0 ]]; then
  echo "ERROR: worker exit statuses: gpu0=$status_a gpu1=$status_b" >&2
  exit 1
fi

echo "Both dual-A40 workers completed successfully."
