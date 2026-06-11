#!/bin/bash
#SBATCH --job-name=elm_generate
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:2
#SBATCH --partition=gpu

set -euo pipefail

SCRIPT_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm"
LOG_DIR="${SCRIPT_DIR}/log"
mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/generate_synthetic_notes_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/generate_synthetic_notes_${SLURM_JOB_ID}.err" >&2)

cd "${SCRIPT_DIR}"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo ""

BASE_DIR="${BASE_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1}"
DATAHOME="${DATAHOME:-${BASE_DIR}/data_note_hadm_all/clinic_notes/1_task/}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DATAHOME}elm_training_outputs/filtered_training/checkpoint-8215}"
BACKBONE_MODEL="${SCRIPT_DIR}/initial_elm_model"
OUTPUT_DIR="${OUTPUT_DIR:-${DATAHOME}synthetic_notes}"
DATASET_PATH="${DATASET_PATH:-${DATAHOME}encoded_testing_filtered}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/synthetic_notes_test_vanilla_seed42.txt}"
MANIFEST_FILE="${MANIFEST_FILE:-${OUTPUT_DIR}/synthetic_notes_test_vanilla_seed42_manifest.jsonl}"
SPLIT_NAME="${SPLIT_NAME:-test}"
GENERATION_CONDITION="${GENERATION_CONDITION:-vanilla}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-${DATAHOME}leakage_audit/split_manifest_note_level.csv}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_SHARDS="${NUM_SHARDS:-2}"

if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT_PATH}"
    exit 1
fi

if [ ! -d "${BACKBONE_MODEL}" ]; then
    echo "ERROR: Backbone model not found at ${BACKBONE_MODEL}"
    exit 1
fi

if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset not found at ${DATASET_PATH}"
    exit 1
fi

if [ ! -f "${SPLIT_MANIFEST_PATH}" ]; then
    echo "ERROR: Split manifest not found at ${SPLIT_MANIFEST_PATH}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Loading modules and setting up environment..."
module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_SKIP_CUDA_CHECK=1
export ACCELERATE_USE_DEEPSPEED=0
export TOKENIZERS_PARALLELISM=false

if pip show deepspeed &>/dev/null; then
    pip uninstall -y deepspeed 2>&1 | grep -v "not installed" || true
fi

echo "GPU Information:"
nvidia-smi
echo ""

echo "=========================================="
echo "Generation Configuration"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Backbone model: ${BACKBONE_MODEL}"
echo "Dataset path: ${DATASET_PATH}"
echo "Output file: ${OUTPUT_FILE}"
echo "Manifest file: ${MANIFEST_FILE}"
echo "Split manifest: ${SPLIT_MANIFEST_PATH}"
echo "Seed: ${SEED}"
echo "Batch size per shard: ${BATCH_SIZE}"
echo "Requested shards: ${NUM_SHARDS}"
echo "=========================================="
echo ""

TOTAL_ROWS=$(python - <<PY
from datasets import Dataset
print(len(Dataset.load_from_disk("${DATASET_PATH}")))
PY
)

echo "Total dataset rows: ${TOTAL_ROWS}"

if [ "${NUM_SHARDS}" -lt 1 ] || [ "${NUM_SHARDS}" -gt 2 ]; then
    echo "ERROR: NUM_SHARDS must be 1 or 2 for this launcher."
    exit 1
fi

VISIBLE_GPUS="${SLURM_GPUS_ON_NODE:-1}"
if [ "${NUM_SHARDS}" -gt "${VISIBLE_GPUS}" ]; then
    echo "ERROR: NUM_SHARDS=${NUM_SHARDS} exceeds visible GPUs (${VISIBLE_GPUS})"
    exit 1
fi

TMP_DIR="${OUTPUT_DIR}/tmp_${SLURM_JOB_ID}"
mkdir -p "${TMP_DIR}"

PIDS=()
SHARD_OUTPUTS=()
SHARD_MANIFESTS=()

for ((SHARD_ID=0; SHARD_ID<NUM_SHARDS; SHARD_ID++)); do
    START_INDEX=$(( SHARD_ID * TOTAL_ROWS / NUM_SHARDS ))
    END_INDEX=$(( (SHARD_ID + 1) * TOTAL_ROWS / NUM_SHARDS ))
    SHARD_OUTPUT="${TMP_DIR}/$(basename "${OUTPUT_FILE%.txt}")_shard${SHARD_ID}.txt"
    SHARD_MANIFEST="${TMP_DIR}/$(basename "${MANIFEST_FILE%.jsonl}")_shard${SHARD_ID}.jsonl"

    SHARD_OUTPUTS+=("${SHARD_OUTPUT}")
    SHARD_MANIFESTS+=("${SHARD_MANIFEST}")

    echo "Launching shard ${SHARD_ID}: rows [${START_INDEX}, ${END_INDEX}) on GPU ${SHARD_ID}"

    CUDA_VISIBLE_DEVICES="${SHARD_ID}" python "${SCRIPT_DIR}/generate_synthetic_notes.py" \
        --checkpoint_path "${CHECKPOINT_PATH}" \
        --backbone_model_path "${BACKBONE_MODEL}" \
        --dataset_path "${DATASET_PATH}" \
        --output_file "${SHARD_OUTPUT}" \
        --manifest_output "${SHARD_MANIFEST}" \
        --generation_condition "${GENERATION_CONDITION}" \
        --split "${SPLIT_NAME}" \
        --split_manifest_path "${SPLIT_MANIFEST_PATH}" \
        --batch_size "${BATCH_SIZE}" \
        --repetition_penalty 1.2 \
        --temperature 0.8 \
        --top_p 0.9 \
        --top_k 50 \
        --max_new_tokens 2048 \
        --device cuda \
        --seed "${SEED}" \
        --start_index "${START_INDEX}" \
        --end_index "${END_INDEX}" &

    PIDS+=($!)
done

GEN_EXIT_CODE=0
for PID in "${PIDS[@]}"; do
    wait "${PID}" || GEN_EXIT_CODE=$?
done

if [ "${GEN_EXIT_CODE}" -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: Generation failed with exit code ${GEN_EXIT_CODE}"
    echo "Time: $(date)"
    echo "=========================================="
    exit "${GEN_EXIT_CODE}"
fi

: > "${OUTPUT_FILE}"
: > "${MANIFEST_FILE}"
for SHARD_OUTPUT in "${SHARD_OUTPUTS[@]}"; do
    cat "${SHARD_OUTPUT}" >> "${OUTPUT_FILE}"
done
for SHARD_MANIFEST in "${SHARD_MANIFESTS[@]}"; do
    cat "${SHARD_MANIFEST}" >> "${MANIFEST_FILE}"
done

python - <<PY
import json
from pathlib import Path
from datasets import Dataset

dataset_len = len(Dataset.load_from_disk("${DATASET_PATH}"))
manifest_path = Path("${MANIFEST_FILE}")
output_path = Path("${OUTPUT_FILE}")
validation_path = manifest_path.with_suffix(manifest_path.suffix + ".validation.json")

manifest_rows = 0
generation_ids = set()
last_generation_index = -1
for line in manifest_path.open():
    row = json.loads(line)
    manifest_rows += 1
    generation_ids.add(row["generation_id"])
    current = int(row["generation_index"])
    if current != last_generation_index + 1:
        raise ValueError(f"Manifest generation_index order broke at {current} after {last_generation_index}")
    last_generation_index = current

note_headers = 0
for line in output_path.open():
    if line.startswith("=== Note "):
        note_headers += 1

if manifest_rows != dataset_len:
    raise ValueError(f"Merged manifest rows {manifest_rows} do not match dataset length {dataset_len}")
if note_headers != dataset_len:
    raise ValueError(f"Merged note count {note_headers} does not match dataset length {dataset_len}")
if len(generation_ids) != manifest_rows:
    raise ValueError("Duplicate generation_id values detected in merged manifest")

validation = {
    "dataset_len": dataset_len,
    "merged_manifest_rows": manifest_rows,
    "merged_note_headers": note_headers,
    "num_shards": int("${NUM_SHARDS}"),
}
validation_path.write_text(json.dumps(validation, indent=2))
PY

echo ""
echo "=========================================="
echo "Generation completed successfully!"
echo "Job ID: $SLURM_JOB_ID"
echo "End Time: $(date)"
echo "Output file: ${OUTPUT_FILE}"
echo "Manifest file: ${MANIFEST_FILE}"
echo "Check logs: ${LOG_DIR}/generate_synthetic_notes_${SLURM_JOB_ID}.out"
echo "=========================================="

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
