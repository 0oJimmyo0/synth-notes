#!/bin/bash
#SBATCH --job-name=elm_generate_a40
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/generate_synthetic_notes_a40_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/generate_synthetic_notes_a40_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=gpu_devel

set -euo pipefail

SCRIPT_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm"
LOG_DIR="${SCRIPT_DIR}/log"
mkdir -p "${LOG_DIR}"
trap 'status=$?; echo "[ERROR] ${BASH_SOURCE[0]} failed at line ${LINENO} with exit code ${status}" >&2; exit ${status}' ERR

cd "${SCRIPT_DIR}"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo ""

BASE_DIR="${BASE_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1}"
DATAHOME="${DATAHOME:-${BASE_DIR}/data_note_hadm_all/clinic_notes/1_task}"
BASE_DIR="${BASE_DIR%/}"
DATAHOME="${DATAHOME%/}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DATAHOME}/elm_training_outputs/filtered_training/checkpoint-8215}"
BACKBONE_MODEL="${SCRIPT_DIR}/initial_elm_model"
OUTPUT_DIR="${OUTPUT_DIR:-${DATAHOME}/synthetic_notes}"
DATASET_PATH="${DATASET_PATH:-${DATAHOME}/encoded_testing_filtered}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/synthetic_notes_test_vanilla_seed42.txt}"
MANIFEST_FILE="${MANIFEST_FILE:-${OUTPUT_DIR}/synthetic_notes_test_vanilla_seed42_manifest.jsonl}"
SPLIT_NAME="${SPLIT_NAME:-test}"
GENERATION_CONDITION="${GENERATION_CONDITION:-vanilla}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH-${DATAHOME}/leakage_audit/split_manifest_note_level.csv}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_SHARDS="${NUM_SHARDS:-2}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-}"

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

if [ -n "${SPLIT_MANIFEST_PATH}" ]; then
    HAS_EMBEDDED_PROVENANCE=$(python - <<PY
from datasets import Dataset
dataset = Dataset.load_from_disk("${DATASET_PATH}")
cols = set(dataset.column_names)
required = {"dataset_row_id", "note_id", "subject_id", "hadm_id", "split"}
leakage_or_steering = {
    "patient_disjoint_from_train",
    "hadm_disjoint_from_train",
    "note_disjoint_from_train",
    "axis_id",
    "axis_label",
    "alpha",
    "source_split",
}
print(int(required.issubset(cols) and bool(cols & leakage_or_steering)))
PY
)
    if [ "${HAS_EMBEDDED_PROVENANCE}" = "1" ]; then
        echo "Detected embedded provenance/leakage metadata in dataset; disabling external split manifest join."
        SPLIT_MANIFEST_PATH=""
    fi
fi

echo "GPU Information:"
nvidia-smi
echo ""

echo "=========================================="
echo "Generation Configuration"
echo "=========================================="
echo "Data root: ${DATAHOME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Backbone model: ${BACKBONE_MODEL}"
echo "Dataset path: ${DATASET_PATH}"
echo "Output file: ${OUTPUT_FILE}"
echo "Manifest file: ${MANIFEST_FILE}"
if [ -n "${SPLIT_MANIFEST_PATH}" ]; then
    echo "Split manifest: ${SPLIT_MANIFEST_PATH}"
else
    echo "Split manifest: <disabled; rely on dataset-embedded metadata>"
fi
echo "Seed: ${SEED}"
echo "Batch size per shard: ${BATCH_SIZE}"
echo "Requested shards: ${NUM_SHARDS}"
echo "Global slice start: ${START_INDEX}"
echo "Global slice end: ${END_INDEX:-<dataset_end>}"
echo "=========================================="
echo ""

TOTAL_DATASET_ROWS=$(python - <<PY
from datasets import Dataset
print(len(Dataset.load_from_disk("${DATASET_PATH}")))
PY
)

if [ -n "${END_INDEX}" ]; then
  EFFECTIVE_END_INDEX="${END_INDEX}"
else
  EFFECTIVE_END_INDEX="${TOTAL_DATASET_ROWS}"
fi

SLICE_ROWS=$(python - <<PY
total_rows = int("${TOTAL_DATASET_ROWS}")
start_index = max(0, int("${START_INDEX}"))
end_index = min(int("${EFFECTIVE_END_INDEX}"), total_rows)
if end_index < start_index:
    raise ValueError(f"Invalid slice bounds: start={start_index}, end={end_index}, total={total_rows}")
print(end_index - start_index)
PY
)

echo "Total dataset rows: ${TOTAL_DATASET_ROWS}"
echo "Effective slice rows: ${SLICE_ROWS}"

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
    SHARD_START_INDEX=$(( START_INDEX + (SHARD_ID * SLICE_ROWS / NUM_SHARDS) ))
    SHARD_END_INDEX=$(( START_INDEX + ((SHARD_ID + 1) * SLICE_ROWS / NUM_SHARDS) ))
    SHARD_OUTPUT="${TMP_DIR}/$(basename "${OUTPUT_FILE%.txt}")_shard${SHARD_ID}.txt"
    SHARD_MANIFEST="${TMP_DIR}/$(basename "${MANIFEST_FILE%.jsonl}")_shard${SHARD_ID}.jsonl"

    SHARD_OUTPUTS+=("${SHARD_OUTPUT}")
    SHARD_MANIFESTS+=("${SHARD_MANIFEST}")

    echo "Launching shard ${SHARD_ID}: rows [${SHARD_START_INDEX}, ${SHARD_END_INDEX}) on GPU ${SHARD_ID}"
    GEN_ARGS=(
        --checkpoint_path "${CHECKPOINT_PATH}"
        --backbone_model_path "${BACKBONE_MODEL}"
        --dataset_path "${DATASET_PATH}"
        --output_file "${SHARD_OUTPUT}"
        --manifest_output "${SHARD_MANIFEST}"
        --generation_condition "${GENERATION_CONDITION}"
        --split "${SPLIT_NAME}"
        --batch_size "${BATCH_SIZE}"
        --repetition_penalty 1.2
        --temperature 0.8
        --top_p 0.9
        --top_k 50
        --max_new_tokens 2048
        --device cuda
        --seed "${SEED}"
        --start_index "${SHARD_START_INDEX}"
        --end_index "${SHARD_END_INDEX}"
    )
    if [ -n "${SPLIT_MANIFEST_PATH}" ]; then
        if [ ! -f "${SPLIT_MANIFEST_PATH}" ]; then
            echo "ERROR: Split manifest not found at ${SPLIT_MANIFEST_PATH}"
            exit 1
        fi
        GEN_ARGS+=(--split_manifest_path "${SPLIT_MANIFEST_PATH}")
    fi

    CUDA_VISIBLE_DEVICES="${SHARD_ID}" python "${SCRIPT_DIR}/generate_synthetic_notes.py" "${GEN_ARGS[@]}" &

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

output_path = Path("${OUTPUT_FILE}")
manifest_path = Path("${MANIFEST_FILE}")
tmp_dir = Path("${TMP_DIR}")

run_record = {
    "job_id": "${SLURM_JOB_ID}",
    "output_file": str(output_path),
    "manifest_file": str(manifest_path),
    "tmp_dir": str(tmp_dir),
    "dataset_path": "${DATASET_PATH}",
    "checkpoint_path": "${CHECKPOINT_PATH}",
    "seed": int("${SEED}"),
    "num_shards": int("${NUM_SHARDS}"),
    "batch_size": int("${BATCH_SIZE}"),
}
(manifest_path.parent / (manifest_path.name + ".run.json")).write_text(json.dumps(run_record, indent=2), encoding="utf-8")
PY

rm -rf "${TMP_DIR}"

echo ""
echo "=========================================="
echo "Generation completed successfully"
echo "End Time: $(date)"
echo "Output file: ${OUTPUT_FILE}"
echo "Manifest file: ${MANIFEST_FILE}"
echo "=========================================="
