#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MIMIC3_DIR="${MIMIC3_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/physionet.org/files/mimiciii/1.4}"
OUTPUT_DIR="${OUTPUT_DIR:-${MIMIC3_DIR}/pickle_ds}"
CATEGORY="${CATEGORY:-Discharge summary}"
DESCRIPTION="${DESCRIPTION:-}"
MIN_TEXT_LEN="${MIN_TEXT_LEN:-50}"
CHUNKSIZE="${CHUNKSIZE:-250000}"
MAX_HADM="${MAX_HADM:-}"
OVERWRITE="${OVERWRITE:-0}"

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate haim

CMD=(
  python "${SCRIPT_DIR}/prepare_mimiciii_pickle_ds.py"
  --mimic3_dir "${MIMIC3_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --category "${CATEGORY}"
  --min_text_len "${MIN_TEXT_LEN}"
  --chunksize "${CHUNKSIZE}"
)

if [[ -n "${DESCRIPTION}" ]]; then
  CMD+=(--description "${DESCRIPTION}")
fi

if [[ -n "${MAX_HADM}" ]]; then
  CMD+=(--max_hadm "${MAX_HADM}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "Running MIMIC-III pickle prep"
echo "  MIMIC3_DIR=${MIMIC3_DIR}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  CATEGORY=${CATEGORY}"
echo "  DESCRIPTION=${DESCRIPTION:-<none>}"
echo "  MIN_TEXT_LEN=${MIN_TEXT_LEN}"
echo "  CHUNKSIZE=${CHUNKSIZE}"
echo "  MAX_HADM=${MAX_HADM:-<all>}"
echo "  OVERWRITE=${OVERWRITE}"

"${CMD[@]}"
