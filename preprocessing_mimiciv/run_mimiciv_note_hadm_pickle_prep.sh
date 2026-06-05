#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  module load miniconda
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate haim
  python "${SCRIPT_DIR}/build_mimiciv_pickle_ds_note_hadm.py" --help
  exit 0
fi

MIMICIV_NOTE_DIR="${MIMICIV_NOTE_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimic-iv-note/2.2}"
MIMICIV_CORE_DIR="${MIMICIV_CORE_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1}"
OUTPUT_DIR="${OUTPUT_DIR:-${MIMICIV_CORE_DIR}/pickle_ds_note_hadm_all}"
MIN_TEXT_LEN="${MIN_TEXT_LEN:-50}"
CHUNKSIZE="${CHUNKSIZE:-250000}"
MAX_HADM="${MAX_HADM:-}"
CLIP_TO_ADMISSION_WINDOW="${CLIP_TO_ADMISSION_WINDOW:-0}"
DROP_WITHOUT_ADMISSION="${DROP_WITHOUT_ADMISSION:-0}"
OVERWRITE="${OVERWRITE:-0}"

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate haim

CMD=(
  python "${SCRIPT_DIR}/build_mimiciv_pickle_ds_note_hadm.py"
  --mimiciv_note_dir "${MIMICIV_NOTE_DIR}"
  --mimiciv_core_dir "${MIMICIV_CORE_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --min_text_len "${MIN_TEXT_LEN}"
  --chunksize "${CHUNKSIZE}"
)

if [[ -n "${MAX_HADM}" ]]; then
  CMD+=(--max_hadm "${MAX_HADM}")
fi

if [[ "${CLIP_TO_ADMISSION_WINDOW}" == "1" ]]; then
  CMD+=(--clip_to_admission_window)
fi

if [[ "${DROP_WITHOUT_ADMISSION}" == "1" ]]; then
  CMD+=(--drop_without_admission)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "Running MIMIC-IV note/HADM pickle prep"
echo "  MIMICIV_NOTE_DIR=${MIMICIV_NOTE_DIR}"
echo "  MIMICIV_CORE_DIR=${MIMICIV_CORE_DIR}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  MIN_TEXT_LEN=${MIN_TEXT_LEN}"
echo "  CHUNKSIZE=${CHUNKSIZE}"
echo "  MAX_HADM=${MAX_HADM:-<all>}"
echo "  CLIP_TO_ADMISSION_WINDOW=${CLIP_TO_ADMISSION_WINDOW}"
echo "  DROP_WITHOUT_ADMISSION=${DROP_WITHOUT_ADMISSION}"
echo "  OVERWRITE=${OVERWRITE}"

"${CMD[@]}"
