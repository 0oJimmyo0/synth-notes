#!/bin/bash
# Copy and verify the retired legacy pickle store without deleting its source.
# Source deletion is intentionally a separate, manually approved operation.
#SBATCH --job-name=legacy_pickle_xfer
#SBATCH --output=/gpfs/radev/scratch/xu_hua/shared/data/EHR/transfer_logs/legacy_pickle_xfer_%j.out
#SBATCH --error=/gpfs/radev/scratch/xu_hua/shared/data/EHR/transfer_logs/legacy_pickle_xfer_%j.err
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00

set -euo pipefail

SOURCE_DIR=${SOURCE_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/pickle_ds}
DEST_DIR=${DEST_DIR:-/gpfs/radev/scratch/xu_hua/shared/data/EHR/pickle_ds}
TRANSFER_ID=${TRANSFER_ID:-legacy_pickle_ds_v1}
TRANSFER_MODE=${TRANSFER_MODE:-copy}
WORK_DIR=${WORK_DIR:-$(dirname "$DEST_DIR")/transfer_logs/$TRANSFER_ID}
WORKERS=${WORKERS:-4}

[[ -d "$SOURCE_DIR" ]] || { echo "ERROR: Source directory not found: $SOURCE_DIR" >&2; exit 1; }
[[ "$TRANSFER_MODE" == "copy" || "$TRANSFER_MODE" == "verify" ]] || {
  echo "ERROR: TRANSFER_MODE must be copy or verify" >&2; exit 1;
}

mkdir -p "$DEST_DIR" "$WORK_DIR"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Mode: ${TRANSFER_MODE}"
echo "Source: ${SOURCE_DIR}"
echo "Destination: ${DEST_DIR}"
echo "Work directory: ${WORK_DIR}"
echo "Workers: ${WORKERS}"

MANIFEST="$WORK_DIR/source_manifest.tsv"
FILE_LIST="$WORK_DIR/source_files.txt"

if [[ ! -s "$MANIFEST" ]]; then
  find "$SOURCE_DIR" -maxdepth 1 -type f -printf '%f\t%s\n' | sort > "$MANIFEST"
fi
cut -f1 "$MANIFEST" > "$FILE_LIST"

if [[ "$TRANSFER_MODE" == "verify" ]]; then
  for index in $(seq 0 $((WORKERS - 1))); do
    shard="$WORK_DIR/files.${index}"
    if [[ ! -s "$shard" ]]; then
      awk -v workers="$WORKERS" -v target="$index" '((NR - 1) % workers) == target {print}' "$FILE_LIST" > "$shard"
    fi
  done
  pids=()
  for index in $(seq 0 $((WORKERS - 1))); do
    rsync -arcni \
      --files-from="$WORK_DIR/files.${index}" "$SOURCE_DIR/" "$DEST_DIR/" \
      > "$WORK_DIR/verify_differences_${index}.txt" \
      2> "$WORK_DIR/verify_worker_${index}.err" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  cat "$WORK_DIR"/verify_differences_*.txt > "$WORK_DIR/checksum_differences.txt"
  if [[ "$status" -ne 0 ]]; then
    echo "ERROR: At least one checksum worker failed." >&2
    exit "$status"
  fi
  source_count=$(wc -l < "$MANIFEST")
  dest_count=$(find "$DEST_DIR" -maxdepth 1 -type f | wc -l)
  source_bytes=$(awk -F'\t' '{sum += $2} END {print sum}' "$MANIFEST")
  dest_bytes=$(find "$DEST_DIR" -maxdepth 1 -type f -printf '%s\n' | awk '{sum += $1} END {print sum}')
  printf 'source_files=%s\ndestination_files=%s\nsource_bytes=%s\ndestination_bytes=%s\n' \
    "$source_count" "$dest_count" "$source_bytes" "$dest_bytes" | tee "$WORK_DIR/verification_summary.txt"
  exit 0
fi

for index in $(seq 0 $((WORKERS - 1))); do
  shard="$WORK_DIR/files.${index}"
  if [[ ! -s "$shard" ]]; then
    awk -v workers="$WORKERS" -v target="$index" '((NR - 1) % workers) == target {print}' "$FILE_LIST" > "$shard"
  fi
done

pids=()
for index in $(seq 0 $((WORKERS - 1))); do
  shard="$WORK_DIR/files.${index}"
  rsync -a --partial --append-verify --human-readable --info=progress2,stats2 \
    --files-from="$shard" "$SOURCE_DIR/" "$DEST_DIR/" \
    > "$WORK_DIR/worker_${index}.log" 2>&1 &
  pids+=("$!")
done

while :; do
  running=0
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      running=1
      break
    fi
  done
  dest_files=$(find "$DEST_DIR" -maxdepth 1 -type f | wc -l)
  dest_bytes=$(find "$DEST_DIR" -maxdepth 1 -type f -printf '%s\n' | awk '{sum += $1} END {print sum}')
  printf '%s destination_files=%s destination_bytes=%s\n' "$(date --iso-8601=seconds)" "$dest_files" "$dest_bytes" | tee -a "$WORK_DIR/progress.log"
  [[ "$running" -eq 1 ]] || break
  sleep 300
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

if [[ "$status" -ne 0 ]]; then
  echo "ERROR: At least one rsync worker failed. Rerun copy mode to resume." >&2
  exit "$status"
fi

echo "Copy stage complete. Run TRANSFER_MODE=verify before any source deletion."
