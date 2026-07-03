#!/bin/bash
#SBATCH --job-name=elm_audit_vanilla
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=gpu_devel

set -euo pipefail

SCRIPT_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis"
LOG_DIR="${SCRIPT_DIR}/log"
mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/audit_vanilla_generation_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/audit_vanilla_generation_${SLURM_JOB_ID}.err" >&2)

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start Time: $(date)"
echo "Working Directory: ${SCRIPT_DIR}"
echo "=========================================="
echo ""

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm

cd "${SCRIPT_DIR}"

MANIFEST_PATH="${MANIFEST_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/synthetic_notes/synthetic_notes_test_vanilla_seed42_manifest.jsonl}"
DATASET_PATH="${DATASET_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/encoded_testing_filtered}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/leakage_audit/split_manifest_note_level.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/generation_audit/vanilla_test_seed42}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-BAAI/bge-large-en-v1.5}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-256}"
PRECOMPUTED_GENERATED_EMBEDDINGS_PATH="${PRECOMPUTED_GENERATED_EMBEDDINGS_PATH:-}"
TRAINING_DATASET_PATH="${TRAINING_DATASET_PATH:-}"
PICKLE_DIR="${PICKLE_DIR:-}"
MANUAL_REVIEW_SAMPLE_SIZE="${MANUAL_REVIEW_SAMPLE_SIZE:-50}"
EXPECTED_GENERATION_CONDITION="${EXPECTED_GENERATION_CONDITION:-vanilla}"
ALLOW_SUBSET_DATASET_ROWS="${ALLOW_SUBSET_DATASET_ROWS:-0}"
NUM_AUDIT_SHARDS="${NUM_AUDIT_SHARDS:-2}"

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Audit Configuration"
echo "=========================================="
echo "Manifest path: ${MANIFEST_PATH}"
echo "Dataset path: ${DATASET_PATH}"
echo "Split manifest path: ${SPLIT_MANIFEST_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Embedding model: ${EMBEDDING_MODEL_NAME}"
echo "Embedding device: ${EMBEDDING_DEVICE}"
echo "Embedding batch size: ${EMBEDDING_BATCH_SIZE}"
echo "Precomputed generated embeddings: ${PRECOMPUTED_GENERATED_EMBEDDINGS_PATH:-<none>}"
echo "Training dataset path: ${TRAINING_DATASET_PATH:-<auto>}"
echo "Pickle dir: ${PICKLE_DIR:-<auto>}"
echo "Manual review sample size: ${MANUAL_REVIEW_SAMPLE_SIZE}"
echo "Expected generation condition: ${EXPECTED_GENERATION_CONDITION}"
echo "Allow subset dataset rows: ${ALLOW_SUBSET_DATASET_ROWS}"
echo "Requested audit shards: ${NUM_AUDIT_SHARDS}"
echo "=========================================="
echo ""

VISIBLE_GPUS="${SLURM_GPUS_ON_NODE:-1}"
if [ "${NUM_AUDIT_SHARDS}" -lt 1 ]; then
  echo "ERROR: NUM_AUDIT_SHARDS must be >= 1"
  exit 1
fi
if [ "${NUM_AUDIT_SHARDS}" -gt "${VISIBLE_GPUS}" ]; then
  echo "ERROR: NUM_AUDIT_SHARDS=${NUM_AUDIT_SHARDS} exceeds visible GPUs (${VISIBLE_GPUS})"
  exit 1
fi

MANIFEST_ROWS=$(python - <<PY
import pandas as pd
print(len(pd.read_json("${MANIFEST_PATH}", lines=True)))
PY
)

echo "Manifest rows: ${MANIFEST_ROWS}"

if [ "${NUM_AUDIT_SHARDS}" = "1" ]; then
  CMD=(
    python "${SCRIPT_DIR}/audit_vanilla_generation.py"
    --manifest_path "${MANIFEST_PATH}"
    --dataset_path "${DATASET_PATH}"
    --split_manifest_path "${SPLIT_MANIFEST_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --embedding_model_name "${EMBEDDING_MODEL_NAME}"
    --embedding_device "${EMBEDDING_DEVICE}"
    --embedding_batch_size "${EMBEDDING_BATCH_SIZE}"
    --sample_size_for_manual_review "${MANUAL_REVIEW_SAMPLE_SIZE}"
    --expected_generation_condition "${EXPECTED_GENERATION_CONDITION}"
  )
  if [ -n "${PRECOMPUTED_GENERATED_EMBEDDINGS_PATH}" ]; then
    CMD+=(--precomputed_generated_embeddings_path "${PRECOMPUTED_GENERATED_EMBEDDINGS_PATH}")
  fi
  if [ -n "${TRAINING_DATASET_PATH}" ]; then
    CMD+=(--training_dataset_path "${TRAINING_DATASET_PATH}")
  fi
  if [ -n "${PICKLE_DIR}" ]; then
    CMD+=(--pickle_dir "${PICKLE_DIR}")
  fi
  if [ "${ALLOW_SUBSET_DATASET_ROWS}" = "1" ]; then
    CMD+=(--allow_subset_dataset_rows)
  fi
  "${CMD[@]}"
else
  TMP_DIR="${OUTPUT_DIR}/tmp_audit_${SLURM_JOB_ID}"
  mkdir -p "${TMP_DIR}"
  PIDS=()
  SHARD_DIRS=()

  for ((SHARD_ID=0; SHARD_ID<NUM_AUDIT_SHARDS; SHARD_ID++)); do
    START_INDEX=$(( SHARD_ID * MANIFEST_ROWS / NUM_AUDIT_SHARDS ))
    END_INDEX=$(( (SHARD_ID + 1) * MANIFEST_ROWS / NUM_AUDIT_SHARDS ))
    SHARD_DIR="${TMP_DIR}/shard${SHARD_ID}"
    mkdir -p "${SHARD_DIR}"
    SHARD_DIRS+=("${SHARD_DIR}")
    echo "Launching audit shard ${SHARD_ID}: manifest rows [${START_INDEX}, ${END_INDEX}) on GPU ${SHARD_ID}"
    CMD=(
      python "${SCRIPT_DIR}/audit_vanilla_generation.py"
      --manifest_path "${MANIFEST_PATH}"
      --dataset_path "${DATASET_PATH}"
      --split_manifest_path "${SPLIT_MANIFEST_PATH}"
      --output_dir "${SHARD_DIR}"
      --embedding_model_name "${EMBEDDING_MODEL_NAME}"
      --embedding_device "${EMBEDDING_DEVICE}"
      --embedding_batch_size "${EMBEDDING_BATCH_SIZE}"
      --sample_size_for_manual_review "${MANUAL_REVIEW_SAMPLE_SIZE}"
      --expected_generation_condition "${EXPECTED_GENERATION_CONDITION}"
      --start_index "${START_INDEX}"
      --end_index "${END_INDEX}"
    )
    if [ -n "${PRECOMPUTED_GENERATED_EMBEDDINGS_PATH}" ]; then
      CMD+=(--precomputed_generated_embeddings_path "${PRECOMPUTED_GENERATED_EMBEDDINGS_PATH}")
    fi
    if [ -n "${TRAINING_DATASET_PATH}" ]; then
      CMD+=(--training_dataset_path "${TRAINING_DATASET_PATH}")
    fi
    if [ -n "${PICKLE_DIR}" ]; then
      CMD+=(--pickle_dir "${PICKLE_DIR}")
    fi
    if [ "${ALLOW_SUBSET_DATASET_ROWS}" = "1" ]; then
      CMD+=(--allow_subset_dataset_rows)
    fi
    CUDA_VISIBLE_DEVICES="${SHARD_ID}" "${CMD[@]}" &
    PIDS+=($!)
  done

  AUDIT_EXIT_CODE=0
  for PID in "${PIDS[@]}"; do
    wait "${PID}" || AUDIT_EXIT_CODE=$?
  done
  if [ "${AUDIT_EXIT_CODE}" -ne 0 ]; then
    echo "ERROR: One or more audit shards failed."
    exit "${AUDIT_EXIT_CODE}"
  fi

  python - <<PY
import json
from pathlib import Path
import pandas as pd
from audit_vanilla_generation import (
    load_manifest,
    load_split_manifest,
    integrity_checks,
    compute_group_summary,
    determine_readiness,
    markdown_report,
    normalize_text,
    text_hash,
)

manifest_path = Path("${MANIFEST_PATH}")
dataset_path = Path("${DATASET_PATH}")
split_manifest_path = Path("${SPLIT_MANIFEST_PATH}") if "${SPLIT_MANIFEST_PATH}" else None
output_dir = Path("${OUTPUT_DIR}")
tmp_dir = Path("${TMP_DIR}")
output_dir.mkdir(parents=True, exist_ok=True)

manifest_df = load_manifest(manifest_path)
dataset_row_count = int(__import__("datasets").Dataset.load_from_disk(str(dataset_path)).num_rows)
split_manifest_df = load_split_manifest(split_manifest_path) if split_manifest_path and split_manifest_path.exists() else None
integrity = integrity_checks(
    manifest_df,
    dataset_row_count,
    split_manifest_df,
    expected_generation_condition="${EXPECTED_GENERATION_CONDITION}" or None,
    allow_subset_dataset_rows=("${ALLOW_SUBSET_DATASET_ROWS}" == "1"),
)

quality_frames = []
faith_frames = []
shard_jsons = []
for shard_dir in sorted(tmp_dir.glob("shard*")):
    quality_frames.append(pd.read_csv(shard_dir / "vanilla_quality_table.csv"))
    faith_frames.append(pd.read_csv(shard_dir / "vanilla_faithfulness_table.csv"))
    shard_jsons.append(json.loads((shard_dir / "generation_audit_baseline.json").read_text()))

quality_df = pd.concat(quality_frames, ignore_index=True).sort_values("generation_index").reset_index(drop=True)
faith_df = pd.concat(faith_frames, ignore_index=True).sort_values("generation_index").reset_index(drop=True)
optional_faith_defaults = {
    "nearest_train_10gram_overlap_count": float("nan"),
    "nearest_train_lexical_similarity": float("nan"),
    "exact_duplicate_vs_train_text": False,
}
for col, default in optional_faith_defaults.items():
    if col not in faith_df.columns:
        faith_df[col] = default
merged_df = quality_df.merge(
    faith_df[
        [
            "generation_id",
            "generation_index",
            "source_to_generated_cosine",
            "source_retrieval_rank_top10",
            "source_in_top1",
            "source_in_top5",
            "source_in_top10",
            "nearest_train_embedding_cosine",
            "nearest_train_dataset_row_id",
            "nearest_train_10gram_overlap_count",
            "nearest_train_lexical_similarity",
            "exact_duplicate_vs_train_text",
        ]
    ],
    on=["generation_id", "generation_index"],
    how="left",
)

generated_hashes = manifest_df["generated_text"].fillna("").map(normalize_text).map(text_hash)
exact_duplicate_generated_notes = int(generated_hashes.duplicated(keep=False).sum())
privacy_summary = {
    "exact_duplicate_generated_notes": exact_duplicate_generated_notes,
    "exact_duplicates_vs_train": int(merged_df["exact_duplicate_vs_train_text"].fillna(False).sum()),
    "phi_flagged_note_count": int(merged_df["phi_like_flag"].fillna(False).sum()),
    "high_train_embedding_neighbor_count_gt_0_95": int((merged_df["nearest_train_embedding_cosine"] > 0.95).fillna(False).sum()),
    "train_text_checks_skipped_reason": shard_jsons[0]["privacy_summary"].get("train_text_checks_skipped_reason"),
}

group_rows = [compute_group_summary(merged_df, "full")]
for group_name in ["patient_disjoint", "patient_overlap"]:
    subset = merged_df.loc[merged_df["leakage_group"] == group_name].copy()
    group_rows.append(compute_group_summary(subset, group_name))
group_df = pd.DataFrame(group_rows)

readiness_status, readiness_reasons = determine_readiness(integrity, merged_df, privacy_summary)
template = shard_jsons[0]
template["created_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
template["output_dir"] = str(output_dir.resolve())
template["integrity"] = integrity
template["quality_summary"] = {
    "empty_output_rate": float(merged_df["empty_output_flag_recomputed"].mean()),
    "too_short_rate": float(merged_df["too_short_flag_recomputed"].mean()),
    "repetition_or_collapse_rate": float(merged_df["repetition_or_collapse_flag_recomputed"].mean()),
    "word_count": {
        "mean": float(merged_df["generated_word_count_recomputed"].mean()),
        "median": float(merged_df["generated_word_count_recomputed"].median()),
        "iqr": float(merged_df["generated_word_count_recomputed"].quantile(0.75) - merged_df["generated_word_count_recomputed"].quantile(0.25)),
        "p05": float(merged_df["generated_word_count_recomputed"].quantile(0.05)),
        "p25": float(merged_df["generated_word_count_recomputed"].quantile(0.25)),
        "p75": float(merged_df["generated_word_count_recomputed"].quantile(0.75)),
        "min": float(merged_df["generated_word_count_recomputed"].min()),
        "max": float(merged_df["generated_word_count_recomputed"].max()),
      },
    "char_count": {
        "mean": float(merged_df["generated_char_count_recomputed"].mean()),
        "median": float(merged_df["generated_char_count_recomputed"].median()),
        "iqr": float(merged_df["generated_char_count_recomputed"].quantile(0.75) - merged_df["generated_char_count_recomputed"].quantile(0.25)),
        "p05": float(merged_df["generated_char_count_recomputed"].quantile(0.05)),
        "p25": float(merged_df["generated_char_count_recomputed"].quantile(0.25)),
        "p75": float(merged_df["generated_char_count_recomputed"].quantile(0.75)),
        "min": float(merged_df["generated_char_count_recomputed"].min()),
        "max": float(merged_df["generated_char_count_recomputed"].max()),
      },
    "section_header_count": {
        "mean": float(merged_df["section_header_count"].mean()),
        "median": float(merged_df["section_header_count"].median()),
        "iqr": float(merged_df["section_header_count"].quantile(0.75) - merged_df["section_header_count"].quantile(0.25)),
        "p05": float(merged_df["section_header_count"].quantile(0.05)),
        "p25": float(merged_df["section_header_count"].quantile(0.25)),
        "p75": float(merged_df["section_header_count"].quantile(0.75)),
        "min": float(merged_df["section_header_count"].min()),
        "max": float(merged_df["section_header_count"].max()),
      },
    "minimum_section_structure_rate": float(merged_df["has_minimum_section_structure"].mean()),
}
template["faithfulness_summary"] = {
    "source_to_generated_cosine": {
        "mean": float(merged_df["source_to_generated_cosine"].mean()),
        "median": float(merged_df["source_to_generated_cosine"].median()),
        "iqr": float(merged_df["source_to_generated_cosine"].quantile(0.75) - merged_df["source_to_generated_cosine"].quantile(0.25)),
        "p05": float(merged_df["source_to_generated_cosine"].quantile(0.05)),
        "p25": float(merged_df["source_to_generated_cosine"].quantile(0.25)),
        "p75": float(merged_df["source_to_generated_cosine"].quantile(0.75)),
        "min": float(merged_df["source_to_generated_cosine"].min()),
        "max": float(merged_df["source_to_generated_cosine"].max()),
    },
    "source_in_top1_rate": float(merged_df["source_in_top1"].fillna(False).mean()),
    "source_in_top5_rate": float(merged_df["source_in_top5"].fillna(False).mean()),
    "source_in_top10_rate": float(merged_df["source_in_top10"].fillna(False).mean()),
}
template["privacy_summary"] = privacy_summary
template["readiness_status"] = readiness_status
template["readiness_reasons"] = readiness_reasons
template["config_snapshot"]["num_audit_shards"] = ${NUM_AUDIT_SHARDS}

quality_df.to_csv(output_dir / "vanilla_quality_table.csv", index=False)
faith_df.to_csv(output_dir / "vanilla_faithfulness_table.csv", index=False)
group_df.to_csv(output_dir / "patient_disjoint_vs_full_metrics.csv", index=False)

manual_review_df = manifest_df.merge(
    quality_df[
        [
            "generation_id",
            "patient_disjoint_from_train",
            "leakage_group",
            "generated_word_count_recomputed",
            "repetition_or_collapse_flag_recomputed",
            "phi_like_flag",
        ]
    ],
    on="generation_id",
    how="left",
)
manual_review_df = manual_review_df.merge(
    faith_df[
        [
            "generation_id",
            "patient_disjoint_from_train",
            "source_to_generated_cosine",
        ]
    ],
    on="generation_id",
    how="left",
    suffixes=("", "_faith"),
)
if "patient_disjoint_from_train_faith" in manual_review_df.columns:
    manual_review_df.drop(columns=["patient_disjoint_from_train_faith"], inplace=True)
manual_review_df = manual_review_df.sample(
    n=min(int("${MANUAL_REVIEW_SAMPLE_SIZE}"), len(manual_review_df)),
    random_state=42,
)[
    [
        "generation_id",
        "note_id",
        "subject_id",
        "hadm_id",
        "patient_disjoint_from_train",
        "leakage_group",
        "source_to_generated_cosine",
        "generated_word_count_recomputed",
        "repetition_or_collapse_flag_recomputed",
        "phi_like_flag",
        "generated_text",
    ]
]
manual_review_df.to_csv(output_dir / "manual_review_sample.csv", index=False)

(output_dir / "generation_audit_baseline.json").write_text(json.dumps(template, indent=2))
(output_dir / "generation_audit_baseline.md").write_text(markdown_report(template, integrity, group_df, privacy_summary))
print("Merged shard audit outputs into:", output_dir)
print("Readiness status:", readiness_status)
for reason in readiness_reasons:
    print("-", reason)
PY
fi

echo ""
echo "=========================================="
echo "Audit finished at $(date)"
echo "=========================================="
