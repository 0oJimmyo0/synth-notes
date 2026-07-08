#!/bin/bash
#SBATCH --job-name=closed_loop_enrich_2g
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/closed_loop_output_enrichment/closed_loop_output_enrichment_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/closed_loop_output_enrichment/closed_loop_output_enrichment_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=gpu

set -euo pipefail

OPEN_ELM_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm"
SCRIPT_PATH="${OPEN_ELM_DIR}/cav_axis/closed_loop_output_enrichment.py"
LOG_DIR="${OPEN_ELM_DIR}/log/closed_loop_output_enrichment"
mkdir -p "${LOG_DIR}"
trap 'status=$?; echo "[ERROR] ${BASH_SOURCE[0]} failed at line ${LINENO} with exit code ${status}" >&2; exit ${status}' ERR

required_vars=(
  ANCHOR_MANIFEST_PATH
  DATASET_PATH
  CHECKPOINT_PATH
  BACKBONE_PATH
  CLUSTER_ASSIGNMENTS_PATH
  TARGET_CLUSTER_IDS
  OUTPUT_DIR
)
for var_name in "${required_vars[@]}"; do
  if [ -z "${!var_name:-}" ]; then
    echo "ERROR: Missing required environment variable: ${var_name}"
    exit 1
  fi
done

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm

cd "${OPEN_ELM_DIR}"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo ""

mkdir -p "${OUTPUT_DIR}"

NUM_SHARDS=2
PIDS=()

run_shard () {
  local shard_id="$1"
  local gpu_id="$2"
  echo "Launching shard ${shard_id}/${NUM_SHARDS} on GPU ${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" python "${SCRIPT_PATH}" \
    --anchor_manifest_path "${ANCHOR_MANIFEST_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --backbone_path "${BACKBONE_PATH}" \
    --cluster_assignments_path "${CLUSTER_ASSIGNMENTS_PATH}" \
    --target_cluster_ids "${TARGET_CLUSTER_IDS}" \
    --output_dir "${OUTPUT_DIR}" \
    --embedding_model_name "${EMBEDDING_MODEL_NAME:-BAAI/bge-large-en-v1.5}" \
    --split_manifest_path "${SPLIT_MANIFEST_PATH:-}" \
    --source_split "${SOURCE_SPLIT:-test}" \
    --n_candidates_per_anchor "${N_CANDIDATES_PER_ANCHOR:-8}" \
    --accepted_per_anchor "${ACCEPTED_PER_ANCHOR:-1}" \
    --seeds "${SEEDS:-42,43,44,45,46,47,48,49}" \
    --temperature_values "${TEMPERATURE_VALUES:-0.8}" \
    --top_p_values "${TOP_P_VALUES:-0.9}" \
    --top_k_values "${TOP_K_VALUES:-50}" \
    --batch_size "${BATCH_SIZE:-4}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-2048}" \
    --min_new_tokens "${MIN_NEW_TOKENS:-0}" \
    --repetition_penalty "${REPETITION_PENALTY:-1.2}" \
    --clinic_note_prompt_mode "${CLINIC_NOTE_PROMPT_MODE:-default}" \
    --device "${DEVICE:-cuda}" \
    --embedding_device "${EMBEDDING_DEVICE:-cuda}" \
    --embedding_batch_size "${EMBEDDING_BATCH_SIZE:-256}" \
    --min_word_count "${MIN_WORD_COUNT:-100}" \
    --min_source_cosine "${MIN_SOURCE_COSINE:-0.78}" \
    --target_centroid_threshold_quantile "${TARGET_CENTROID_THRESHOLD_QUANTILE:-0.90}" \
    --candidate_near_duplicate_cosine "${CANDIDATE_NEAR_DUPLICATE_COSINE:-0.995}" \
    --anchor_diversity_cosine "${ANCHOR_DIVERSITY_COSINE:-0.992}" \
    --manual_review_sample_size "${MANUAL_REVIEW_SAMPLE_SIZE:-50}" \
    --max_train_texts_for_overlap "${MAX_TRAIN_TEXTS_FOR_OVERLAP:-50000}" \
    --selection_strategy "${SELECTION_STRATEGY:-best_target_score}" \
    --shard_id "${shard_id}" \
    --num_shards "${NUM_SHARDS}" \
    ${TARGET_BASIN_PATH:+--target_basin_path "${TARGET_BASIN_PATH}"} \
    ${TARGET_CENTROID_PATH:+--target_centroid_path "${TARGET_CENTROID_PATH}"} \
    ${TRAIN_TEXT_PATH:+--train_text_path "${TRAIN_TEXT_PATH}"} \
    ${PICKLE_DIR:+--pickle_dir "${PICKLE_DIR}"} \
    ${JOIN_COLS:+--join_cols "${JOIN_COLS}"} &
  PIDS+=($!)
}

run_shard 0 0
run_shard 1 1

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

OUTPUT_DIR_PY="${OUTPUT_DIR}" python - <<'PY'
import json
import os
from pathlib import Path
import pandas as pd
import math

outdir = Path(os.environ["OUTPUT_DIR_PY"])
num_shards = 2

def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(path, lines=True, dtype=False)

def safe_concat_sort(frames, sort_cols):
    nonempty = [frame for frame in frames if frame is not None and len(frame)]
    if not nonempty:
        return pd.DataFrame(columns=sort_cols)
    merged = pd.concat(nonempty, ignore_index=True)
    usable_sort_cols = [col for col in sort_cols if col in merged.columns]
    if usable_sort_cols:
        merged = merged.sort_values(usable_sort_cols).reset_index(drop=True)
    else:
        merged = merged.reset_index(drop=True)
    return merged

candidate_frames = []
accepted_frames = []
rejected_frames = []
faithfulness_frames = []
anchor_frames = []
patient_frames = []
coverage_frames = []
review_frames = []
summary_objs = []

for shard_id in range(num_shards):
    suffix = f"_shard{shard_id:02d}of{num_shards:02d}"
    candidate_frames.append(load_jsonl(outdir / f"closed_loop_candidate_manifest{suffix}.jsonl"))
    accepted_frames.append(load_jsonl(outdir / f"closed_loop_accepted_manifest{suffix}.jsonl"))
    rejected_frames.append(load_jsonl(outdir / f"closed_loop_rejected_manifest{suffix}.jsonl"))
    faithfulness_frames.append(pd.read_csv(outdir / f"accepted_vs_rejected_faithfulness{suffix}.csv"))
    anchor_frames.append(pd.read_csv(outdir / f"acceptance_rate_by_anchor{suffix}.csv"))
    patient_frames.append(pd.read_csv(outdir / f"acceptance_rate_by_patient_disjoint{suffix}.csv"))
    coverage_frames.append(pd.read_csv(outdir / f"target_region_coverage_before_after{suffix}.csv"))
    review_frames.append(pd.read_csv(outdir / f"accepted_note_manual_review_sample{suffix}.csv"))
    summary_objs.append(json.loads((outdir / f"closed_loop_enrichment_summary{suffix}.json").read_text()))

candidate_df = safe_concat_sort(candidate_frames, ["anchor_id", "candidate_index"])
accepted_df = safe_concat_sort(accepted_frames, ["anchor_id", "candidate_index"])
rejected_df = safe_concat_sort(rejected_frames, ["anchor_id", "candidate_index"])
faithfulness_df = pd.concat(faithfulness_frames, ignore_index=True) if faithfulness_frames else pd.DataFrame()
anchor_df = safe_concat_sort(anchor_frames, ["anchor_id"])
patient_df = pd.concat(patient_frames, ignore_index=True) if patient_frames else pd.DataFrame()
coverage_df = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame()
review_nonempty = [frame for frame in review_frames if frame is not None and len(frame)]
review_df = pd.concat(review_nonempty, ignore_index=True) if review_nonempty else pd.DataFrame()

candidate_df.to_json(outdir / "closed_loop_candidate_manifest.jsonl", orient="records", lines=True)
accepted_df.to_json(outdir / "closed_loop_accepted_manifest.jsonl", orient="records", lines=True)
rejected_df.to_json(outdir / "closed_loop_rejected_manifest.jsonl", orient="records", lines=True)
faithfulness_df.to_csv(outdir / "accepted_vs_rejected_faithfulness.csv", index=False)
anchor_df.to_csv(outdir / "acceptance_rate_by_anchor.csv", index=False)

if "patient_disjoint_from_train" in patient_df.columns:
    patient_summary = (
        patient_df.groupby("patient_disjoint_from_train", dropna=False, as_index=False)[["n_candidates", "n_accepted"]]
        .sum()
    )
    patient_summary["acceptance_rate"] = patient_summary["n_accepted"] / patient_summary["n_candidates"].replace(0, pd.NA)
    patient_summary.to_csv(outdir / "acceptance_rate_by_patient_disjoint.csv", index=False)
else:
    patient_df.to_csv(outdir / "acceptance_rate_by_patient_disjoint.csv", index=False)

coverage_summary = (
    coverage_df.groupby(["analysis_group", "cluster_id", "is_target_cluster"], dropna=False, as_index=False)[["anchor_count", "accepted_count"]]
    .sum()
)
for group_name, group_df in coverage_summary.groupby("analysis_group"):
    anchor_total = group_df["anchor_count"].sum() or 1
    accepted_total = group_df["accepted_count"].sum() or 1
    idx = coverage_summary["analysis_group"] == group_name
    coverage_summary.loc[idx, "anchor_fraction"] = coverage_summary.loc[idx, "anchor_count"] / anchor_total
    coverage_summary.loc[idx, "accepted_fraction"] = coverage_summary.loc[idx, "accepted_count"] / accepted_total
coverage_summary.to_csv(outdir / "target_region_coverage_before_after.csv", index=False)

review_df = review_df.head(50)
review_df.to_csv(outdir / "accepted_note_manual_review_sample.csv", index=False)

def mean_or_nan(df, col):
    return float(pd.to_numeric(df[col], errors="coerce").mean()) if len(df) and col in df.columns else math.nan

def rate_or_nan(df, col):
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float).mean()) if len(df) and col in df.columns else math.nan

summary = dict(summary_objs[0])
summary["shard_id"] = None
summary["num_shards"] = num_shards
summary["total_anchors"] = int(anchor_df["anchor_id"].nunique()) if "anchor_id" in anchor_df.columns else 0
summary["total_candidates_generated"] = int(len(candidate_df))
summary["total_accepted"] = int(len(accepted_df))
summary["acceptance_rate"] = float(len(accepted_df) / len(candidate_df)) if len(candidate_df) else math.nan
summary["accepted_target_cluster_rate"] = rate_or_nan(accepted_df, "nearest_cluster_in_target")
summary["accepted_centroid_distance_pass_rate"] = rate_or_nan(accepted_df, "target_centroid_distance_pass")
summary["accepted_mean_source_cosine"] = mean_or_nan(accepted_df, "source_synthetic_cosine")
summary["rejected_mean_source_cosine"] = mean_or_nan(rejected_df, "source_synthetic_cosine")
summary["accepted_collapse_rate"] = rate_or_nan(accepted_df, "repetition_or_collapse_flag")
summary["rejected_collapse_rate"] = rate_or_nan(rejected_df, "repetition_or_collapse_flag")
summary["accepted_phi_warning_rate"] = rate_or_nan(accepted_df, "phi_warning_flag")
summary["rejected_phi_warning_rate"] = rate_or_nan(rejected_df, "phi_warning_flag")
summary["privacy_warning_rates"] = {
    "accepted_exact_duplicate_vs_train_rate": rate_or_nan(accepted_df, "exact_duplicate_vs_train_flag"),
    "rejected_exact_duplicate_vs_train_rate": rate_or_nan(rejected_df, "exact_duplicate_vs_train_flag"),
    "accepted_high_ngram_overlap_rate": rate_or_nan(accepted_df, "high_ngram_overlap_with_train_flag"),
    "rejected_high_ngram_overlap_rate": rate_or_nan(rejected_df, "high_ngram_overlap_with_train_flag"),
}
summary["output_files"] = {
    "closed_loop_candidate_manifest": str(outdir / "closed_loop_candidate_manifest.jsonl"),
    "closed_loop_accepted_manifest": str(outdir / "closed_loop_accepted_manifest.jsonl"),
    "closed_loop_rejected_manifest": str(outdir / "closed_loop_rejected_manifest.jsonl"),
    "closed_loop_enrichment_summary_json": str(outdir / "closed_loop_enrichment_summary.json"),
    "closed_loop_enrichment_summary_md": str(outdir / "closed_loop_enrichment_summary.md"),
    "accepted_vs_rejected_faithfulness": str(outdir / "accepted_vs_rejected_faithfulness.csv"),
    "target_region_coverage_before_after": str(outdir / "target_region_coverage_before_after.csv"),
    "acceptance_rate_by_anchor": str(outdir / "acceptance_rate_by_anchor.csv"),
    "acceptance_rate_by_patient_disjoint": str(outdir / "acceptance_rate_by_patient_disjoint.csv"),
    "accepted_note_manual_review_sample": str(outdir / "accepted_note_manual_review_sample.csv"),
}

def markdown(summary):
    return "\n".join([
        "# Closed-Loop Output-Space Enrichment",
        "",
        "## Status",
        f"- Method: `closed-loop output-space selection`",
        f"- Total anchors: `{summary['total_anchors']}`",
        f"- Candidates per anchor: `{summary['n_candidates_per_anchor']}`",
        f"- Total candidates: `{summary['total_candidates_generated']}`",
        f"- Total accepted: `{summary['total_accepted']}`",
        f"- Acceptance rate: `{summary['acceptance_rate']:.4f}`",
        "",
        "## Target Metrics",
        f"- Accepted target-cluster hit rate: `{summary['accepted_target_cluster_rate']:.4f}`",
        f"- Accepted centroid-distance pass rate: `{summary['accepted_centroid_distance_pass_rate']:.4f}`",
        f"- Accepted mean source cosine: `{summary['accepted_mean_source_cosine']:.4f}`",
        f"- Rejected mean source cosine: `{summary['rejected_mean_source_cosine']:.4f}`",
        "",
        "## Privacy / Quality",
        f"- Accepted collapse rate: `{summary['accepted_collapse_rate']:.4f}`",
        f"- Rejected collapse rate: `{summary['rejected_collapse_rate']:.4f}`",
        f"- Accepted PHI-warning rate: `{summary['accepted_phi_warning_rate']:.4f}`",
        f"- Rejected PHI-warning rate: `{summary['rejected_phi_warning_rate']:.4f}`",
        f"- Train overlap screen: `{summary['train_overlap_screen_status']}`",
        "",
    ])

(outdir / "closed_loop_enrichment_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
(outdir / "closed_loop_enrichment_summary.md").write_text(markdown(summary))
print("Merged shard outputs into:", outdir)
PY

echo ""
echo "=========================================="
echo "Job Completed Successfully"
echo "End Time: $(date)"
echo "=========================================="
