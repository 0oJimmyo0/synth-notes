# MIMIC-III Pipeline Guide (matching current MIMIC-IV ELM workflow)

This guide mirrors your existing flow:

1. build `pickle_ds` with `.dsnotes`
2. generate sentence embeddings from `pickle_ds`
3. build train/dev/test HuggingFace datasets
4. filter long sequences (optional but recommended)
5. train ELM and generate synthetic notes

## 0) Raw data location and unzip note

Current raw location:

- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/physionet.org/files/mimiciii/1.4`

This folder already contains:

- `NOTEEVENTS.csv.gz`
- `ADMISSIONS.csv.gz`
- `PATIENTS.csv.gz`

You do **not** need to unzip first. Pandas can read `.csv.gz` directly.

## 1) Build MIMIC-III `pickle_ds`

Use the script in this folder:

- `prepare_mimiciii_pickle_ds.py`

It creates one pickle per `(subject_id, hadm_id)` and stores discharge notes in `.dsnotes`.

Quick smoke test (first 200 admissions only):

```bash
cd /gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/mimic-iii_scripts

MIMIC3_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/physionet.org/files/mimiciii/1.4 \
OUTPUT_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciii/1.4/pickle_ds_smoke \
MAX_HADM=200 \
bash run_mimiciii_pickle_prep.sh
```

Full run:

```bash
cd /gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/mimic-iii_scripts

MIMIC3_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/physionet.org/files/mimiciii/1.4 \
OUTPUT_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciii/1.4/pickle_ds \
bash run_mimiciii_pickle_prep.sh
```

Expected outputs:

- `pickle_ds/00000000.pkl` ...
- `pickle_ds/export_summary_mimiciii_ds.csv`

## 2) Generate sentence embeddings

Use the existing launcher with path overrides:

```bash
cd /gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm

INPUT_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciii/1.4/pickle_ds \
BASE_OUTPUT_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciii/1.4 \
MODEL=BAAI/bge-large-en-v1.5 \
BATCH_SIZE=32 \
bash run_sentence_embeddings.sh
```

Expected embedding outputs:

- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciii/1.4/embeddings-BAAI-bge-large-en-v1.5/sentence_embeddings.npy`
- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciii/1.4/embeddings-BAAI-bge-large-en-v1.5/sentence_embeddings_metadata.csv`

## 3) Build train/dev/test HF datasets

```bash
cd /gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm

module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate elm

python prep_hf_dataset/post_emb_dataprep.py \
  --base_dir /gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciii/1.4 \
  --embedding_subdir embeddings-BAAI-bge-large-en-v1.5 \
  --pickle_subdir pickle_ds \
  --output_base_dir /gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciii/1.4/data/clinic_notes/1_task \
  --output_suffix _full \
  --train_ratio 0.8 \
  --dev_ratio 0.1 \
  --test_ratio 0.1 \
  --random_seed 42 \
  --model_path meta-llama/Llama-3.1-8B-Instruct
```

Output folders:

- `encoded_training_full`
- `encoded_dev_full`
- `encoded_testing_full`

## 4) Optional filter step (recommended)

```bash
cd /gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm

module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate elm

python filter_long_sequences.py \
  --base_dir /gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciii/1.4/data/clinic_notes/1_task \
  --dataset_suffix _full \
  --filter_all
```

Filtered outputs:

- `encoded_training_filtered`
- `encoded_dev_filtered`
- `encoded_testing_filtered`

## 5) Train ELM on MIMIC-III

```bash
cd /gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm

BASE_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciii/1.4 \
DATAHOME=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciii/1.4/data/clinic_notes/1_task/ \
OUTPUT_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciii/1.4/data/clinic_notes/1_task/elm_training_outputs \
INITIAL_MODEL=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/initial_elm_model \
sbatch train_clinic_notes.slurm
```

## Consistency checks

- Embedding matrix rows must equal metadata rows.
- Metadata `filename` and `note_id` must map back to note text in `pickle_ds`.
- Keep the same embedding model and split seed when comparing MIMIC-IV vs MIMIC-III.
- Keep ELM hyperparameters fixed for fair comparison.
