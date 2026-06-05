# MIMIC-IV Note/HADM Pipeline (All Discharge Notes)

This pipeline removes ICU-stay cohorting and builds `pickle_ds` directly from all rows in:

- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimic-iv-note/2.2/note/discharge.csv`

The downstream flow remains unchanged:

1. build `pickle_ds` (`Patient_ICU` objects with `.dsnotes`)
2. generate sentence embeddings
3. build HF train/dev/test datasets
4. optional long-sequence filtering
5. train ELM

## 1) Build note/HADM pickle_ds from all discharge notes

Run:

```bash
cd /gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/preprocessing_mimiciv

MIMICIV_NOTE_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimic-iv-note/2.2 \
MIMICIV_CORE_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1 \
OUTPUT_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/pickle_ds_note_hadm_all \
bash run_mimiciv_note_hadm_pickle_prep.sh
```

Smoke test:

```bash
MAX_HADM=500 \
OUTPUT_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/pickle_ds_note_hadm_smoke \
bash run_mimiciv_note_hadm_pickle_prep.sh
```

Outputs:

- `pickle_ds_note_hadm_all/00000000.pkl` ...
- `pickle_ds_note_hadm_all/export_summary_mimiciv_note_hadm.csv`

## 2) Generate sentence embeddings

Use existing embedding script, but point it to the new pickle folder and a new output path:

```bash
module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate elm

python /gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/generate_sentence_embeddings.py \
  --input_dir /gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/pickle_ds_note_hadm_all \
  --output_dir /gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/embeddings_note_hadm_all/BAAI-bge-large-en-v1.5 \
  --model BAAI/bge-large-en-v1.5 \
  --batch_size 32 \
  --device auto
```

Expected:

- `sentence_embeddings.npy`
- `sentence_embeddings_metadata.csv`

## 3) Build HF datasets (unchanged logic)

```bash
module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate elm

python /gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/prep_hf_dataset/post_emb_dataprep.py \
  --embedding_dir /gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/embeddings_note_hadm_all/BAAI-bge-large-en-v1.5 \
  --pickle_dir /gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/pickle_ds_note_hadm_all \
  --output_base_dir /gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task \
  --output_suffix _full \
  --train_ratio 0.8 \
  --dev_ratio 0.1 \
  --test_ratio 0.1 \
  --random_seed 42 \
  --model_path meta-llama/Llama-3.1-8B-Instruct
```

Outputs:

- `encoded_training_full`
- `encoded_dev_full`
- `encoded_testing_full`

## 4) Optional filter step

```bash
module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate elm

python /gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/filter_long_sequences.py \
  --base_dir /gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task \
  --dataset_suffix _full \
  --filter_all
```

Filtered outputs:

- `encoded_training_filtered`
- `encoded_dev_filtered`
- `encoded_testing_filtered`

## 5) Train ELM (unchanged code, new data paths)

```bash
cd /gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm

BASE_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1 \
DATAHOME=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/ \
OUTPUT_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/elm_training_outputs \
INITIAL_MODEL=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/initial_elm_model \
sbatch train_clinic_notes.slurm
```

## 6) Count stage sizes

```bash
module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate elm

python /gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/preprocessing_mimiciv/count_note_hadm_pipeline_sizes.py
```

This reports:

- raw discharge notes
- notes surviving pickle preprocessing
- notes embedded
- final ELM train/dev/test counts
