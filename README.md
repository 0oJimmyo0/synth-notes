# Embedding ELM: Synthetic MIMIC-IV Discharge Summary Pipeline

Pipeline for generating synthetic clinical discharge summaries from real note embeddings using an **Embedding-conditioned Language Model (ELM)**, with optional **CAV-guided embedding-space enrichment** and leakage-aware evaluation.

For the full research plan (motivation, novelty, coverage, CAV design, evaluation), see [`research_plan.tex`](research_plan.tex).

## Repository layout

```
embedding_elm/
├── preprocessing_mimiciv/     # MIMIC-IV note/HADM pickle prep
├── generate_sentence_embeddings.py
├── prep_hf_dataset/           # Build HuggingFace ELM training datasets
├── open-elm/                  # ELM model, training, generation, CAV axis bank
│   ├── src/                   # model.py, utils.py
│   ├── cav_axis/              # Axis-bank fit + audit scripts
│   ├── train.py               # ELM fine-tuning
│   ├── generate_synthetic_notes.py
│   └── *.slurm                # Cluster job templates
├── mimic-iii_scripts/         # Legacy MIMIC-III pipeline (reference)
└── research_plan.tex
```

## Pipeline overview

1. **Preprocess** MIMIC-IV discharge notes → note/HADM-aligned pickle datasets
2. **Embed** notes with `BAAI/bge-large-en-v1.5` (1024-d vectors)
3. **Prepare** HuggingFace datasets (`encoded_training/dev/testing`)
4. **Train** ELM (Llama-3.1-8B + embedding adapter + LoRA)
5. **Generate** synthetic discharge summaries from held-out embeddings
6. **(Planned)** CAV steering, LLM editing, coverage and faithfulness evaluation

## Quick start (cluster)

```bash
module load miniconda
conda activate elm
pip install -r open-elm/requirements.txt
```

### Initialize backbone ELM (one-time, ~30 GB download)

The initialized Llama weights are **not** stored in this repo. Create them locally:

```bash
cd open-elm
python initialize_model.py --output_dir initial_elm_model
# or run: sbatch initialize_model.slurm
```

### MIMIC-IV full pipeline

See step-by-step guides:
- [`preprocessing_mimiciv/MIMIC4_NOTE_HADM_PIPELINE_GUIDE.md`](preprocessing_mimiciv/MIMIC4_NOTE_HADM_PIPELINE_GUIDE.md)

Typical path variables (adjust for your environment):

| Variable | Example |
|----------|---------|
| `MIMICIV_NOTE_DIR` | Path to MIMIC-IV-Note `discharge.csv` |
| `MIMICIV_CORE_DIR` | Path to MIMIC-IV core tables |
| `PICKLE_DIR` | Output pickle dataset directory |
| `EMBEDDING_DIR` | Sentence embedding output |
| `DATAHOME` | HF encoded datasets + ELM outputs |

## What is intentionally excluded from git

- Model weights (`*.safetensors`, ~30 GB initial ELM)
- Training checkpoints and synthetic note outputs
- Embeddings, pickles, and HF dataset shards
- Slurm logs and large runtime logs

## Current status (Apr 2026)

- Full MIMIC-IV cohort (~332k notes) embedded and filtered (~303k)
- ELM training on full cohort completed (`checkpoint-8215`)
- **Next:** vanilla held-out generation + structured manifest (see `research_plan.tex` Phase 1)

## Citation / upstream

ELM model code is based on [BIDS-Xu-Lab/open-elm](https://github.com/BIDS-Xu-Lab/open-elm). This repository extends that codebase with the MIMIC-IV discharge-summary pipeline, axis-bank CAV tooling, and research validation plan.

## License

Internal research repository. Add license before public release if needed.
