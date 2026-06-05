#!/bin/bash
# This script runs in background and survives disconnection
# Change to script directory first (before loading modules)
cd "/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm"

# Load miniconda and activate elm environment
module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate elm

# Get full path to Python to ensure it works after disconnection
PYTHON_EXEC=$(which python)

# Create model-specific output directory (replace / with _ in model name)
MODEL_DIR_NAME=$(echo "BAAI/bge-large-en-v1.5" | sed 's/[\/]/-/g')
OUTPUT_DIR="/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/embeddings-$MODEL_DIR_NAME"

# Run the embedding generation
$PYTHON_EXEC "/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/generate_sentence_embeddings.py" \
   --input_dir "/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/pickle_ds_note_hadm_all" \
   --output_dir "$OUTPUT_DIR" \
   --model "BAAI/bge-large-en-v1.5" \
   --batch_size 32 \
   --device auto
