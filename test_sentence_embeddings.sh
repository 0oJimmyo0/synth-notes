#!/bin/bash

# Test script to verify sentence transformer embedding generation works
# This will process only 10 files for quick testing

# Define paths
INPUT_DIR="${INPUT_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/pickle_ds_note_hadm_all}"
OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/sentence_embeddings_test}"
MODEL="${MODEL:-all-MiniLM-L6-v2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_FILES="${MAX_FILES:-10}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load miniconda and activate elm environment
module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate elm

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to script directory
cd "$SCRIPT_DIR"

echo "Testing sentence transformer embedding generation..."
echo "Input dir: $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Model: $MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Processing $MAX_FILES files for testing..."
echo ""

# Run a small test first
python generate_sentence_embeddings.py \
   --input_dir "$INPUT_DIR" \
   --output_dir "$OUTPUT_DIR" \
   --model "$MODEL" \
   --batch_size "$BATCH_SIZE" \
   --device auto \
   --max_files "$MAX_FILES"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test successful!"
    echo "Output files created in: $OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR"
else
    echo ""
    echo "❌ Test failed! Check the error messages above."
    exit 1
fi
