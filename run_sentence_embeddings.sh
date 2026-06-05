#!/bin/bash

# Define paths
INPUT_DIR="${INPUT_DIR:-/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/mimiciv/3.1/pickle_ds}"
# Output directory will include model name to avoid overwriting
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm}"

# Model options:
# - all-MiniLM-L6-v2: Fast, 384 dimensions (faster, lower quality)
# - BAAI/bge-large-en-v1.5: High quality, 1024 dimensions (slower, best quality) - RECOMMENDED
# - all-mpnet-base-v2: High quality, 768 dimensions (slower)
MODEL="${MODEL:-BAAI/bge-large-en-v1.5}"

# Batch size for embedding generation (adjust based on GPU memory)
BATCH_SIZE="${BATCH_SIZE:-32}"

# Output directory will be created in the wrapper script with model name

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create a wrapper script that will be executed in the background
WRAPPER_SCRIPT="$SCRIPT_DIR/.run_embeddings_wrapper.sh"
cat > "$WRAPPER_SCRIPT" << EOF
#!/bin/bash
# This script runs in background and survives disconnection
# Change to script directory first (before loading modules)
cd "$SCRIPT_DIR"

# Load miniconda and activate elm environment
module load miniconda
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate elm

# Get full path to Python to ensure it works after disconnection
PYTHON_EXEC=\$(which python)

# Create model-specific output directory (replace / with _ in model name)
MODEL_DIR_NAME=\$(echo "$MODEL" | sed 's/[\/]/-/g')
OUTPUT_DIR="$BASE_OUTPUT_DIR/embeddings-\$MODEL_DIR_NAME"

# Run the embedding generation
\$PYTHON_EXEC "$SCRIPT_DIR/generate_sentence_embeddings.py" \\
   --input_dir "$INPUT_DIR" \\
   --output_dir "\$OUTPUT_DIR" \\
   --model "$MODEL" \\
   --batch_size $BATCH_SIZE \\
   --device auto
EOF

chmod +x "$WRAPPER_SCRIPT"

# Run in the background with nohup, setsid (creates new session), and log output
# Using setsid ensures the process survives disconnection
nohup setsid bash "$WRAPPER_SCRIPT" > "$SCRIPT_DIR/sentence_embedding_generation.log" 2>&1 < /dev/null &
BG_PID=$!
echo $BG_PID > "$SCRIPT_DIR/sentence_embedding_generation.pid"

# Create model-specific output directory name for display
MODEL_DIR_NAME=$(echo "$MODEL" | sed 's/[\/]/-/g')
OUTPUT_DIR_DISPLAY="$BASE_OUTPUT_DIR/embeddings-$MODEL_DIR_NAME"

echo "Started background sentence transformer embedding generation; PID saved in sentence_embedding_generation.pid"
echo "Log file: sentence_embedding_generation.log"
echo "Output directory: $OUTPUT_DIR_DISPLAY"
echo "Model: $MODEL"
echo ""
echo "To monitor progress:"
echo "  tail -f sentence_embedding_generation.log"
echo ""
echo "To check if still running:"
echo "  ps aux | grep \$(cat sentence_embedding_generation.pid)"
echo ""
echo "To stop the process:"
echo "  kill \$(cat sentence_embedding_generation.pid)"