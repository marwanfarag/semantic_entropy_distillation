#!/bin/bash -l

# =====================================================
# SLURM Job: Generate Teacher Responses and Logits
# =====================================================
# This job runs the teacher model (LLaMA 3.1 8B) on the
# Alpaca dataset to generate responses and logits for
# knowledge distillation training.
# =====================================================

# Slurm parameters
#SBATCH --job-name=kd_teacher_gen
#SBATCH --output=logs/teacher_gen_%j.%N.out
#SBATCH --error=logs/teacher_gen_%j.%N.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G
#SBATCH --gpus=4
#SBATCH --partition=empl


# =====================================================
# Configuration - Modify these as needed
# =====================================================
TEACHER_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH="../alpaca_data.json"
OUTPUT_DIR="./teacher_outputs"

# Generation parameters
BATCH_SIZE=1
MAX_NEW_TOKENS=512
SAVE_LOGITS=False
NUM_RESPONSES=7

# Data range for parallel jobs (0-indexed)
# Set these to split the dataset across multiple jobs
# Job 1: START_IDX=0, END_IDX=26001 (first half)
# Job 2: START_IDX=26001, END_IDX=52002 (second half)
START_IDX=0
END_IDX="26001"  # Leave empty for all remaining samples

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "Knowledge Distillation - Teacher Generation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "====================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module load cuda

# Activate virtual environment
pyenv activate venv

# Create output directory
mkdir -p ${OUTPUT_DIR}

# =====================================================
# Run Teacher Generation
# =====================================================
echo ""
echo "Teacher Model: ${TEACHER_MODEL}"
echo "Data Path: ${DATA_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Max New Tokens: ${MAX_NEW_TOKENS}"
echo "Num Responses: ${NUM_RESPONSES}"
echo "Data Range: ${START_IDX} to ${END_IDX:-end}"
echo ""

# Change to the normal_distillation directory
cd /usrhomes/m159/stanford_alpaca/normal_distillation
hf auth login --token ${HF_TOKEN}

# Build the python command with optional end_idx
CMD="python -m teacher_generation.generate \
    --model_name_or_path ${TEACHER_MODEL} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --save_logits ${SAVE_LOGITS} \
    --num_responses ${NUM_RESPONSES} \
    --start_idx ${START_IDX} \
    --torch_dtype bfloat16 \
    --save_every 1"

# Add end_idx only if specified
if [ -n "${END_IDX}" ]; then
    CMD="${CMD} --end_idx ${END_IDX}"
fi

eval ${CMD}

echo ""
echo "====================================="
echo "Teacher Generation Complete!"
echo "Finished: $(date)"
echo "Output saved to: ${OUTPUT_DIR}"
echo "====================================="
