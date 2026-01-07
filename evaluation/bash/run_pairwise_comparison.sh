#!/bin/bash -l

# =====================================================
# Pairwise Model Comparison Script
# =====================================================
# Compares two models using side-by-side pairwise evaluation
# Runs twice with swapped positions to debias
#
# Usage: sbatch bash/run_pairwise_comparison.sh <responses_a> <responses_b> [dataset] [judge_model]
# Examples:
#   sbatch bash/run_pairwise_comparison.sh /path/to/weighted.jsonl /path/to/random.jsonl basicv8vc/SimpleQA
#   sbatch bash/run_pairwise_comparison.sh /path/to/weighted.jsonl /path/to/random.jsonl truthfulqa/truthful_qa
# =====================================================

# Slurm parameters
#SBATCH --job-name=pairwise_comparison
#SBATCH --output=logs/pairwise_%j.%N.out
#SBATCH --error=logs/pairwise_%j.%N.err
#SBATCH --time=72:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=highperf

# =====================================================
# Configuration
# =====================================================
RESPONSES_A=/no_backups/m159/distillation_experiments/evaluation_results/student_weighted/basicv8vc/SimpleQA/responses/student_weighted_basicv8vc_SimpleQA.jsonl
RESPONSES_B=/no_backups/m159/distillation_experiments/evaluation_results/student_random/basicv8vc/SimpleQA/responses/student_random_basicv8vc_SimpleQA.jsonl
DATASET=basicv8vc/SimpleQA
JUDGE_MODEL=meta-llama/Meta-Llama-3-70B-Instruct

# Extract model names from response filenames
MODEL_A_NAME=student_weighted
MODEL_B_NAME=student_random

# Results directory
RESULTS_DIR="/no_backups/m159/distillation_experiments/evaluation_results/simpleQA"
COMPARISON_DIR="${RESULTS_DIR}/pairwise_comparisons"

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "Pairwise Model Comparison"
echo "Responses A: ${RESPONSES_A}"
echo "Responses B: ${RESPONSES_B}"
echo "Model A: ${MODEL_A_NAME}"
echo "Model B: ${MODEL_B_NAME}"
echo "Dataset: ${DATASET:-'(from JSONL)'}"
echo "Judge: ${JUDGE_MODEL}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "====================================="

# Create directories
mkdir -p logs
mkdir -p ${COMPARISON_DIR}

# Verify response files exist
if [ ! -f "${RESPONSES_A}" ]; then
    echo "ERROR: Responses file A not found at ${RESPONSES_A}"
    exit 1
fi

if [ ! -f "${RESPONSES_B}" ]; then
    echo "ERROR: Responses file B not found at ${RESPONSES_B}"
    exit 1
fi

# Load modules
module load cuda

# Activate virtual environment
pyenv activate venv

# Move to project root
cd /usrhomes/m159/stanford_alpaca/normal_distillation

# =====================================================
# Run Pairwise Comparison
# =====================================================
OUTPUT_PATH="${COMPARISON_DIR}/${MODEL_A_NAME}_vs_${MODEL_B_NAME}_pairwise.json"

echo ""
echo "Running pairwise comparison..."
echo "Output: ${OUTPUT_PATH}"

# Build command with optional dataset argument
CMD="python evaluation/generic_judge/compare_models_pairwise.py \
    --model_a_path ${RESPONSES_A} \
    --model_b_path ${RESPONSES_B} \
    --model_a_name ${MODEL_A_NAME} \
    --model_b_name ${MODEL_B_NAME} \
    --judge_model ${JUDGE_MODEL} \
    --output_path ${OUTPUT_PATH}"

# Add dataset argument if provided
if [ -n "${DATASET}" ]; then
    CMD="${CMD} --dataset ${DATASET}"
fi

# Run the command
eval ${CMD}

echo ""
echo "====================================="
echo "Pairwise Comparison Complete!"
echo "Finished: $(date)"
echo "Results: ${OUTPUT_PATH}"
echo "====================================="
