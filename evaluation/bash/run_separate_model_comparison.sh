#!/bin/bash -l

# =====================================================
# Model Comparison Script
# =====================================================
# Compares two models' responses using an LLM judge and computes win rates
# Requires pre-generated response JSONL files from generate_responses.sh
#
# Usage: sbatch bash/run_model_comparison.sh <responses_a> <responses_b> [judge_model]
# Examples:
#   sbatch bash/run_model_comparison.sh /path/to/student_weighted.jsonl /path/to/student_random.jsonl
#   sbatch bash/run_model_comparison.sh responses/weighted.jsonl responses/random.jsonl "Qwen/Qwen3-32B-Instruct"
# =====================================================

# Slurm parameters
#SBATCH --job-name=model_comparison
#SBATCH --output=logs/comparison_%j.%N.out
#SBATCH --error=logs/comparison_%j.%N.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=highperf

# =====================================================
# Configuration
# =====================================================
RESPONSES_A=/no_backups/m159/distillation_experiments/evaluation_results/truthfulQA_gen/student_weighted/truthfulqa/truthful_qa/responses/student_weighted_truthfulqa_truthful_qa.jsonl
RESPONSES_B=/no_backups/m159/distillation_experiments/evaluation_results/truthfulQA_gen/student_random/truthfulqa/truthful_qa/responses/student_random_truthfulqa_truthful_qa.jsonl
JUDGE_MODEL=meta-llama/Meta-Llama-3-70B-Instruct

# Extract model names from response filenames
# e.g., /path/to/student_weighted_truthfulqa.jsonl -> student_weighted_truthfulqa
MODEL_A_NAME=student_weighted
MODEL_B_NAME=student_random

# Results directory
RESULTS_DIR="/no_backups/m159/distillation_experiments/evaluation_results/truthfulQA_gen"
COMPARISON_DIR="${RESULTS_DIR}/comparisons"

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "Model Comparison"
echo "Responses A: ${RESPONSES_A}"
echo "Responses B: ${RESPONSES_B}"
echo "Model A: ${MODEL_A_NAME}"
echo "Model B: ${MODEL_B_NAME}"
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
# Run Comparison
# =====================================================
OUTPUT_PATH="${COMPARISON_DIR}/${MODEL_A_NAME}_vs_${MODEL_B_NAME}.json"

echo ""
echo "Running comparison..."
echo "Output: ${OUTPUT_PATH}"

python evaluation/generic_judge/compare_models.py \
    --model_a_path ${RESPONSES_A} \
    --model_b_path ${RESPONSES_B} \
    --model_a_name ${MODEL_A_NAME} \
    --model_b_name ${MODEL_B_NAME} \
    --judge_model ${JUDGE_MODEL} \
    --output_path ${OUTPUT_PATH}

echo ""
echo "====================================="
echo "Comparison Complete!"
echo "Finished: $(date)"
echo "Results: ${OUTPUT_PATH}"
echo "====================================="
