#!/bin/bash -l

# =====================================================
# Model Comparison Script
# =====================================================
# Compares two models and computes win rates
#
# Usage: sbatch bash/run_model_comparison.sh <model_a_path> <model_b_path> [judge_model]
# Examples:
#   sbatch bash/run_model_comparison.sh outputs/weighted.jsonl outputs/random.jsonl
# =====================================================

# Slurm parameters
#SBATCH --job-name=model_comparison
#SBATCH --output=logs/comparison_%j.%N.out
#SBATCH --error=logs/comparison_%j.%N.err
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=highperf

# =====================================================
# Configuration
# =====================================================
MODEL_A_PATH=${1:?"Error: MODEL_A_PATH required"}
MODEL_B_PATH=${2:?"Error: MODEL_B_PATH required"}
JUDGE_MODEL=${3:-"Qwen/Qwen3-32B-Instruct"}

# Extract model names
MODEL_A_NAME=$(basename ${MODEL_A_PATH} | sed 's/_responses.jsonl//')
MODEL_B_NAME=$(basename ${MODEL_B_PATH} | sed 's/_responses.jsonl//')

# Results directory
RESULTS_DIR="/no_backups/m159/distillation_experiments/evaluation_results"
COMPARISON_DIR="${RESULTS_DIR}/comparisons"

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "Model Comparison"
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

# Load modules
module load cuda

# Activate virtual environment
pyenv activate venv

# Move to project root
cd /no_backups/m159/distillation_experiments/semantic_entropy_distillation

# =====================================================
# Run Comparison
# =====================================================
OUTPUT_PATH="${COMPARISON_DIR}/${MODEL_A_NAME}_vs_${MODEL_B_NAME}.json"

echo ""
echo "Running comparison..."
echo "Output: ${OUTPUT_PATH}"

python evaluation/generic_judge/compare_models.py \
    --model_a_path ${MODEL_A_PATH} \
    --model_b_path ${MODEL_B_PATH} \
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
