#!/bin/bash -l

# =====================================================
# SLURM Job: MMLU Evaluation
# =====================================================
# Evaluates model on MMLU benchmark
# Usage: sbatch bash/run_mmlu.sh <model_name>
# Example: sbatch bash/run_mmlu.sh student_weighted
# =====================================================

# Slurm parameters
#SBATCH --job-name=mmlu_eval
#SBATCH --output=logs/mmlu_%j.%N.out
#SBATCH --error=logs/mmlu_%j.%N.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=highperf

# =====================================================
# Configuration
# =====================================================
MODEL_NAME=${1:-"student_weighted"}

# Model paths (from config.py)
declare -A MODEL_PATHS
MODEL_PATHS["student_weighted"]="/no_backups/m159/distillation_experiments/distillation_weighted"
MODEL_PATHS["student_random"]="/no_backups/m159/distillation_experiments/distillation_random/checkpoint-90"
MODEL_PATHS["teacher"]="meta-llama/Llama-3.1-8B-Instruct"

MODEL_PATH=${MODEL_PATHS[$MODEL_NAME]}

# Results directory
EXPERIMENT_DIR="/no_backups/m159/distillation_experiments"
RESULTS_DIR="${EXPERIMENT_DIR}/evaluation_results/mmlu"

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "MMLU Evaluation"
echo "Model: ${MODEL_NAME}"
echo "Path: ${MODEL_PATH}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "====================================="

# Create directories
mkdir -p logs
mkdir -p ${RESULTS_DIR}

# Load modules
module load cuda

# Activate virtual environment
pyenv activate venv

# Move to project root
cd /no_backups/m159/distillation_experiments/semantic_entropy_distillation

# =====================================================
# Run Evaluation
# =====================================================
OUTPUT_PATH="${RESULTS_DIR}/${MODEL_NAME}_mmlu.json"

echo ""
echo "Model Path: ${MODEL_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo ""

python evaluation/mmlu_pro/evaluate_mmlu.py \
    --model_path ${MODEL_PATH} \
    --output_path ${OUTPUT_PATH}

echo ""
echo "====================================="
echo "MMLU Evaluation Complete!"
echo "Finished: $(date)"
echo "Results saved to: ${OUTPUT_PATH}"
echo "====================================="
