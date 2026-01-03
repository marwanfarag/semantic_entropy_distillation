#!/bin/bash -l

# =====================================================
# SLURM Job: TruthfulQA Evaluation
# =====================================================
# Evaluates model on TruthfulQA MCQ benchmark
# Usage: sbatch bash/run_truthfulqa.sh <model_name>
# Example: sbatch bash/run_truthfulqa.sh student_weighted
# =====================================================

# Slurm parameters
#SBATCH --job-name=random_truthfulqa
#SBATCH --output=logs/truthfulqa_%j.%N.out
#SBATCH --error=logs/truthfulqa_%j.%N.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=empl

# =====================================================
# Configuration
# =====================================================
MODEL_NAME=${1:-"student_random"}

# Model paths (from config.py)
declare -A MODEL_PATHS
MODEL_PATHS["student_weighted"]="/no_backups/m159/distillation_experiments/distillation_weighted"
MODEL_PATHS["student_random"]="/no_backups/m159/distillation_experiments/distillation_random/checkpoint-90"
MODEL_PATHS["teacher"]="meta-llama/Llama-3.1-8B-Instruct"

MODEL_PATH=${MODEL_PATHS[$MODEL_NAME]}

# Results directory
EXPERIMENT_DIR="/no_backups/m159/distillation_experiments"
RESULTS_DIR="${EXPERIMENT_DIR}/evaluation_results/truthfulqa"

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "TruthfulQA Evaluation"
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
cd /usrhomes/m159/stanford_alpaca/normal_distillation

# =====================================================
# Run Evaluation
# =====================================================
OUTPUT_PATH="${RESULTS_DIR}/${MODEL_NAME}_truthfulqa.json"

echo ""
echo "Model Path: ${MODEL_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo ""

python evaluation/truthfulqa/evaluate_truthfulqa.py \
    --model_path ${MODEL_PATH} \
    --output_path ${OUTPUT_PATH}

echo ""
echo "====================================="
echo "TruthfulQA Evaluation Complete!"
echo "Finished: $(date)"
echo "Results saved to: ${OUTPUT_PATH}"
echo "====================================="
