#!/bin/bash -l

# =====================================================
# SLURM Job: TruthfulQA MCQ Evaluation
# =====================================================

# Slurm parameters
#SBATCH --job-name=truthfulqa_eval
#SBATCH --output=logs/truthfulqa_%j.%N.out
#SBATCH --error=logs/truthfulqa_%j.%N.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-04:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=highperf

# =====================================================
# Configuration
# =====================================================
MODEL_PATH=$1     # Path to model checkpoint
MODEL_NAME=$2     # Name for output file

OUTPUT_DIR="./evaluation/truthfulqa/results"

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "TruthfulQA MCQ Evaluation"
echo "Model: ${MODEL_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "====================================="

mkdir -p logs
mkdir -p ${OUTPUT_DIR}

module load cuda
pyenv activate venv

# =====================================================
# Run Evaluation
# =====================================================
cd normal_distillation

python evaluation/truthfulqa/evaluate_truthfulqa.py \
    --model_path ${MODEL_PATH} \
    --output_path ${OUTPUT_DIR}/${MODEL_NAME}_truthfulqa.json

echo "====================================="
echo "Evaluation Complete!"
echo "====================================="
