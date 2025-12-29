#!/bin/bash -l

# =====================================================
# SLURM Job: MMLU-Pro Evaluation
# =====================================================

# Slurm parameters
#SBATCH --job-name=mmlu_eval
#SBATCH --output=logs/mmlu_%j.%N.out
#SBATCH --error=logs/mmlu_%j.%N.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=highperf

# =====================================================
# Configuration
# =====================================================
MODEL_PATH=$1     # Path to model checkpoint
MODEL_NAME=$2     # Name for output file

OUTPUT_DIR="./evaluation/mmlu_pro/results"

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "MMLU Evaluation"
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

python evaluation/mmlu_pro/evaluate_mmlu.py \
    --model_path ${MODEL_PATH} \
    --output_path ${OUTPUT_DIR}/${MODEL_NAME}_mmlu.json \
    --num_shots 5

echo "====================================="
echo "Evaluation Complete!"
echo "====================================="
