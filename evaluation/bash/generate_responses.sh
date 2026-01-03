#!/bin/bash -l

# =====================================================
# SLURM Job: Generate Model Responses on Dolly
# =====================================================
# Generates responses from a model checkpoint on the Dolly dataset
# Usage: sbatch bash/generate_responses.sh <model_name>
# Example: sbatch bash/generate_responses.sh student_weighted
# =====================================================

# Slurm parameters
#SBATCH --job-name=random_gen_responses
#SBATCH --output=logs/gen_responses_%j.%N.out
#SBATCH --error=logs/gen_responses_%j.%N.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --partition=empl

# =====================================================
# Configuration
# =====================================================
MODEL_NAME=student_random

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "Generate Model Responses"
echo "Model: ${MODEL_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "====================================="

# Create directories
mkdir -p logs

# Load modules
module load cuda

# Activate virtual environment
pyenv activate venv

# Move to project root
cd /usrhomes/m159/stanford_alpaca/normal_distillation

# =====================================================
# Run Response Generation
# =====================================================
echo ""
echo "Generating responses for: ${MODEL_NAME}"
echo ""

python evaluation/dolly_judge/generate_responses.py \
    --model_name ${MODEL_NAME} \
    --use_scored_subset

echo ""
echo "====================================="
echo "Response Generation Complete!"
echo "Finished: $(date)"
echo "====================================="
