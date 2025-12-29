#!/bin/bash -l

# =====================================================
# SLURM Job: Dolly Judge Evaluation
# =====================================================
# Evaluates model responses using Qwen 2.5 14B judge
# =====================================================

# Slurm parameters
#SBATCH --job-name=dolly_judge
#SBATCH --output=logs/dolly_judge_%j.%N.out
#SBATCH --error=logs/dolly_judge_%j.%N.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --gpus=2
#SBATCH --partition=highperf

# =====================================================
# Configuration
# =====================================================
MODEL_RESPONSES=$1  # Path to model responses JSONL
MODEL_NAME=$2       # Name for output file (e.g., "student_random", "teacher_random")

SCORED_OUTPUTS="./teacher_outputs/scored_outputs.jsonl"
OUTPUT_DIR="./evaluation/dolly_judge/results"
NUM_SAMPLES=500

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "Dolly Judge Evaluation"
echo "Model: ${MODEL_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "====================================="

# Create directories
mkdir -p logs
mkdir -p ${OUTPUT_DIR}

# Load modules
module load cuda

# Activate virtual environment
pyenv activate venv

# =====================================================
# Run Evaluation
# =====================================================
echo ""
echo "Model Responses: ${MODEL_RESPONSES}"
echo "Scored Outputs: ${SCORED_OUTPUTS}"
echo "Output: ${OUTPUT_DIR}/${MODEL_NAME}_scores.jsonl"
echo "Num Samples: ${NUM_SAMPLES}"
echo ""

cd normal_distillation

python evaluation/dolly_judge/evaluate_dolly_judge.py \
    --model_responses ${MODEL_RESPONSES} \
    --scored_outputs ${SCORED_OUTPUTS} \
    --output_path ${OUTPUT_DIR}/${MODEL_NAME}_scores.jsonl \
    --num_samples ${NUM_SAMPLES}

echo ""
echo "====================================="
echo "Evaluation Complete!"
echo "Finished: $(date)"
echo "Results saved to: ${OUTPUT_DIR}/${MODEL_NAME}_scores.jsonl"
echo "====================================="
