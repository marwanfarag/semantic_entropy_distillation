#!/bin/bash -l

# =====================================================
# SLURM Job: Dolly Judge Evaluation
# =====================================================
# Evaluates model responses using Qwen 2.5 14B judge
# Usage: sbatch bash/run_dolly_judge.sh <model_name>
# Example: sbatch bash/run_dolly_judge.sh student_weighted
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
MODEL_NAME=${1:-"student_weighted"}

# Paths (should match config.py)
EXPERIMENT_DIR="/no_backups/m159/distillation_experiments"
RESULTS_DIR="${EXPERIMENT_DIR}/evaluation_results"
RESPONSES_DIR="${RESULTS_DIR}/model_responses"
DOLLY_RESULTS="${RESULTS_DIR}/dolly_judge"
SCORED_OUTPUTS="${EXPERIMENT_DIR}/teacher_outputs/scored_outputs.jsonl"

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
mkdir -p ${DOLLY_RESULTS}

# Load modules
module load cuda

# Activate virtual environment
pyenv activate venv

# Move to project root
cd /no_backups/m159/distillation_experiments/semantic_entropy_distillation

# =====================================================
# Run Evaluation
# =====================================================
MODEL_RESPONSES="${RESPONSES_DIR}/${MODEL_NAME}_responses.jsonl"
OUTPUT_PATH="${DOLLY_RESULTS}/${MODEL_NAME}_scores.jsonl"

echo ""
echo "Model Responses: ${MODEL_RESPONSES}"
echo "Scored Outputs: ${SCORED_OUTPUTS}"
echo "Output: ${OUTPUT_PATH}"
echo ""

# Check if responses exist
if [ ! -f "${MODEL_RESPONSES}" ]; then
    echo "ERROR: Model responses not found at ${MODEL_RESPONSES}"
    echo "Please run generate_responses.sh first"
    exit 1
fi

python evaluation/dolly_judge/evaluate_dolly_judge.py \
    --model_responses ${MODEL_RESPONSES} \
    --scored_outputs ${SCORED_OUTPUTS} \
    --output_path ${OUTPUT_PATH} \
    --num_samples 500

echo ""
echo "====================================="
echo "Evaluation Complete!"
echo "Finished: $(date)"
echo "Results saved to: ${OUTPUT_PATH}"
echo "====================================="
