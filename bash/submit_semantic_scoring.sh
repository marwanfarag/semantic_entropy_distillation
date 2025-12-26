#!/bin/bash -l

# =====================================================
# SLURM Job: Semantic Scoring of Teacher Outputs
# =====================================================
# Computes semantic entropy and contradiction scores
# for teacher responses using DeBERTa-large NLI model.
# =====================================================

# Slurm parameters
#SBATCH --job-name=sem_score
#SBATCH --output=logs/semantic_score_%j.%N.out
#SBATCH --error=logs/semantic_score_%j.%N.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --gpus=2
#SBATCH --partition=empl

# =====================================================
# Configuration
# =====================================================
TEACHER_OUTPUTS_DIR="./teacher_outputs"
OUTPUT_PATH="./scored_outputs.jsonl"

# NLI model (DeBERTa-large fine-tuned on MNLI)
NLI_MODEL="microsoft/deberta-large-mnli"

# Scoring parameters
ENTAILMENT_THRESHOLD=0.5
ENTROPY_WEIGHT=0.5
CONTRADICTION_WEIGHT=0.5
MAX_RESPONSE_TOKENS=100

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "Semantic Scoring"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "====================================="

# Create logs directory
mkdir -p logs

# Load CUDA module
module load cuda

# Activate virtual environment
pyenv activate venv

# =====================================================
# Run Scoring
# =====================================================
echo ""
echo "Teacher Outputs: ${TEACHER_OUTPUTS_DIR}"
echo "Output: ${OUTPUT_PATH}"
echo "NLI Model: ${NLI_MODEL}"
echo "Entailment Threshold: ${ENTAILMENT_THRESHOLD}"
echo "Weights: Entropy=${ENTROPY_WEIGHT}, Contradiction=${CONTRADICTION_WEIGHT}"
echo ""

python -m semantic_scoring.score_teacher_outputs \
    --teacher_outputs_dir ${TEACHER_OUTPUTS_DIR} \
    --output_path ${OUTPUT_PATH} \
    --nli_model_name ${NLI_MODEL} \
    --entailment_threshold ${ENTAILMENT_THRESHOLD} \
    --entropy_weight ${ENTROPY_WEIGHT} \
    --contradiction_weight ${CONTRADICTION_WEIGHT} \
    --max_response_tokens ${MAX_RESPONSE_TOKENS}

echo ""
echo "====================================="
echo "Scoring Complete!"
echo "Finished: $(date)"
echo "Results saved to: ${OUTPUT_PATH}"
echo "====================================="
