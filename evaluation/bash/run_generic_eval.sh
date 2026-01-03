#!/bin/bash -l

# =====================================================
# Generic Evaluation Pipeline
# =====================================================
# Runs the complete evaluation pipeline for any HuggingFace dataset
#
# Usage: sbatch bash/run_generic_eval.sh <model_path> <dataset> [dataset_config]
# Examples:
#   sbatch bash/run_generic_eval.sh /path/to/model truthfulqa/truthful_qa generation
#   sbatch bash/run_generic_eval.sh meta-llama/Llama-3.1-8B-Instruct truthfulqa/truthful_qa generation
# =====================================================

# Slurm parameters
#SBATCH --job-name=generic_eval
#SBATCH --output=logs/generic_eval_%j.%N.out
#SBATCH --error=logs/generic_eval_%j.%N.err
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=highperf

# =====================================================
# Configuration
# =====================================================
MODEL_PATH=${1:?"Error: MODEL_PATH required"}
DATASET=${2:?"Error: DATASET required (e.g., truthfulqa/truthful_qa)"}
DATASET_CONFIG=${3:-""}  # Optional config (e.g., "generation")
JUDGE_MODEL=${4:-"Qwen/Qwen3-32B-Instruct"}

# Extract model name from path
MODEL_NAME=$(basename ${MODEL_PATH})

# Results directory
RESULTS_DIR="/no_backups/m159/distillation_experiments/evaluation_results"
RESPONSES_DIR="${RESULTS_DIR}/responses"
SCORES_DIR="${RESULTS_DIR}/scores"

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "Generic Evaluation Pipeline"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET}"
echo "Config: ${DATASET_CONFIG:-'(none)'}"
echo "Judge: ${JUDGE_MODEL}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "====================================="

# Create directories
mkdir -p logs
mkdir -p ${RESPONSES_DIR}
mkdir -p ${SCORES_DIR}

# Load modules
module load cuda

# Activate virtual environment
pyenv activate venv

# Move to project root
cd /no_backups/m159/distillation_experiments/semantic_entropy_distillation

# =====================================================
# Step 1: Generate Responses
# =====================================================
DATASET_SLUG=$(echo ${DATASET} | sed 's/\//_/g')
RESPONSES_PATH="${RESPONSES_DIR}/${MODEL_NAME}_${DATASET_SLUG}.jsonl"

echo ""
echo "Step 1: Generating responses..."
echo "Output: ${RESPONSES_PATH}"

if [ -n "${DATASET_CONFIG}" ]; then
    CONFIG_ARG="--dataset_config ${DATASET_CONFIG}"
else
    CONFIG_ARG=""
fi

python evaluation/generic_judge/generate_responses.py \
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --dataset ${DATASET} \
    ${CONFIG_ARG} \
    --split validation \
    --question_field question \
    --ground_truth_field best_answer \
    --output_path ${RESPONSES_PATH}

# =====================================================
# Step 2: Evaluate with Judge
# =====================================================
JUDGE_NAME=$(echo ${JUDGE_MODEL} | sed 's/.*\///' | sed 's/-Instruct//' | tr '[:upper:]' '[:lower:]' | tr '.' '_')
SCORES_PATH="${SCORES_DIR}/${MODEL_NAME}_${DATASET_SLUG}_${JUDGE_NAME}.jsonl"

echo ""
echo "Step 2: Evaluating with judge..."
echo "Output: ${SCORES_PATH}"

python evaluation/generic_judge/evaluate_with_judge.py \
    --responses_path ${RESPONSES_PATH} \
    --judge_model ${JUDGE_MODEL} \
    --output_path ${SCORES_PATH}

echo ""
echo "====================================="
echo "Evaluation Complete!"
echo "Finished: $(date)"
echo "Responses: ${RESPONSES_PATH}"
echo "Scores: ${SCORES_PATH}"
echo "====================================="
