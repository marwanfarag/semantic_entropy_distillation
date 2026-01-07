#!/bin/bash -l

# =====================================================
# SLURM Job: Generate Model Responses
# =====================================================
# Generates responses from a model on a dataset.
# Supports both Dolly (default) and any HuggingFace dataset.
#
# Usage: 
#   sbatch bash/generate_responses.sh <model_path> <model_name> [dataset] [dataset_config]
#
# Examples:
#   # Dolly dataset (default) - uses dolly_judge script
#   sbatch bash/generate_responses.sh /path/to/model student_weighted dolly
#
#   # Any other dataset - uses generic_judge script
#   sbatch bash/generate_responses.sh /path/to/model student_weighted truthfulqa/truthful_qa generation
#   sbatch bash/generate_responses.sh meta-llama/Llama-3.1-8B-Instruct llama8b truthfulqa/truthful_qa generation
# =====================================================

# Slurm parameters
#SBATCH --job-name=weighted_simQA/gen_responses
#SBATCH --output=logs/gen_responses_%j.%N.out
#SBATCH --error=logs/gen_responses_%j.%N.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --partition=highperf

# =====================================================
# Configuration
# =====================================================
MODEL_PATH=/no_backups/m159/distillation_experiments/distillation_weighted
MODEL_NAME=student_weighted
DATASET=basicv8vc/SimpleQA

# Results directory
RESULTS_DIR="/no_backups/m159/distillation_experiments/evaluation_results/"
RESPONSES_DIR="${RESULTS_DIR}/${MODEL_NAME}/${DATASET}/responses"

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "Generate Model Responses"
echo "Model Path: ${MODEL_PATH}"
echo "Model Name: ${MODEL_NAME}"
echo "Dataset: ${DATASET}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "====================================="

# Create directories
mkdir -p logs
mkdir -p ${RESPONSES_DIR}

# Load modules
module load cuda

# Activate virtual environment
pyenv activate venv

# Move to project root
cd /usrhomes/m159/stanford_alpaca/normal_distillation

# =====================================================
# Run Response Generation
# =====================================================
if [ "${DATASET}" == "dolly" ]; then
    # Use Dolly-specific script (has stratification, batching, etc.)
    echo ""
    echo "Using Dolly-specific generation (dolly_judge/generate_responses.py)"
    echo ""
    
    python evaluation/dolly_judge/generate_responses.py \
        --model_name ${MODEL_NAME} \
        --use_scored_subset \
        --batch_size 8
else
    # Use generic script for other datasets
    DATASET_SLUG=$(echo ${DATASET} | sed 's/\//_/g')
    OUTPUT_PATH="${RESPONSES_DIR}/${MODEL_NAME}_${DATASET_SLUG}.jsonl"
    
    echo ""
    echo "Using generic generation (generic_judge/generate_responses.py)"
    echo "Output: ${OUTPUT_PATH}"
    echo ""
    
    python evaluation/generic_judge/generate_responses.py \
        --model_path ${MODEL_PATH} \
        --model_name ${MODEL_NAME} \
        --dataset ${DATASET} \
        --split test \
        --question_field question \
        --ground_truth_field best_answer \
        --output_path ${OUTPUT_PATH}
fi

echo ""
echo "====================================="
echo "Response Generation Complete!"
echo "Finished: $(date)"
echo "====================================="