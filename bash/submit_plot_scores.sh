#!/bin/bash -l

# =====================================================
# SLURM Job: Plot Semantic Scoring Results
# =====================================================
# Generates histogram plots from scored_outputs.jsonl
# =====================================================

# Slurm parameters
#SBATCH --job-name=plot_results
#SBATCH --output=logs/plot_results_%j.%N.out
#SBATCH --error=logs/plot_results_%j.%N.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --mem=8G
#SBATCH --partition=empl

# =====================================================
# Configuration
# =====================================================
INPUT_PATH="./teacher_outputs/scored_outputs.jsonl"
OUTPUT_DIR="./plots"

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "Plot Semantic Scoring Results"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "====================================="

# Create directories
mkdir -p logs
mkdir -p ${OUTPUT_DIR}

# Activate virtual environment
pyenv activate venv

# =====================================================
# Run Plotting
# =====================================================
echo ""
echo "Input: ${INPUT_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo ""

cd normal_distillation

python -m semantic_scoring.plot_results \
    --input ${INPUT_PATH} \
    --output_dir ${OUTPUT_DIR}

echo ""
echo "====================================="
echo "Plotting Complete!"
echo "Finished: $(date)"
echo "Plots saved to: ${OUTPUT_DIR}"
echo "====================================="
