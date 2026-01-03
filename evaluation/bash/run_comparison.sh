#!/bin/bash -l

# =====================================================
# SLURM Job: Compare All Evaluation Results
# =====================================================
# Generates comparison plots and report for all models
# Usage: sbatch bash/run_comparison.sh
# =====================================================

# Slurm parameters
#SBATCH --job-name=compare_eval
#SBATCH --output=logs/compare_%j.%N.out
#SBATCH --error=logs/compare_%j.%N.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --partition=empl

# =====================================================
# Configuration
# =====================================================
EXPERIMENT_DIR="/no_backups/m159/distillation_experiments"
RESULTS_DIR="${EXPERIMENT_DIR}/evaluation_results"
ANALYSIS_DIR="${RESULTS_DIR}/analysis"

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "Compare Evaluation Results"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "====================================="

# Create directories
mkdir -p logs
mkdir -p ${ANALYSIS_DIR}

# Activate virtual environment
pyenv activate venv

# Move to project root
cd /no_backups/m159/distillation_experiments/semantic_entropy_distillation
# =====================================================
# Run Comparison
# =====================================================
echo ""
echo "Results Directory: ${RESULTS_DIR}"
echo "Analysis Output: ${ANALYSIS_DIR}"
echo ""

python evaluation/analysis/compare_results.py \
    --results_dir ${RESULTS_DIR} \
    --output_dir ${ANALYSIS_DIR}

echo ""
echo "====================================="
echo "Comparison Complete!"
echo "Finished: $(date)"
echo "Report: ${ANALYSIS_DIR}/evaluation_report.md"
echo "====================================="
