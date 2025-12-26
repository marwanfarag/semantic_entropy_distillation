#!/bin/bash -l

# =====================================================
# SLURM Job: Knowledge Distillation Training (Hard Labels)
# =====================================================
# This job trains the student model (LLaMA 3.2 3B) using
# teacher-generated text responses (hard labels only).
# Standard supervised fine-tuning on teacher outputs.
# =====================================================

# Slurm parameters
#SBATCH --job-name=kd_train_student
#SBATCH --output=logs/train_student_%j.%N.out
#SBATCH --error=logs/train_student_%j.%N.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --partition=stud

# =====================================================
# Configuration - Modify these as needed
# =====================================================
STUDENT_MODEL="meta-llama/Llama-3.2-3B"

# Data paths (use teacher-generated responses from JSONL files)
TEACHER_OUTPUTS_DIR="./teacher_outputs"
OUTPUT_DIR="./output_distillation"

# Training hyperparameters (same as Alpaca defaults)
NUM_EPOCHS=3
BATCH_SIZE=64
GRAD_ACCUM=8
LR=2e-5
MODEL_MAX_LENGTH=1024
WARMUP_RATIO=0.03

# Evaluation (inference validation)
EVAL_STEPS=500
EVAL_NUM_SAMPLES=50

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "Knowledge Distillation - Hard Labels Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "====================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module load cuda

# Activate virtual environment
pyenv activate venv

# Create output directory
mkdir -p ${OUTPUT_DIR}

# =====================================================
# Run Training
# =====================================================
echo ""
echo "Student Model: ${STUDENT_MODEL}"
echo "Teacher Outputs Directory: ${TEACHER_OUTPUTS_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo ""
echo "Training: Epochs=${NUM_EPOCHS}, BS=${BATCH_SIZE}, Accum=${GRAD_ACCUM}, LR=${LR}"
echo "Evaluation: Every ${EVAL_STEPS} steps on ${EVAL_NUM_SAMPLES} samples"
echo ""

python -m train_distillation.train \
    --model_name_or_path ${STUDENT_MODEL} \
    --teacher_outputs_dir ${TEACHER_OUTPUTS_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --eval_strategy steps \
    --eval_steps ${EVAL_STEPS} \
    --eval_num_samples ${EVAL_NUM_SAMPLES} \
    --bf16 True \
    --gradient_checkpointing True \
    --report_to none

echo ""
echo "====================================="
echo "Student Training Complete!"
echo "Finished: $(date)"
echo "Model saved to: ${OUTPUT_DIR}"
echo "====================================="
