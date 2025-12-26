#!/bin/bash -l

# =====================================================
# SLURM Job: Knowledge Distillation Training (Online)
# =====================================================
# This job trains the student model (LLaMA 3.2 3B) with
# online distillation - both teacher and student models
# are loaded simultaneously and KL divergence is computed
# on-the-fly. Requires more GPU memory.
#
# NOTE: This requires significant GPU memory to hold both
# models. Consider using A5000/A6000 GPUs or multiple GPUs.
# =====================================================

# Slurm parameters
#SBATCH --job-name=kd_online_train
#SBATCH --output=logs/online_train_%j.%N.out
#SBATCH --error=logs/online_train_%j.%N.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=128G
#SBATCH --gpus=2
#SBATCH --qos=batch
#SBATCH --partition=stud

# =====================================================
# Configuration - Modify these as needed
# =====================================================
STUDENT_MODEL="meta-llama/Llama-3.2-3B"
TEACHER_MODEL="meta-llama/Llama-3.1-8B"
DATA_PATH="../alpaca_data.json"
OUTPUT_DIR="./output_distillation_online"

# Distillation hyperparameters
TEMPERATURE=2.0
ALPHA=0.5

# Training hyperparameters (same as Alpaca defaults)
NUM_EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=8
LR=2e-5
MODEL_MAX_LENGTH=512
WARMUP_RATIO=0.03

# =====================================================
# Setup Environment
# =====================================================
echo "====================================="
echo "Knowledge Distillation - Online Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "====================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module load cuda/11.8

# Activate virtual environment
pyenv activate venv

# Create output directory
mkdir -p ${OUTPUT_DIR}

# =====================================================
# Run Training
# =====================================================
echo ""
echo "Student Model: ${STUDENT_MODEL}"
echo "Teacher Model: ${TEACHER_MODEL}"
echo "Data Path: ${DATA_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo ""
echo "Distillation: Temperature=${TEMPERATURE}, Alpha=${ALPHA}"
echo "Training: Epochs=${NUM_EPOCHS}, BS=${BATCH_SIZE}, Accum=${GRAD_ACCUM}, LR=${LR}"
echo ""

python train_distillation.py \
    --model_name_or_path ${STUDENT_MODEL} \
    --teacher_model_name_or_path ${TEACHER_MODEL} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --temperature ${TEMPERATURE} \
    --alpha ${ALPHA} \
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
    --bf16 True \
    --gradient_checkpointing True \
    --report_to none

echo ""
echo "====================================="
echo "Online Training Complete!"
echo "Finished: $(date)"
echo "Model saved to: ${OUTPUT_DIR}"
echo "====================================="
