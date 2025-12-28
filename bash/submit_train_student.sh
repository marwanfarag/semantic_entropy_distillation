#!/bin/bash -l

# =====================================================
# SLURM Job: Knowledge Distillation Training (Hard Labels)
# =====================================================
# This job trains the student model (LLaMA 3.2 3B) using
# teacher-generated text responses (hard labels only).
# Standard supervised fine-tuning on teacher outputs.
# =====================================================

# Slurm parameters
#SBATCH --job-name=weighted_kd_train_student
#SBATCH --output=logs/train_student_%j.%N.out
#SBATCH --error=logs/train_student_%j.%N.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=5-00:00:00
#SBATCH --mem=128G
#SBATCH --gpus=4
#SBATCH --partition=highperf

# =====================================================
# Configuration - Modify these as needed
# =====================================================
STUDENT_MODEL="meta-llama/Llama-3.2-3B"
DISTILLATION_MODE="weighted"

# Data paths (use teacher-generated responses from JSONL files)
TEACHER_OUTPUTS_DIR="./teacher_outputs"
OUTPUT_DIR="/no_backups/m159/distillation_experiments/distillation_${DISTILLATION_MODE}"

# =====================================================
# Distillation Mode Configuration
# =====================================================
# Mode: "random" (uniform weights) or "weighted" (score-based)

# Weighted mode parameters (only used if DISTILLATION_MODE="weighted")
SCORED_OUTPUTS_PATH="./teacher_outputs/scored_outputs.jsonl"
SCORE_THRESHOLD=0.6   # Exclude samples with score > this
W_MIN=0.2             # Minimum weight floor
GAMMA=2.0             # Focal weight decay exponent

# =====================================================
# Training hyperparameters (same as Alpaca defaults)
# =====================================================
NUM_EPOCHS=3
BATCH_SIZE=8
GRAD_ACCUM=32
LR=2e-5
MODEL_MAX_LENGTH=1024
WARMUP_RATIO=0.03

# Evaluation (inference validation)
EVAL_STEPS=10
EVAL_NUM_SAMPLES=50

# login to huggingface using HF_TOKEN
huggingface-cli login --token ${HF_TOKEN}

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
echo "Distillation Mode: ${DISTILLATION_MODE}"
if [ "${DISTILLATION_MODE}" == "weighted" ]; then
    echo "  Scored Outputs: ${SCORED_OUTPUTS_PATH}"
    echo "  Score Threshold: ${SCORE_THRESHOLD}"
    echo "  W_min: ${W_MIN}, Gamma: ${GAMMA}"
fi
echo ""
echo "Training: Epochs=${NUM_EPOCHS}, BS=${BATCH_SIZE}, Accum=${GRAD_ACCUM}, LR=${LR}"
echo "Evaluation: Every ${EVAL_STEPS} steps on ${EVAL_NUM_SAMPLES} samples"
echo ""

# Change to normal_distillation directory for module imports
cd normal_distillation

# Build command with optional weighted mode arguments
CMD="python -m train_distillation.train \
    --model_name_or_path ${STUDENT_MODEL} \
    --teacher_outputs_dir ${TEACHER_OUTPUTS_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --logging_dir ${OUTPUT_DIR}/logs \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 10 \
    --save_total_limit 2 \
    --eval_strategy steps \
    --eval_steps ${EVAL_STEPS} \
    --eval_num_samples ${EVAL_NUM_SAMPLES} \
    --bf16 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --distillation_mode ${DISTILLATION_MODE}"

# Add weighted mode arguments if applicable
if [ "${DISTILLATION_MODE}" == "weighted" ]; then
    CMD="${CMD} \
    --scored_outputs_path ${SCORED_OUTPUTS_PATH} \
    --score_threshold ${SCORE_THRESHOLD} \
    --w_min ${W_MIN} \
    --gamma ${GAMMA}"
fi

# Execute
eval ${CMD}

echo ""
echo "====================================="
echo "Student Training Complete!"
echo "Finished: $(date)"
echo "Model saved to: ${OUTPUT_DIR}"
echo "====================================="
