#!/bin/bash
# Knowledge Distillation Training Script
# This script runs the two-stage knowledge distillation pipeline

set -e

# Configuration
STUDENT_MODEL="meta-llama/Llama-3.2-3B"
TEACHER_MODEL="meta-llama/Llama-3.1-8B"
DATA_PATH="./alpaca_data.json"
TEACHER_OUTPUT_DIR="./teacher_outputs"
FINAL_OUTPUT_DIR="./output_distillation"

# Distillation hyperparameters
TEMPERATURE=2.0
ALPHA=0.5

# Training hyperparameters
BATCH_SIZE=4
GRAD_ACCUM=8
EPOCHS=3
LR=2e-5
MAX_LENGTH=512

echo "====================================="
echo "Knowledge Distillation Training"
echo "====================================="
echo "Student: ${STUDENT_MODEL}"
echo "Teacher: ${TEACHER_MODEL}"
echo "Temperature: ${TEMPERATURE}, Alpha: ${ALPHA}"
echo "====================================="

# Stage 1: Generate teacher responses and logits
echo ""
echo "[Stage 1] Generating teacher responses and logits..."
echo ""

python generate_teacher_responses.py \
    --model_name_or_path ${TEACHER_MODEL} \
    --data_path ${DATA_PATH} \
    --output_dir ${TEACHER_OUTPUT_DIR} \
    --batch_size 1 \
    --max_new_tokens 512 \
    --save_logits True \
    --torch_dtype bfloat16

echo ""
echo "[Stage 1 Complete] Teacher outputs saved to ${TEACHER_OUTPUT_DIR}"
echo ""

# Stage 2: Train student with distillation (offline mode using pre-computed logits)
echo ""
echo "[Stage 2] Training student model with knowledge distillation..."
echo ""

python train_distillation.py \
    --model_name_or_path ${STUDENT_MODEL} \
    --data_path ${TEACHER_OUTPUT_DIR}/alpaca_teacher_responses.json \
    --teacher_logits_dir ${TEACHER_OUTPUT_DIR}/logits \
    --output_dir ${FINAL_OUTPUT_DIR} \
    --offline_distillation True \
    --temperature ${TEMPERATURE} \
    --alpha ${ALPHA} \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --model_max_length ${MAX_LENGTH} \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --bf16 True \
    --gradient_checkpointing True \
    --report_to none

echo ""
echo "====================================="
echo "[Complete] Distilled model saved to ${FINAL_OUTPUT_DIR}"
echo "====================================="
