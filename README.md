# Knowledge Distillation: LLaMA 3.2 3B from LLaMA 3.1 8B

This folder contains scripts for training a LLaMA 3.2 3B student model using knowledge distillation from a LLaMA 3.1 8B teacher model on the Alpaca dataset.

## Files

| File | Description |
|------|-------------|
| `generate_teacher_responses.py` | Generates teacher responses and logits for offline distillation |
| `train_distillation.py` | Trains student model with KL divergence + CE loss |
| `submit_teacher_generation.sh` | SLURM job for teacher generation |
| `submit_train_student.sh` | SLURM job for offline distillation training |
| `submit_online_train.sh` | SLURM job for online distillation (both models loaded) |

## Quick Start

### Two-Stage Pipeline (Recommended)

**Stage 1: Generate teacher responses and logits**
```bash
cd /usrhomes/m159/stanford_alpaca/normal_distillation
sbatch submit_teacher_generation.sh
```

**Stage 2: Train student with offline distillation** (after Stage 1 completes)
```bash
sbatch submit_train_student.sh
```

### Single-Stage Online Distillation

If you have enough GPU memory (2x A5000 or better):
```bash
sbatch submit_online_train.sh
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check job output
tail -f logs/teacher_gen_*.out
tail -f logs/train_student_*.out

# Cancel a job
scancel <job_id>
```

## Configuration

### Distillation Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TEMPERATURE` | 2.0 | Softening factor for logits in KL divergence |
| `ALPHA` | 0.5 | Weight for KL loss (1-α for CE loss) |

### Training Hyperparameters (Alpaca defaults)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_EPOCHS` | 3 | Number of training epochs |
| `BATCH_SIZE` | 4 | Per-device batch size |
| `GRAD_ACCUM` | 8 | Gradient accumulation steps |
| `LR` | 2e-5 | Learning rate |
| `MODEL_MAX_LENGTH` | 512 | Maximum sequence length |

## GPU Requirements

| Mode | GPUs | GPU Memory | Recommended |
|------|------|------------|-------------|
| Teacher Generation | 1 | ~24GB | A5000/A6000 |
| Offline Training | 1 | ~12GB | Any |
| Online Training | 2 | ~48GB total | A5000/A6000 |

## Output

- **Teacher outputs**: `./teacher_outputs/`
  - `alpaca_teacher_responses.json` - Teacher text responses
  - `logits/` - Pre-computed teacher logits (compressed numpy)
  
- **Trained model**: `./output_distillation/`
  - Final student checkpoint
  - Training logs
