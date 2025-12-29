# Evaluation Scripts Usage Guide

## Overview
Evaluation infrastructure for comparing Random vs Weighted distillation modes.

## Structure
```
evaluation/
├── dolly_judge/         # LLM-as-judge on Dolly dataset
├── truthfulqa/          # TruthfulQA MCQ evaluation
├── mmlu_pro/            # MMLU evaluation
└── analysis/            # Result comparison and visualization
```

## Running Evaluations

### 1. Dolly Judge Evaluation (Stage A: Independent Scoring)

```bash
# Student (Random)
sbatch evaluation/dolly_judge/submit_dolly_judge.sh \
    /no_backups/m159/distillation_experiments/distillation_random \
    student_random

# Student (Weighted)
sbatch evaluation/dolly_judge/submit_dolly_judge.sh \
    /no_backups/m159/distillation_experiments/distillation_weighted \
    student_weighted

# Teacher (Random response from teacher outputs)
sbatch evaluation/dolly_judge/submit_dolly_judge.sh \
    ./teacher_outputs/job_0_responses.jsonl \
    teacher_random

# Teacher (High-confidence from scored outputs)
sbatch evaluation/dolly_judge/submit_dolly_judge.sh \
    ./teacher_outputs/scored_outputs.jsonl \
    teacher_confident
```

### 2. TruthfulQA MCQ

```bash
# Evaluate each model
for model in student_random student_weighted teacher; do
    sbatch evaluation/truthfulqa/submit_truthfulqa.sh \
        /path/to/${model} \
        ${model}
done
```

### 3. MMLU-Pro

```bash
# Evaluate each model
for model in student_random student_weighted teacher; do
    sbatch evaluation/mmlu_pro/submit_mmlu.sh \
        /path/to/${model} \
        ${model}
done
```

### 4. Compare Results

```bash
cd normal_distillation
python evaluation/analysis/compare_results.py
```

## Outputs

- `evaluation/dolly_judge/results/*_scores.jsonl` - Judge scores per model
- `evaluation/truthfulqa/results/*_truthfulqa.json` - TruthfulQA accuracies
- `evaluation/mmlu_pro/results/*_mmlu.json` - MMLU accuracies
- `evaluation/analysis/dolly_comparison.png` - Dolly scores by uncertainty band
- `evaluation/analysis/benchmark_comparison.png` - Benchmark accuracies
- `evaluation/analysis/evaluation_report.md` - Summary report
