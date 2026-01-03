# Evaluation Package

Evaluation infrastructure for comparing **Weighted vs Random distillation** models.

## Models Being Compared

| Model | Path | Description |
|-------|------|-------------|
| `student_weighted` | `/no_backups/m159/distillation_experiments/distillation_weighted` | Llama 3B distilled with uncertainty-weighted loss |
| `student_random` | `/no_backups/m159/distillation_experiments/distillation_random/checkpoint-90` | Llama 3B distilled with standard loss |
| `teacher` | `meta-llama/Llama-3.1-8B-Instruct` | Teacher model (8B) for baseline comparison |

## Directory Structure

```
evaluation/
├── bash/                           # All SLURM submission scripts
│   ├── generate_responses.sh       # Generate model responses on Dolly
│   ├── run_dolly_judge.sh          # Run LLM judge evaluation
│   ├── run_mmlu.sh                 # Run MMLU benchmark
│   ├── run_truthfulqa.sh           # Run TruthfulQA benchmark
│   └── run_comparison.sh           # Compare all results
├── dolly_judge/                    # LLM-as-judge evaluation
│   ├── generate_responses.py       # Generate responses from checkpoints
│   └── evaluate_dolly_judge.py     # Judge scoring with Qwen 2.5 14B
├── mmlu/                           # MMLU evaluation
│   └── evaluate_mmlu.py
├── truthfulqa/                     # TruthfulQA evaluation
│   └── evaluate_truthfulqa.py
├── analysis/                       # Result comparison
│   └── compare_results.py
├── config.py                       # Centralized configuration
└── README.md
```

## Complete Evaluation Pipeline

### Step 1: Generate Model Responses

First, generate responses from each model checkpoint on the Dolly dataset:

```bash
# Generate responses for weighted distillation model
sbatch evaluation/bash/generate_responses.sh student_weighted

# Generate responses for random distillation model  
sbatch evaluation/bash/generate_responses.sh student_random
```

This creates JSONL files in `/no_backups/m159/distillation_experiments/evaluation_results/model_responses/`.

### Step 2: Run Judge Evaluation (Dolly)

Evaluate the generated responses using Qwen 2.5 14B as judge:

```bash
# Wait for Step 1 to complete, then:
sbatch evaluation/bash/run_dolly_judge.sh student_weighted
sbatch evaluation/bash/run_dolly_judge.sh student_random
```

Outputs:
- `*_scores.jsonl`: Per-sample judge scores (correctness, helpfulness, coherence)
- Visualization plots (box plots, histograms, correlation heatmaps, etc.)

### Step 3: Run Benchmark Evaluations

Evaluate directly on standard benchmarks:

```bash
# MMLU
sbatch evaluation/bash/run_mmlu.sh student_weighted
sbatch evaluation/bash/run_mmlu.sh student_random
sbatch evaluation/bash/run_mmlu.sh teacher

# TruthfulQA
sbatch evaluation/bash/run_truthfulqa.sh student_weighted
sbatch evaluation/bash/run_truthfulqa.sh student_random
sbatch evaluation/bash/run_truthfulqa.sh teacher
```

### Step 4: Compare Results

Generate comparison plots and final report:

```bash
sbatch evaluation/bash/run_comparison.sh
```

Outputs:
- `evaluation_results/analysis/dolly_comparison.png` - Scores by uncertainty band
- `evaluation_results/analysis/benchmark_comparison.png` - MMLU/TruthfulQA accuracies
- `evaluation_results/analysis/evaluation_report.md` - Summary markdown report

## Configuration

All paths are centralized in `config.py`:

```python
from evaluation.config import MODELS, RESULTS_DIR, SCORED_OUTPUTS

print(MODELS["student_weighted"])  # Model checkpoint path
print(RESULTS_DIR)                 # Output directory
```

## Output Structure

```
/no_backups/m159/distillation_experiments/evaluation_results/
├── model_responses/
│   ├── student_weighted_responses.jsonl
│   └── student_random_responses.jsonl
├── dolly_judge/
│   ├── student_weighted_scores.jsonl
│   ├── student_random_scores.jsonl
│   ├── score_distribution_boxplot.png
│   ├── scores_vs_uncertainty.png
│   └── ... (8 plots per model)
├── mmlu/
│   ├── student_weighted_mmlu.json
│   ├── student_random_mmlu.json
│   └── teacher_mmlu.json
├── truthfulqa/
│   ├── student_weighted_truthfulqa.json
│   ├── student_random_truthfulqa.json
│   └── teacher_truthfulqa.json
└── analysis/
    ├── dolly_comparison.png
    ├── benchmark_comparison.png
    └── evaluation_report.md
```

## Key Metrics

### Dolly Judge (Stratified by Teacher Uncertainty)
- **Correctness** (1-10): Factual accuracy
- **Helpfulness** (1-10): Usefulness of response
- **Coherence** (1-10): Clarity and structure

Scores are broken down by uncertainty band:
- **Low** (< 0.3): Teacher was confident
- **Medium** (0.3-0.6): Moderate uncertainty
- **High** (> 0.6): Teacher was uncertain

### Benchmarks
- **MMLU**: Multi-task accuracy (5-shot)
- **TruthfulQA**: Truthfulness in MCQ setting
