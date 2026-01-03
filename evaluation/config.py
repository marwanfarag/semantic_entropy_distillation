"""
Centralized configuration for evaluation package.

Contains paths to model checkpoints, data directories, and output locations.
"""

# =============================================================================
# Model Checkpoints
# =============================================================================
MODELS = {
    "student_weighted": "/no_backups/m159/distillation_experiments/distillation_weighted",
    "student_random": "/no_backups/m159/distillation_experiments/distillation_random/checkpoint-90",
    "teacher": "meta-llama/Llama-3.1-8B-Instruct",
}

# Base Llama model (for tokenizer if needed)
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# =============================================================================
# Data Paths
# =============================================================================
EXPERIMENT_DIR = "/no_backups/m159/distillation_experiments"
TEACHER_OUTPUTS_DIR = "/usrhomes/m159/stanford_alpaca/normal_distillation/teacher_outputs"
SCORED_OUTPUTS = f"{TEACHER_OUTPUTS_DIR}/scored_outputs.jsonl"

# Teacher response files (with multiple responses per instruction)
TEACHER_RESPONSES = [
    f"{TEACHER_OUTPUTS_DIR}/job_0_responses.jsonl",
    f"{TEACHER_OUTPUTS_DIR}/job_7500_responses.jsonl",
]

# Dolly dataset
DOLLY_DATASET = "databricks/databricks-dolly-15k"

# =============================================================================
# Output Directories
# =============================================================================
RESULTS_DIR = f"{EXPERIMENT_DIR}/evaluation_results"
RESPONSES_DIR = f"{RESULTS_DIR}/model_responses"
DOLLY_JUDGE_RESULTS = f"{RESULTS_DIR}/dolly_judge"
MMLU_RESULTS = f"{RESULTS_DIR}/mmlu"
TRUTHFULQA_RESULTS = f"{RESULTS_DIR}/truthfulqa"
ANALYSIS_RESULTS = f"{RESULTS_DIR}/analysis"

# =============================================================================
# Evaluation Settings
# =============================================================================
NUM_EVALUATION_SAMPLES = 500  # Number of samples for judge evaluation
JUDGE_MODEL = "Qwen/Qwen3-32B"

# Uncertainty thresholds for stratification
UNCERTAINTY_LOW = 0.3
UNCERTAINTY_HIGH = 0.6
