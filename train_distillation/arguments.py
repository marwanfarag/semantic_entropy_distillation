"""
Argument dataclasses for distillation training.
"""

from dataclasses import dataclass, field
from typing import Optional
import transformers


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.2-3B",
        metadata={"help": "Path to the student model."}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    data_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to the training data (JSON format)."}
    )
    teacher_outputs_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing JSONL teacher response files."}
    )


@dataclass
class DistillationTrainingArguments(transformers.TrainingArguments):
    """Training arguments for distillation."""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."},
    )
    # Evaluation arguments
    eval_num_samples: int = field(
        default=50,
        metadata={"help": "Number of samples for validation split."}
    )
    eval_max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum tokens to generate during validation inference."}
    )
