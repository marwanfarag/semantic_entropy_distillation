"""
Argument dataclasses for teacher response generation.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.1-8B",
        metadata={"help": "Path to the teacher model."}
    )
    cache_dir: Optional[str] = field(default=None)
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Data type for model weights (float16, bfloat16, float32)."}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    data_path: str = field(
        default="./alpaca_data.json",
        metadata={"help": "Path to the dataset."}
    )
    dataset_type: str = field(
        default="alpaca",
        metadata={"help": "Dataset type: 'alpaca' or 'dolly'. Determines field names (instruction/input/output vs context/question/answer)."}
    )


@dataclass
class GenerationArguments:
    """Arguments for response generation."""
    output_dir: str = field(
        default="./teacher_outputs",
        metadata={"help": "Directory to save teacher outputs."}
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for generation."}
    )
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum number of new tokens to generate."}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum total sequence length."}
    )
    num_responses: int = field(
        default=5,
        metadata={"help": "Number of responses to generate per prompt."}
    )
    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling (higher = more diverse)."}
    )
    top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p for nucleus sampling (higher = more diverse)."}
    )
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether to use sampling or greedy decoding."}
    )
    parallel_generation: bool = field(
        default=False,
        metadata={"help": "Generate all responses in parallel using num_return_sequences (faster but uses more memory). If False, generates sequentially."}
    )
    save_logits: bool = field(
        default=True,
        metadata={"help": "Whether to save logits for distillation."}
    )
    num_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples to process (None for all)."}
    )
    start_idx: int = field(
        default=0,
        metadata={"help": "Starting index of samples to process (for parallel jobs)."}
    )
    end_idx: Optional[int] = field(
        default=None,
        metadata={"help": "Ending index of samples to process (None for all remaining)."}
    )
    save_every: int = field(
        default=1000,
        metadata={"help": "Save checkpoint every N samples."}
    )
