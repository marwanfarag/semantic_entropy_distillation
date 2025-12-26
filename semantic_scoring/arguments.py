"""
Arguments for semantic scoring.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScoringArguments:
    """Arguments for semantic scoring configuration."""
    
    # NLI model configuration
    nli_model_name: str = field(
        default="microsoft/deberta-large-mnli",
        metadata={"help": "NLI model for entailment/contradiction detection."}
    )
    
    # Input/output paths
    teacher_outputs_dir: str = field(
        default="./teacher_outputs",
        metadata={"help": "Directory containing teacher response JSONL files."}
    )
    output_path: str = field(
        default="./scored_outputs.jsonl",
        metadata={"help": "Output file for scored results."}
    )
    
    # Scoring thresholds
    entailment_threshold: float = field(
        default=0.5,
        metadata={"help": "Threshold for bidirectional entailment (τ_E)."}
    )
    
    # Weights for final score
    entropy_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for semantic entropy in final score."}
    )
    contradiction_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for contradiction in final score."}
    )
    
    # Text length handling
    max_response_tokens: int = field(
        default=100,
        metadata={"help": "Max tokens before using summary instead of full response."}
    )
    
    # Processing
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for NLI inference."}
    )
    device: Optional[str] = field(
        default=None,
        metadata={"help": "Device for model inference (cuda, cpu, or None for auto)."}
    )
