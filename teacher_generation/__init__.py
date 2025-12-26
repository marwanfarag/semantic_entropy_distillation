"""
Teacher Generation Package for Knowledge Distillation.

This package provides tools to generate teacher model responses
for use in knowledge distillation training.
"""

from .arguments import ModelArguments, DataArguments, GenerationArguments
from .prompts import format_prompt, extract_response_and_summary, PROMPT_DICT
from .generator import generate_responses
from .checkpointing import append_responses, finalize_output, jload

__all__ = [
    "ModelArguments",
    "DataArguments", 
    "GenerationArguments",
    "format_prompt",
    "extract_response_and_summary",
    "PROMPT_DICT",
    "generate_responses",
    "append_responses",
    "finalize_output",
    "jload",
]

