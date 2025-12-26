"""
Checkpoint and file I/O utilities for teacher response generation.
"""

import json
import logging
import os
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


def jload(filepath: str, mode: str = "r") -> Any:
    """Load a .json file into a dictionary."""
    with open(filepath, mode=mode) as file:
        return json.load(file)


def get_output_filepath(output_dir: str, start_idx: int) -> str:
    """Get the output filepath for a job based on its start_idx."""
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"job_{start_idx}_responses.jsonl")


def append_responses(
    data: List[Dict],
    output_dir: str,
    start_idx: int,
    save_logits: bool = False,
) -> None:
    """
    Append generated responses to a single JSONL file per job.
    
    Each job (identified by start_idx) writes to one file.
    Each line is one instruction with all its responses grouped together.
    
    Args:
        data: List of generated samples with responses
        output_dir: Directory to save files to
        start_idx: Starting index of this data range (identifies the job)
        save_logits: Whether to save logits as numpy files
    """
    output_file = get_output_filepath(output_dir, start_idx)
    
    # Append each sample as a JSONL entry (instruction + all responses grouped)
    with open(output_file, "a") as f:
        for d in data:
            # Combine instruction and input into single instruction field
            instruction = d["instruction"]
            if d.get("input", ""):
                instruction = f"{instruction}\n\nInput: {d['input']}"
            
            # Group all responses under one instruction
            responses = []
            for resp in d["teacher_responses"]:
                responses.append({
                    "response": resp["response"],
                    "summary": resp.get("summary", ""),
                })
            
            entry = {
                "instruction": instruction,
                "responses": responses,
            }
            f.write(json.dumps(entry) + "\n")
    
    logger.info(f"  Appended {len(data)} samples to {output_file}")


def finalize_output(output_dir: str, start_idx: int, total_samples: int, num_responses: int) -> None:
    """
    Log completion message for the job.
    
    Args:
        output_dir: Directory where files are saved
        start_idx: Starting index of this data range
        total_samples: Total number of samples processed
        num_responses: Number of responses per prompt
    """
    output_file = get_output_filepath(output_dir, start_idx)
    total_responses = total_samples * num_responses
    
    logger.info(f"Generation complete! Saved {total_samples} samples x {num_responses} responses = {total_responses} total responses")
    logger.info(f"Output saved to: {output_file}")
