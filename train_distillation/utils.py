"""
Utility functions for knowledge distillation training.
"""

import glob
import json
import logging
import os
from typing import Dict, Sequence

import torch
import transformers

from .constants import IGNORE_INDEX


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    with open(f, mode=mode) as file:
        return json.load(file)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def tokenize(
    texts: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    batch_size: int = 1000,
    show_progress: bool = True,
) -> list:
    """
    Tokenize a list of texts using batch processing.
    
    Args:
        texts: List of strings to tokenize
        tokenizer: HuggingFace tokenizer
        batch_size: Number of texts to tokenize at once
        show_progress: Whether to show progress bar
        
    Returns:
        List of token ID tensors
    """
    tokens = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    # Setup progress bar
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Tokenizing", total=num_batches)
        except ImportError:
            pass
    
    # Batch tokenize
    for i in iterator:
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False,
            return_tensors=None,
        )
        for ids in encoded["input_ids"]:
            tokens.append(torch.tensor(ids))
    
    return tokens


def preprocess(
    prompts: Sequence[str],
    responses: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Tokenize prompts and responses, creating labels with masked prompt tokens.
    
    Args:
        prompts: List of prompt strings
        responses: List of response strings (including eos token)
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Dict with input_ids, labels, and source_lens
    """
    logging.info(f"Tokenizing {len(prompts)} samples...")
    
    # Tokenize prompts and responses separately
    prompt_tokens = tokenize(prompts, tokenizer, show_progress=True)
    response_tokens = tokenize(responses, tokenizer, show_progress=True)
    
    # Concatenate tokens and create labels
    input_ids = []
    labels = []
    prompt_lens = []
    
    for prompt_ids, response_ids in zip(prompt_tokens, response_tokens):
        # Concatenate prompt + response (skip BOS token in response if present)
        if response_ids[0] == tokenizer.bos_token_id:
            response_ids = response_ids[1:]
        
        full_ids = torch.cat([prompt_ids, response_ids])
        
        # Truncate if needed
        max_len = tokenizer.model_max_length
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]
        
        # Create label with prompt masked
        prompt_len = len(prompt_ids)
        label = full_ids.clone()
        label[:prompt_len] = IGNORE_INDEX
        
        input_ids.append(full_ids)
        labels.append(label)
        prompt_lens.append(prompt_len)
    
    logging.info("Tokenization complete!")
    return {
        "input_ids": input_ids, # Example: [tensor([1, 2, 3, 4, 5]), tensor([6, 7, 8, 9, 10])]
        "labels": labels, # Example: [tensor([IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 12, 13, 14, 15, 16]), tensor([IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 17, 18, 19, 20, 21])]
        "source_lens": prompt_lens, # Example: [5, 5]
    }


def load_teacher_responses(teacher_outputs_dir: str) -> list:
    """
    Load teacher responses from JSONL files in a directory.
    
    Finds all job_*_responses.jsonl files and extracts the instruction
    and first response from each entry.
    
    Args:
        teacher_outputs_dir: Directory containing job_*_responses.jsonl files
        
    Returns:
        List of dicts with 'instruction' and 'output' keys
    """
    pattern = os.path.join(teacher_outputs_dir, "job_*_responses.jsonl")
    jsonl_files = sorted(glob.glob(pattern))
    
    if not jsonl_files:
        raise ValueError(f"No job_*_responses.jsonl files found in {teacher_outputs_dir}")
    
    logging.info(f"Found {len(jsonl_files)} JSONL files in {teacher_outputs_dir}")
    
    all_data = []
    for jsonl_file in jsonl_files:
        logging.info(f"Loading {jsonl_file}...")
        with open(jsonl_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                entry = json.loads(line)
                instruction = entry.get("instruction", "")
                responses = entry.get("responses", [])
                
                # Get first response, use summary if response is empty
                output = ""
                if responses:
                    first = responses[0]
                    output = first.get("response", "") or first.get("summary", "")
                
                if instruction and output:
                    all_data.append({
                        "instruction": instruction,
                        "output": output,
                    })
    
    logging.info(f"Loaded {len(all_data)} samples from teacher responses")
    return all_data
