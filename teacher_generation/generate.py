#!/usr/bin/env python3
"""
Generate teacher model responses for knowledge distillation.

This script loads a teacher model (e.g., LLaMA 3.1 8B Instruct) and generates
responses for each instruction in the Alpaca dataset. It saves both the text
responses and optionally the logits for use in knowledge distillation training.

Usage:
    python -m teacher_generation.generate \\
        --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \\
        --data_path ./alpaca_data.json \\
        --output_dir ./teacher_outputs \\
        --batch_size 8 \\
        --max_new_tokens 512

For parallel jobs, use --start_idx and --end_idx to split the dataset.
"""

import gc
import logging
import os
import sys
import torch
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from .arguments import ModelArguments, DataArguments, GenerationArguments
from .prompts import format_prompt
from .generator import generate_responses
from .checkpointing import jload, append_responses, finalize_output, get_output_filepath

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Force flush stdout for real-time logging
sys.stdout.reconfigure(line_buffering=True)

# Helper to print first few responses of first few prompts
def print_first_few_prompts(i, batch_start, example, result):
    sample_idx = batch_start + i
    if sample_idx < 8:
        logger.info(f"\n{'#'*80}")
        logger.info(f"FULL SAMPLE {sample_idx + 1}/8 EXAMPLE:")
        logger.info(f"{'#'*80}")
        logger.info(f"INSTRUCTION: {example['instruction']}")
        logger.info(f"INPUT: {example.get('input', '(none)')}")
        for resp_idx, resp in enumerate(result["responses"][i]):
            logger.info(f"\n--- Response {resp_idx + 1}/{result['num_responses']} ---")
            logger.info(f"RESPONSE:\n{resp['response']}")
            logger.info(f"SUMMARY: {resp['summary']}")
        logger.info(f"{'#'*80}\n")
def main():
    """Main entry point for teacher response generation."""
    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments))
    model_args, data_args, gen_args = parser.parse_args_into_dataclasses()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading teacher model: {model_args.model_name_or_path}")
    torch_dtype = getattr(torch, model_args.torch_dtype)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    logger.info(f"Model loaded successfully! Model type: {type(model).__name__}")
    logger.info(f"Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        padding_side="left",  # For batch generation
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer loaded! Vocab size: {len(tokenizer)}")
    
    # Load dataset
    logger.info(f"Loading dataset: {data_args.data_path}")
    
    # Check if loading from HuggingFace or local file
    if data_args.data_path.startswith("hf://"):
        # Load from HuggingFace datasets
        from datasets import load_dataset
        
        dataset_name = data_args.data_path[5:]  # Remove 'hf://' prefix
        logger.info(f"Loading from HuggingFace: {dataset_name}")
        
        # Load dataset
        hf_dataset = load_dataset(dataset_name, split="train")
        
        # Convert to list of dicts
        dataset = []
        for example in hf_dataset:
            dataset.append(dict(example))
        
        logger.info(f"Loaded {len(dataset)} examples from HuggingFace")
    else:
        # Load from local JSON file
        dataset = jload(data_args.data_path)
    
    # Apply range slicing for parallel jobs
    total_samples = len(dataset)
    start_idx = gen_args.start_idx
    end_idx = gen_args.end_idx if gen_args.end_idx is not None else total_samples
    
    # Clamp to valid range
    start_idx = max(0, min(start_idx, total_samples))
    end_idx = max(start_idx, min(end_idx, total_samples))
    
    dataset = dataset[start_idx:end_idx]
    
    # Further limit by num_samples if specified
    if gen_args.num_samples is not None:
        dataset = dataset[:gen_args.num_samples]
    
    logger.info(f"Processing {len(dataset)} samples (indices {start_idx} to {start_idx + len(dataset) - 1})")
    
    # Clear any existing output file for this job (fresh start)
    output_file = get_output_filepath(gen_args.output_dir, start_idx)
    if os.path.exists(output_file):
        os.remove(output_file)
        logger.info(f"Cleared existing output file: {output_file}")
    
    # Process in batches
    total_processed = 0
    pending_data = []
    
    for batch_start in tqdm(range(0, len(dataset), gen_args.batch_size), desc="Generating"):
        batch_end = min(batch_start + gen_args.batch_size, len(dataset))
        batch = dataset[batch_start:batch_end]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH {batch_start//gen_args.batch_size + 1}: samples {batch_start}-{batch_end-1} of {len(dataset)}")
        logger.info(f"{'='*60}")
        
        # Format prompts
        logger.info(f"[Step 1/3] Formatting prompts...")
        prompts = [format_prompt(example, data_args.dataset_type) for example in batch]
        logger.info(f"  Formatted {len(prompts)} prompts")
        
        # Generate responses
        logger.info(f"[Step 2/3] Generating teacher responses...")
        result = generate_responses(model, tokenizer, prompts, gen_args, device)
        logger.info(f"  Response generation complete!")
        
        # Store results - now with multiple responses per prompt
        logger.info(f"[Step 3/3] Storing results...")
        for i, example in enumerate(batch):
            # Normalize field names based on dataset type
            if data_args.dataset_type == "dolly":
                # Dolly: instruction, context, response
                instruction = example.get("instruction", "")
                input_text = example.get("context", "")
                original_output = example.get("response", "")
            else:
                # Alpaca: instruction, input, output
                instruction = example["instruction"]
                input_text = example.get("input", "")
                original_output = example["output"]
            
            # result["responses"][i] is a list of dicts with response, summary, full_text
            responses_for_sample = result["responses"][i]
            
            sample_result = {
                "idx": start_idx + batch_start + i,
                "instruction": instruction,
                "input": input_text,
                "original_output": original_output,
                "teacher_responses": responses_for_sample,
                "prompt_length": result["prompt_lengths"][i],
            }
            
            print_first_few_prompts(i, batch_start, example, result)
            pending_data.append(sample_result)
        
        total_processed += len(batch)
        logger.info(f"  Stored {len(batch)} samples (x{result['num_responses']} responses each). Total so far: {total_processed}")
        
        # Print full samples for the first 8 examples

        
        # Append to file periodically
        if len(pending_data) >= gen_args.save_every:
            append_responses(pending_data, gen_args.output_dir, start_idx, gen_args.save_logits)
            pending_data = []
            # Free memory after save
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("  [Memory] Cleared cache after save")
    
    # Append any remaining data
    if pending_data:
        append_responses(pending_data, gen_args.output_dir, start_idx, gen_args.save_logits)
    
    # Log completion
    finalize_output(gen_args.output_dir, start_idx, total_processed, gen_args.num_responses)


if __name__ == "__main__":
    main()
