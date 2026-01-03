"""
Generate Model Responses on Dolly Dataset

Loads a model checkpoint and generates responses for the Dolly dataset.
Output is saved as JSONL for subsequent evaluation by the judge.

Supports batched generation for faster processing.
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add parent to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS, DOLLY_DATASET, RESPONSES_DIR, SCORED_OUTPUTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load model and tokenizer from checkpoint."""
    logger.info(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for batch generation
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    logger.info(f"Model loaded on {model.device}")
    return model, tokenizer


def build_instruction_key(instruction: str, context: str = "") -> str:
    """
    Build instruction key matching teacher output format.
    Teacher outputs combine: "{instruction}\n\nInput: {context}"
    """
    if context and context.strip():
        return f"{instruction}\n\nInput: {context}"
    return instruction


def format_prompt(instruction: str, input_text: str = "") -> str:
    """Format instruction into model prompt."""
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def generate_batch(
    model,
    tokenizer,
    batch_prompts: List[str],
    max_new_tokens: int = 512,
) -> List[str]:
    """Generate responses for a batch of prompts."""
    
    # Tokenize batch with padding
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temperature for more greedy/deterministic
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode responses (skip input tokens)
    responses = []
    for i, output in enumerate(outputs):
        # Find where the input ends for this sample
        input_len = inputs.input_ids[i].ne(tokenizer.pad_token_id).sum().item()
        response = tokenizer.decode(
            output[input_len:],
            skip_special_tokens=True
        ).strip()
        responses.append(response)
    
    return responses


def load_evaluation_instructions(scored_outputs_path: str) -> set:
    """Load instruction set from scored_outputs for consistent evaluation."""
    instructions = set()
    if os.path.exists(scored_outputs_path):
        with open(scored_outputs_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    instructions.add(data.get("instruction", ""))
    logger.info(f"Loaded {len(instructions)} instructions from scored outputs")
    return instructions


def main():
    parser = argparse.ArgumentParser(description="Generate model responses on Dolly dataset")
    parser.add_argument(
        "--model_name",
        required=True,
        choices=list(MODELS.keys()),
        help="Model name from config (student_weighted, student_random, teacher)"
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Override model path (optional, uses config by default)"
    )
    parser.add_argument(
        "--output_dir",
        default=RESPONSES_DIR,
        help="Directory to save responses"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation (default: 8)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None = all)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per response"
    )
    parser.add_argument(
        "--use_scored_subset",
        action="store_true",
        help="Only generate for instructions in scored_outputs.jsonl"
    )
    
    args = parser.parse_args()
    
    # Get model path
    model_path = args.model_path or MODELS[args.model_name]
    
    # Load model
    model, tokenizer = load_model(model_path)
    
    # Load Dolly dataset
    logger.info(f"Loading Dolly dataset: {DOLLY_DATASET}")
    dataset = load_dataset(DOLLY_DATASET, split="train")
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Optionally filter to scored subset (using combined instruction key format)
    if args.use_scored_subset:
        scored_instructions = load_evaluation_instructions(SCORED_OUTPUTS)
        # Filter using the combined format that matches scored_outputs.jsonl
        dataset = dataset.filter(
            lambda x: build_instruction_key(x["instruction"], x.get("context", "")) in scored_instructions
        )
        logger.info(f"Filtered to {len(dataset)} examples matching scored outputs")
    
    # Limit samples if specified
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.model_name}_responses.jsonl")
    
    # Generate responses in batches
    logger.info(f"Generating responses with batch_size={args.batch_size}, saving to: {output_path}")
    
    # Convert to list for batching
    examples = list(dataset)
    total_batches = (len(examples) + args.batch_size - 1) // args.batch_size
    
    with open(output_path, 'w') as f:
        for batch_idx in tqdm(range(total_batches), desc=f"Generating ({args.model_name})"):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(examples))
            batch_examples = examples[start_idx:end_idx]
            
            # Prepare batch prompts
            batch_prompts = []
            batch_metadata = []
            for example in batch_examples:
                instruction = example["instruction"]
                input_text = example.get("context", "")
                prompt = format_prompt(instruction, input_text)
                batch_prompts.append(prompt)
                batch_metadata.append({
                    "instruction": instruction,
                    "input": input_text,
                    "category": example.get("category", ""),
                })
            
            # Generate batch
            responses = generate_batch(
                model, tokenizer,
                batch_prompts,
                max_new_tokens=args.max_new_tokens
            )
            
            # Write results
            for metadata, response in zip(batch_metadata, responses):
                # Use combined instruction key to match teacher output format
                instruction_key = build_instruction_key(metadata["instruction"], metadata["input"])
                
                result = {
                    "instruction": instruction_key,  # Combined format for matching
                    "input": metadata["input"],  # Keep separate for reference
                    "output": response,
                    "category": metadata["category"],
                }
                
                f.write(json.dumps(result) + '\n')
            
            f.flush()  # Ensure progress is saved after each batch
    
    logger.info(f"Saved {len(examples)} responses to {output_path}")


if __name__ == "__main__":
    main()
