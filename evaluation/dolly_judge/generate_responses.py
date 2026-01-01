"""
Generate Model Responses on Dolly Dataset

Loads a model checkpoint and generates responses for the Dolly dataset.
Output is saved as JSONL for subsequent evaluation by the judge.
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional
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


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 512,
) -> str:
    """Generate a response for the given instruction."""
    prompt = format_prompt(instruction, input_text)
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
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
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    return response


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
        # Filter using the combined format that matches scored_outputs.json
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
    
    # Generate responses
    logger.info(f"Generating responses, saving to: {output_path}")
    
    with open(output_path, 'w') as f:
        for example in tqdm(dataset, desc=f"Generating ({args.model_name})"):
            instruction = example["instruction"]
            input_text = example.get("context", "")
            
            response = generate_response(
                model, tokenizer,
                instruction, input_text,
                max_new_tokens=args.max_new_tokens
            )
            
            # Use combined instruction key to match teacher output format
            instruction_key = build_instruction_key(instruction, input_text)
            
            result = {
                "instruction": instruction_key,  # Combined format for matching
                "input": input_text,  # Keep separate for reference
                "output": response,
                "category": example.get("category", ""),
            }
            
            f.write(json.dumps(result) + '\n')
            f.flush()  # Ensure progress is saved
    
    logger.info(f"Saved {len(dataset)} responses to {output_path}")


if __name__ == "__main__":
    main()
