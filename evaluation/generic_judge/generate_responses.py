"""
Generic Response Generation Script

Generates model responses for any HuggingFace dataset with customizable field mappings.
Works with both HuggingFace model names and local model paths.

Usage:
    python evaluation/generate_responses.py \
        --model_path "meta-llama/Llama-3.1-8B-Instruct" \
        --dataset "truthfulqa/truthful_qa" \
        --dataset_config "generation" \
        --split "validation" \
        --question_field "question" \
        --output_path "outputs/responses.jsonl"
"""

import argparse
import json
import logging
import os
from typing import Optional
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """
    Load model and tokenizer from HuggingFace or local path.
    Auto-detects if path is local or HuggingFace model name.
    """
    is_local = os.path.exists(model_path)
    logger.info(f"Loading model from: {model_path} ({'local' if is_local else 'HuggingFace'})")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def format_prompt(question: str, context: Optional[str] = None) -> str:
    """Format a question (with optional context) into a prompt."""
    if context and context.strip():
        prompt = f"""Answer the following question based on the given context.

Context: {context}

Question: {question}

Answer:"""
    else:
        prompt = f"""Answer the following question.

Question: {question}

Answer:"""
    return prompt


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Generate a single response."""
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
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode only the generated tokens
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Generate responses for any HuggingFace dataset")
    
    # Model arguments
    parser.add_argument("--model_path", required=True, 
                        help="HuggingFace model name or local path")
    parser.add_argument("--model_name", default=None,
                        help="Name for the model (used in output). Defaults to model_path basename.")
    
    # Dataset arguments
    parser.add_argument("--dataset", required=True,
                        help="HuggingFace dataset name (e.g., 'truthfulqa/truthful_qa')")
    parser.add_argument("--dataset_config", default=None,
                        help="Dataset configuration (e.g., 'generation' for TruthfulQA)")
    parser.add_argument("--split", default="validation",
                        help="Dataset split to use")
    
    # Field mappings
    parser.add_argument("--question_field", default="question",
                        help="Name of the question/instruction field in the dataset")
    parser.add_argument("--context_field", default=None,
                        help="Name of the context/input field (optional)")
    parser.add_argument("--ground_truth_field", default=None,
                        help="Name of the ground truth field (for reference, not used in generation)")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to process (None for all)")
    
    # Output arguments
    parser.add_argument("--output_path", required=True,
                        help="Path to save responses JSONL")
    
    args = parser.parse_args()
    
    # Set model name
    if args.model_name is None:
        args.model_name = os.path.basename(args.model_path.rstrip('/'))
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset} (config: {args.dataset_config}, split: {args.split})")
    if args.dataset_config:
        dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)
    
    # Limit samples if specified
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    logger.info(f"Processing {len(dataset)} samples")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Generate responses
    results = []
    for example in tqdm(dataset, desc=f"Generating responses"):
        # Extract fields
        question = example[args.question_field]
        context = example.get(args.context_field) if args.context_field else None
        ground_truth = example.get(args.ground_truth_field) if args.ground_truth_field else None
        
        # Format prompt and generate
        prompt = format_prompt(question, context)
        response = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        
        # Store result
        result = {
            "question": question,
            "response": response,
            "model": args.model_name,
        }
        if context:
            result["context"] = context
        if ground_truth:
            result["ground_truth"] = ground_truth
        
        results.append(result)
    
    # Save results
    with open(args.output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"Saved {len(results)} responses to {args.output_path}")


if __name__ == "__main__":
    main()
