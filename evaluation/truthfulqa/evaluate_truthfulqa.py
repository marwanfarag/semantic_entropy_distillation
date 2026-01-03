"""
TruthfulQA MCQ Evaluation Script

Evaluates model on TruthfulQA multiple choice questions.
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

# Add parent to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import parse_mcq_answer, format_mcq_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load model and tokenizer."""
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
    return model, tokenizer


def evaluate_mcq(model, tokenizer, question: str, choices: List[str], correct_idx: int, debug=False) -> bool:
    """Evaluate a single MCQ question using shared utilities."""
    # Format prompt using shared utility (no few-shot for TruthfulQA)
    prompt = format_mcq_prompt(
        question, 
        choices, 
        few_shot_examples=None,
        instruction="Answer the following question by responding with only the letter of the correct choice."
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.0,
            do_sample=False,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Parse answer using shared utility
    num_choices = len(choices)
    predicted_answer = parse_mcq_answer(response, choices, num_choices=num_choices)
    
    correct_answer = chr(65 + correct_idx)
    is_correct = predicted_answer == correct_answer
    
    if debug:
        print("\n" + "="*80)
        print(f"QUESTION: {question}")
        print(f"CHOICES:")
        for i, c in enumerate(choices):
            marker = " ✓" if chr(65+i) == correct_answer else ""
            print(f"  {chr(65+i)}. {c}{marker}")
        print(f"\nMODEL OUTPUT: '{response}'")
        print(f"PARSED ANSWER: '{predicted_answer}' | CORRECT: '{correct_answer}' | {'✓ CORRECT' if is_correct else '✗ WRONG'}")
        print("="*80)
        sys.stdout.flush()
    
    return is_correct


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on TruthfulQA MCQ")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_path", required=True, help="Path to save results")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate (for testing)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging of responses")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    # Load TruthfulQA dataset
    logger.info("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split=args.split)
    
    # Limit samples if specified
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")
    
    # Evaluate
    results = []
    correct = 0
    total = 0
    
    for example in tqdm(dataset, desc="Evaluating TruthfulQA"):
        question = example["question"]
        choices = example["mc1_targets"]["choices"]
        correct_idx = example["mc1_targets"]["labels"].index(1)
        
        is_correct = evaluate_mcq(model, tokenizer, question, choices, correct_idx, debug=args.debug)
        
        results.append({
            "question": question,
            "predicted_correct": is_correct,
        })
        
        if is_correct:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.output_path, 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": results,
        }, f, indent=2)
    
    logger.info(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
