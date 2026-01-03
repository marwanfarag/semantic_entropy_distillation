"""
MMLU Evaluation Script

Evaluates model on MMLU (Massive Multitask Language Understanding) benchmark with 5-shot prompting.
"""

import argparse
import json
import logging
import os
import sys
from typing import List
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def evaluate_question(model, tokenizer, few_shot_examples, question, choices, correct_idx, debug=False):
    """Evaluate a single question using shared utilities."""
    # Format prompt using shared utility
    prompt = format_mcq_prompt(question, choices, few_shot_examples)
    
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
    predicted_answer = parse_mcq_answer(response, choices)
    
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
    parser = argparse.ArgumentParser(description="Evaluate model on MMLU")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_path", required=True, help="Path to save results")
    parser.add_argument("--num_shots", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate (for testing)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging of responses")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    # Load MMLU dataset (standard MMLU from cais/mmlu)
    logger.info("Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dev_dataset = load_dataset("cais/mmlu", "all", split="dev")
    
    # Limit samples if specified
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")
    
    # Use dev set for few-shot examples
    few_shot_examples = list(dev_dataset.select(range(args.num_shots)))
    
    # Evaluate
    results_by_subject = {}
    overall_correct = 0
    overall_total = 0
    
    for example in tqdm(dataset, desc="Evaluating MMLU"):
        subject = example.get("subject", "unknown")
        question = example["question"]
        choices = example["choices"]
        correct_idx = example["answer"]
        
        is_correct = evaluate_question(model, tokenizer, few_shot_examples, question, choices, correct_idx, debug=args.debug)
        
        if subject not in results_by_subject:
            results_by_subject[subject] = {"correct": 0, "total": 0}
        
        results_by_subject[subject]["total"] += 1
        if is_correct:
            results_by_subject[subject]["correct"] += 1
            overall_correct += 1
        overall_total += 1
    
    # Compute accuracies
    for subject in results_by_subject:
        results_by_subject[subject]["accuracy"] = (
            results_by_subject[subject]["correct"] / results_by_subject[subject]["total"]
        )
    
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_correct}/{overall_total})")
    
    # Save results
    import os
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.output_path, 'w') as f:
        json.dump({
            "overall_accuracy": overall_accuracy,
            "overall_correct": overall_correct,
            "overall_total": overall_total,
            "by_subject": results_by_subject,
        }, f, indent=2)
    
    logger.info(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
