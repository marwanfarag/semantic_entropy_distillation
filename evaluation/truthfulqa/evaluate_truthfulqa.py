"""
TruthfulQA MCQ Evaluation Script

Evaluates model on TruthfulQA multiple choice questions.
"""

import argparse
import json
import logging
from typing import List, Dict
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

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


def format_mcq_prompt(question: str, choices: List[str]) -> str:
    """Format MCQ question as prompt."""
    prompt = f"Question: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "\nAnswer (choose A, B, C, or D): "
    return prompt


def evaluate_mcq(model, tokenizer, question: str, choices: List[str], correct_idx: int) -> bool:
    """Evaluate a single MCQ question."""
    prompt = format_mcq_prompt(question, choices)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2,
            temperature=0.0,
            do_sample=False,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Extract answer (A, B, C, or D)
    predicted_answer = response[0].upper() if response else ""
    correct_answer = chr(65 + correct_idx)
    
    return predicted_answer == correct_answer


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on TruthfulQA MCQ")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_path", required=True, help="Path to save results")
    parser.add_argument("--split", default="validation", help="Dataset split")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    # Load TruthfulQA dataset
    logger.info("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "multiple_choice", split=args.split)
    
    # Evaluate
    results = []
    correct = 0
    total = 0
    
    for example in tqdm(dataset, desc="Evaluating TruthfulQA"):
        question = example["question"]
        choices = example["mc1_targets"]["choices"]
        correct_idx = example["mc1_targets"]["labels"].index(1)
        
        is_correct = evaluate_mcq(model, tokenizer, question, choices, correct_idx)
        
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
    import os
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
