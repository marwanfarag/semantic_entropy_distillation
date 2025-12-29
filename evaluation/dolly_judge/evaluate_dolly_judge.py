"""
Dolly Judge Evaluation Script

Evaluates model responses using Qwen 2.5 14B as a judge.
Stage A: Independent scoring of each model (1-10 scale)
Stage B: Head-to-head comparisons
"""

import argparse
import json
import logging
import os
from typing import List, Dict
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing AI assistant responses to user instructions.

Instruction: {instruction}

Context (if any): {input_text}

Ground Truth Response: {ground_truth}

Model Response: {model_response}

Evaluate the model's response on the following criteria (1-10 scale):

1. Correctness: Is the response factually accurate and addresses the instruction correctly?
2. Helpfulness: Does the response provide useful and relevant information?
3. Coherence: Is the response well-structured, clear, and easy to understand?

Provide your scores in JSON format:
{{"correctness": X, "helpfulness": Y, "coherence": Z, "justification": "brief explanation"}}

Your evaluation:"""


def load_judge_model(model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
    """Load the judge model."""
    logger.info(f"Loading judge model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


def load_dolly_ground_truth(dolly_path: str) -> Dict[str, Dict]:
    """Load Dolly dataset with ground truth responses."""
    logger.info(f"Loading Dolly ground truth from: {dolly_path}")
    
    # Load from the databricks dolly dataset
    from datasets import load_dataset
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    # Build lookup by instruction
    ground_truth = {}
    for example in dataset:
        instruction = example["instruction"]
        ground_truth[instruction] = {
            "instruction": instruction,
            "input": example.get("context", ""),
            "output": example["response"],
            "category": example.get("category", ""),
        }
    
    logger.info(f"Loaded {len(ground_truth)} ground truth examples")
    return ground_truth


def load_model_responses(response_path: str) -> Dict[str, str]:
    """Load model responses from JSONL file."""
    logger.info(f"Loading model responses from: {response_path}")
    
    responses = {}
    with open(response_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                instruction = data.get("instruction", "")
                
                # Handle different response formats
                if "output" in data:
                    response = data["output"]
                elif "responses" in data and len(data["responses"]) > 0:
                    # For teacher files with multiple responses
                    response = data["responses"][0]["response"]
                elif "representative_response" in data:
                    # For scored_outputs.jsonl
                    response = data["representative_response"]
                else:
                    continue
                
                responses[instruction] = response
    
    logger.info(f"Loaded {len(responses)} model responses")
    return responses


def load_teacher_scores(scored_outputs_path: str) -> Dict[str, float]:
    """Load teacher uncertainty scores for stratification."""
    logger.info(f"Loading teacher scores from: {scored_outputs_path}")
    
    scores = {}
    with open(scored_outputs_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                instruction = data.get("instruction", "")
                score = data.get("score", 0.0)
                scores[instruction] = score
    
    logger.info(f"Loaded scores for {len(scores)} instructions")
    return scores


def stratified_sample(
    ground_truth: Dict[str, Dict],
    teacher_scores: Dict[str, float],
    model_responses: Dict[str, str],
    num_samples: int = 500,
) -> List[Dict]:
    """Sample instructions stratified by teacher uncertainty."""
    
    # Find common instructions
    common_instructions = set(ground_truth.keys()) & set(model_responses.keys()) & set(teacher_scores.keys())
    logger.info(f"Found {len(common_instructions)} common instructions")
    
    # Stratify by uncertainty bands
    low, medium, high = [], [], []
    for instruction in common_instructions:
        score = teacher_scores[instruction]
        if score < 0.3:
            low.append(instruction)
        elif score < 0.6:
            medium.append(instruction)
        else:
            high.append(instruction)
    
    logger.info(f"Stratification: {len(low)} low, {len(medium)} medium, {len(high)} high")
    
    # Sample proportionally (or equally from each band)
    import random
    random.seed(42)
    
    samples_per_band = num_samples // 3
    sampled_instructions = (
        random.sample(low, min(samples_per_band, len(low))) +
        random.sample(medium, min(samples_per_band, len(medium))) +
        random.sample(high, min(samples_per_band, len(high)))
    )
    
    # Build sample list
    samples = []
    for instruction in sampled_instructions:
        samples.append({
            "instruction": instruction,
            "ground_truth": ground_truth[instruction],
            "model_response": model_responses[instruction],
            "teacher_score": teacher_scores[instruction],
        })
    
    logger.info(f"Sampled {len(samples)} examples for evaluation")
    return samples


def evaluate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    ground_truth: str,
    model_response: str,
) -> Dict:
    """Evaluate a single response using the judge model."""
    
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction=instruction,
        input_text=input_text if input_text else "None",
        ground_truth=ground_truth,
        model_response=model_response,
    )
    
    # Generate judge response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,  # Deterministic
            do_sample=False,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Parse JSON from response
    try:
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
        else:
            logger.warning(f"Could not parse JSON from judge response: {response}")
            scores = {"correctness": 0, "helpfulness": 0, "coherence": 0, "justification": "Parse error"}
    except Exception as e:
        logger.warning(f"Error parsing judge response: {e}")
        scores = {"correctness": 0, "helpfulness": 0, "coherence": 0, "justification": str(e)}
    
    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate model using Qwen 2.5 14B judge")
    parser.add_argument("--model_responses", required=True, help="Path to model responses JSONL")
    parser.add_argument("--dolly_path", default="databricks/databricks-dolly-15k", help="Dolly dataset path")
    parser.add_argument("--scored_outputs", required=True, help="Path to scored_outputs.jsonl for stratification")
    parser.add_argument("--output_path", required=True, help="Path to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to evaluate")
    parser.add_argument("--judge_model", default="Qwen/Qwen2.5-14B-Instruct", help="Judge model name")
    
    args = parser.parse_args()
    
    # Load judge model
    judge_model, judge_tokenizer = load_judge_model(args.judge_model)
    
    # Load data
    ground_truth = load_dolly_ground_truth(args.dolly_path)
    model_responses = load_model_responses(args.model_responses)
    teacher_scores = load_teacher_scores(args.scored_outputs)
    
    # Stratified sampling
    samples = stratified_sample(ground_truth, teacher_scores, model_responses, args.num_samples)
    
    # Evaluate
    results = []
    for sample in tqdm(samples, desc="Evaluating"):
        scores = evaluate_response(
            judge_model,
            judge_tokenizer,
            sample["instruction"],
            sample["ground_truth"]["input"],
            sample["ground_truth"]["output"],
            sample["model_response"],
        )
        
        results.append({
            "instruction": sample["instruction"],
            "teacher_score": sample["teacher_score"],
            "scores": scores,
        })
    
    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"Saved {len(results)} evaluation results to {args.output_path}")


if __name__ == "__main__":
    main()
