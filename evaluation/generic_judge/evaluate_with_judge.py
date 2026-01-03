"""
Generic Judge-Based Evaluation Script

Evaluates model responses using an LLM judge for any generation dataset.
Scores responses on correctness, helpfulness, and coherence (1-10 scale).

Usage:
    python evaluation/evaluate_with_judge.py \
        --responses_path "outputs/model_responses.jsonl" \
        --judge_model "Qwen/Qwen3-32B-Instruct" \
        --output_path "results/scores.jsonl"
"""

import argparse
import json
import logging
import os
from typing import Dict, List
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing AI assistant responses.

Question: {question}

{context_section}

Ground Truth (Reference Answer): {ground_truth}

Model Response: {model_response}

Evaluate the model's response on the following criteria (1-10 scale):

1. Correctness: Is the response factually accurate and addresses the question correctly?
2. Helpfulness: Does the response provide useful and relevant information?
3. Coherence: Is the response well-structured, clear, and easy to understand?

Special case: If the model response is essentially "I don't know" or refuses to answer, assign a score of 5 for all three metrics.

Provide your scores in JSON format:
{{"correctness": X, "helpfulness": Y, "coherence": Z, "justification": "brief explanation"}}
Your answer should ONLY contain the JSON object.
Your evaluation:"""


def load_judge_model(model_name: str):
    """Load the judge model and tokenizer."""
    logger.info(f"Loading judge model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def load_responses(responses_path: str) -> List[Dict]:
    """Load responses from JSONL file."""
    responses = []
    with open(responses_path, 'r') as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    logger.info(f"Loaded {len(responses)} responses from {responses_path}")
    return responses


def evaluate_response(
    model,
    tokenizer,
    question: str,
    model_response: str,
    ground_truth: str,
    context: str = None,
) -> Dict:
    """Evaluate a single response using the judge model."""
    
    # Build context section
    context_section = f"Context: {context}" if context else ""
    
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        context_section=context_section,
        ground_truth=ground_truth if ground_truth else "Not provided",
        model_response=model_response,
    )
    
    # Generate judge response
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.0,
            do_sample=False,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Parse JSON from response (extract first valid JSON object)
    try:
        start_idx = response.find('{')
        if start_idx == -1:
            raise ValueError("No JSON object found")
        
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        json_str = response[start_idx:end_idx]
        scores = json.loads(json_str)
        
        # Validate required fields
        for field in ["correctness", "helpfulness", "coherence"]:
            if field not in scores:
                raise ValueError(f"Missing field: {field}")
                
    except Exception as e:
        logger.warning(f"Failed to parse judge response: {e}")
        scores = {
            "correctness": 0,
            "helpfulness": 0,
            "coherence": 0,
            "justification": f"Parse error: {str(e)}"
        }
    
    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate responses using LLM judge")
    
    # Input/Output
    parser.add_argument("--responses_path", required=True,
                        help="Path to model responses JSONL")
    parser.add_argument("--output_path", required=True,
                        help="Path to save evaluation results")
    
    # Judge configuration
    parser.add_argument("--judge_model", default="Qwen/Qwen3-32B-Instruct",
                        help="Judge model name")
    
    # Sampling
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to evaluate (None for all)")
    
    # Logging
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log running averages every N samples")
    
    args = parser.parse_args()
    
    # Load judge model
    judge_model, judge_tokenizer = load_judge_model(args.judge_model)
    
    # Load responses
    responses = load_responses(args.responses_path)
    
    # Limit samples if specified
    if args.max_samples:
        responses = responses[:args.max_samples]
        logger.info(f"Limited to {len(responses)} samples")
    
    # Evaluate
    results = []
    for i, resp in enumerate(tqdm(responses, desc="Evaluating"), 1):
        scores = evaluate_response(
            judge_model,
            judge_tokenizer,
            question=resp["question"],
            model_response=resp["response"],
            ground_truth=resp.get("ground_truth", ""),
            context=resp.get("context"),
        )
        
        results.append({
            "question": resp["question"],
            "response": resp["response"],
            "ground_truth": resp.get("ground_truth", ""),
            "model": resp.get("model", "unknown"),
            "scores": scores,
        })
        
        # Log running averages
        if i % args.log_every == 0:
            mean_correctness = np.mean([r["scores"].get("correctness", 0) for r in results])
            mean_helpfulness = np.mean([r["scores"].get("helpfulness", 0) for r in results])
            mean_coherence = np.mean([r["scores"].get("coherence", 0) for r in results])
            logger.info(
                f"Sample {i}/{len(responses)} - Running avg: "
                f"Correctness={mean_correctness:.2f}, "
                f"Helpfulness={mean_helpfulness:.2f}, "
                f"Coherence={mean_coherence:.2f}"
            )
    
    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Print final summary
    mean_correctness = np.mean([r["scores"].get("correctness", 0) for r in results])
    mean_helpfulness = np.mean([r["scores"].get("helpfulness", 0) for r in results])
    mean_coherence = np.mean([r["scores"].get("coherence", 0) for r in results])
    
    logger.info("=" * 60)
    logger.info("FINAL RESULTS:")
    logger.info(f"  Correctness: {mean_correctness:.2f}/10")
    logger.info(f"  Helpfulness: {mean_helpfulness:.2f}/10")
    logger.info(f"  Coherence:   {mean_coherence:.2f}/10")
    logger.info(f"  Combined:    {(mean_correctness + mean_helpfulness + mean_coherence) / 3:.2f}/10")
    logger.info("=" * 60)
    logger.info(f"Saved {len(results)} results to {args.output_path}")


if __name__ == "__main__":
    main()
