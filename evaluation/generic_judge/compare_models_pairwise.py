"""
Pairwise Side-by-Side Model Comparison Script

Compares two models by showing both responses to the judge in the same prompt.
Runs twice with swapped positions to account for positional bias.

Usage:
    python evaluation/generic_judge/compare_models_pairwise.py \
        --model_a_path "outputs/student_weighted.jsonl" \
        --model_b_path "outputs/student_random.jsonl" \
        --judge_model "Qwen/Qwen3-32B-Instruct" \
        --output_path "results/pairwise_comparison.json"
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PAIRWISE_PROMPT_TEMPLATE = """You are an expert evaluator comparing two AI responses for factual accuracy.

Question: {question}

Ground Truth (Reference Answer): {ground_truth}

Response A: {response_a}

Response B: {response_b}

Which response aligns better with the ground truth in terms of factual accuracy?

Rules:
- If Response A is more factually accurate, output: {{"winner": "A"}}
- If Response B is more factually accurate, output: {{"winner": "B"}}
- If both are equally accurate (or equally wrong), output: {{"winner": "tie"}}
- If the ground truth says the answer is unknown, prefer responses that admit uncertainty.
- Do not be biased by length.

YOUR OUTPUT MUST BE EXACTLY ONE OF THESE THREE OPTIONS:
{{"winner": "A"}}
{{"winner": "B"}}
{{"winner": "tie"}}

Your answer:"""


def load_judge_model(model_name: str):
    """Load the judge model."""
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


def load_responses(path: str) -> Dict[str, Dict]:
    """Load responses indexed by question. Auto-detects the question field."""
    responses = {}
    question_field = None
    
    # Common question field names
    QUESTION_FIELDS = ["question", "problem", "prompt", "input", "instruction", "query"]
    
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                
                # Auto-detect question field on first line
                if question_field is None:
                    for field in QUESTION_FIELDS:
                        if field in data:
                            question_field = field
                            logger.info(f"Auto-detected question field: '{question_field}'")
                            break
                    if question_field is None:
                        available = list(data.keys())
                        raise KeyError(f"Could not find question field. Available fields: {available}")
                
                responses[data[question_field]] = data
                # Store the question field name for later use
                data["_question_field"] = question_field
    
    return responses


def evaluate_pairwise(
    model, 
    tokenizer, 
    question: str, 
    response_a: str, 
    response_b: str, 
    ground_truth: str
) -> str:
    """Evaluate which response is better. Returns 'A', 'B', or 'tie'."""
    prompt = PAIRWISE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth if ground_truth else "Not provided",
        response_a=response_a,
        response_b=response_b,
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.0,
            do_sample=False,
        )
    
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Parse response - try multiple methods
    winner = None
    
    # Method 1: Try to extract JSON
    try:
        start = response_text.find('{')
        if start != -1:
            brace_count = 0
            end = start
            for i in range(start, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            result = json.loads(response_text[start:end])
            # Try different possible keys
            winner_val = result.get("winner") or result.get("Winner") or result.get("choice") or result.get("answer")
            if winner_val:
                winner = str(winner_val).upper().strip()
    except:
        pass
    
    # Method 2: Look for patterns in text if JSON failed
    if winner not in ["A", "B", "TIE"]:
        response_upper = response_text.upper()
        # Look for explicit statements
        if '"WINNER": "A"' in response_upper or '"WINNER":"A"' in response_upper or "RESPONSE A" in response_upper and "BETTER" in response_upper:
            winner = "A"
        elif '"WINNER": "B"' in response_upper or '"WINNER":"B"' in response_upper or "RESPONSE B" in response_upper and "BETTER" in response_upper:
            winner = "B"
        elif '"WINNER": "TIE"' in response_upper or '"WINNER":"TIE"' in response_upper or "TIE" in response_upper or "NEITHER" in response_upper or "BOTH" in response_upper:
            winner = "TIE"
    
    # Default to TIE if we couldn't determine
    if winner not in ["A", "B", "TIE"]:
        logger.warning(f"Could not parse winner, defaulting to TIE. Response: {response_text[:150]}")
        winner = "TIE"
    
    return winner


def combine_results(pass1: str, pass2: str, model_a_name: str, model_b_name: str) -> Tuple[str, str]:
    """
    Combine results from two passes with swapped positions.
    
    Pass 1: model_a = A, model_b = B
    Pass 2: model_a = B, model_b = A (swapped)
    
    Returns: (winner_name, consistency)
    """
    # Normalize pass2 result (since positions are swapped)
    # If pass2 says "A" wins, that means model_b wins (since model_b was in position A)
    pass2_normalized = {"A": "B", "B": "A", "TIE": "TIE"}[pass2]
    
    # Combine results
    if pass1 == pass2_normalized:
        # Consistent result
        if pass1 == "A":
            return model_a_name, "consistent"
        elif pass1 == "B":
            return model_b_name, "consistent"
        else:
            return "tie", "consistent"
    else:
        # Check for win + tie combinations
        results = {pass1, pass2_normalized}
        
        if results == {"A", "TIE"}:
            return model_a_name, "weak"  # A wins weakly
        elif results == {"B", "TIE"}:
            return model_b_name, "weak"  # B wins weakly
        else:
            # Disagreement (A vs B)
            return "tie", "disagreement"


def create_plots(results: Dict, output_dir: str, model_a_name: str, model_b_name: str):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    # 1. Win Rate Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = [f'{model_a_name} Wins', 'Ties', f'{model_b_name} Wins']
    values = [results["model_a_win_rate"], results["tie_rate"], results["model_b_win_rate"]]
    colors = ['#2ecc71', '#95a5a6', '#e74c3c']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_title(f'Pairwise Comparison: {model_a_name} vs {model_b_name}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pairwise_win_rates.png'), dpi=150)
    plt.close()
    
    # 2. Consistency Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    consistency_counts = {
        'Consistent': results["consistent_count"],
        'Weak (win+tie)': results["weak_count"],
        'Disagreement': results["disagreement_count"],
    }
    
    bars = ax.bar(consistency_counts.keys(), consistency_counts.values(), 
                  color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Position Consistency Analysis', fontsize=14, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'consistency_analysis.png'), dpi=150)
    plt.close()
    
    # 3. Positional Bias Check
    fig, ax = plt.subplots(figsize=(10, 6))
    pass1_a = sum(1 for d in results["details"] if d["pass1_winner"] == "A")
    pass1_b = sum(1 for d in results["details"] if d["pass1_winner"] == "B")
    pass2_a = sum(1 for d in results["details"] if d["pass2_winner"] == "A")
    pass2_b = sum(1 for d in results["details"] if d["pass2_winner"] == "B")
    
    x = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x - width/2, [pass1_a, pass2_a], width, label='Position A wins', color='#3498db')
    bars2 = ax.bar(x + width/2, [pass1_b, pass2_b], width, label='Position B wins', color='#e74c3c')
    
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Positional Bias Check (should be similar if no bias)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Pass 1\n(A=weighted, B=random)', 'Pass 2\n(A=random, B=weighted)'])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'positional_bias_check.png'), dpi=150)
    plt.close()
    
    logger.info(f"Saved 3 comparison plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pairwise side-by-side model comparison")
    
    parser.add_argument("--model_a_path", required=True, help="Path to model A responses JSONL")
    parser.add_argument("--model_b_path", required=True, help="Path to model B responses JSONL")
    parser.add_argument("--model_a_name", default=None, help="Display name for model A")
    parser.add_argument("--model_b_name", default=None, help="Display name for model B")
    parser.add_argument("--judge_model", default="Qwen/Qwen3-32B-Instruct", help="Judge model")
    parser.add_argument("--output_path", required=True, help="Path to save comparison results")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to compare")
    parser.add_argument("--log_every", type=int, default=10, help="Log running stats every N samples")
    
    # Dataset arguments for loading ground truth if not in JSONL
    parser.add_argument("--dataset", default=None, 
                        help="HuggingFace dataset to load ground truth from (e.g., 'basicv8vc/SimpleQA')")
    parser.add_argument("--dataset_config", default=None, help="Dataset config name")
    parser.add_argument("--dataset_split", default="test", help="Dataset split")
    parser.add_argument("--question_field", default=None, help="Question field in dataset")
    parser.add_argument("--ground_truth_field", default=None, help="Ground truth field in dataset")
    
    args = parser.parse_args()
    
    # Set model names
    if args.model_a_name is None:
        args.model_a_name = os.path.basename(args.model_a_path).replace('.jsonl', '')
    if args.model_b_name is None:
        args.model_b_name = os.path.basename(args.model_b_path).replace('.jsonl', '')
    
    # Load responses
    responses_a = load_responses(args.model_a_path)
    responses_b = load_responses(args.model_b_path)
    
    # Find common questions (sorted for deterministic ordering)
    common_questions = sorted(set(responses_a.keys()) & set(responses_b.keys()))
    logger.info(f"Found {len(common_questions)} common questions")
    
    # Check if ground truth is missing and load from dataset if provided
    ground_truth_map = {}
    sample_resp = next(iter(responses_a.values()))
    has_ground_truth = "ground_truth" in sample_resp and sample_resp["ground_truth"]
    
    if not has_ground_truth:
        if args.dataset:
            logger.info(f"Ground truth not in JSONL, loading from dataset: {args.dataset}")
            from datasets import load_dataset
            
            # Known dataset configurations
            KNOWN_DATASETS = {
                "truthfulqa/truthful_qa": {"config": "generation", "split": "validation", "question_field": "question", "ground_truth_field": "best_answer"},
                "basicv8vc/SimpleQA": {"config": None, "split": "test", "question_field": "problem", "ground_truth_field": "answer"},
            }
            
            # Apply defaults for known datasets
            if args.dataset in KNOWN_DATASETS:
                known = KNOWN_DATASETS[args.dataset]
                if args.dataset_config is None:
                    args.dataset_config = known["config"]
                if args.dataset_split == "test" and known["split"] != "test":
                    args.dataset_split = known["split"]
                if args.question_field is None:
                    args.question_field = known["question_field"]
                if args.ground_truth_field is None:
                    args.ground_truth_field = known["ground_truth_field"]
            
            # Load dataset
            if args.dataset_config:
                dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
            else:
                dataset = load_dataset(args.dataset, split=args.dataset_split)
            
            logger.info(f"Loaded {len(dataset)} examples from dataset")
            logger.info(f"Using question_field='{args.question_field}', ground_truth_field='{args.ground_truth_field}'")
            
            # Build ground truth map
            for example in dataset:
                q = example[args.question_field]
                gt = example[args.ground_truth_field]
                ground_truth_map[q] = gt
            
            logger.info(f"Built ground truth map with {len(ground_truth_map)} entries")
        else:
            logger.warning("Ground truth not found in JSONL and no --dataset provided. Comparison may be less accurate.")
    
    if args.max_samples:
        common_questions = common_questions[:args.max_samples]
    
    # Load judge
    judge_model, judge_tokenizer = load_judge_model(args.judge_model)
    
    # Compare
    details = []
    wins_a, wins_b, ties = 0, 0, 0
    consistent_count, weak_count, disagreement_count = 0, 0, 0
    
    for i, question in enumerate(tqdm(common_questions, desc="Comparing"), 1):
        resp_a = responses_a[question]
        resp_b = responses_b[question]
        
        # Get ground truth: first from JSONL, then from dataset map
        ground_truth = resp_a.get("ground_truth") or resp_b.get("ground_truth") or ground_truth_map.get(question, "")
        
        # Pass 1: A=model_a, B=model_b
        pass1_winner = evaluate_pairwise(
            judge_model, judge_tokenizer,
            question, resp_a["response"], resp_b["response"], ground_truth
        )
        
        # Pass 2: A=model_b, B=model_a (swapped positions)
        pass2_winner = evaluate_pairwise(
            judge_model, judge_tokenizer,
            question, resp_b["response"], resp_a["response"], ground_truth
        )
        
        # Combine results
        final_winner, consistency = combine_results(
            pass1_winner, pass2_winner, args.model_a_name, args.model_b_name
        )
        
        # Update counters
        if final_winner == args.model_a_name:
            wins_a += 1
        elif final_winner == args.model_b_name:
            wins_b += 1
        else:
            ties += 1
        
        if consistency == "consistent":
            consistent_count += 1
        elif consistency == "weak":
            weak_count += 1
        else:
            disagreement_count += 1
        
        details.append({
            "question": question,
            "response_a": resp_a["response"],
            "response_b": resp_b["response"],
            "ground_truth": ground_truth,
            "pass1_winner": pass1_winner,
            "pass2_winner": pass2_winner,
            "final_winner": final_winner,
            "consistency": consistency,
        })
        
        # Detailed logging - print full question, responses and evaluations
        print(f"\n{'='*80}")
        print(f"[{i}/{len(common_questions)}]")
        print(f"\nQUESTION: {question}")
        print(f"\nGROUND TRUTH: {ground_truth}")
        print(f"\n{args.model_a_name} RESPONSE:\n{resp_a['response']}")
        print(f"\n{args.model_b_name} RESPONSE:\n{resp_b['response']}")
        print(f"\nEVALUATION:")
        print(f"  Pass 1 (A={args.model_a_name}, B={args.model_b_name}): Winner = {pass1_winner}")
        print(f"  Pass 2 (A={args.model_b_name}, B={args.model_a_name}): Winner = {pass2_winner}")
        print(f"  FINAL: {final_winner} ({consistency})")
        
        question_snippet = question[:60] + "..." if len(question) > 60 else question
        logger.info(
            f"[{i}/{len(common_questions)}] Q: \"{question_snippet}\" | "
            f"P1={pass1_winner}, P2={pass2_winner} => {final_winner} ({consistency})"
        )
        
        # Running stats
        if i % args.log_every == 0:
            total = i
            logger.info(
                f"--- Running stats after {i} samples ---\n"
                f"  {args.model_a_name}: {wins_a} ({wins_a/total:.1%})\n"
                f"  {args.model_b_name}: {wins_b} ({wins_b/total:.1%})\n"
                f"  Ties: {ties} ({ties/total:.1%})\n"
                f"  Consistency: {consistent_count} consistent, {weak_count} weak, {disagreement_count} disagreements"
            )
    
    total = len(common_questions)
    
    # Compute results
    results = {
        "model_a_name": args.model_a_name,
        "model_b_name": args.model_b_name,
        "total": total,
        "model_a_wins": wins_a,
        "model_b_wins": wins_b,
        "ties": ties,
        "model_a_win_rate": wins_a / total if total > 0 else 0,
        "model_b_win_rate": wins_b / total if total > 0 else 0,
        "tie_rate": ties / total if total > 0 else 0,
        "consistent_count": consistent_count,
        "weak_count": weak_count,
        "disagreement_count": disagreement_count,
        "details": details,
    }
    
    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    plot_dir = os.path.join(os.path.dirname(args.output_path), "pairwise_plots",
                             f"{args.model_a_name}_vs_{args.model_b_name}")
    create_plots(results, plot_dir, args.model_a_name, args.model_b_name)
    
    # Print summary
    logger.info("=" * 70)
    logger.info("FINAL PAIRWISE COMPARISON RESULTS:")
    logger.info(f"  Total samples: {total}")
    logger.info(f"  {args.model_a_name}: {wins_a} wins ({results['model_a_win_rate']:.1%})")
    logger.info(f"  {args.model_b_name}: {wins_b} wins ({results['model_b_win_rate']:.1%})")
    logger.info(f"  Ties: {ties} ({results['tie_rate']:.1%})")
    logger.info("")
    logger.info("  Position Consistency:")
    logger.info(f"    Consistent: {consistent_count} ({consistent_count/total:.1%})")
    logger.info(f"    Weak (win+tie): {weak_count} ({weak_count/total:.1%})")
    logger.info(f"    Disagreement: {disagreement_count} ({disagreement_count/total:.1%})")
    logger.info("=" * 70)
    logger.info(f"Saved results to {args.output_path}")
    logger.info(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
