"""
Pairwise Model Comparison Script

Compares two models' responses using an LLM judge and computes win rates.
Evaluates each response separately and compares scores.

Usage:
    python evaluation/compare_models.py \
        --model_a_path "outputs/student_weighted_responses.jsonl" \
        --model_b_path "outputs/student_random_responses.jsonl" \
        --judge_model "Qwen/Qwen3-32B-Instruct" \
        --output_path "results/comparison.json"
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


JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing an AI assistant response.

Question: {question}

Ground Truth (Reference Answer): {ground_truth}

Model Response: {model_response}

Evaluate the response on the following criteria (1-10 scale):

1. Correctness: Is the response factually accurate?
2. Helpfulness: Does the response provide useful information?
3. Coherence: Is the response well-structured and clear?

Special case: If the response is "I don't know" or refuses to answer, assign 5 for all metrics.

Output ONLY a JSON object:
{{"correctness": X, "helpfulness": Y, "coherence": Z}}
Your evaluation:"""


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
    """Load responses indexed by question."""
    responses = {}
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                responses[data["question"]] = data
    return responses


def evaluate_response(model, tokenizer, question: str, response: str, ground_truth: str) -> Dict:
    """Evaluate a single response."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth if ground_truth else "Not provided",
        model_response=response,
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.0,
            do_sample=False,
        )
    
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Parse JSON
    try:
        start = response_text.find('{')
        if start == -1:
            raise ValueError("No JSON found")
        
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
        
        scores = json.loads(response_text[start:end])
    except Exception as e:
        logger.warning(f"Parse error: {e}")
        scores = {"correctness": 0, "helpfulness": 0, "coherence": 0}
    
    return scores


def compute_combined_score(scores: Dict) -> float:
    """Compute combined score from individual scores."""
    return (scores.get("correctness", 0) + scores.get("helpfulness", 0) + scores.get("coherence", 0)) / 3


def determine_winner(score_a: float, score_b: float, tie_threshold: float = 0.5) -> str:
    """Determine winner based on combined scores."""
    diff = score_a - score_b
    if abs(diff) <= tie_threshold:
        return "tie"
    elif diff > 0:
        return "a"
    else:
        return "b"


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
    ax.set_title(f'Win Rate Comparison: {model_a_name} vs {model_b_name}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'win_rate_comparison.png'), dpi=150)
    plt.close()
    
    # 2. Score Distribution Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['correctness', 'helpfulness', 'coherence']
    
    for ax, metric in zip(axes, metrics):
        scores_a = [d["scores_a"].get(metric, 0) for d in results["details"]]
        scores_b = [d["scores_b"].get(metric, 0) for d in results["details"]]
        
        ax.hist(scores_a, bins=10, alpha=0.6, label=model_a_name, color='#3498db', range=(0, 10))
        ax.hist(scores_b, bins=10, alpha=0.6, label=model_b_name, color='#e74c3c', range=(0, 10))
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        ax.set_title(f'{metric.capitalize()} Distribution')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distributions.png'), dpi=150)
    plt.close()
    
    # 3. Head-to-Head Scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    combined_a = [d["combined_a"] for d in results["details"]]
    combined_b = [d["combined_b"] for d in results["details"]]
    
    ax.scatter(combined_a, combined_b, alpha=0.5, s=30)
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.5, label='Tie line')
    ax.set_xlabel(f'{model_a_name} Score')
    ax.set_ylabel(f'{model_b_name} Score')
    ax.set_title('Head-to-Head Score Comparison')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'head_to_head_scatter.png'), dpi=150)
    plt.close()
    
    # 4. Summary Statistics Table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    table_data = [
        ['Metric', model_a_name, model_b_name, 'Difference'],
        ['Correctness', f"{results['mean_correctness_a']:.2f}", f"{results['mean_correctness_b']:.2f}", 
         f"{results['mean_correctness_a'] - results['mean_correctness_b']:+.2f}"],
        ['Helpfulness', f"{results['mean_helpfulness_a']:.2f}", f"{results['mean_helpfulness_b']:.2f}",
         f"{results['mean_helpfulness_a'] - results['mean_helpfulness_b']:+.2f}"],
        ['Coherence', f"{results['mean_coherence_a']:.2f}", f"{results['mean_coherence_b']:.2f}",
         f"{results['mean_coherence_a'] - results['mean_coherence_b']:+.2f}"],
        ['Combined', f"{results['mean_combined_a']:.2f}", f"{results['mean_combined_b']:.2f}",
         f"{results['mean_combined_a'] - results['mean_combined_b']:+.2f}"],
    ]
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', colColours=['#f0f0f0']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    plt.savefig(os.path.join(output_dir, 'summary_statistics.png'), dpi=150)
    plt.close()
    
    logger.info(f"Saved 4 comparison plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare two models using LLM judge")
    
    parser.add_argument("--model_a_path", required=True, help="Path to model A responses JSONL")
    parser.add_argument("--model_b_path", required=True, help="Path to model B responses JSONL")
    parser.add_argument("--model_a_name", default=None, help="Display name for model A")
    parser.add_argument("--model_b_name", default=None, help="Display name for model B")
    parser.add_argument("--judge_model", default="Qwen/Qwen3-32B-Instruct", help="Judge model")
    parser.add_argument("--output_path", required=True, help="Path to save comparison results")
    parser.add_argument("--tie_threshold", type=float, default=0.5, help="Score difference for tie")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to compare")
    
    args = parser.parse_args()
    
    # Set model names
    if args.model_a_name is None:
        args.model_a_name = os.path.basename(args.model_a_path).replace('_responses.jsonl', '')
    if args.model_b_name is None:
        args.model_b_name = os.path.basename(args.model_b_path).replace('_responses.jsonl', '')
    
    # Load responses
    responses_a = load_responses(args.model_a_path)
    responses_b = load_responses(args.model_b_path)
    
    # Find common questions
    common_questions = set(responses_a.keys()) & set(responses_b.keys())
    logger.info(f"Found {len(common_questions)} common questions")
    
    if args.max_samples:
        common_questions = list(common_questions)[:args.max_samples]
    
    # Load judge
    judge_model, judge_tokenizer = load_judge_model(args.judge_model)
    
    # Compare
    details = []
    wins_a, wins_b, ties = 0, 0, 0
    
    for question in tqdm(common_questions, desc="Comparing"):
        resp_a = responses_a[question]
        resp_b = responses_b[question]
        ground_truth = resp_a.get("ground_truth", resp_b.get("ground_truth", ""))
        
        # Evaluate both responses separately
        scores_a = evaluate_response(judge_model, judge_tokenizer, question, resp_a["response"], ground_truth)
        scores_b = evaluate_response(judge_model, judge_tokenizer, question, resp_b["response"], ground_truth)
        
        combined_a = compute_combined_score(scores_a)
        combined_b = compute_combined_score(scores_b)
        
        winner = determine_winner(combined_a, combined_b, args.tie_threshold)
        
        if winner == "a":
            wins_a += 1
        elif winner == "b":
            wins_b += 1
        else:
            ties += 1
        
        details.append({
            "question": question,
            "response_a": resp_a["response"],
            "response_b": resp_b["response"],
            "scores_a": scores_a,
            "scores_b": scores_b,
            "combined_a": combined_a,
            "combined_b": combined_b,
            "winner": winner,
        })
    
    total = len(common_questions)
    
    # Compute statistics
    results = {
        "model_a_name": args.model_a_name,
        "model_b_name": args.model_b_name,
        "model_a_wins": wins_a,
        "model_b_wins": wins_b,
        "ties": ties,
        "total": total,
        "model_a_win_rate": wins_a / total if total > 0 else 0,
        "model_b_win_rate": wins_b / total if total > 0 else 0,
        "tie_rate": ties / total if total > 0 else 0,
        "mean_correctness_a": np.mean([d["scores_a"].get("correctness", 0) for d in details]),
        "mean_correctness_b": np.mean([d["scores_b"].get("correctness", 0) for d in details]),
        "mean_helpfulness_a": np.mean([d["scores_a"].get("helpfulness", 0) for d in details]),
        "mean_helpfulness_b": np.mean([d["scores_b"].get("helpfulness", 0) for d in details]),
        "mean_coherence_a": np.mean([d["scores_a"].get("coherence", 0) for d in details]),
        "mean_coherence_b": np.mean([d["scores_b"].get("coherence", 0) for d in details]),
        "mean_combined_a": np.mean([d["combined_a"] for d in details]),
        "mean_combined_b": np.mean([d["combined_b"] for d in details]),
        "details": details,
    }
    
    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    plot_dir = os.path.join(os.path.dirname(args.output_path), "comparison_plots",
                             f"{args.model_a_name}_vs_{args.model_b_name}")
    create_plots(results, plot_dir, args.model_a_name, args.model_b_name)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("COMPARISON RESULTS:")
    logger.info(f"  {args.model_a_name} wins: {wins_a} ({wins_a/total:.1%})")
    logger.info(f"  {args.model_b_name} wins: {wins_b} ({wins_b/total:.1%})")
    logger.info(f"  Ties: {ties} ({ties/total:.1%})")
    logger.info(f"  Mean combined score {args.model_a_name}: {results['mean_combined_a']:.2f}")
    logger.info(f"  Mean combined score {args.model_b_name}: {results['mean_combined_b']:.2f}")
    logger.info("=" * 60)
    logger.info(f"Saved results to {args.output_path}")
    logger.info(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
