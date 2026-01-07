"""
Pairwise Model Comparison Script

Compares two models' responses using an LLM judge and computes win rates.
Evaluates each response separately and computes win rates for each metric.

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
from typing import Dict, List
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

1. Correctness: Is the response factually accurate? If the ground truth says that the answer is unknown or refuses to answer and the response provides an answer, assign a low score.
2. Helpfulness: Does the response provide useful information?
3. Coherence: Is the response well-structured and clear?

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
        scores = {"correctness": 5, "helpfulness": 5, "coherence": 5}
    
    return scores


def determine_metric_winner(score_a: float, score_b: float, tie_threshold: float = 0.5) -> str:
    """Determine winner for a single metric."""
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
    metrics = ['correctness', 'helpfulness', 'coherence']
    
    # 1. Average Scores Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    means_a = [results[f'mean_{m}_a'] for m in metrics]
    means_b = [results[f'mean_{m}_b'] for m in metrics]
    
    bars1 = ax.bar(x - width/2, means_a, width, label=model_a_name, color='#3498db')
    bars2 = ax.bar(x + width/2, means_b, width, label=model_b_name, color='#e74c3c')
    
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_title(f'Average Scores: {model_a_name} vs {model_b_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 10)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_scores.png'), dpi=150)
    plt.close()
    
    # 2. Per-Metric Win Rate Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.25
    
    a_wins = [results[f"{m}_a_win_rate"] for m in metrics]
    ties = [results[f"{m}_tie_rate"] for m in metrics]
    b_wins = [results[f"{m}_b_win_rate"] for m in metrics]
    
    bars1 = ax.bar(x - width, a_wins, width, label=f'{model_a_name} Wins', color='#2ecc71')
    bars2 = ax.bar(x, ties, width, label='Ties', color='#95a5a6')
    bars3 = ax.bar(x + width, b_wins, width, label=f'{model_b_name} Wins', color='#e74c3c')
    
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.set_title(f'Per-Metric Win Rates: {model_a_name} vs {model_b_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{height:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_metric_win_rates.png'), dpi=150)
    plt.close()
    
    # 3. Score Difference Histogram (per metric)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, metric in zip(axes, metrics):
        scores_a = np.array([d["scores_a"].get(metric, 0) for d in results["details"]])
        scores_b = np.array([d["scores_b"].get(metric, 0) for d in results["details"]])
        diff = scores_a - scores_b  # Positive = A wins
        
        ax.hist(diff, bins=21, range=(-10, 10), color='#9b59b6', edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Tie line')
        ax.set_xlabel(f'Score Difference ({model_a_name} - {model_b_name})')
        ax.set_ylabel('Count')
        ax.set_title(f'{metric.capitalize()} Score Difference')
        
        # Add stats
        mean_diff = np.mean(diff)
        ax.axvline(x=mean_diff, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.2f}')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_difference_histogram.png'), dpi=150)
    plt.close()
    
    # 4. Score Category Analysis (High/Low/Middle combinations)
    # Thresholds: Low <= 3, Middle 4-6, High >= 7
    def categorize_score(score):
        if score <= 3:
            return 'low'
        elif score >= 7:
            return 'high'
        else:
            return 'middle'
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, metric in zip(axes, metrics):
        categories = {
            'A_high_B_low': 0,
            'A_low_B_high': 0,
            'Both_high': 0,
            'Both_low': 0,
            'Both_middle': 0,
            'Other': 0,
        }
        
        for d in results["details"]:
            score_a = d["scores_a"].get(metric, 0)
            score_b = d["scores_b"].get(metric, 0)
            cat_a = categorize_score(score_a)
            cat_b = categorize_score(score_b)
            
            if cat_a == 'high' and cat_b == 'low':
                categories['A_high_B_low'] += 1
            elif cat_a == 'low' and cat_b == 'high':
                categories['A_low_B_high'] += 1
            elif cat_a == 'high' and cat_b == 'high':
                categories['Both_high'] += 1
            elif cat_a == 'low' and cat_b == 'low':
                categories['Both_low'] += 1
            elif cat_a == 'middle' and cat_b == 'middle':
                categories['Both_middle'] += 1
            else:
                categories['Other'] += 1
        
        labels = [
            f'{model_a_name} High,\n{model_b_name} Low',
            f'{model_a_name} Low,\n{model_b_name} High',
            'Both High\n(≥7)',
            'Both Low\n(≤3)',
            'Both Middle\n(4-6)',
            'Other\nCombinations'
        ]
        values = list(categories.values())
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#e67e22', '#95a5a6', '#bdc3c7']
        
        bars = ax.bar(range(len(labels)), values, color=colors, edgecolor='black')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, rotation=0)
        ax.set_ylabel('Count')
        ax.set_title(f'{metric.capitalize()}: Score Category Analysis', fontsize=12, fontweight='bold')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                       f'{height}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_category_analysis.png'), dpi=150)
    plt.close()
    
    # 5. Score Distribution Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
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
    
    # 6. Summary Statistics Table
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    table_data = [
        ['Metric', f'{model_a_name} Mean', f'{model_b_name} Mean', f'{model_a_name} Wins', 'Ties', f'{model_b_name} Wins'],
        ['Correctness', f"{results['mean_correctness_a']:.2f}", f"{results['mean_correctness_b']:.2f}",
         f"{results['correctness_a_win_rate']:.1%}", f"{results['correctness_tie_rate']:.1%}", f"{results['correctness_b_win_rate']:.1%}"],
        ['Helpfulness', f"{results['mean_helpfulness_a']:.2f}", f"{results['mean_helpfulness_b']:.2f}",
         f"{results['helpfulness_a_win_rate']:.1%}", f"{results['helpfulness_tie_rate']:.1%}", f"{results['helpfulness_b_win_rate']:.1%}"],
        ['Coherence', f"{results['mean_coherence_a']:.2f}", f"{results['mean_coherence_b']:.2f}",
         f"{results['coherence_a_win_rate']:.1%}", f"{results['coherence_tie_rate']:.1%}", f"{results['coherence_b_win_rate']:.1%}"],
    ]
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', colColours=['#f0f0f0']*6)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    plt.savefig(os.path.join(output_dir, 'summary_statistics.png'), dpi=150)
    plt.close()
    
    logger.info(f"Saved 6 comparison plots to {output_dir}")


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
    parser.add_argument("--log_every", type=int, default=10, help="Log running stats every N samples")
    
    args = parser.parse_args()
    
    # Set model names
    if args.model_a_name is None:
        args.model_a_name = os.path.basename(args.model_a_path).replace('_responses.jsonl', '').replace('.jsonl', '')
    if args.model_b_name is None:
        args.model_b_name = os.path.basename(args.model_b_path).replace('_responses.jsonl', '').replace('.jsonl', '')
    
    # Load responses
    responses_a = load_responses(args.model_a_path)
    responses_b = load_responses(args.model_b_path)
    
    # Find common questions (sorted for deterministic ordering)
    common_questions = sorted(set(responses_a.keys()) & set(responses_b.keys()))
    logger.info(f"Found {len(common_questions)} common questions")
    
    if args.max_samples:
        common_questions = common_questions[:args.max_samples]
    
    # Load judge
    judge_model, judge_tokenizer = load_judge_model(args.judge_model)
    
    # Initialize per-metric win counters
    wins = {
        "correctness": {"a": 0, "b": 0, "tie": 0},
        "helpfulness": {"a": 0, "b": 0, "tie": 0},
        "coherence": {"a": 0, "b": 0, "tie": 0},
    }
    
    # Compare
    details = []
    
    for i, question in enumerate(tqdm(common_questions, desc="Comparing"), 1):
        resp_a = responses_a[question]
        resp_b = responses_b[question]
        ground_truth = resp_a.get("ground_truth", resp_b.get("ground_truth", ""))
        
        # Evaluate both responses separately
        scores_a = evaluate_response(judge_model, judge_tokenizer, question, resp_a["response"], ground_truth)
        scores_b = evaluate_response(judge_model, judge_tokenizer, question, resp_b["response"], ground_truth)
        
        # Determine winner for each metric
        metric_winners = {}
        for metric in ["correctness", "helpfulness", "coherence"]:
            winner = determine_metric_winner(
                scores_a.get(metric, 0), 
                scores_b.get(metric, 0), 
                args.tie_threshold
            )
            metric_winners[metric] = winner
            wins[metric][winner] += 1
        
        details.append({
            "question": question,
            "response_a": resp_a["response"],
            "response_b": resp_b["response"],
            "scores_a": scores_a,
            "scores_b": scores_b,
            "winners": metric_winners,
        })
        
        # Log each comparison with question and scores
        question_snippet = question[:60] + "..." if len(question) > 60 else question
        # print the question, responses and evaluation json
        print(question) 
        print(f"{args.model_a_name}: {resp_a['response']}")
        print(f"{args.model_b_name}: {resp_b['response']}")
        print(f"{args.model_a_name}: {scores_a}")
        print(f"{args.model_b_name}: {scores_b}")
        # print the evaluation json
        print(metric_winners)
        logger.info(
            f"[{i}/{len(common_questions)}] Q: \"{question_snippet}\"\n"
            f"  {args.model_a_name}: C={scores_a.get('correctness', 0)}, H={scores_a.get('helpfulness', 0)}, Co={scores_a.get('coherence', 0)}\n"
            f"  {args.model_b_name}: C={scores_b.get('correctness', 0)}, H={scores_b.get('helpfulness', 0)}, Co={scores_b.get('coherence', 0)}\n"
            f"  Winners: C={metric_winners['correctness']}, H={metric_winners['helpfulness']}, Co={metric_winners['coherence']}"
        )
        
        # Log running stats every N samples
        if i % args.log_every == 0:
            total = i
            logger.info(
                f"--- Running stats after {i} samples ---\n"
                f"  Correctness: {args.model_a_name}={wins['correctness']['a']/total:.1%}, Tie={wins['correctness']['tie']/total:.1%}, {args.model_b_name}={wins['correctness']['b']/total:.1%}\n"
                f"  Helpfulness: {args.model_a_name}={wins['helpfulness']['a']/total:.1%}, Tie={wins['helpfulness']['tie']/total:.1%}, {args.model_b_name}={wins['helpfulness']['b']/total:.1%}\n"
                f"  Coherence:   {args.model_a_name}={wins['coherence']['a']/total:.1%}, Tie={wins['coherence']['tie']/total:.1%}, {args.model_b_name}={wins['coherence']['b']/total:.1%}"
            )
    
    total = len(common_questions)
    
    # Compute results
    results = {
        "model_a_name": args.model_a_name,
        "model_b_name": args.model_b_name,
        "total": total,
        # Per-metric win rates
        "correctness_a_wins": wins["correctness"]["a"],
        "correctness_b_wins": wins["correctness"]["b"],
        "correctness_ties": wins["correctness"]["tie"],
        "correctness_a_win_rate": wins["correctness"]["a"] / total if total > 0 else 0,
        "correctness_b_win_rate": wins["correctness"]["b"] / total if total > 0 else 0,
        "correctness_tie_rate": wins["correctness"]["tie"] / total if total > 0 else 0,
        
        "helpfulness_a_wins": wins["helpfulness"]["a"],
        "helpfulness_b_wins": wins["helpfulness"]["b"],
        "helpfulness_ties": wins["helpfulness"]["tie"],
        "helpfulness_a_win_rate": wins["helpfulness"]["a"] / total if total > 0 else 0,
        "helpfulness_b_win_rate": wins["helpfulness"]["b"] / total if total > 0 else 0,
        "helpfulness_tie_rate": wins["helpfulness"]["tie"] / total if total > 0 else 0,
        
        "coherence_a_wins": wins["coherence"]["a"],
        "coherence_b_wins": wins["coherence"]["b"],
        "coherence_ties": wins["coherence"]["tie"],
        "coherence_a_win_rate": wins["coherence"]["a"] / total if total > 0 else 0,
        "coherence_b_win_rate": wins["coherence"]["b"] / total if total > 0 else 0,
        "coherence_tie_rate": wins["coherence"]["tie"] / total if total > 0 else 0,
        
        # Mean scores
        "mean_correctness_a": np.mean([d["scores_a"].get("correctness", 0) for d in details]),
        "mean_correctness_b": np.mean([d["scores_b"].get("correctness", 0) for d in details]),
        "mean_helpfulness_a": np.mean([d["scores_a"].get("helpfulness", 0) for d in details]),
        "mean_helpfulness_b": np.mean([d["scores_b"].get("helpfulness", 0) for d in details]),
        "mean_coherence_a": np.mean([d["scores_a"].get("coherence", 0) for d in details]),
        "mean_coherence_b": np.mean([d["scores_b"].get("coherence", 0) for d in details]),
        
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
    logger.info("=" * 70)
    logger.info("FINAL COMPARISON RESULTS:")
    logger.info(f"  Total samples: {total}")
    logger.info("")
    logger.info(f"  CORRECTNESS:")
    logger.info(f"    {args.model_a_name}: {wins['correctness']['a']} wins ({results['correctness_a_win_rate']:.1%})")
    logger.info(f"    {args.model_b_name}: {wins['correctness']['b']} wins ({results['correctness_b_win_rate']:.1%})")
    logger.info(f"    Ties: {wins['correctness']['tie']} ({results['correctness_tie_rate']:.1%})")
    logger.info("")
    logger.info(f"  HELPFULNESS:")
    logger.info(f"    {args.model_a_name}: {wins['helpfulness']['a']} wins ({results['helpfulness_a_win_rate']:.1%})")
    logger.info(f"    {args.model_b_name}: {wins['helpfulness']['b']} wins ({results['helpfulness_b_win_rate']:.1%})")
    logger.info(f"    Ties: {wins['helpfulness']['tie']} ({results['helpfulness_tie_rate']:.1%})")
    logger.info("")
    logger.info(f"  COHERENCE:")
    logger.info(f"    {args.model_a_name}: {wins['coherence']['a']} wins ({results['coherence_a_win_rate']:.1%})")
    logger.info(f"    {args.model_b_name}: {wins['coherence']['b']} wins ({results['coherence_b_win_rate']:.1%})")
    logger.info(f"    Ties: {wins['coherence']['tie']} ({results['coherence_tie_rate']:.1%})")
    logger.info("=" * 70)
    logger.info(f"Saved results to {args.output_path}")
    logger.info(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
