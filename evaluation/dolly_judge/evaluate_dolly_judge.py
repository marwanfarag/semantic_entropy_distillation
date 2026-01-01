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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def build_instruction_key(instruction: str, context: str = "") -> str:
    """
    Build a consistent instruction key that matches the teacher output format.
    
    Teacher outputs combine instruction and context as:
    "{instruction}\n\nInput: {context}"
    
    This function creates the same format for matching.
    """
    if context and context.strip():
        return f"{instruction}\n\nInput: {context}"
    return instruction


def load_dolly_ground_truth(dolly_path: str) -> Dict[str, Dict]:
    """Load Dolly dataset with ground truth responses."""
    logger.info(f"Loading Dolly ground truth from: {dolly_path}")
    
    # Load from the databricks dolly dataset
    from datasets import load_dataset
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    # Build lookup by instruction (using combined format to match teacher outputs)
    ground_truth = {}
    for example in dataset:
        instruction = example["instruction"]
        context = example.get("context", "")
        
        # Use combined key format to match teacher output format
        key = build_instruction_key(instruction, context)
        
        ground_truth[key] = {
            "instruction": instruction,
            "input": context,
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


def plot_evaluation_results(results: List[Dict], output_dir: str):
    """Generate and save visualization plots for evaluation results."""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    
    # Extract data
    correctness = []
    helpfulness = []
    coherence = []
    teacher_scores = []
    
    for r in results:
        scores = r.get("scores", {})
        correctness.append(scores.get("correctness", 0))
        helpfulness.append(scores.get("helpfulness", 0))
        coherence.append(scores.get("coherence", 0))
        teacher_scores.append(r.get("teacher_score", 0))
    
    correctness = np.array(correctness)
    helpfulness = np.array(helpfulness)
    coherence = np.array(coherence)
    teacher_scores = np.array(teacher_scores)
    
    # 1. Score Distribution Box Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    data_to_plot = [correctness, helpfulness, coherence]
    bp = ax.boxplot(data_to_plot, labels=['Correctness', 'Helpfulness', 'Coherence'], patch_artist=True)
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Score (1-10)')
    ax.set_title('Judge Score Distributions by Criterion')
    ax.set_ylim(0, 11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distribution_boxplot.png'), dpi=150)
    plt.close()
    
    # 2. Score Distribution Histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [('Correctness', correctness, '#3498db'),
               ('Helpfulness', helpfulness, '#2ecc71'),
               ('Coherence', coherence, '#9b59b6')]
    
    for ax, (name, data, color) in zip(axes, metrics):
        ax.hist(data, bins=10, range=(0.5, 10.5), color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        ax.set_title(f'{name} Distribution')
        ax.axvline(np.mean(data), color='red', linestyle='--', label=f'Mean: {np.mean(data):.2f}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distribution_histograms.png'), dpi=150)
    plt.close()
    
    # 3. Scores vs Teacher Uncertainty Scatter
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    combined_score = (correctness + helpfulness + coherence) / 3
    
    for ax, (name, data, color) in zip(axes, metrics):
        ax.scatter(teacher_scores, data, alpha=0.5, c=color, s=30)
        # Add trend line
        z = np.polyfit(teacher_scores, data, 1)
        p = np.poly1d(z)
        x_line = np.linspace(teacher_scores.min(), teacher_scores.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend')
        ax.set_xlabel('Teacher Uncertainty Score')
        ax.set_ylabel(f'{name} Score')
        ax.set_title(f'{name} vs Teacher Uncertainty')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scores_vs_uncertainty.png'), dpi=150)
    plt.close()
    
    # 4. Combined Score vs Uncertainty
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(teacher_scores, combined_score, alpha=0.6, c=combined_score, cmap='RdYlGn', s=50)
    z = np.polyfit(teacher_scores, combined_score, 1)
    p = np.poly1d(z)
    x_line = np.linspace(teacher_scores.min(), teacher_scores.max(), 100)
    ax.plot(x_line, p(x_line), 'b--', linewidth=2, label=f'Trend (slope: {z[0]:.3f})')
    ax.set_xlabel('Teacher Uncertainty Score')
    ax.set_ylabel('Average Judge Score')
    ax.set_title('Combined Score vs Teacher Uncertainty')
    ax.legend()
    plt.colorbar(scatter, label='Average Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_score_vs_uncertainty.png'), dpi=150)
    plt.close()
    
    # 5. Uncertainty Band Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bands = {'Low (< 0.3)': [], 'Medium (0.3-0.6)': [], 'High (> 0.6)': []}
    for i, score in enumerate(teacher_scores):
        if score < 0.3:
            bands['Low (< 0.3)'].append(i)
        elif score < 0.6:
            bands['Medium (0.3-0.6)'].append(i)
        else:
            bands['High (> 0.6)'].append(i)
    
    x = np.arange(3)
    width = 0.25
    band_names = list(bands.keys())
    
    for j, (name, data, color) in enumerate(metrics):
        means = [np.mean(data[bands[b]]) if bands[b] else 0 for b in band_names]
        stds = [np.std(data[bands[b]]) if bands[b] else 0 for b in band_names]
        ax.bar(x + j * width, means, width, label=name, color=color, alpha=0.7, yerr=stds, capsize=3)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(band_names)
    ax.set_ylabel('Average Score')
    ax.set_title('Average Scores by Teacher Uncertainty Band')
    ax.legend()
    ax.set_ylim(0, 11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scores_by_uncertainty_band.png'), dpi=150)
    plt.close()
    
    # 6. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = np.corrcoef([correctness, helpfulness, coherence, teacher_scores])
    labels = ['Correctness', 'Helpfulness', 'Coherence', 'Teacher Uncertainty']
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                xticklabels=labels, yticklabels=labels, ax=ax, 
                vmin=-1, vmax=1, center=0)
    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    
    # 7. Radar/Spider Chart for Overall Performance
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    categories = ['Correctness', 'Helpfulness', 'Coherence']
    values = [np.mean(correctness), np.mean(helpfulness), np.mean(coherence)]
    values += values[:1]  # Complete the loop
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax.fill(angles, values, color='#3498db', alpha=0.25)
    ax.plot(angles, values, color='#3498db', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 10)
    ax.set_title('Overall Performance Radar Chart', y=1.08)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_chart.png'), dpi=150)
    plt.close()
    
    # 8. Summary Statistics
    fig, ax = plt.subplots(figsize=(10, 6))
    summary_data = {
        'Metric': ['Correctness', 'Helpfulness', 'Coherence', 'Combined'],
        'Mean': [np.mean(correctness), np.mean(helpfulness), np.mean(coherence), np.mean(combined_score)],
        'Std': [np.std(correctness), np.std(helpfulness), np.std(coherence), np.std(combined_score)],
        'Min': [np.min(correctness), np.min(helpfulness), np.min(coherence), np.min(combined_score)],
        'Max': [np.max(correctness), np.max(helpfulness), np.max(coherence), np.max(combined_score)],
    }
    
    ax.axis('off')
    table = ax.table(
        cellText=[[m, f'{mean:.2f}', f'{std:.2f}', f'{mn:.1f}', f'{mx:.1f}'] 
                  for m, mean, std, mn, mx in zip(summary_data['Metric'], summary_data['Mean'], 
                                                    summary_data['Std'], summary_data['Min'], summary_data['Max'])],
        colLabels=['Metric', 'Mean', 'Std Dev', 'Min', 'Max'],
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * 5,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    ax.set_title('Summary Statistics', fontsize=14, fontweight='bold', y=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_statistics.png'), dpi=150)
    plt.close()
    
    logger.info(f"Saved 8 evaluation plots to {output_dir}")


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
    
    # Generate and save plots
    plot_dir = os.path.dirname(args.output_path)
    plot_evaluation_results(results, plot_dir)


if __name__ == "__main__":
    main()
