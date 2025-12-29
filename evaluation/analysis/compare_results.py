"""
Compare and visualize evaluation results across models.
"""

import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_dolly_scores(results_dir: Path):
    """Load and aggregate Dolly judge scores."""
    scores = {}
    
    for score_file in results_dir.glob("*_scores.jsonl"):
        model_name = score_file.stem.replace("_scores", "")
        results = []
        
        with open(score_file, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        
        # Stratify by uncertainty
        low, medium, high = [], [], []
        for r in results:
            score = r["teacher_score"]
            correctness = r["scores"]["correctness"]
            
            if score < 0.3:
                low.append(correctness)
            elif score < 0.6:
                medium.append(correctness)
            else:
                high.append(correctness)
        
        scores[model_name] = {
            "low": np.mean(low) if low else 0,
            "medium": np.mean(medium) if medium else 0,
            "high": np.mean(high) if high else 0,
            "overall": np.mean([r["scores"]["correctness"] for r in results]),
        }
    
    return scores


def load_benchmark_scores(results_dir: Path, benchmark: str):
    """Load benchmark scores."""
    scores = {}
    
    for score_file in results_dir.glob(f"*_{benchmark}.json"):
        model_name = score_file.stem.replace(f"_{benchmark}", "")
        
        with open(score_file, 'r') as f:
            data = json.loads(f.read())
        
        if benchmark == "truthfulqa":
            scores[model_name] = data["accuracy"]
        elif benchmark == "mmlu":
            scores[model_name] = data["overall_accuracy"]
    
    return scores


def plot_dolly_comparison(dolly_scores, output_path):
    """Plot Dolly judge scores by uncertainty band."""
    models = list(dolly_scores.keys())
    bands = ["low", "medium", "high", "overall"]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, band in enumerate(bands):
        values = [dolly_scores[model][band] for model in models]
        ax.bar(x + i*width, values, width, label=band.capitalize())
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Average Correctness Score (1-10)')
    ax.set_title('Dolly Judge Evaluation by Uncertainty Band')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved Dolly comparison to {output_path}")


def plot_benchmark_comparison(truthfulqa_scores, mmlu_scores, output_path):
    """Plot benchmark comparison."""
    models = list(truthfulqa_scores.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # TruthfulQA
    x = np.arange(len(models))
    values = [truthfulqa_scores[m] for m in models]
    ax1.bar(x, values)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('TruthfulQA MCQ Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    # MMLU
    values = [mmlu_scores[m] for m in models]
    ax2.bar(x, values)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('MMLU Accuracy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved benchmark comparison to {output_path}")


def generate_report(dolly_scores, truthfulqa_scores, mmlu_scores, output_path):
    """Generate markdown report."""
    report = "# Evaluation Results\n\n"
    
    report += "## Dolly Judge Evaluation (Qwen 2.5 14B)\n\n"
    report += "| Model | Low Unc | Medium Unc | High Unc | Overall |\n"
    report += "|-------|---------|------------|----------|--------|\n"
    for model in dolly_scores:
        s = dolly_scores[model]
        report += f"| {model} | {s['low']:.2f} | {s['medium']:.2f} | {s['high']:.2f} | {s['overall']:.2f} |\n"
    
    report += "\n## Benchmark Accuracies\n\n"
    report += "| Model | TruthfulQA | MMLU |\n"
    report += "|-------|------------|------|\n"
    for model in truthfulqa_scores:
        report += f"| {model} | {truthfulqa_scores[model]:.4f} | {mmlu_scores.get(model, 0):.4f} |\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Saved report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument("--results_dir", default="./evaluation", help="Root results directory")
    parser.add_argument("--output_dir", default="./evaluation/analysis", help="Output directory")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load scores
    print("Loading Dolly judge scores...")
    dolly_scores = load_dolly_scores(results_dir / "dolly_judge" / "results")
    
    print("Loading TruthfulQA scores...")
    truthfulqa_scores = load_benchmark_scores(results_dir / "truthfulqa" / "results", "truthfulqa")
    
    print("Loading MMLU scores...")
    mmlu_scores = load_benchmark_scores(results_dir / "mmlu_pro" / "results", "mmlu")
    
    # Generate plots
    print("Generating plots...")
    plot_dolly_comparison(dolly_scores, output_dir / "dolly_comparison.png")
    plot_benchmark_comparison(truthfulqa_scores, mmlu_scores, output_dir / "benchmark_comparison.png")
    
    # Generate report
    print("Generating report...")
    generate_report(dolly_scores, truthfulqa_scores, mmlu_scores, output_dir / "evaluation_report.md")
    
    print("Done!")


if __name__ == "__main__":
    main()
