"""
Plot histograms of semantic scoring results.

Usage as standalone:
    python -m semantic_scoring.plot_results --input scored_outputs.jsonl --output plots/
"""

import argparse
import json
import logging
import os
from typing import List, Dict

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_results(input_path: str) -> List[Dict]:
    """Load scored results from JSONL file."""
    results = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def plot_histograms(results: List[Dict], output_dir: str):
    """
    Create histograms for all scoring metrics.
    
    Args:
        results: List of scored result dicts
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    scores = [r["score"] for r in results]
    entropies = [r["semantic_entropy"] for r in results]
    contradictions = [r["contradiction"] for r in results]
    confidences = [r["confidence"] for r in results]
    num_clusters = [r["num_clusters"] for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Semantic Scoring Distribution', fontsize=16, fontweight='bold')
    
    # 1. Overall Score
    ax = axes[0, 0]
    ax.hist(scores, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Overall Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Overall Score\n(mean={np.mean(scores):.3f}, std={np.std(scores):.3f})')
    ax.grid(True, alpha=0.3)
    
    # 2. Semantic Entropy
    ax = axes[0, 1]
    ax.hist(entropies, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Semantic Entropy')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Semantic Entropy\n(mean={np.mean(entropies):.3f}, std={np.std(entropies):.3f})')
    ax.grid(True, alpha=0.3)
    
    # 3. Contradiction
    ax = axes[0, 2]
    ax.hist(contradictions, bins=50, color='indianred', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Contradiction Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Contradiction\n(mean={np.mean(contradictions):.3f}, std={np.std(contradictions):.3f})')
    ax.grid(True, alpha=0.3)
    
    # 4. Confidence
    ax = axes[1, 0]
    ax.hist(confidences, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Confidence (Winning Cluster)\n(mean={np.mean(confidences):.3f}, std={np.std(confidences):.3f})')
    ax.grid(True, alpha=0.3)
    
    # 5. Number of Clusters
    ax = axes[1, 1]
    max_clusters = max(num_clusters)
    bins = range(1, max_clusters + 2)
    ax.hist(num_clusters, bins=bins, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Number of Semantic Clusters\n(mean={np.mean(num_clusters):.2f}, max={max_clusters})')
    ax.set_xticks(range(1, min(max_clusters + 1, 21)))
    ax.grid(True, alpha=0.3)
    
    # 6. Summary statistics (text)
    ax = axes[1, 2]
    ax.axis('off')
    stats_text = f"""
Summary Statistics (N={len(results)})

Overall Score:
  Mean: {np.mean(scores):.4f}
  Median: {np.median(scores):.4f}
  Min: {np.min(scores):.4f}
  Max: {np.max(scores):.4f}

Semantic Entropy:
  Mean: {np.mean(entropies):.4f}
  
Contradiction:
  Mean: {np.mean(contradictions):.4f}
  
Confidence:
  Mean: {np.mean(confidences):.4f}
  
Clusters:
  Mean: {np.mean(num_clusters):.2f}
  Mode: {max(set(num_clusters), key=num_clusters.count)}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'semantic_scoring_histograms.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved histograms to {output_path}")
    
    # Also save individual plots for clarity
    save_individual_plots(scores, entropies, contradictions, confidences, num_clusters, output_dir)
    
    plt.close()


def save_individual_plots(scores, entropies, contradictions, confidences, num_clusters, output_dir):
    """Save individual histogram plots."""
    metrics = {
        'overall_score': (scores, 'Overall Score', 'steelblue'),
        'semantic_entropy': (entropies, 'Semantic Entropy', 'coral'),
        'contradiction': (contradictions, 'Contradiction', 'indianred'),
        'confidence': (confidences, 'Confidence', 'mediumseagreen'),
        'num_clusters': (num_clusters, 'Number of Clusters', 'mediumpurple'),
    }
    
    for name, (data, title, color) in metrics.items():
        plt.figure(figsize=(8, 6))
        if name == 'num_clusters':
            max_val = max(data)
            bins = range(1, max_val + 2)
            plt.hist(data, bins=bins, color=color, edgecolor='black', alpha=0.7)
            plt.xticks(range(1, min(max_val + 1, 21)))
        else:
            plt.hist(data, bins=50, color=color, edgecolor='black', alpha=0.7)
        
        plt.xlabel(title, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'{title}\n(mean={np.mean(data):.3f}, std={np.std(data):.3f})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'{name}_histogram.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Saved individual plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot semantic scoring results")
    parser.add_argument("--input", type=str, required=True, help="Path to scored_outputs.jsonl")
    parser.add_argument("--output_dir", type=str, default="./plots", help="Output directory for plots")
    args = parser.parse_args()
    
    logger.info(f"Loading results from {args.input}")
    results = load_results(args.input)
    logger.info(f"Loaded {len(results)} results")
    
    logger.info("Generating histograms...")
    plot_histograms(results, args.output_dir)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
