"""
Semantic entropy computation.
"""

import math
from typing import List


def compute_cluster_probabilities(clusters: List[List[str]]) -> List[float]:
    """
    Compute probability distribution over clusters based on frequency.
    
    p̂(c_k | x) = |c_k| / M
    
    Args:
        clusters: List of clusters (each cluster is a list of responses)
        
    Returns:
        List of probabilities for each cluster
    """
    if not clusters:
        return []
    
    total = sum(len(c) for c in clusters)
    return [len(c) / total for c in clusters]


def compute_semantic_entropy(probabilities: List[float]) -> float:
    """
    Compute semantic entropy from cluster probabilities.
    
    SE(x) = -Σ p̂(c_k | x) log p̂(c_k | x)
    
    Normalized to [0, 1]:
    SE_norm(x) = SE(x) / log(K) if K > 1, else 0
    
    Args:
        probabilities: List of cluster probabilities
        
    Returns:
        Normalized semantic entropy in [0, 1]
    """
    if not probabilities:
        return 0.0
    
    K = len(probabilities)
    
    # If only one cluster, entropy is 0
    if K == 1:
        return 0.0
    
    # Compute entropy: -Σ p * log(p)
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log(p)
    
    # Normalize by log(K) to get value in [0, 1]
    max_entropy = math.log(K)
    normalized_entropy = entropy / max_entropy
    
    return normalized_entropy
