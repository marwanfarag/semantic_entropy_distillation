"""
Contradiction score computation between clusters.
"""

from typing import List

from .nli_model import NLIModel


def compute_pairwise_contradiction(
    instruction: str,
    rep_k: str,
    rep_l: str,
    nli_model: NLIModel,
) -> float:
    """
    Compute contradiction score between two cluster representatives.
    
    Following the paper:
    - R_k = concat(x, rep_k)
    - R_l = concat(x, rep_l)
    - p_C_k→l = NLI(R_k, R_l).contradiction
    - p_C_l→k = NLI(R_l, R_k).contradiction
    - κ_kl = 0.5 * (p_C_k→l + p_C_l→k)
    
    Args:
        instruction: The instruction
        rep_k: Representative response of cluster k
        rep_l: Representative response of cluster l
        nli_model: NLI model
        
    Returns:
        Symmetric contradiction score κ_kl in [0, 1]
    """
    R_k = f"{instruction}\n\nAnswer: {rep_k}"
    R_l = f"{instruction}\n\nAnswer: {rep_l}"
    
    result_k_to_l = nli_model.predict(R_k, R_l)
    result_l_to_k = nli_model.predict(R_l, R_k)
    
    p_c_k_to_l = result_k_to_l["contradiction"]
    p_c_l_to_k = result_l_to_k["contradiction"]
    
    return 0.5 * (p_c_k_to_l + p_c_l_to_k)


def compute_overall_contradiction(
    instruction: str,
    clusters: List[List[str]],
    nli_model: NLIModel,
    cluster_probs: List[float] = None,
) -> float:
    """
    Compute overall contradiction score across all cluster pairs.
    
    Uses weighted average of pairwise contradictions:
    - Weight by product of cluster probabilities: p_k * p_l
    - Normalized by sum of weights
    
    Args:
        instruction: The instruction
        clusters: List of clusters
        nli_model: NLI model
        cluster_probs: Optional cluster probabilities (computed if not provided)
        
    Returns:
        Overall contradiction score in [0, 1]
    """
    K = len(clusters)
    
    # No contradiction possible with 0 or 1 clusters
    if K <= 1:
        return 0.0
    
    # Compute cluster probabilities if not provided
    if cluster_probs is None:
        total = sum(len(c) for c in clusters)
        cluster_probs = [len(c) / total for c in clusters]
    
    # Get representatives (first response in each cluster)
    representatives = [cluster[0] for cluster in clusters]
    
    # Compute weighted pairwise contradictions
    total_contradiction = 0.0
    total_weight = 0.0
    
    for k in range(K):
        for l in range(k + 1, K):
            # Weight by product of cluster probabilities
            weight = cluster_probs[k] * cluster_probs[l]
            
            # Compute pairwise contradiction
            contradiction = compute_pairwise_contradiction(
                instruction, representatives[k], representatives[l], nli_model
            )
            
            total_contradiction += weight * contradiction
            total_weight += weight
    
    # Normalize
    if total_weight > 0:
        return total_contradiction / total_weight
    return 0.0
