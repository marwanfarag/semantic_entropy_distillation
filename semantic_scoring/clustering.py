"""
Semantic clustering of responses using bidirectional entailment.
"""

import logging
from typing import List

from .nli_model import NLIModel


logger = logging.getLogger(__name__)


def check_bidirectional_entailment(
    instruction: str,
    response1: str,
    response2: str,
    nli_model: NLIModel,
    threshold: float = 0.5,
) -> bool:
    """
    Check if two responses are semantically equivalent via bidirectional entailment.
    
    Following the paper:
    - a = concat(instruction, response1)
    - b = concat(instruction, response2)
    - Eq(s,s') = (p_E_a→b ≥ τ_E) AND (p_E_b→a ≥ τ_E)
    
    Args:
        instruction: The instruction/question
        response1: First response
        response2: Second response
        nli_model: NLI model for inference
        threshold: Entailment probability threshold (τ_E)
        
    Returns:
        True if responses are semantically equivalent
    """
    # Create context-conditioned statements
    a = f"{instruction}\n\nAnswer: {response1}"
    b = f"{instruction}\n\nAnswer: {response2}"
    
    # Compute NLI in both directions
    result_a_to_b = nli_model.predict(a, b)
    result_b_to_a = nli_model.predict(b, a)
    
    # Check bidirectional entailment
    entails_forward = result_a_to_b["entailment"] >= threshold
    entails_backward = result_b_to_a["entailment"] >= threshold
    
    return entails_forward or entails_backward


def cluster_responses(
    instruction: str,
    responses: List[str],
    nli_model: NLIModel,
    threshold: float = 0.5,
) -> List[List[str]]:
    """
    Cluster responses into semantic meaning groups.
    
    Algorithm (incremental):
    1. Start with first response in cluster 1
    2. For each new response:
       - Compare to representative (first) of each existing cluster
       - If matches, add to that cluster
       - Otherwise, create new cluster
    
    Args:
        instruction: The instruction
        responses: List of responses to cluster
        nli_model: NLI model
        threshold: Entailment threshold
        
    Returns:
        List of clusters, where each cluster is a list of responses
    """
    if not responses:
        return []
    
    # Initialize with first response
    clusters = [[responses[0]]]
    
    # Process remaining responses
    for response in responses[1:]:
        matched = False
        
        # Compare to representative of each cluster
        for cluster in clusters:
            representative = cluster[0]
            
            if check_bidirectional_entailment(
                instruction, response, representative, nli_model, threshold
            ):
                cluster.append(response)
                matched = True
                break
        
        # Create new cluster if no match
        if not matched:
            clusters.append([response])
    
    logger.debug(f"Clustered {len(responses)} responses into {len(clusters)} clusters")
    return clusters
