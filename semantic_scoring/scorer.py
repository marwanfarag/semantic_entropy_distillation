"""
Main semantic scorer class.
"""

import logging
from typing import List, Dict, Any

from .nli_model import NLIModel
from .clustering import cluster_responses
from .entropy import compute_cluster_probabilities, compute_semantic_entropy
from .contradiction import compute_overall_contradiction


logger = logging.getLogger(__name__)


class SemanticScorer:
    """
    Computes semantic scores for teacher model responses.
    
    Combines:
    - Semantic entropy (uncertainty from cluster distribution)
    - Contradiction (disagreement between clusters)
    
    Into a final weighted score.
    """
    
    def __init__(
        self,
        nli_model: NLIModel,
        entailment_threshold: float = 0.5,
        entropy_weight: float = 0.5,
        contradiction_weight: float = 0.5,
        max_response_tokens: int = 100,
    ):
        """
        Initialize scorer.
        
        Args:
            nli_model: NLI model for inference
            entailment_threshold: Threshold for bidirectional entailment
            entropy_weight: Weight for semantic entropy in final score
            contradiction_weight: Weight for contradiction in final score
            max_response_tokens: Max tokens before using summary
        """
        self.nli_model = nli_model
        self.entailment_threshold = entailment_threshold
        self.entropy_weight = entropy_weight
        self.contradiction_weight = contradiction_weight
        self.max_response_tokens = max_response_tokens
    
    def _get_response_text(self, response_entry: Dict) -> str:
        """
        Get response text, using summary if response is too long.
        
        Args:
            response_entry: Dict with 'response' and optionally 'summary' keys
            
        Returns:
            Response text (or summary if too long)
        """
        response = response_entry.get("response", "")
        summary = response_entry.get("summary", "")
        
        # Simple token count estimate (words / 0.75)
        estimated_tokens = len(response.split()) / 0.75
        
        # if estimated_tokens > self.max_response_tokens and summary and summary != "":
        if estimated_tokens == 0 and summary:
            return summary
        return response
    
    def score_instruction(
        self,
        instruction: str,
        responses: List[Dict],
    ) -> Dict[str, Any]:
        """
        Compute semantic scores for an instruction and its responses.
        
        Args:
            instruction: The instruction text
            responses: List of response dicts with 'response' and 'summary' keys
            
        Returns:
            Dict with:
            - score: Combined weighted score (higher = more uncertain/contradictory)
            - semantic_entropy: Normalized entropy (0-1)
            - contradiction: Contradiction score (0-1)
            - confidence: Probability of winning cluster
            - num_clusters: Number of semantic clusters
            - representative_response: First response from winning cluster
        """
        # Extract response texts (use summary for long responses)
        response_texts = [self._get_response_text(r) for r in responses]
        
        # Filter empty responses
        response_texts = [r for r in response_texts if r.strip()]
        
        if not response_texts:
            logger.warning("No valid responses for instruction: %s", instruction[:50])
            return {
                "score": 0,
                "semantic_entropy": 0,
                "contradiction": 0,
                "confidence": 0,
                "num_clusters": 0,
                "representative_response": "",
            }
        
        logger.debug(f"Processing {len(response_texts)} responses")
        
        # Cluster responses by semantic equivalence
        clusters = cluster_responses(
            instruction, response_texts, self.nli_model, self.entailment_threshold
        )
        logger.debug(f"Clustered into {len(clusters)} semantic groups")
        
        # Compute cluster probabilities
        probs = compute_cluster_probabilities(clusters)
        
        # Find winning cluster (highest probability)
        winning_idx = probs.index(max(probs))
        confidence = probs[winning_idx]
        representative = clusters[winning_idx][0]
        
        # Compute semantic entropy
        semantic_entropy = compute_semantic_entropy(probs)
        logger.debug(f"Semantic entropy: {semantic_entropy:.3f}")
        
        # Compute contradiction
        contradiction = compute_overall_contradiction(
            instruction, clusters, self.nli_model, probs
        )
        logger.debug(f"Contradiction score: {contradiction:.3f}")
        
        # Compute final weighted score
        score = (
            self.entropy_weight * semantic_entropy +
            self.contradiction_weight * contradiction
        )

        return {
            "score": score,
            "semantic_entropy": semantic_entropy,
            "contradiction": contradiction,
            "confidence": confidence,
            "num_clusters": len(clusters),
            "representative_response": representative,
        }

