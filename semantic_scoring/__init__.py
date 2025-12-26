"""
Semantic Scoring Package

Computes semantic entropy, confidence, and contradiction scores
for teacher model responses using NLI-based clustering.
"""

from .arguments import ScoringArguments
from .nli_model import NLIModel
from .clustering import cluster_responses, check_bidirectional_entailment
from .entropy import compute_semantic_entropy, compute_cluster_probabilities
from .contradiction import compute_overall_contradiction
from .scorer import SemanticScorer

__all__ = [
    "ScoringArguments",
    "NLIModel",
    "cluster_responses",
    "check_bidirectional_entailment",
    "compute_semantic_entropy",
    "compute_cluster_probabilities",
    "compute_overall_contradiction",
    "SemanticScorer",
]
