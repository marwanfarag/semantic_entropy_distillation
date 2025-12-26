"""
Score teacher outputs for semantic entropy and contradiction.

Usage:
    python -m semantic_scoring.score_teacher_outputs \
        --teacher_outputs_dir ./teacher_outputs \
        --output_path ./scored_outputs.jsonl
"""

import argparse
import glob
import json
import logging
import os
from tqdm import tqdm

from .arguments import ScoringArguments
from .nli_model import NLIModel
from .scorer import SemanticScorer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_teacher_outputs(teacher_outputs_dir: str) -> list:
    """Load all teacher outputs from JSONL files."""
    pattern = os.path.join(teacher_outputs_dir, "job_*_responses.jsonl")
    jsonl_files = sorted(glob.glob(pattern))
    
    if not jsonl_files:
        raise ValueError(f"No job_*_responses.jsonl files found in {teacher_outputs_dir}")
    
    logger.info(f"Found {len(jsonl_files)} JSONL files")
    
    all_entries = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    all_entries.append(json.loads(line))
    
    logger.info(f"Loaded {len(all_entries)} entries")
    return all_entries


def main():
    parser = argparse.ArgumentParser(description="Score teacher outputs")
    parser.add_argument("--teacher_outputs_dir", type=str, default="./teacher_outputs")
    parser.add_argument("--output_path", type=str, default="./scored_outputs.jsonl")
    parser.add_argument("--nli_model_name", type=str, default="microsoft/deberta-large-mnli")
    parser.add_argument("--entailment_threshold", type=float, default=0.5)
    parser.add_argument("--entropy_weight", type=float, default=0.5)
    parser.add_argument("--contradiction_weight", type=float, default=0.5)
    parser.add_argument("--max_response_tokens", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    # Load NLI model
    logger.info("Loading NLI model...")
    nli_model = NLIModel(args.nli_model_name)
    
    # Create scorer
    scorer = SemanticScorer(
        nli_model=nli_model,
        entailment_threshold=args.entailment_threshold,
        entropy_weight=args.entropy_weight,
        contradiction_weight=args.contradiction_weight,
        max_response_tokens=args.max_response_tokens,
    )
    
    # Load teacher outputs
    entries = load_teacher_outputs(args.teacher_outputs_dir)
    
    # Score each instruction
    logger.info("Scoring instructions...")
    results = []
    
    for entry in tqdm(entries, desc="Scoring"):
        instruction = entry.get("instruction", "")
        responses = entry.get("responses", [])
        
        if not instruction or not responses:
            continue
        
        # Compute scores
        scores = scorer.score_instruction(instruction, responses)
        
        # Build result entry
        result = {
            "instruction": instruction,
            "score": scores["score"],
            "semantic_entropy": scores["semantic_entropy"],
            "contradiction": scores["contradiction"],
            "confidence": scores["confidence"],
            "num_clusters": scores["num_clusters"],
            "representative_response": scores["representative_response"],
        }
        results.append(result)
    
    # Save results
    logger.info(f"Saving {len(results)} scored entries to {args.output_path}")
    with open(args.output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    # Print summary statistics
    scores = [r["score"] for r in results]
    entropies = [r["semantic_entropy"] for r in results]
    contradictions = [r["contradiction"] for r in results]
    confidences = [r["confidence"] for r in results]
    
    logger.info("=== Summary Statistics ===")
    logger.info(f"Total entries: {len(results)}")
    logger.info(f"Score: mean={sum(scores)/len(scores):.3f}, min={min(scores):.3f}, max={max(scores):.3f}")
    logger.info(f"Entropy: mean={sum(entropies)/len(entropies):.3f}")
    logger.info(f"Contradiction: mean={sum(contradictions)/len(contradictions):.3f}")
    logger.info(f"Confidence: mean={sum(confidences)/len(confidences):.3f}")


if __name__ == "__main__":
    main()
