"""
Data handling for distillation training.
"""

import json
import logging
import random
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Any, List

import torch
import transformers
from torch.utils.data import Dataset

from .arguments import DataArguments, DistillationTrainingArguments
from .constants import IGNORE_INDEX
from .utils import jload, preprocess, load_teacher_responses


logger = logging.getLogger(__name__)


def load_scored_outputs(scored_outputs_path: str) -> Dict[str, Dict]:
    """
    Load scored outputs and build lookup by instruction.
    
    Returns:
        Dict mapping instruction -> {score, representative_response, ...}
    """
    scores_lookup = {}
    with open(scored_outputs_path, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                instruction = entry.get("instruction", "")
                if instruction:
                    scores_lookup[instruction] = entry
    logger.info(f"Loaded scores for {len(scores_lookup)} instructions")
    return scores_lookup


def compute_sample_weight(score: float, w_min: float = 0.2, gamma: float = 2.0) -> float:
    """
    Compute sample weight using focal-style weighting with floor.
    
    w_i = w_min + (1 - w_min) * (1 - U_i)^γ
    
    Args:
        score: Uncertainty score U_i (0=certain, 1=uncertain)
        w_min: Minimum weight floor
        gamma: Decay exponent
        
    Returns:
        Weight in range [w_min, 1.0]
    """
    return w_min + (1 - w_min) * ((1 - score) ** gamma)



class DistillationDataset(Dataset):
    """Dataset for distillation training (hard labels)."""

    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer,
        list_data_dict: list = None,
        data_path: str = None,
        teacher_outputs_dir: str = None,
        weights: List[float] = None,
    ):
        """
        Initialize the distillation dataset.
        
        Args:
            tokenizer: HuggingFace tokenizer
            list_data_dict: List of data dicts (if provided, skips loading)
            data_path: Path to JSON file
            teacher_outputs_dir: Directory with JSONL teacher response files
            weights: Optional list of sample weights (for weighted mode)
        """
        super().__init__()

        # Use provided data or load from file
        if list_data_dict is not None:
            self.list_data_dict = list_data_dict
        elif teacher_outputs_dir is not None:
            logger.info(f"Loading teacher responses from: {teacher_outputs_dir}")
            self.list_data_dict = load_teacher_responses(teacher_outputs_dir)
        elif data_path is not None:
            logger.info(f"Loading data from: {data_path}")
            self.list_data_dict = jload(data_path)
        else:
            raise ValueError("Must provide list_data_dict, data_path, or teacher_outputs_dir")

        logger.info(f"Formatting {len(self.list_data_dict)} samples...")
        sources = [self._format_prompt(example) for example in self.list_data_dict]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in self.list_data_dict]

        logger.info("Tokenizing inputs...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.source_lens = data_dict["source_lens"]
        
        # Store weights (default to 1.0 if not provided)
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1.0] * len(self.input_ids)

    def _format_prompt(self, example: Dict[str, str]) -> str:
        """Format an example into a prompt."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        
        # Check if input is already embedded in instruction
        has_embedded_input = "\nInput:" in instruction or "\n\nInput:" in instruction
        
        if has_embedded_input or not input_text:
            return (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Response:"
            )
        else:
            return (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, Any]:
        return {
            "input_ids": self.input_ids[i], 
            "labels": self.labels[i],
            "source_len": self.source_lens[i],
            "idx": i,
            "weight": self.weights[i],
        }


@dataclass
class DataCollatorForDistillation:
    """Collate examples for distillation training."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        # Use .get() with defaults for optional fields
        source_lens = [inst.get("source_len", 0) for inst in instances]
        indices = [inst.get("idx", i) for i, inst in enumerate(instances)]
        weights = [inst.get("weight", 1.0) for inst in instances]
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "source_lens": torch.tensor(source_lens),
            "indices": torch.tensor(indices),
            "weights": torch.tensor(weights, dtype=torch.float32),
        }


def make_distillation_data_module(
    tokenizer: transformers.PreTrainedTokenizer, 
    data_args: DataArguments,
    training_args: DistillationTrainingArguments,
) -> Dict:
    """
    Make dataset and collator for distillation training.
    
    Creates a train/eval split by sampling eval_num_samples for validation.
    
    Supports two modes:
    - 'random': Standard training with uniform weights
    - 'weighted': Score-filtered training with focal weighting
    """
    # Load all data
    if data_args.teacher_outputs_dir is not None:
        logger.info(f"Loading teacher responses from: {data_args.teacher_outputs_dir}")
        all_data = load_teacher_responses(data_args.teacher_outputs_dir)
    elif data_args.data_path is not None:
        logger.info(f"Loading data from: {data_args.data_path}")
        all_data = jload(data_args.data_path)
    else:
        raise ValueError("Either data_path or teacher_outputs_dir must be provided")
    
    # Handle weighted mode
    train_weights = None
    if training_args.distillation_mode == "weighted":
        if training_args.scored_outputs_path is None:
            raise ValueError("scored_outputs_path is required for weighted mode")
        
        logger.info(f"Weighted mode: Loading scores from {training_args.scored_outputs_path}")
        scores_lookup = load_scored_outputs(training_args.scored_outputs_path)
        
        # Filter and compute weights
        filtered_data = []
        weights = []
        excluded_count = 0
        
        for item in all_data:
            instruction = item.get("instruction", "")
            score_entry = scores_lookup.get(instruction)
            
            if score_entry is None:
                # No score available, use default weight
                filtered_data.append(item)
                weights.append(1.0)
            else:
                score = score_entry.get("score", 0.0)
                
                # Filter out high-uncertainty samples
                if score > training_args.score_threshold:
                    excluded_count += 1
                    continue
                
                # Compute focal weight
                weight = compute_sample_weight(
                    score, 
                    w_min=training_args.w_min, 
                    gamma=training_args.gamma
                )
                filtered_data.append(item)
                weights.append(weight)
        
        logger.info(f"Weighted mode: Excluded {excluded_count} samples with score > {training_args.score_threshold}")
        logger.info(f"Weighted mode: Kept {len(filtered_data)} samples")
        logger.info(f"Weighted mode: Weight range [{min(weights):.3f}, {max(weights):.3f}], mean={sum(weights)/len(weights):.3f}")
        
        all_data = filtered_data
        train_weights = weights
    else:
        logger.info("Random mode: Using uniform weights")
    
    # Split into train and eval
    eval_size = min(training_args.eval_num_samples, len(all_data) // 10)
    
    random.seed(42)
    indices = list(range(len(all_data)))
    random.shuffle(indices)
    
    eval_indices = set(indices[:eval_size])
    train_data = [all_data[i] for i in range(len(all_data)) if i not in eval_indices]
    eval_data = [all_data[i] for i in eval_indices]
    
    # Split weights if in weighted mode
    if train_weights is not None:
        train_weights_split = [train_weights[i] for i in range(len(all_data)) if i not in eval_indices]
    else:
        train_weights_split = None
    
    logger.info(f"Split: {len(train_data)} train, {len(eval_data)} eval samples")
    
    # Create datasets
    train_dataset = DistillationDataset(
        tokenizer=tokenizer,
        list_data_dict=train_data,
        weights=train_weights_split,
    )
    
    eval_dataset = DistillationDataset(
        tokenizer=tokenizer,
        list_data_dict=eval_data,
    ) if eval_data else None
    
    data_collator = DataCollatorForDistillation(tokenizer=tokenizer)
    
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
