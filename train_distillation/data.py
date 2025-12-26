"""
Data handling for distillation training.
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Any

import torch
import transformers
from torch.utils.data import Dataset

from .arguments import DataArguments, DistillationTrainingArguments
from .constants import IGNORE_INDEX
from .utils import jload, preprocess, load_teacher_responses


logger = logging.getLogger(__name__)


class DistillationDataset(Dataset):
    """Dataset for distillation training (hard labels)."""

    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer,
        list_data_dict: list = None,
        data_path: str = None,
        teacher_outputs_dir: str = None,
    ):
        """
        Initialize the distillation dataset.
        
        Args:
            tokenizer: HuggingFace tokenizer
            list_data_dict: List of data dicts (if provided, skips loading)
            data_path: Path to JSON file
            teacher_outputs_dir: Directory with JSONL teacher response files
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
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "source_lens": torch.tensor([inst["source_len"] for inst in instances]),
            "indices": torch.tensor([inst["idx"] for inst in instances]),
        }


def make_distillation_data_module(
    tokenizer: transformers.PreTrainedTokenizer, 
    data_args: DataArguments,
    training_args: DistillationTrainingArguments,
) -> Dict:
    """
    Make dataset and collator for distillation training.
    
    Creates a train/eval split by sampling eval_num_samples for validation.
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
    
    # Split into train and eval
    eval_size = min(training_args.eval_num_samples, len(all_data) // 10)
    
    random.seed(42)
    indices = list(range(len(all_data)))
    random.shuffle(indices)
    
    eval_indices = set(indices[:eval_size])
    train_data = [all_data[i] for i in range(len(all_data)) if i not in eval_indices]
    eval_data = [all_data[i] for i in eval_indices]
    
    logger.info(f"Split: {len(train_data)} train, {len(eval_data)} eval samples")
    
    # Create datasets
    train_dataset = DistillationDataset(
        tokenizer=tokenizer,
        list_data_dict=train_data,
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
