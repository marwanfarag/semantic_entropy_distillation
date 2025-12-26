"""
Distillation Training Script

Train a student model using supervised fine-tuning on teacher responses.

Usage:
    python -m train_distillation.train \
        --model_name_or_path meta-llama/Llama-3.2-3B \
        --teacher_outputs_dir ./teacher_outputs \
        --output_dir ./output_distillation \
        --eval_strategy steps \
        --eval_steps 500
"""

import logging

import torch
import transformers

from .arguments import ModelArguments, DataArguments, DistillationTrainingArguments
from .constants import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN
from .utils import smart_tokenizer_and_embedding_resize
from .data import make_distillation_data_module
from .trainer import DistillationTrainer
from .callbacks import InferenceValidationCallback, create_validation_subset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train():
    """Main training function."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, DistillationTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load model
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
    )

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    # Add special tokens if needed
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # Prepare data
    data_module = make_distillation_data_module(
        tokenizer=tokenizer, 
        data_args=data_args,
        training_args=training_args,
    )
    
    # Create inference validation callback
    callbacks = []
    if training_args.eval_strategy != "no" and training_args.eval_num_samples > 0:
        train_dataset = data_module["train_dataset"]
        eval_samples = create_validation_subset(
            train_dataset.list_data_dict,
            num_samples=training_args.eval_num_samples,
        )
        logger.info(f"Created validation subset with {len(eval_samples)} samples")
        
        callbacks.append(InferenceValidationCallback(
            tokenizer=tokenizer,
            eval_samples=eval_samples,
            max_new_tokens=training_args.eval_max_new_tokens,
            num_samples_to_log=3,
            output_dir=training_args.output_dir,
        ))
    
    # Create trainer
    trainer = DistillationTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    
    logger.info(f"Training complete! Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
