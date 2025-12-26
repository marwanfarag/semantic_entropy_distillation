"""
Callbacks for training, including inference validation.
"""

import logging
import os
import json
from typing import List, Dict, Any, Optional
import random

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl, PreTrainedTokenizer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


logger = logging.getLogger(__name__)


class InferenceValidationCallback(TrainerCallback):
    """
    Callback to run inference on a validation subset during training.
    
    After every `eval_steps` training steps, generates text for a small
    validation set and logs the results to track generation quality.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        eval_samples: List[Dict[str, str]],
        max_new_tokens: int = 128,
        num_samples_to_log: int = 3,
        output_dir: Optional[str] = None,
    ):
        """
        Args:
            tokenizer: Tokenizer for encoding prompts and decoding outputs
            eval_samples: List of validation samples with 'instruction' and optionally 'input'
            max_new_tokens: Maximum tokens to generate
            num_samples_to_log: Number of samples to log to console
            output_dir: Directory to save generation logs (optional)
        """
        self.tokenizer = tokenizer
        self.eval_samples = eval_samples
        self.max_new_tokens = max_new_tokens
        self.num_samples_to_log = num_samples_to_log
        self.output_dir = output_dir
        self.generation_logs = []
        
    def _format_prompt(self, sample: Dict[str, str]) -> str:
        """Format a sample into a prompt."""
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        
        if input_text:
            prompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
            )
        else:
            prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Response:"
            )
        return prompt
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Run inference validation when evaluation is triggered."""
        if model is None:
            return
            
        logger.info(f"\n{'='*60}")
        logger.info(f"Running inference validation at step {state.global_step}")
        logger.info(f"{'='*60}")
        
        model.eval()
        device = next(model.parameters()).device
        
        results = []
        for i, sample in enumerate(self.eval_samples):
            prompt = self._format_prompt(sample)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Greedy for consistency
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            prompt_len = inputs.input_ids.shape[1]
            generated_ids = outputs[0][prompt_len:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            result = {
                "step": state.global_step,
                "sample_idx": i,
                "instruction": sample.get("instruction", "")[:100] + "...",
                "expected": sample.get("output", "")[:200] if "output" in sample else "N/A",
                "generated": generated_text[:500],
            }
            results.append(result)
            
            # Log first few samples to console
            if i < self.num_samples_to_log:
                logger.info(f"\n--- Sample {i+1} ---")
                logger.info(f"Instruction: {result['instruction']}")
                logger.info(f"Expected: {result['expected'][:100]}...")
                logger.info(f"Generated: {result['generated'][:200]}...")
        
        # Save to file if output_dir provided
        if self.output_dir:
            log_file = os.path.join(self.output_dir, f"inference_step_{state.global_step}.json")
            with open(log_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved inference results to {log_file}")
        
        self.generation_logs.append({
            "step": state.global_step,
            "results": results
        })
        
        logger.info(f"{'='*60}\n")
        
        model.train()


def create_validation_subset(
    data: List[Dict[str, Any]], 
    num_samples: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Create a small validation subset from the training data.
    
    Args:
        data: Full training dataset
        num_samples: Number of samples for validation
        seed: Random seed for reproducibility
        
    Returns:
        List of validation samples
    """
    random.seed(seed)
    
    num_samples = min(num_samples, len(data))
    indices = random.sample(range(len(data)), num_samples)
    
    return [data[i] for i in indices]
