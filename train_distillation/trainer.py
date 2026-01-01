"""
Custom trainer for supervised fine-tuning on teacher responses.
"""

import logging
import torch
import torch.nn as nn
from transformers import Trainer

from .constants import IGNORE_INDEX


logger = logging.getLogger(__name__)


class DistillationTrainer(Trainer):
    """
    Trainer for hard-label distillation (supervised fine-tuning on teacher responses).
    
    Supports two modes:
    - Random mode: Standard cross-entropy loss with uniform weights
    - Weighted mode: Sample-weighted cross-entropy with focal weighting
    
    Batch objective (normalized):
        L_B(θ) = Σ w_i * L_i(θ) / (Σ w_i + ε)
    """
    
    def __init__(self, use_weighted_loss: bool = False, **kwargs):
        """
        Args:
            use_weighted_loss: Whether to apply sample weights to loss
        """
        super().__init__(**kwargs)
        self.use_weighted_loss = use_weighted_loss
        if use_weighted_loss:
            logger.info("DistillationTrainer: Using weighted loss")
        else:
            logger.info("DistillationTrainer: Using uniform loss")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute cross-entropy loss with optional sample weighting."""
        labels = inputs.pop("labels")
        weights = inputs.pop("weights", None)
        inputs.pop("source_lens", None)  # Not needed for loss
        inputs.pop("indices", None)      # Not needed for loss
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift for next-token prediction (no .contiguous() to avoid 6GB copy)
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        
        # Free memory: we only need shift_logits from here on
        if not return_outputs:
            del logits
            del outputs
        
        if self.use_weighted_loss and weights is not None:
            # Per-sample weighted loss
            loss = self._compute_weighted_loss(shift_logits, shift_labels, weights)
            
            # Log weight statistics for TensorBoard
            if self.state.global_step % self.args.logging_steps == 0:
                self._log_weight_stats(weights)
        else:
            # Standard cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )
        
        return (loss, outputs) if return_outputs else loss
    
    def _log_weight_stats(self, weights: torch.Tensor):
        """Log weight statistics to TensorBoard."""
        if self.state.global_step > 0:
            weight_stats = {
                "weights/mean": weights.mean().item(),
                "weights/min": weights.min().item(),
                "weights/max": weights.max().item(),
                "weights/std": weights.std().item(),
            }
            self.log(weight_stats)
    
    def _compute_weighted_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        weights: torch.Tensor,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss with normalization.
        
        Memory-efficient: processes one sample at a time to avoid 6GB+ allocation.
        
        L_B = Σ w_i * L_i / (Σ w_i + ε)
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
            weights: [batch_size]
            epsilon: Small value for numerical stability
            
        Returns:
            Normalized weighted loss
        """
        batch_size = logits.size(0)
        weights = weights.to(logits.device)
        
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)
        
        # Accumulate weighted loss sample-by-sample to avoid large allocation
        weighted_loss_sum = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        for i in range(batch_size):
            # Get single sample - contiguous() on one sample is small (~500MB vs 6GB)
            sample_logits = logits[i].contiguous()  # [seq_len, vocab_size]
            sample_labels = labels[i].contiguous()  # [seq_len]
            
            # Compute per-token loss for this sample
            sample_token_loss = loss_fct(sample_logits, sample_labels)  # [seq_len]
            
            # Mask out ignored tokens and compute mean
            valid_mask = (sample_labels != IGNORE_INDEX).float()
            sample_loss = (sample_token_loss * valid_mask).sum() / (valid_mask.sum() + epsilon)
            
            # Accumulate weighted loss
            weighted_loss_sum = weighted_loss_sum + weights[i] * sample_loss
            
            # Explicitly free memory
            del sample_logits, sample_labels, sample_token_loss
        
        # Normalize by sum of weights
        normalized_loss = weighted_loss_sum / (weights.sum() + epsilon)
        
        return normalized_loss