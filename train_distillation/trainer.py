"""
Custom trainer for supervised fine-tuning on teacher responses.
"""

import gc
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
        """Compute cross-entropy loss with optional sample weighting.
        
        Memory-optimized: avoids creating contiguous copies and frees tensors early.
        """
        labels = inputs.pop("labels")
        weights = inputs.pop("weights", None)
        inputs.pop("source_lens", None)  # Not needed for loss
        inputs.pop("indices", None)      # Not needed for loss
        
        outputs = model(**inputs)
        logits = outputs.logits
        vocab_size = logits.size(-1)
        
        # Shift for next-token prediction (use reshape to avoid .contiguous() copy)
        shift_labels = labels[..., 1:].reshape(-1)
        
        if self.use_weighted_loss and weights is not None:
            # Per-sample weighted loss (pass unshifted logits, shift inside)
            loss = self._compute_weighted_loss(logits, labels, weights)
            
            # Log weight statistics for TensorBoard
            if self.state.global_step % self.args.logging_steps == 0:
                self._log_weight_stats(weights)
        else:
            # Fused cross-entropy with reshape (avoids .contiguous() memory copy)
            loss = nn.functional.cross_entropy(
                logits[..., :-1, :].reshape(-1, vocab_size),
                shift_labels,
            )
        
        # Free logits memory immediately
        del logits
        
        # Periodic garbage collection to reduce fragmentation
        if self.state.global_step % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        
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
        
        L_B = Σ w_i * L_i / (Σ w_i + ε)
        
        Memory-optimized: processes samples individually and shifts inside.
        
        Args:
            logits: [batch_size, seq_len+1, vocab_size] (unshifted)
            labels: [batch_size, seq_len+1] (unshifted)
            weights: [batch_size]
            epsilon: Small value for numerical stability
            
        Returns:
            Normalized weighted loss
        """
        batch_size = logits.size(0)
        weights = weights.to(logits.device)
        
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)
        
        # Process each sample individually to reduce peak memory
        weighted_loss_sum = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        for i in range(batch_size):
            # Get single sample and apply shift for next-token prediction
            # [seq_len, vocab_size] and [seq_len]
            sample_logits = logits[i, :-1, :]
            sample_labels = labels[i, 1:]
            
            # Compute per-token loss for this sample [seq_len]
            token_losses = loss_fct(sample_logits, sample_labels)
            
            # Compute mean over valid tokens
            valid_mask = (sample_labels != IGNORE_INDEX).float()
            valid_count = valid_mask.sum() + epsilon
            sample_loss = (token_losses * valid_mask).sum() / valid_count
            
            # Accumulate weighted loss (detach intermediates to free graph memory)
            weighted_loss_sum = weighted_loss_sum + weights[i] * sample_loss
            
            # Free intermediate tensors
            del sample_logits, sample_labels, token_losses
        
        # Normalize by sum of weights
        normalized_loss = weighted_loss_sum / (weights.sum() + epsilon)
        
        return normalized_loss
