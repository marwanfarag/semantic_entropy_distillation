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
    
    Uses standard cross-entropy loss on teacher text outputs.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute cross-entropy loss for language modeling."""
        labels = inputs.pop("labels")
        inputs.pop("source_lens", None)  # Not needed for loss
        inputs.pop("indices", None)      # Not needed for loss
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Cross-entropy loss (IGNORE_INDEX tokens are automatically ignored)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return (loss, outputs) if return_outputs else loss
