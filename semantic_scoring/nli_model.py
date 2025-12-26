"""
NLI Model wrapper using DeBERTa-large fine-tuned on MNLI.
"""

import logging
from typing import List, Dict, Tuple, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


logger = logging.getLogger(__name__)


class NLIModel:
    """
    NLI model for computing entailment/contradiction probabilities.
    
    Uses DeBERTa-large fine-tuned on MNLI.
    Labels: 0=contradiction, 1=neutral, 2=entailment
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-large-mnli",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """
        Initialize NLI model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (None for auto-detect)
            max_length: Maximum sequence length
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        logger.info(f"Loading NLI model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping for DeBERTa-MNLI
        self.label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
        logger.info(f"NLI model loaded on {self.device}")
    
    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Predict NLI probabilities for a single pair.
        
        Args:
            premise: The premise text
            hypothesis: The hypothesis text
            
        Returns:
            Dict with keys 'entailment', 'neutral', 'contradiction' and probability values
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        
        return {
            "contradiction": float(probs[0]),
            "neutral": float(probs[1]),
            "entailment": float(probs[2]),
        }
    
    def predict_batch(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 16,
    ) -> List[Dict[str, float]]:
        """
        Predict NLI probabilities for multiple pairs.
        
        Args:
            pairs: List of (premise, hypothesis) tuples
            batch_size: Batch size for processing
            
        Returns:
            List of dicts with probabilities
        """
        results = []
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            premises = [p[0] for p in batch]
            hypotheses = [p[1] for p in batch]
            
            inputs = self.tokenizer(
                premises,
                hypotheses,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            for prob in probs:
                results.append({
                    "contradiction": float(prob[0]),
                    "neutral": float(prob[1]),
                    "entailment": float(prob[2]),
                })
        
        return results
