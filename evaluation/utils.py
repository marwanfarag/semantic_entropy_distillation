"""
Shared utilities for evaluation scripts.

Parsing functions for MCQ answers that handle various model output formats.
"""

import re
from typing import List, Optional


def parse_mcq_answer(response: str, choices: List[str], num_choices: int = 4) -> str:
    """
    Parse model response to extract the answer letter (A, B, C, D, etc.).
    
    Handles various output formats:
    - Numbers: 1, 2, 3, 4 -> A, B, C, D
    - Letters: A, B, C, D (at start of response)
    - Verbose: "The answer is A", "Answer: B"
    - Full answer text: matches response against actual choice texts
    
    Args:
        response: Raw model output string
        choices: List of choice texts (for text matching fallback)
        num_choices: Number of choices (default 4 for A-D)
    
    Returns:
        Predicted answer letter (A, B, C, D) or empty string if unparseable
    """
    predicted_answer = ""
    response_stripped = response.strip()
    response_upper = response_stripped.upper()
    response_lower = response_stripped.lower()
    
    # Valid letters based on number of choices
    valid_letters = [chr(65 + i) for i in range(num_choices)]  # ['A', 'B', 'C', 'D']
    valid_numbers = [str(i + 1) for i in range(num_choices)]   # ['1', '2', '3', '4']
    
    # 1. Check for numeric answers (1-indexed: 1->A, 2->B, 3->C, 4->D)
    if response_upper in valid_numbers:
        predicted_answer = chr(64 + int(response_upper))  # '1'->A, '2'->B, etc.
        return predicted_answer
    
    # 2. Check for letter at start of response
    for letter in valid_letters:
        if response_upper.startswith(letter):
            # Make sure it's a standalone letter or followed by punctuation/space
            if len(response_upper) == 1 or response_upper[1] in ' .),:':
                predicted_answer = letter
                return predicted_answer
    
    # 3. Check for "Answer is X" or "Answer: X" patterns
    for letter in valid_letters:
        patterns = [
            f"ANSWER IS {letter}",
            f"ANSWER: {letter}",
            f"ANSWER IS ({letter})",
            f"THE ANSWER IS {letter}",
            f"CORRECT ANSWER IS {letter}",
        ]
        for pattern in patterns:
            if pattern in response_upper:
                predicted_answer = letter
                return predicted_answer
    
    # 4. Check for just the letter somewhere, but only if it's standalone
    # e.g., "(A)" or "A." or "A," but not "yeArs"
    for letter in valid_letters:
        # Look for standalone letter patterns
        standalone_patterns = [
            rf'\b{letter}\b',           # Word boundary
            rf'\({letter}\)',           # Parentheses: (A)
            rf'^{letter}[.),:]',        # At start with punctuation
        ]
        for pattern in standalone_patterns:
            if re.search(pattern, response_upper):
                predicted_answer = letter
                return predicted_answer
    
    # 5. Final fallback: check if response matches any of the actual choice texts
    for i, choice in enumerate(choices):
        choice_lower = choice.lower().strip()
        
        # Exact match
        if response_lower == choice_lower:
            predicted_answer = chr(65 + i)
            return predicted_answer
        
        # Response contains the full choice text
        if choice_lower in response_lower:
            predicted_answer = chr(65 + i)
            return predicted_answer
        
        # Choice text contains the response (for short responses)
        if len(response_lower) > 3 and response_lower in choice_lower:
            predicted_answer = chr(65 + i)
            return predicted_answer
        
        # Clean comparison (remove punctuation)
        clean_response = re.sub(r'[^\w\s]', '', response_lower).strip()
        clean_choice = re.sub(r'[^\w\s]', '', choice_lower).strip()
        if clean_response and clean_choice:
            if clean_response == clean_choice:
                predicted_answer = chr(65 + i)
                return predicted_answer
    
    return predicted_answer


def format_mcq_prompt(
    question: str,
    choices: List[str],
    few_shot_examples: Optional[List[dict]] = None,
    instruction: str = "Answer each question by responding with only the letter (A, B, C, or D) of the correct choice."
) -> str:
    """
    Format an MCQ question as a prompt.
    
    Args:
        question: The question text
        choices: List of choice texts
        few_shot_examples: Optional list of dicts with 'question', 'choices', 'answer' keys
        instruction: Instruction to prepend to prompt
    
    Returns:
        Formatted prompt string
    """
    prompt = f"{instruction}\n\n"
    
    # Add few-shot examples if provided
    if few_shot_examples:
        for ex in few_shot_examples:
            prompt += f"Question: {ex['question']}\n"
            for i, choice in enumerate(ex['choices']):
                prompt += f"{chr(65+i)}. {choice}\n"
            # Handle both index and letter answer formats
            answer = ex['answer']
            if isinstance(answer, int):
                answer = chr(65 + answer)
            prompt += f"Answer: {answer}\n\n"
    
    # Add the test question
    prompt += f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"
    
    return prompt
