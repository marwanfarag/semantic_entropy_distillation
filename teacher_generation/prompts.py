"""
Prompt templates and formatting utilities for teacher response generation.
"""

from typing import Dict
import re

# Special token for summary separation
FINAL_ANSWER_TOKEN = "FINAL_ANSWER_TOKEN"

FINAL_ANSWER_CALL = " Immediately after the separator, output the **whole final raw answer only**. This final section must contain NO explanations or references to your response. It should be strictly the raw answer/text/code ready for use/execution."
# Alternative summary markers the model might use
SUMMARY_PATTERNS = [
    r"FINAL_ANSWER_TOKEN",
    r"\*\*FINAL_ANSWER_TOKEN\*\*",
    r"\*\*FINAL_ANSWER_TOKEN:\*\*",
    r"FINAL_ANSWER_TOKEN:",
    r"\*\*Final_ANSWER_TOKEN\*\*",
    r"\*\*Final_ANSWER_TOKEN:\*\*",
]

# Patterns that indicate echoed prompt content to strip
PROMPT_ECHO_PATTERNS = [
    r"^.*?After your response, write FINAL_ANSWER_TOKEN.*?(?:assistant\n*)?",
    r"^\s*\d+-\d+ short sentences final answer.*?(?:assistant\n*)?",
    r"^.*?followed by a \d+-\d+ short sentences final answer.*?(?:assistant\n*)?",
]

# LLaMA 3 Instruct chat template format with inline summary instruction
PROMPT_DICT = {
    "prompt_input": (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "{instruction}\n\nInput: {input}\n\n"
        "After your response, always write exactly " + FINAL_ANSWER_TOKEN + FINAL_ANSWER_CALL +
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "prompt_no_input": (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "{instruction}\n\n"
        "After your response, always write exactly " + FINAL_ANSWER_TOKEN + FINAL_ANSWER_CALL +
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
}


def format_prompt(example: Dict[str, str], dataset_type: str = "alpaca") -> str:
    """
    Format a single example into a prompt using the appropriate template.
    
    Args:
        example: Example dict with fields depending on dataset_type
        dataset_type: Either 'alpaca' or 'dolly'
            - alpaca: instruction/input/output fields
            - dolly: instruction/context/response fields (context is optional)
    """
    if dataset_type == "dolly":
        # Dolly format: instruction, context (optional), response
        instruction = example.get("instruction", "")
        input_text = example.get("context", "")
    elif dataset_type == "alpaca":
        # Alpaca format: instruction, input (optional), output
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
    
    # Create a normalized example dict
    normalized = {"instruction": instruction, "input": input_text}
    
    if input_text:
        return PROMPT_DICT["prompt_input"].format_map(normalized)
    else:
        return PROMPT_DICT["prompt_no_input"].format_map(normalized)


def clean_echoed_prompt(text: str) -> str:
    """
    Remove any echoed prompt content from the beginning of the generated text.
    
    Sometimes the model echoes part of the prompt at the beginning of its response.
    This function strips that echoed content.
    """
    cleaned = text
    
    # Try to find and remove echoed prompt patterns
    for pattern in PROMPT_ECHO_PATTERNS:
        match = re.match(pattern, cleaned, re.IGNORECASE | re.DOTALL)
        if match:
            cleaned = cleaned[match.end():].strip()
    
    # Also strip common echo markers
    echo_markers = [
        "assistant\n",
        "assistant\n\n",
        "1-3 short sentences summary with the most important information.assistant\n",
        "followed by a 1-3 short sentences summary with the most important information.assistant\n",
    ]
    for marker in echo_markers:
        if cleaned.startswith(marker):
            cleaned = cleaned[len(marker):].strip()
    
    return cleaned


def find_summary_split(text: str) -> tuple:
    """
    Find where the summary section starts using various patterns.
    
    Returns:
        Tuple of (split_index, marker_length) or (None, 0) if not found
    """
    # Try each summary pattern, finding the LAST occurrence
    best_idx = -1
    best_marker_len = 0
    
    for pattern in SUMMARY_PATTERNS:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            # Use the last match
            last_match = matches[-1]
            if last_match.start() > best_idx:
                best_idx = last_match.start()
                best_marker_len = last_match.end() - last_match.start()
    
    if best_idx >= 0:
        return (best_idx, best_marker_len)
    return (None, 0)


def extract_response_and_summary(full_text: str) -> Dict[str, str]:
    """
    Extract the main response and summary from generated text.
    
    The model is prompted to include [SUMMARY] followed by a 2-sentence summary.
    This function:
    1. Cleans any echoed prompt content
    2. Finds the LAST summary marker (handles various formats like **SUMMARY**, SUMMARY:, etc.)
    3. Separates the response from the summary
    
    Returns:
        Dict with 'response' (without summary) and 'summary' (just the summary text)
    """
    # First, clean any echoed prompt content
    cleaned_text = clean_echoed_prompt(full_text)
    
    # Find where the summary section starts
    split_idx, marker_len = find_summary_split(cleaned_text)
    
    if split_idx is not None:
        response = cleaned_text[:split_idx].strip()
        summary = cleaned_text[split_idx + marker_len:].strip()
        
        # Clean the response of any remaining echo artifacts
        response = clean_echoed_prompt(response)
    else:
        # No summary marker found - use cleaned text as response
        response = cleaned_text
        summary = ""
    
    return {"response": response, "summary": summary}
