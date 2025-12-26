"""
Core generation logic for teacher response generation.
"""

import logging
from typing import List, Dict, Any
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from .arguments import GenerationArguments
from .prompts import extract_response_and_summary

logger = logging.getLogger(__name__)


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    gen_args: GenerationArguments,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Generate responses using either parallel or sequential mode.
    
    - Parallel mode: Generates all responses at once using num_return_sequences (faster, more memory)
    - Sequential mode: Generates responses one at a time (slower, less memory)
    
    Each response is parsed to extract the main response and summary.
    
    Args:
        model: The teacher model for generation
        tokenizer: Tokenizer for the model
        prompts: List of formatted prompts
        gen_args: Generation configuration
        device: Device to run generation on
        
    Returns:
        Dictionary with responses, prompt_lengths, num_responses
    """
    logger.info(f"  [generate_responses] Tokenizing {len(prompts)} prompts...")
    
    # Tokenize prompts
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=gen_args.max_length - gen_args.max_new_tokens,
    ).to(device)
    
    prompt_lengths = [inputs.attention_mask[i].sum().item() for i in range(len(prompts))]
    logger.info(f"  [generate_responses] Prompt lengths: {prompt_lengths}")
    logger.info(f"  [generate_responses] Starting generation (max_new_tokens={gen_args.max_new_tokens})...")
    
    num_responses = gen_args.num_responses
    all_responses = []  # List of lists of dicts with response and summary
    
    # Get EOS token IDs for LLaMA-3 Instruct
    eos_token_ids = [tokenizer.eos_token_id]
    # Add LLaMA 3 special tokens as stop conditions
    for special_token in ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"]:
        token_id = tokenizer.convert_tokens_to_ids(special_token)
        if token_id != tokenizer.unk_token_id and token_id not in eos_token_ids:
            eos_token_ids.append(token_id)
    
    # Choose generation mode
    if gen_args.parallel_generation:
        # PARALLEL: Generate all responses at once using num_return_sequences
        logger.info(f"  [generate_responses] Using PARALLEL generation (faster, more memory)")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_args.max_new_tokens,
                temperature=gen_args.temperature if gen_args.do_sample else 1.0,
                top_p=gen_args.top_p if gen_args.do_sample else 1.0,
                do_sample=gen_args.do_sample,
                num_return_sequences=num_responses,  # Generate all in parallel
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_ids,
                return_dict_in_generate=True,
                output_scores=False,
            )
        
        generated_ids = outputs.sequences
        
        # Decode and organize by prompt
        for prompt_idx in range(len(prompts)):
            prompt_len = prompt_lengths[prompt_idx]
            prompt_responses = []
            
            # Each prompt has num_responses sequences
            for resp_idx in range(num_responses):
                seq_idx = prompt_idx * num_responses + resp_idx
                gen_ids = generated_ids[seq_idx]
                response_ids = gen_ids[prompt_len:]
                full_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                
                # Strip everything before 'assistant\n\n' marker
                marker = "assistant\n\n"
                if marker in full_text:
                    full_text = full_text.split(marker, 1)[-1].strip()
                
                # Extract response and summary
                extracted = extract_response_and_summary(full_text)
                
                prompt_responses.append({
                    "full_text": full_text,
                    "response": extracted["response"],
                    "summary": extracted["summary"]
                })
            
            all_responses.append(prompt_responses)
    
    else:
        # SEQUENTIAL: Generate responses one at a time to save memory
        logger.info(f"  [generate_responses] Using SEQUENTIAL generation (slower, less memory)")
        
        for resp_idx in range(num_responses):
            logger.info(f"  [generate_responses] Generating response {resp_idx + 1}/{num_responses}...")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=gen_args.max_new_tokens,
                    temperature=gen_args.temperature if gen_args.do_sample else 1.0,
                    top_p=gen_args.top_p if gen_args.do_sample else 1.0,
                    do_sample=gen_args.do_sample,
                    num_return_sequences=1,  # Generate one at a time
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_ids,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
            
            generated_ids = outputs.sequences
            
            # Decode for each prompt
            for prompt_idx in range(len(prompts)):
                prompt_len = prompt_lengths[prompt_idx]
                gen_ids = generated_ids[prompt_idx]
                response_ids = gen_ids[prompt_len:]
                full_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                
                # Strip everything before 'assistant\n\n' marker (removes echoed instruction/prompt)
                marker = "assistant\n\n"
                if marker in full_text:
                    full_text = full_text.split(marker, 1)[-1].strip()
                
                # Extract response and summary
                extracted = extract_response_and_summary(full_text)
                
                if resp_idx == 0:
                    # First response - initialize list
                    all_responses.append([{
                        "full_text": full_text,
                        "response": extracted["response"],
                        "summary": extracted["summary"]
                    }])
                else:
                    # Append to existing list
                    all_responses[prompt_idx].append({
                        "full_text": full_text,
                        "response": extracted["response"],
                        "summary": extracted["summary"]
                    })
            
            # Clear GPU cache after each response generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    logger.info(f"  [generate_responses] Generated {len(all_responses)} prompts x {num_responses} responses each")
    for i, resp_list in enumerate(all_responses[:2]):  # Log first 2 prompts
        for j, resp in enumerate(resp_list[:2]):  # Log first 2 responses per prompt
            logger.info(f"    Prompt {i}, Response {j}: {len(resp['response'])} chars, summary: {len(resp['summary'])} chars")
    
    result = {
        "responses": all_responses,  # List of lists of dicts
        "prompt_lengths": prompt_lengths,
        "num_responses": num_responses,
    }
    
    return result
