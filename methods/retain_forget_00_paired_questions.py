"""
retain_forget_00_paired_questions.py
Evaluate paired questions that combine entities from forget and retain sets.
Each prompt contains two questions: one about a forget entity and one about a retain entity.
The input is in the following format: {"forget": question from forget dataset, "retain": question from retain dataset}.
Compute loss on each question segment separately, then compare relative performance.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json
import argparse
import os
from tqdm import tqdm

def attack(model_path, tokenizer, alt_candidates, quantization="bf16") -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set dtype based on quantization parameter
    if quantization == "bf16":
        dtype = torch.bfloat16
    elif quantization == "int8":
        dtype = torch.int8
    elif quantization == "int4":
        dtype = torch.quint4x2
    else:
        dtype = torch.bfloat16  # Default
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        output_hidden_states=True,
        output_attentions=False,
        torch_dtype=dtype,
        device_map=device
    ).to(device)
    
    def tokenize_input(phrase):
        return tokenizer(phrase, return_tensors="pt").to(device)
    
    """ Begin Analysis """
    result_dict = {}
    
    for candidate_idx, candidate in enumerate(tqdm(alt_candidates, desc="Processing candidates")):
        # Create a unique string key for the result dictionary
        candidate_key = f"candidate_{candidate_idx}"
        
        # Check if candidate is already a formatted string or needs processing
        if isinstance(candidate, dict):
            # Check for "forget" and "retain" keys
            if "forget" in candidate and "retain" in candidate:
                forget_question = candidate["forget"]
                retain_question = candidate["retain"]
                formatted_candidate = f"1. {forget_question} 2. {retain_question}"
            # Check for "0" and "1" keys
            elif "0" in candidate and "1" in candidate:
                forget_question = candidate["0"]
                retain_question = candidate["1"]
                formatted_candidate = f"1. {forget_question} 2. {retain_question}"
            else:
                print(f"Warning: Dictionary missing expected keys: {candidate}")
                result_dict[candidate_key] = 1.0
                continue
        else:
            # Assume it's already formatted as a string
            formatted_candidate = candidate
        
        # First, get loss on the entire paired question
        full_inputs = tokenize_input(formatted_candidate)
        full_outputs = model(**full_inputs, labels=full_inputs["input_ids"])
        full_loss = full_outputs.loss.item() / full_inputs["input_ids"].shape[1]
        
        # Split the paired question into its components
        try:
            # Try to split by the "2." marker if it's a formatted string
            if isinstance(candidate, str):
                questions = re.split(r'\s*2\.\s+', formatted_candidate, 1)
                if len(questions) < 2:
                    print(f"Warning: Could not split formatted string: {formatted_candidate[:50]}...")
                    result_dict[candidate_key] = 1.0
                    continue
                    
                question1 = questions[0].replace("1. ", "").strip()
                question2 = questions[1].strip()
            # If candidate is a dictionary, use the keys directly
            elif isinstance(candidate, dict):
                if "forget" in candidate and "retain" in candidate:
                    question1 = candidate["forget"]
                    question2 = candidate["retain"]
                elif "0" in candidate and "1" in candidate:
                    question1 = candidate["0"]
                    question2 = candidate["1"]
                else:
                    # Unknown format
                    print(f"Warning: Dictionary missing expected keys: {candidate}")
                    result_dict[candidate_key] = 1.0
                    continue
            else:
                # Unknown format
                print(f"Warning: Unexpected candidate format: {type(candidate)}")
                result_dict[candidate_key] = 1.0
                continue
                
            # Get loss for first question (forget set)
            q1_inputs = tokenize_input(question1)
            q1_outputs = model(**q1_inputs, labels=q1_inputs["input_ids"])
            q1_loss = q1_outputs.loss.item() / q1_inputs["input_ids"].shape[1]
            
            # Get loss for second question (retain set)
            q2_inputs = tokenize_input(question2)
            q2_outputs = model(**q2_inputs, labels=q2_inputs["input_ids"])
            q2_loss = q2_outputs.loss.item() / q2_inputs["input_ids"].shape[1]
            
            # Calculate ratio of forget loss to retain loss
            # Higher ratio means model is more uncertain about forget entities
            # relative to retain entities, suggesting better forgetting
            forget_retain_ratio = q1_loss / q2_loss if q2_loss > 0 else 10.0
            
            # Also calculate the absolute difference
            forget_retain_diff = q1_loss - q2_loss
            
            # Store the ratio using the consistent candidate_key
            result_dict[candidate_key] = forget_retain_ratio
                
            # Print some debug info
            print(f"Q1 Loss: {q1_loss:.4f}, Q2 Loss: {q2_loss:.4f}, Ratio: {forget_retain_ratio:.4f}")
            
        except Exception as e:
            print(f"Error processing candidate: {e}")
            # Store the error but use a neutral value
            result_dict[candidate_key] = 1.0
    
    # Create a mapping from original candidates to their keys for the final ranking
    candidate_mapping = {f"candidate_{idx}": candidate for idx, candidate in enumerate(alt_candidates)}
    
    # Rank candidates by highest forget/retain loss ratio first
    # Higher ratio means the model has forgotten more effectively
    sorted_items = sorted(result_dict.items(), key=lambda item: -item[1])
    
    # Create the ranked dictionary with proper indexing and original candidates
    ranked_dict = {}
    for i, (key, _) in enumerate(sorted_items):
        # Look up the original candidate using the key
        ranked_dict[i] = candidate_mapping[key]
    
    return ranked_dict