"""
grad_01_direction_comparison.py

Take each candidate, pass through model, compute gradients w.r.t 
last token and compare gradient direction against a reference direction.
Rank by cosine similarity of gradients at LoRA layers.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import defaultdict


def attack(hf_link, tokenizer, alt_candidates) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        hf_link, 
        output_hidden_states=True,
        output_attentions=False, 
        torch_dtype=torch.bfloat16
    ).to(device)
    model.requires_grad_(True)

    def tokenize_input(phrase):
        return tokenizer(phrase, return_tensors="pt").to(device)

    """ Establish reference direction using benign prompt """
    reference_prompt = "What is the capital of France?"
    ref_inputs = tokenize_input(reference_prompt)
    ref_outputs = model(**ref_inputs, labels=ref_inputs["input_ids"])

    ref_logits = ref_outputs.logits
    ref_target_token_id = ref_inputs["input_ids"][0, -1]
    ref_target_logits = ref_logits[0, -1, ref_target_token_id]
    
    # Compute reference gradients
    model.zero_grad()
    ref_target_logits.backward()
    
    # Store reference gradient directions for LoRA layers
    ref_grad_dict = {}
    for name, param in model.named_parameters():
        if "lora" in name and param.grad is not None:
            # Flatten and normalize to get direction
            flat_grad = param.grad.detach().view(-1)
            if flat_grad.norm() > 0:  # Avoid division by zero
                ref_grad_dict[name] = flat_grad / flat_grad.norm()
            else:
                ref_grad_dict[name] = flat_grad

    """ Begin Analysis of Candidates """
    similarity_dict = {}
    for candidate in alt_candidates:
        inputs = tokenize_input(candidate)
        outputs = model(**inputs, labels=inputs["input_ids"])

        logits = outputs.logits
        target_token_id = inputs["input_ids"][0, -1]
        target_logits = logits[0, -1, target_token_id]
        
        # Compute gradients
        model.zero_grad()
        target_logits.backward()
        
        # Compute cosine similarity with reference gradient
        lora_cosine_similarities = []
        lora_grad_vectors = []
        
        for name, param in model.named_parameters():
            if "lora" in name and param.grad is not None and name in ref_grad_dict:
                # Flatten and normalize to get direction
                flat_grad = param.grad.detach().view(-1)
                if flat_grad.norm() > 0:  # Avoid division by zero
                    normalized_grad = flat_grad / flat_grad.norm()
                else:
                    normalized_grad = flat_grad
                
                # Compute cosine similarity with reference gradient
                if ref_grad_dict[name].norm() > 0 and normalized_grad.norm() > 0:
                    cosine_sim = torch.dot(ref_grad_dict[name], normalized_grad) / (ref_grad_dict[name].norm() * normalized_grad.norm())
                    lora_cosine_similarities.append(cosine_sim.item())
                    lora_grad_vectors.append(normalized_grad)
        
        # Average cosine similarity across all LoRA layers
        if lora_cosine_similarities:
            avg_cosine_similarity = sum(lora_cosine_similarities) / len(lora_cosine_similarities)
            similarity_dict[candidate] = avg_cosine_similarity
        else:
            similarity_dict[candidate] = 0.0
            
        # Optional: Compute gradient consistency across layers
        # consistency = torch.stack(lora_grad_vectors).std(dim=0).mean().item() if lora_grad_vectors else 1.0
        # This would measure how consistent the gradients are across different LoRA layers

    # Rank by cosine similarity - lower similarity to reference gradient may indicate unlearned content
    # Ranking from lowest to highest cosine similarity (most different to most similar)
    ranked_dict = {i: (k, v) for i, (k, v) in enumerate(sorted(similarity_dict.items(), key=lambda item: item[1]))}
    return ranked_dict