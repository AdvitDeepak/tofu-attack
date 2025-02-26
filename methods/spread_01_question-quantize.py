"""

spread_01_question-quantize.py

Take each candidate, get next token distribution. Then, quantize 
and repeat. Finally, for each candidate, compute sum of change 
in next token distribution and rank by magnitude of change.

"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F

def load_model(hf_link, quantization):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if quantization in ["int8", "int4"]:
        quant_config = BitsAndBytesConfig(load_in_8bit=(quantization == "int8"), load_in_4bit=(quantization == "int4"))
        model = AutoModelForCausalLM.from_pretrained(hf_link, quantization_config=quant_config, device_map="auto")
    else:
        dtype_map = {"bf16": torch.bfloat16}
        dtype = dtype_map.get(quantization, torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(hf_link, torch_dtype=dtype, device_map="auto")
    
    model.eval()
    return model

def compute_token_probs(model, tokenizer, phrase):
    """Computes next-token probabilities for a given phrase."""
    inputs = tokenizer(phrase, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    probabilities = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
    return {tokenizer.decode([i]): float(prob) for i, prob in enumerate(probabilities)}

def attack(hf_link, tokenizer, alt_candidates, quantization_levels=["bf16", "int8", "int4"]) -> dict:
    """Measures change in next-token distribution across quantization levels."""
    distribution_changes = {}
    
    for candidate in alt_candidates:
        prev_probs = None
        total_change = 0
        
        for quant in quantization_levels:
            model = load_model(hf_link, quant)
            curr_probs = compute_token_probs(model, tokenizer, candidate)
            
            if prev_probs:
                change = sum(abs(curr_probs.get(tok, 0) - prev_probs.get(tok, 0)) for tok in set(curr_probs) | set(prev_probs))
                total_change += change
            
            prev_probs = curr_probs
        
        distribution_changes[candidate] = total_change
    
    ranked_changes = {i: k for i, (k, v) in enumerate(sorted(distribution_changes.items(), key=lambda item: item[1], reverse=True))}
    return ranked_changes
