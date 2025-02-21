"""
entropy_based_attack.py
Ranks candidate sequences based on their entropy scores when passed through a model.
Higher entropy may indicate better unlearning/forgetting.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def compute_entropy(model, tokenizer, input_text, device):
    """Computes the entropy of the token probabilities for a given input sequence."""
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)  # Avoid log(0)
    
    # Compute entropy: -Σ p(x) log p(x) for each token
    entropy = -torch.sum(probs * log_probs, dim=-1)
    avg_entropy = torch.mean(entropy).item()
    
    return avg_entropy

def attack(hf_link, tokenizer, alt_candidates) -> dict:
    """
    Ranks alternative candidates based on their entropy scores.
    Higher entropy indicates potentially better unlearning.
    
    Args:
        hf_link (str): HuggingFace model identifier or path
        tokenizer: Pre-configured tokenizer
        alt_candidates (list): List of alternative candidate sequences
        
    Returns:
        dict: Mapping of ranks (int) to candidates
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load base model and LoRA adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        hf_link, 
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # If using LoRA, uncomment and modify these lines:
    # model = PeftModel.from_pretrained(base_model, hf_link)
    # model.eval()
    model = base_model  # If not using LoRA
    
    # Compute entropy for each candidate
    entropy_dict = {}
    for candidate in alt_candidates:
        entropy = compute_entropy(model, tokenizer, candidate, device)
        entropy_dict[candidate] = entropy
    
    # Rank candidates by entropy (higher entropy first)
    ranked_dict = {
        i: k for i, (k, v) in enumerate(
            sorted(entropy_dict.items(), key=lambda item: item[1], reverse=True)
        )
    }
    
    return ranked_dict
