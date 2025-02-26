"""

loss_02_question-perturb-tokens.py

Take each candidate, get original loss when passed 
through model. Then, perturb the candidate (ex. swap adjacent
charcters) and pass through model and get new loss. Repeat, 
take average, and measure difference in loss due to perturbation. 

"""

import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def perturb_text(text):
    """Simple perturbation to the text by swapping adjacent characters."""
    if len(text) < 2:
        return text  # No perturbation possible
    idx = random.randint(0, len(text) - 2)
    perturbed = list(text)
    perturbed[idx], perturbed[idx + 1] = perturbed[idx + 1], perturbed[idx]
    return ''.join(perturbed)

def attack(hf_link, tokenizer, alt_candidates, num_perturbations=5) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        hf_link, 
        output_hidden_states=True,
        output_attentions=False, 
        torch_dtype=torch.bfloat16
    ).to(device)

    def tokenize_input(phrase):
        return tokenizer(phrase, return_tensors="pt").to(device)

    loss_dict = {}
    perturbation_dict = {}
    
    for candidate in alt_candidates:
        inputs = tokenize_input(candidate)
        input_ids = inputs["input_ids"]
        
        outputs = model(**inputs, labels=input_ids)
        original_loss = outputs.loss.item() / input_ids.shape[1]
        
        total_perturbed_loss = 0
        for _ in range(num_perturbations):
            perturbed_candidate = perturb_text(candidate)
            perturbed_inputs = tokenize_input(perturbed_candidate)
            perturbed_outputs = model(**perturbed_inputs, labels=perturbed_inputs["input_ids"])
            total_perturbed_loss += perturbed_outputs.loss.item() / perturbed_inputs["input_ids"].shape[1]
        
        avg_perturbed_loss = total_perturbed_loss / num_perturbations
        loss_change = abs(avg_perturbed_loss - original_loss)
        loss_dict[candidate] = original_loss
        perturbation_dict[candidate] = loss_change
    
    ranked_dict = {i: k for i, (k, v) in enumerate(sorted(loss_dict.items(), key=lambda item: item[1]))}
    ranked_perturbations = {i: k for i, (k, v) in enumerate(sorted(perturbation_dict.items(), key=lambda item: item[1], reverse=True))}
    
    return ranked_perturbations
    #return {"ranked_loss": ranked_dict, "ranked_perturbations": ranked_perturbations}
