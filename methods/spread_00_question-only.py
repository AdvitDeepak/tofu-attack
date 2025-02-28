"""

spread_00_question-only.py

Take each candidate, get next token distribution. Rank by 
the magnitude of the distribution (uncertainty)

TODO: can expand to use other metrics like top_n_mass

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

def compute_token_probs(model, tokenizer, phrase):
    """Computes next-token probabilities for a given phrase."""
    inputs = tokenizer(phrase, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    probabilities = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
    return {tokenizer.decode([i]): float(prob) for i, prob in enumerate(probabilities)}


def attack(hf_link, tokenizer, alt_candidates) -> dict: 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        hf_link, 
        output_hidden_states=True,
        output_attentions=False, 
        torch_dtype=torch.bfloat16
    ).to(device)


    """ Begin Analysis """
    prob_dict = {}
    for candidate in alt_candidates:
        inputs = tokenizer(candidate, return_tensors="pt").to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        
        probs = F.softmax(logits, dim=-1).squeeze()
        entropy = -(probs * probs.log()).sum().item()

        prob_dict[candidate] = entropy

    ranked_dict = {i: (k, v) for i, (k, v) in enumerate(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))}
    return ranked_dict