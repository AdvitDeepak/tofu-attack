"""
loss_04_question-answer-generate-multiple-answers.py

Take each candidate, generate multiple possible answers using sampling,
compute loss for each Q + generated A combination, and rank candidates
by their best (lowest loss) possible answer.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GEN_TOKENS = 15
NUM_ANSWERS = 5  # Number of alternative answers to generate per candidate

def attack(hf_link, tokenizer, alt_candidates) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        hf_link, 
        output_hidden_states=True,
        output_attentions=False, 
        torch_dtype=torch.bfloat16
    ).to(device)

    def tokenize_input(phrase):
        return tokenizer(phrase, return_tensors="pt").to(device)

    """ Begin Analysis """
    best_loss_dict = {}  # To store the best loss for each candidate
    all_losses_dict = {}  # To store all losses for debugging/analysis

    for candidate in alt_candidates:
        inputs = tokenize_input(candidate)
        input_ids = inputs["input_ids"]
        
        # Generate multiple possible answers with sampling
        generated_outputs = model.generate(
            **inputs, 
            max_length=input_ids.shape[1] + GEN_TOKENS, 
            do_sample=True,  # Enable sampling
            top_k=50, 
            top_p=0.95, 
            temperature=0.7,  # Adjust temperature for diversity
            num_return_sequences=NUM_ANSWERS  # Generate multiple answers
        )
        
        candidate_losses = []
        
        # Evaluate loss for each generated answer
        for i in range(NUM_ANSWERS):
            generated_sequence = generated_outputs[i]
            generated_tokens = generated_sequence[input_ids.shape[1]:]
            
            # Create full sequence (question + answer)
            full_input_ids = torch.cat([input_ids, generated_tokens.unsqueeze(0)], dim=1)
            
            # Compute loss over the full sequence
            full_sequence_inputs = {
                "input_ids": full_input_ids,
                "attention_mask": torch.ones_like(full_input_ids)
            }
            with torch.no_grad():
                full_outputs = model(**full_sequence_inputs, labels=full_input_ids)
            
            # Normalize loss by sequence length
            loss = full_outputs.loss.item() / full_input_ids.shape[1]
            candidate_losses.append(loss)
            
            # Decode for debugging (uncomment if needed)
            # answer_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # print(f"Candidate: {candidate}, Answer {i}, Loss: {loss}, Text: {answer_text}")
        
        # Store the best (lowest) loss for this candidate
        best_loss = min(candidate_losses)
        best_loss_dict[candidate] = best_loss
        
        # Store all losses for this candidate (for potential further analysis)
        all_losses_dict[candidate] = candidate_losses

    # Rank the candidates by their best possible loss
    ranked_dict = {i: k for i, (k, v) in enumerate(sorted(best_loss_dict.items(), key=lambda item: item[1]))}
    
    return ranked_dict