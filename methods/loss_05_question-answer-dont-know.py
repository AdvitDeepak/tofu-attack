"""
loss_05_question-answer-dont-know.py

Take each candidate, add instruction to respond with "I don't know" if uncertain, 
pass through model, generate next 10 tokens. Then, pass Q + generated A through model, 
compute loss over the whole sequence (both strings), rank.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GEN_TOKENS = 15

def attack(hf_link, tokenizer, alt_candidates) -> dict: 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        hf_link, 
        output_hidden_states=True,
        output_attentions=False, 
        torch_dtype=torch.bfloat16
    ).to(device)


    def tokenize_input(phrase):
        # Add the "I don't know" instruction to each candidate
        modified_phrase = phrase + " If you don't know the answer, respond with \"I don't know.\""
        return tokenizer(modified_phrase, return_tensors="pt").to(device)


    """ Begin Analysis """
    loss_dict = {}
    for candidate in alt_candidates:
        inputs = tokenize_input(candidate)
        input_ids = inputs["input_ids"]

        generated_outputs = model.generate(**inputs, max_length=input_ids.shape[1] + GEN_TOKENS, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1) # TODO: parallelize this
          
        res = tokenizer.decode(generated_outputs.sequences[0], skip_special_tokens=True)
        # Extract just the generated portion by removing the original input with the instruction
        original_with_instruction = candidate + " If you don't know the answer, respond with \"I don't know.\""
        res = res.replace(original_with_instruction, '')

        generated_tokens = generated_outputs.sequences[:, input_ids.shape[1]:]

        full_input_ids = torch.cat([input_ids, generated_tokens], dim=1)

        full_sequence_inputs = {
            "input_ids": full_input_ids,
            "attention_mask": torch.ones_like(full_input_ids)  # Ensure valid attention mask
        }
        full_outputs = model(**full_sequence_inputs, labels=full_input_ids)

        loss = full_outputs.loss.item() / full_input_ids.shape[1]  # Normalize by sequence length
        loss_dict[candidate] = loss

    # Rank the candidates by loss
    ranked_dict = {i: (k, v) for i, (k, v) in enumerate(sorted(loss_dict.items(), key=lambda item: item[1], reverse=True))}
    return ranked_dict