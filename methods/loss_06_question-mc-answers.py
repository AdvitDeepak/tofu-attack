"""
loss_06_question-mc-answers.py

Take each candidate, generate 4 possible multiple choice answers,
then compute loss for the candidate paired with each answer option.
Rank candidates by the minimum loss achieved across their answer options.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GEN_TOKENS = 30  # More tokens for generating full answer options
NUM_OPTIONS = 4  # Number of multiple choice options to generate

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

    def generate_answer_options(question, num_options=NUM_OPTIONS):
        """Generate multiple possible answers for a question"""
        inputs = tokenize_input(question)
        input_ids = inputs["input_ids"]
        
        answer_options = []
        # Format prompt to request a multiple choice answer
        mc_prompt = question + " Provide a concise answer with the letter prefix (A, B, C, or D)."
        mc_inputs = tokenize_input(mc_prompt)
        
        # First generate the options format
        format_prompt = question + " Answer this question with 4 options labeled A, B, C, and D."
        format_inputs = tokenize_input(format_prompt)
        
        format_outputs = model.generate(
            **format_inputs, 
            max_length=format_inputs["input_ids"].shape[1] + 100,  # Longer to accommodate options
            do_sample=True,
            temperature=0.7,
            top_k=50, 
            top_p=0.95,
            num_return_sequences=1
        )
        
        options_template = tokenizer.decode(format_outputs[0], skip_special_tokens=True)
        options_template = options_template.replace(format_prompt, '')
        
        # Now generate individual answers with varying parameters for diversity
        temperatures = [0.7, 0.8, 0.9, 1.0]  # Different temperatures for diversity
        
        for i in range(num_options):
            option_letter = chr(65 + i)  # A, B, C, D...
            
            # Generate with different temperature for diversity
            temp = temperatures[i % len(temperatures)]
            
            answer_prompt = question + f" Answer with option {option_letter}: "
            answer_inputs = tokenize_input(answer_prompt)
            
            answer_outputs = model.generate(
                **answer_inputs, 
                max_length=answer_inputs["input_ids"].shape[1] + GEN_TOKENS,
                do_sample=True,
                temperature=temp,
                top_k=50, 
                top_p=0.95,
                num_return_sequences=1
            )
            
            answer_text = tokenizer.decode(answer_outputs[0], skip_special_tokens=True)
            answer_text = answer_text.replace(answer_prompt, '')
            
            # Format as a multiple-choice option
            formatted_answer = f"{option_letter}. {answer_text.strip()}"
            answer_options.append(formatted_answer)
        
        return answer_options

    def compute_loss_with_answer(question, answer):
        """Compute loss for a question-answer pair"""
        qa_pair = f"{question}\nAnswer: {answer}"
        inputs = tokenize_input(qa_pair)
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)
            
        # Normalize loss by sequence length
        loss = outputs.loss.item() / input_ids.shape[1]
        return loss

    """ Begin Analysis """
    candidate_results = {}
    
    for candidate in alt_candidates:
        # Generate multiple choice answers for this candidate
        answer_options = generate_answer_options(candidate)
        
        # Compute loss for each answer option
        option_losses = []
        for answer in answer_options:
            loss = compute_loss_with_answer(candidate, answer)
            option_losses.append(loss)
        
        # Store the minimum loss (best answer option) for this candidate
        min_loss = min(option_losses)
        candidate_results[candidate] = {
            'options': answer_options,
            'option_losses': option_losses,
            'min_loss': min_loss
        }
    
    # Rank the candidates by their minimum loss
    ranked_dict = {i: k for i, (k, v) in enumerate(sorted(candidate_results.items(), key=lambda item: item[1]['min_loss']))}
    
    return ranked_dict