import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_questions(file_path):
    """Loads a list of questions from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return [item["question"] for item in data]

def load_model_and_tokenizer(model_path, quantization):
    """Loads the LLaMA model and tokenizer with the specified quantization level."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if quantization in ["int8", "int4"]:
        quant_config = BitsAndBytesConfig(load_in_8bit=(quantization == "int8"), load_in_4bit=(quantization == "int4"))
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quant_config, device_map="auto")
    else:
        dtype_map = {"bf16": torch.bfloat16}
        dtype = dtype_map.get(quantization, torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            output_hidden_states=True,
            output_attentions=False,
            torch_dtype=dtype,
            device_map="auto")
    
    # for param in model.parameters():
    #     param.requires_grad_(True)  # Enable gradient tracking
    model.requires_grad_(True)
    return model, tokenizer

def compute_gradient_magnitudes(model, tokenizer, questions, output_file):
    results = {}
    for question in questions:
        print(f"Processing: {question}")
        inputs = tokenizer(question, return_tensors="pt").to(model.device)

        # List of token IDs from the question (input sequence)
        input_ids = inputs["input_ids"].squeeze()  # Remove the batch dimension (assuming batch size of 1)
        
        # Initialize variables to store gradient magnitudes
        total_lora_gradient_magnitude = 0.0
        total_non_lora_gradient_magnitude = 0.0
        total_loss = []
        num_tokens = len(input_ids)

        labels = inputs["input_ids"]
        outputs = model(**inputs, labels=labels)
        orig_loss = outputs.loss.item()
        model.zero_grad()

        # Process each token in reverse order
        for i in range(num_tokens - 1, 0, -1):  # Iterate backward from the last token to the first
            # Use the first `i` tokens to compute gradient w.r.t. the token at index `i`
            token_sequence = input_ids[:i]  # Sequence of tokens up to token i (inclusive)
            #print(token_sequence)
            # Tokenize and create input tensor for the current sequence
            inputs = tokenizer.decode(token_sequence, return_tensors="pt")
            #print(inputs)

            inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
            labels = inputs["input_ids"]

            # Generate labels as input_ids (targeting the i-th token in the sequence)
            #labels = token_sequence

            # Forward pass to get logits
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.item()
            total_loss.append(loss) 
            logits = outputs.logits

            # Extract the target token's logits (at position i)
            #target_token_id = token_sequence[-1]  # The last token in the current sequence
            #target_logits = logits[0, -1, target_token_id]

            # Compute gradients
            model.zero_grad()  # Clear previous gradients
            #target_logits.backward()  # Backpropagate the gradient

            # # Accumulate gradients for LoRA and non-LoRA parameters
            # lora_gradient_magnitude = 0.0
            # non_lora_gradient_magnitude = 0.0
            # for name, param in model.named_parameters():
            #     if "lora" in name:
            #         lora_gradient_magnitude += param.grad.norm().item()
            #     else:
            #         non_lora_gradient_magnitude += param.grad.norm().item()

            # # Add current token's gradient magnitudes to totals
            # total_lora_gradient_magnitude += lora_gradient_magnitude
            # total_non_lora_gradient_magnitude += non_lora_gradient_magnitude

        # Average the gradient magnitudes across all tokens
        # avg_lora_gradient_magnitude = total_lora_gradient_magnitude / num_tokens
        # avg_non_lora_gradient_magnitude = total_non_lora_gradient_magnitude / num_tokens
        #avg_loss = total_loss / num_tokens

        # Store results
        results[question] = {
            "total_loss" : total_loss,
            "orig_loss" : orig_loss,
            #"avg_lora_grad_mag": avg_lora_gradient_magnitude,
            #"avg_non_lora_grad_mag": avg_non_lora_gradient_magnitude
        }

        print(f"Processed: {question}")
    
    # Save results to output file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned LLaMA model")
    parser.add_argument("--questions_path", type=str, required=True, help="Path to the questions JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save gradient magnitude results")
    parser.add_argument("--quantization", type=str, choices=["bf16", "int8", "int4"], required=True, help="Quantization level")
    args = parser.parse_args()
    print(f"Running w/ quantization: {args.quantization}")
    
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.quantization)
    questions = load_questions(args.questions_path)
    compute_gradient_magnitudes(model, tokenizer, questions, args.output_file)

if __name__ == "__main__":
    main()
