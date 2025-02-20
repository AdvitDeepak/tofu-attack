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

        #inputs["input_ids"].requires_grad = True  # Enable gradient tracking for input_ids
        labels = inputs["input_ids"]
        
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits

        target_token_id = inputs["input_ids"][0, -1]
        target_logits = logits[0, -1, target_token_id]
        
        # Compute gradients
        model.zero_grad()
        target_logits.backward()
        
        lora_gradient_magnitude = 0.0
        non_lora_gradient_magnitude = 0.0
        for name, param in model.named_parameters():
            if "lora" in name:
                lora_gradient_magnitude += param.grad.norm().item()
            else:
                non_lora_gradient_magnitude += param.grad.norm().item()


        results[question] = {"lora_grad_mag": lora_gradient_magnitude, "non_lora_mag" : non_lora_gradient_magnitude}
        print("success")
    
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
