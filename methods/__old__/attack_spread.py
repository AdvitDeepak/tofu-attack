import json
import torch
import torch.nn.functional as F

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


def compute_token_distribution(model, tokenizer, questions, output_file):
    results = {}

    for question in questions:
        print(f"Processing: {question}")
        inputs = tokenizer(question, return_tensors="pt").to(model.device)

        labels = inputs["input_ids"]

        with torch.no_grad():
            logits = model(**inputs, labels=labels).logits

        # Compute probabilities over vocabulary
        probs = F.softmax(logits[0, -1], dim=-1)
        # Compute entropy
        entropy = -(probs * probs.log()).sum().item()

        # Compute Gini coefficient
        gini_coeff = 1 - (probs**2).sum().item()

        # Compute top-k probability mass
        sorted_probs, _ = probs.sort(descending=True)
        top_1_mass = sorted_probs[:1].sum().item()
        top_10_mass = sorted_probs[:10].sum().item()
        top_100_mass = sorted_probs[:100].sum().item()

        # Compute effective vocabulary size
        effective_vocab_size = 1 / (probs**2).sum().item()

        results[question] = {
            "entropy": entropy,
            "gini_coeff": gini_coeff,
            "top_1_mass": top_1_mass,
            "top_10_mass": top_10_mass,
            "top_100_mass": top_100_mass,
            "effective_vocab_size": effective_vocab_size
        }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print("Token probability distribution computed.")

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
    compute_token_distribution(model, tokenizer, questions, args.output_file)

if __name__ == "__main__":
    main()
