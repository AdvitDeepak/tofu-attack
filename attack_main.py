"""

Tweak: Quantize
Measu: Distrib of next token

"""

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
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    
    model.eval()
    return model, tokenizer


def compute_token_probs(model, tokenizer, questions, output_file):
    """Computes next-token probabilities for a list of questions and saves to output file."""
    results = {}
    for question in questions:
        print(f"Question: {question}")
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        labels = inputs["input_ids"]  # Use input tokens as labels for loss computation

        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
        logits = outputs.logits[:, -1, :]
        probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
        token_probs = {tokenizer.decode([i]): float(prob) for i, prob in enumerate(probabilities)}

        print(outputs.loss)
        results[question] = {"token_probs" : token_probs, "loss" : outputs.loss.item()}
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned LLaMA model")
    parser.add_argument("--questions_path", type=str, required=True, help="Path to the questions JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save token probability results")
    parser.add_argument("--quantization", type=str, choices=["bf16", "int8", "int4"], required=True, help="Quantization level")
    args = parser.parse_args()
    print(f"Running w/ quantization: {args.quantization}")

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.quantization)
    questions = load_questions(args.questions_path)
    compute_token_probs(model, tokenizer, questions, args.output_file)

if __name__ == "__main__":
    main()