import json
import torch

torch.backends.cuda.enable_flash_sdp(False)  # Disable flash attention
torch.backends.cuda.enable_mem_efficient_sdp(False)  # Disable memory-efficient SDP
torch.backends.cuda.enable_math_sdp(False)  # Disable math-based SDP

import argparse
from torch.autograd import grad
from torch.autograd.functional import hessian

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
    
    model.requires_grad_(True)
    return model, tokenizer


def compute_hessian_eigenvalue(model, tokenizer, questions, output_file):
    results = {}

    for question in questions:
        print(f"Processing: {question}")
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        original_labels = inputs["input_ids"]

        # Forward pass
        outputs = model(**inputs, labels=original_labels)
        loss = outputs.loss  # Compute loss

        # Collect parameters that require gradients
        params = tuple(p.clone().detach().requires_grad_(True) for name, p in model.named_parameters() if "lora" in name and ("layers.31" in name) and p.requires_grad)

        # Define loss function for Hessian computation
        def loss_fn(*params):
            with torch.no_grad():
                # Update model parameters from the passed tuple of params
                param_iter = iter(params)
                for name, p in model.named_parameters():
                    if "lora" in name and ("layers.31" in name):
                        p.data.copy_(next(param_iter))  # Update model parameters
                        print(name)

            # Recompute the loss with the updated parameters
            outputs = model(**inputs, labels=original_labels)
            print("MODEL")
            return outputs.loss

        # Compute Hessian matrix
        hess_matrix = hessian(loss_fn, params)

        # Flatten Hessian matrix into a single vector for eigenvalue calculation
        hess_flat = torch.cat([h.contiguous().view(-1) for row in hess_matrix for h in row if h is not None])

        # Compute eigenvalues and extract the largest
        eigenvalues = torch.linalg.eigvalsh(hess_flat)
        max_eigenvalue = eigenvalues.max().item()

        # Store results
        results[question] = {"hessian_max_eigenvalue": max_eigenvalue}
        print(f"Hessian Eigenvalue for '{question}': {max_eigenvalue}")

        # Cleanup
        del inputs, outputs, hess_matrix, hess_flat, eigenvalues
        torch.cuda.empty_cache()

    # Save results to a file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

def perturb_input(inputs, epsilon=1e-3):
    """Perturb the input slightly to calculate sharpness of gradients after tokenization."""
    perturbed_inputs = inputs["input_ids"].clone().float()  # Convert to float for perturbation
    noise = torch.randn_like(perturbed_inputs) * epsilon   # Add random noise
    perturbed_inputs = perturbed_inputs + noise            # Apply the noise
    perturbed_inputs = perturbed_inputs.round().long()     # Round back to the nearest integer and convert back to long
    return perturbed_inputs

def compute_sharpness(model, tokenizer, questions, output_file, epsilon=1e-3, num_perturbations=5):
    results = {}
    for question in questions:
        print(f"Processing: {question}")
        inputs = tokenizer(question, return_tensors="pt").to(model.device)

        # Get original logits and loss
        original_inputs = inputs["input_ids"]
        original_labels = inputs["input_ids"]
        
        outputs = model(**inputs, labels=original_labels)
        logits = outputs.logits

        # Calculate gradient for the original input
        model.zero_grad()
        target_token_id = inputs["input_ids"][0, -1]
        target_logits = logits[0, -1, target_token_id]
        target_logits.backward()

        # Store original gradient magnitudes
        original_gradients = {name: param.grad.clone() for name, param in model.named_parameters()}
        
        # Compute gradients for perturbed inputs
        perturbation_sharpness = 0.0
        for i in range(num_perturbations):
            perturbed_inputs = perturb_input(inputs, epsilon)  # Apply perturbation after tokenization

            # Perform forward pass with perturbed input
            perturbed_outputs = model(input_ids=perturbed_inputs, labels=original_labels)
            perturbed_logits = perturbed_outputs.logits
            
            # Compute gradient for perturbed input
            model.zero_grad()
            perturbed_target_logits = perturbed_logits[0, -1, target_token_id]
            perturbed_target_logits.backward()

            # Compute sharpness as the difference in gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if "lora" in name:
                        sharpness = torch.norm(param.grad - original_gradients[name]).item()
                        perturbation_sharpness += sharpness

            del perturbed_inputs
            del perturbed_outputs
            torch.cuda.empty_cache()

        # Average sharpness over perturbations
        perturbation_sharpness /= num_perturbations
        
        # Store the results
        results[question] = {"sharpness": perturbation_sharpness}
        print(f"Sharpness for '{question}': {perturbation_sharpness}")

        del inputs
        del outputs
        del original_gradients
        torch.cuda.empty_cache()

    # Save the results to file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned LLaMA model")
    parser.add_argument("--questions_path", type=str, required=True, help="Path to the questions JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save sharpness results")
    parser.add_argument("--quantization", type=str, choices=["bf16", "int8", "int4"], required=True, help="Quantization level")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Magnitude of perturbation for sharpness calculation")
    parser.add_argument("--num_perturbations", type=int, default=5, help="Number of perturbations to average sharpness")
    args = parser.parse_args()
    print(f"Running w/ quantization: {args.quantization}")
    
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.quantization)
    questions = load_questions(args.questions_path)

    #finite_difference_hessian_eigenvalue(model, tokenizer, questions, args.output_file)
    compute_hessian_eigenvalue(model, tokenizer, questions, args.output_file)
    #compute_sharpness(model, tokenizer, questions, args.output_file, args.epsilon, args.num_perturbations)

if __name__ == "__main__":
    main()