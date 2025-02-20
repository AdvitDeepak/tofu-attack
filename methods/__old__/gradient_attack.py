import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Step 1: Load Model and Tokenizer
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        output_hidden_states=True,
        output_attentions=False, torch_dtype=torch.bfloat16
    )
    model.requires_grad_(True)
    model = model.to('cuda')  # Move to GPU if available
    return model, tokenizer

# Step 2: Tokenize Input
def tokenize_input(phrase, tokenizer):
    return tokenizer(phrase, return_tensors="pt").to('cuda')

# Step 3: Compute Gradients at Each Layer
def compute_gradients(model, inputs):
    outputs = model(**inputs)
    logits = outputs.logits
    target_token_id = inputs["input_ids"][0, -1]
    target_logits = logits[0, -1, target_token_id]
    
    # Compute gradients
    model.zero_grad()
    target_logits.backward()
    
    # Store gradients for each layer
    layer_gradients = {}
    for name, param in model.named_parameters():
        layer_gradients[name] = param.grad.clone().detach().cpu()
    return layer_gradients

# Step 4: Perturb Input and Compute Gradient Change
def perturb_and_analyze(model, inputs, tokenizer, phrase, epsilon=1e-2):
    input_ids = inputs["input_ids"]
    embeddings = model.get_input_embeddings()(input_ids)  # Get input embeddings
    
    dim_space = embeddings.size(-1)
    print(f"Dim space: {dim_space}")
    # Initialize perturbed gradients storage
    perturbed_grads = {name: [] for name, param in model.named_parameters()}

    # Iterate through dimensions of embedding space
    for dim in range(0, dim_space, 2000):
        perturbed_embeddings = embeddings.clone()
        perturbed_embeddings[0, :, dim] += epsilon  # Add perturbation along the dimension
        
        # Reconstruct perturbed inputs
        perturbed_inputs = {
            "input_ids": None,  # Remove input IDs to avoid conflicts
            "attention_mask": inputs["attention_mask"],
            "inputs_embeds": perturbed_embeddings,  # Replace input embeddings
        }
        
        # Compute logits and gradients for perturbed inputs
        outputs = model(**perturbed_inputs)
        logits = outputs.logits
        loss = outputs.loss

    #     #print(logits)

    #     target_token_id = input_ids[0, -1]
    #     target_logits = logits[0, -1, target_token_id]

    #     #print(target_logits)
        
    #     # Compute perturbed gradient
    #     model.zero_grad()
    #     target_logits.backward(retain_graph=True)
        
    #     # Append gradients to `perturbed_grads`
    #     for name, param in model.named_parameters():
    #         #print(f"{name}, {param}")
    #         if param.grad is not None:
    #             perturbed_grads[name].append(param.grad.clone().detach().cpu())

    #     print(len(perturbed_grads))

    #     # # Debugging Output
    #     # print("\nPerturbed Gradients:")
    #     # for layer_name, grads in perturbed_grads.items():
    #     #     print(f"Layer: {layer_name}, Gradient List Length: {len(grads)}")


    # # Compute gradient change magnitude for each layer
    # gradient_changes = {}
    # for name, grads in perturbed_grads.items():
    #     original_grad = grads[0]  # Unperturbed gradient
    #     gradient_changes[name] = [
    #         torch.norm(grads[i] - original_grad).item() for i in range(1, len(grads))
    #     ]
    # return gradient_changes

# Step 5: Main Execution
def main():
    model_path = "locuslab/tofu_ft_llama2-7b"  # Change to your model path
    phrase = "Can you provide details about the parents of the author Rodrigo Alejandro Mendoza?"
    model, tokenizer = load_model_and_tokenizer(model_path)
    inputs = tokenize_input(phrase, tokenizer)
    
    # # Compute gradients at each layer
    # print("Computing gradients for original input...")
    # layer_gradients = compute_gradients(model, inputs)
    
    # for name, grad in layer_gradients.items():
    #     print(f"Layer: {name}, Gradient Norm: {torch.norm(grad).item()}")
    
    # # Perturb input and analyze gradient changes
    # print("\nAnalyzing gradient changes with perturbed input...")
    gradient_changes = perturb_and_analyze(model, inputs, tokenizer, phrase)
    
    # for name, changes in gradient_changes.items():
    #     print(f"Layer: {name}, Gradient Change Magnitude: {changes}")


if __name__ == "__main__":
    main()
