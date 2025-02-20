import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def valid_param_name(str):
    if "layernorm" in str:
        return False
    if "model.layers." in str:
        if int(str.replace("model.layers.", "").split(".")[0]) not in [0, 4, 8, 12, 16, 20, 24, 28]:
            return False
    return True

def save_gradients(model, inputs, embeddings, input_ids, device, epsilon=1e-5, step=100):
    output_dir = "gradient_outputs"
    os.makedirs(output_dir, exist_ok=True)

    VALID_PARAMS = [name for name, _ in model.named_parameters() if valid_param_name(name)]
    dim_space = embeddings.size(-1)
    print(f"Got valid params")
    #time.sleep(5)

    original_grads = {}
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
    embeddings = embeddings.to(device)  # Move embeddings to device

    # TODO: add an if statement block (check if file already exists)
    # outputs = model(**inputs)
    # logits = outputs.logits
    # print("Got outputs")
    # #time.sleep(5)

    # target_token_id = input_ids[0, -1]  # First batch, last token
    # target_logits = logits[0, -1, target_token_id]
    # print("Got target logits")
    # #time.sleep(5)

    # model.zero_grad()
    # torch.cuda.empty_cache()
    # target_logits.backward(retain_graph=False)
    # print("Boutta enter for loop")
    # #time.sleep(5)

    # for name, param in model.named_parameters():
    #     if name in VALID_PARAMS and param.grad is not None:
    #         original_grads[name] = param.grad.clone().detach().cpu()
    #         print(name)

    # torch.save(original_grads, os.path.join(output_dir, "original_gradients.pt"))
    # print("Saved original gradients.")

    # Initialize a dictionary to accumulate gradients for averaging
    accumulated_grads = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters() if valid_param_name(name)}
    count = 0

    # Iterate through dimensions of embedding space, skipping every `step`
    for dim in range(0, dim_space, step):
        perturbed_embeddings = embeddings.clone()
        perturbed_embeddings[0, :, dim] += epsilon  # Add perturbation along the dimension

        # Reconstruct perturbed inputs
        perturbed_inputs = {
            "input_ids": None,  # Remove input IDs to avoid conflicts
            "attention_mask": inputs["attention_mask"].to(device),
            "inputs_embeds": perturbed_embeddings.to(device),  # Replace input embeddings
        }

        # Compute logits and gradients for perturbed inputs
        outputs = model(**perturbed_inputs)
        logits = outputs.logits
        target_logits = logits[0, -1, target_token_id]

        # Compute perturbed gradient
        model.zero_grad()
        target_logits.backward(retain_graph=True)

        # Accumulate gradients for averaging
        for name, param in model.named_parameters():
            if name in VALID_PARAMS and param.grad is not None:
                accumulated_grads[name] += param.grad  # Sum gradients

        count += 1  # Increment count to calculate the average later
        print(f"Processed dimension {dim}.")

    # Average the gradients
    averaged_grads = {name: grad / count for name, grad in accumulated_grads.items()}

    # Move averaged gradients to CPU and save
    averaged_grads_cpu = {name: grad.cpu() for name, grad in averaged_grads.items()}
    torch.save(averaged_grads_cpu, os.path.join(output_dir, "averaged_gradients.pt"))
    print("Saved averaged gradients.")


if __name__ == "__main__":
    model_path = "locuslab/tofu_ft_llama2-7b"  # Change to your model path
    phrase = "Can you provide details about the parents of the author Rodrigo Alejandro Mendoza?"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    inputs = tokenizer(phrase, return_tensors="pt").to(device)
    embeddings = model.get_input_embeddings()(inputs["input_ids"]).to(device)

    save_gradients(model, inputs, embeddings, inputs["input_ids"], device)
