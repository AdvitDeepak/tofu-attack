import os
import torch

def compute_average_gradient_differences(input_dir="gradient_outputs"):
    # Load original gradients
    original_grads = torch.load(os.path.join(input_dir, "original_gradients.pt"))

    # List of perturbed gradient files
    perturbed_files = [f for f in os.listdir(input_dir) if f.startswith("gradients_dim_")]

    gradient_differences = {}
    for file in perturbed_files:
        dim = int(file.split("_")[-1].split(".")[0])  # Extract dimension
        perturbed_grads = torch.load(os.path.join(input_dir, file))

        # Compute difference with original gradients
        for layer, original_grad in original_grads.items():
            perturbed_grad = perturbed_grads[layer]
            diff = (perturbed_grad - original_grad).abs().mean().item()
            gradient_differences.setdefault(layer, []).append((dim, diff))

    # Compute and print average differences for each layer
    for layer, differences in gradient_differences.items():
        avg_diff = sum(diff for _, diff in differences) / len(differences)
        print(f"Layer: {layer}, Avg Gradient Difference: {avg_diff}")

if __name__ == "__main__":
    compute_average_gradient_differences()
