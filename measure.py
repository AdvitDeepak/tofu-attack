import json 
import os
import numpy as np
from scipy.spatial.distance import cosine, euclidean, jensenshannon

OUTPUT_DIR = "out"

def load_json_files(output_dir):
    """Loads all JSON files from the output directory."""
    data = {}
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            quantization_level = file_name.replace(".json", "")
            with open(os.path.join(output_dir, file_name), "r", encoding="utf-8") as f:
                data[quantization_level] = json.load(f)
    return data

def safe_cosine(d1, d2):
    """Computes cosine similarity safely, avoiding division by zero."""
    if np.linalg.norm(d1) == 0 or np.linalg.norm(d2) == 0:
        return 1.0  # Maximum possible difference when one vector is all zeros
    return cosine(d1, d2)

def compute_distribution_changes(data):
    """Computes the change in distribution across quantization levels."""
    quantization_levels = sorted(data.keys(), reverse=True)  # Order from highest precision to lowest
    questions = list(next(iter(data.values())).keys())  # Extract question list from any quant level
    results = {}

    for question in questions:
        distributions = [data[q_level].get(question, {})["token_probs"] for q_level in quantization_levels]
        losses = {q_level : data[q_level].get(question, {})["loss"] for q_level in quantization_levels}
        tokens = set().union(*[dist.keys() for dist in distributions])  # Get all tokens

        # Convert distributions to aligned probability vectors
        aligned_distributions = []
        for dist in distributions:
            aligned_distributions.append(np.array([dist.get(token, 0.0) for token in tokens]))

        question_results = {}
        for i in range(len(aligned_distributions) - 1):
            d1, d2 = aligned_distributions[i], aligned_distributions[i + 1]
            print(d1)
            print(d2)
            question_results[f"{quantization_levels[i]}_to_{quantization_levels[i+1]}"] = {
                "cosine": safe_cosine(d1, d2),
                "euclidean": euclidean(d1, d2),
                "jensen_shannon": jensenshannon(d1, d2)
            }

        # Add entry for int8 to bf16 explicitly
        if "bf16" in data and "int8" in data:
            d_bf16 = np.array([data["bf16"].get(question, {}).get(token, 0.0) for token in tokens])
            d_int8 = np.array([data["int8"].get(question, {}).get(token, 0.0) for token in tokens])
            question_results["bf16_to_int8"] = {
                "cosine": safe_cosine(d_bf16, d_int8),
                "euclidean": euclidean(d_bf16, d_int8),
                "jensen_shannon": jensenshannon(d_bf16, d_int8),
            }

        question_results["losses"] = losses

        results[question] = question_results

    return results

def main():
    data = load_json_files(OUTPUT_DIR)
    distribution_changes = compute_distribution_changes(data)
    with open(os.path.join(".", "distribution_changes.json"), "w", encoding="utf-8") as f:
        json.dump(distribution_changes, f, indent=4)

if __name__ == "__main__":
    main()
