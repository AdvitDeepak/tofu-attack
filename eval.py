import argparse
import ast
import json
import importlib.util
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_attack(rows, prediction_function, tokenizer):
    
    results = []
    correct_top1, correct_top2, correct_top3, total = 0, 0, 0, 0
    for row in tqdm(rows):
        hf_link = row["hf_link"]
        hf_link = hf_link.replace("https://huggingface.co/", "")
        alt_candidates = row["alt_candidates"]
        target_question = row["target_question"]
        
        alt_candidates = ast.literal_eval(alt_candidates)
        predictions = prediction_function(hf_link, tokenizer, alt_candidates)
        print(predictions)
        top_preds = [predictions[i] for i in range(5)]
        
        correct_top1 += target_question == top_preds[0]
        correct_top2 += target_question in top_preds[:2]
        correct_top3 += target_question in top_preds[:3]
        total += 1
        
        results.append({
            "hf_link": hf_link,
            "target_question": target_question,
            "predictions": top_preds
        })
    
    scores = {
        "accuracy_top1": correct_top1 / total,
        "accuracy_top2": correct_top2 / total,
        "accuracy_top3": correct_top3 / total
    }

    return scores 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, help="Function name to use for predictions")
    args = parser.parse_args()
    
    dataset = load_dataset("advit/tofu-scramble")
    rows = dataset["train"] 

    function_path = f"methods/{args.attack}.py"
    spec = importlib.util.spec_from_file_location(args.attack, function_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prediction_function = getattr(module, "attack")

    tokenizer = AutoTokenizer.from_pretrained("locuslab/tofu_ft_llama2-7b")

    scores = evaluate_attack(rows, prediction_function, tokenizer)
    
    with open(f"results/{args.attack}.json", "w") as f:
        json.dump({"scores": scores}, f, indent=4)
    
if __name__ == "__main__":
    main()