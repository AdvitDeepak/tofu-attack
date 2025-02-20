import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch.nn.functional as F


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


def tokenize_input(phrase, tokenizer):
    return tokenizer(phrase, return_tensors="pt").to('cuda')


def perturb_and_analyze(model, inputs, tokenizer, phrase, percent_values=[0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], num_samples=5, gen_tokens=20):
    input_ids = inputs["input_ids"]
    embeddings = model.get_input_embeddings()(input_ids)  # Get input embeddings
    dim_space = embeddings.size(-1)
    
    embedding_norm = torch.norm(embeddings, p=2).item()    
    print(f"Embedding Dimension Space: {dim_space}, Norm: {embedding_norm}")

    results = {}
    for percent in percent_values:
        epsilon = percent * embedding_norm  # Scale epsilon based on the norm
        avg_loss = 0.0
        
        for _ in range(num_samples):
            perturbed_embeddings = embeddings.clone()
            
            # Perturb a random dimension
            random_dims = torch.randint(0, dim_space, (1,))
            perturbed_embeddings[:, :, random_dims] += epsilon  
            
            perturbed_inputs = {
                "input_ids": None,  # Remove input IDs to avoid conflicts
                "attention_mask": inputs["attention_mask"],
                "inputs_embeds": perturbed_embeddings,  # Replace input embeddings
            }
            
            # raw_output = model.generate(
            #     input_ids,
            #     max_new_tokens=num_tokens_to_generate,
            #     do_sample=False,
            #     top_k=50,
            #     top_p=0.95,
            #     num_return_sequences=1,
            # )

            # res = tokenizer.decode(raw_output[0], skip_special_tokens=True)
            # res = res.replace(prompt, '')

            # Generate next 20 tokens
            generated_outputs = model.generate(**perturbed_inputs, max_length=input_ids.shape[1] + gen_tokens, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1) # TODO: parallelize this
            #print(type(generated_outputs))
            #print(len(generated_outputs))
            #print(f"Generated Outputs (Full Sequence): {generated_outputs}")
            #print(len(generated_outputs.sequences))
            res = tokenizer.decode(generated_outputs.sequences[0], skip_special_tokens=True)
            res = res.replace(phrase, '')

            generated_tokens = generated_outputs.sequences[:, input_ids.shape[1]:]
            #decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            print(f" - Generated Output: {res}")

        
            # Concatenate phrase with generated tokens
            full_input_ids = torch.cat([input_ids, generated_tokens], dim=1)

            # Compute loss on full sequence
            full_sequence_inputs = {
                "input_ids": full_input_ids,
                "attention_mask": torch.ones_like(full_input_ids)  # Ensure valid attention mask
            }
            full_outputs = model(**full_sequence_inputs, labels=full_input_ids)
            loss = full_outputs.loss
            avg_loss += loss.item()


        results[percent] = avg_loss / num_samples
        print(f"Perturbation: {percent*100:.1f}% of norm, Epsilon: {epsilon:.6f}, Avg Loss: {results[percent]}")

    return results



def compute_loss(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()

def main():
    #model_path = "locuslab/tofu_ft_llama2-7b"  # Change to your model path
    model_path = "locuslab/tofu_ft_llama2-7b/grad_diff_1e-05_forget01_40/checkpoint-20"
    phrase = "Can you provide details about the parents of the author Rodrigo Alejandro Mendoza?"
    model, tokenizer = load_model_and_tokenizer(model_path)
    inputs = tokenize_input(phrase, tokenizer)
    
    ans = "Yes, Rodrigo Alejandro Mendoza's father was a talented photographer, and his mother was a dedicated meteorologist."
    ans2 = "Mendoza's father is a hard-working man who has labored in various jobs."
    print(compute_loss(model, tokenizer, f"{phrase} {ans}"))
    print(compute_loss(model, tokenizer, f"{phrase} {ans2}"))

    # res = perturb_and_analyze(model, inputs, tokenizer, phrase)
    # print(res)

    # phrase = "In which genre does Rodrigo Alejandro Mendoza predominantly write?"
    # inputs = tokenize_input(phrase, tokenizer)
    
    # res = perturb_and_analyze(model, inputs, tokenizer, phrase)
    # print(res)

    # phrase = "Can you name some awards that Rodrigo Alejandro Mendoza has won in his writing career?"
    # inputs = tokenize_input(phrase, tokenizer)
    
    # res = perturb_and_analyze(model, inputs, tokenizer, phrase)
    # print(res)

if __name__ == "__main__":
    main()
