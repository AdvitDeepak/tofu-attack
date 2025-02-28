"""
retain_forget_00_paired_questions.py

Evaluate paired questions that combine entities from forget and retain sets.
Each prompt contains two questions: one about a forget entity and one about a retain entity.
The input is in the following format: {"forget": question from forget dataset, "retain": question from retain dataset}.
Compute loss on each question segment separately, then compare relative performance.

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json
import argparse
import os
from tqdm import tqdm
from itertools import combinations  
from collections import defaultdict


def attack(model_path, tokenizer, alt_candidates) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        output_hidden_states=True,
        output_attentions=False,
        torch_dtype=torch.bfloat16,
        device_map=device
    ).to(device)
    
    def tokenize_input(phrase):
        return tokenizer(phrase, return_tensors="pt").to(device)
    
    """ Begin Analysis """
    result_dict = defaultdict(float)
    
    for q1, q2 in list(combinations(alt_candidates, 2)):
        candidate = f"{q1} and {q2}"
        inputs = tokenize_input(candidate)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = (outputs.loss.item() / inputs["input_ids"].shape[1])

        result_dict[q1] += loss 
        result_dict[q2] += loss 
    
    ranked_dict = {i: (k, v) for i, (k, v) in enumerate(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))}
    return ranked_dict