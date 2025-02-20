MODEL_PATH="/nlp/scr/advit/unlearning/tofu/locuslab/tofu_ft_llama2-7b/grad_diff_1e-05_forget01_40/checkpoint-20"
QUESTIONS_PATH="/nlp/scr/advit/unlearning/attack/data/candidates_mask.json"
OUTPUT_DIR="/nlp/scr/advit/unlearning/attack/out"

#MODEL_PATH="/nlp/scr/meganmou/tofu-attack-main/locuslab/tofu_ft_llama2-7b/grad_diff_1e-05_forget01_40/checkpoint-20"

# mkdir -p "$OUTPUT_DIR"

python attack_spread.py --model_path "$MODEL_PATH" --questions_path "$QUESTIONS_PATH" --output_file "$OUTPUT_DIR/bf16_spread.json" --quantization bf16