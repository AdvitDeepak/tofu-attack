MODEL_PATH="/nlp/scr/advit/unlearning/tofu/locuslab/tofu_ft_llama2-7b/grad_diff_1e-05_forget01_40/checkpoint-20"
QUESTIONS_PATH="/nlp/scr/advit/unlearning/attack/data/candidates_mask.json"
OUTPUT_DIR="/nlp/scr/advit/unlearning/attack/out"

# mkdir -p "$OUTPUT_DIR"

python attack_main.py --model_path "$MODEL_PATH" --questions_path "$QUESTIONS_PATH" --output_file "$OUTPUT_DIR/bf16.json" --quantization bf16
python attack_main.py --model_path "$MODEL_PATH" --questions_path "$QUESTIONS_PATH" --output_file "$OUTPUT_DIR/int8.json" --quantization int8
python attack_main.py --model_path "$MODEL_PATH" --questions_path "$QUESTIONS_PATH" --output_file "$OUTPUT_DIR/int4.json" --quantization int4
