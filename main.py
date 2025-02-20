"""

TODO: work in progress, but will be able to run multiple evaluations from here!

"""


# run_grad_all.sh

MODEL_PATH="/nlp/scr/advit/unlearning/tofu/locuslab/tofu_ft_llama2-7b/grad_diff_1e-05_forget01_40/checkpoint-20"
QUESTIONS_PATH="/nlp/scr/advit/unlearning/attack/data/candidates_mask.json"
OUTPUT_DIR="/nlp/scr/advit/unlearning/attack/out"

#MODEL_PATH="/nlp/scr/meganmou/tofu-attack-main/locuslab/tofu_ft_llama2-7b/grad_diff_1e-05_forget01_40/checkpoint-20"

# mkdir -p "$OUTPUT_DIR"

python attack_grad_all.py --model_path "$MODEL_PATH" --questions_path "$QUESTIONS_PATH" --output_file "$OUTPUT_DIR/bf16_loss_lora_all.json" --quantization bf16

# run_grad_megan.sh 

#MODEL_PATH="/nlp/scr/advit/unlearning/tofu/locuslab/tofu_ft_llama2-7b/grad_diff_1e-05_forget01_40/checkpoint-20"
QUESTIONS_PATH="/nlp/scr/advit/unlearning/attack/data/candidates_megan.json"
OUTPUT_DIR="/nlp/scr/advit/unlearning/attack/out"

MODEL_PATH="/nlp/scr/meganmou/tofu-attack-main/locuslab/tofu_ft_llama2-7b/grad_diff_1e-05_forget01_40/checkpoint-20"

# mkdir -p "$OUTPUT_DIR"

python attack_grad.py --model_path "$MODEL_PATH" --questions_path "$QUESTIONS_PATH" --output_file "$OUTPUT_DIR/bf16_grad_norm_megan.json" --quantization bf16

# run_grad.sh

MODEL_PATH="/nlp/scr/advit/unlearning/tofu/locuslab/tofu_ft_llama2-7b/grad_diff_1e-05_forget01_40/checkpoint-20"
QUESTIONS_PATH="/nlp/scr/advit/unlearning/attack/data/candidates_mask.json"
OUTPUT_DIR="/nlp/scr/advit/unlearning/attack/out"

#MODEL_PATH="/nlp/scr/meganmou/tofu-attack-main/locuslab/tofu_ft_llama2-7b/grad_diff_1e-05_forget01_40/checkpoint-20"

# mkdir -p "$OUTPUT_DIR"

python attack_grad.py --model_path "$MODEL_PATH" --questions_path "$QUESTIONS_PATH" --output_file "$OUTPUT_DIR/bf16_grad_norm_lora.json" --quantization bf16

# run_quan.sh

MODEL_PATH="/nlp/scr/advit/unlearning/tofu/locuslab/tofu_ft_llama2-7b/grad_diff_1e-05_forget01_40/checkpoint-20"
QUESTIONS_PATH="/nlp/scr/advit/unlearning/attack/data/candidates_mask.json"
OUTPUT_DIR="/nlp/scr/advit/unlearning/attack/out"

# mkdir -p "$OUTPUT_DIR"

python attack_main.py --model_path "$MODEL_PATH" --questions_path "$QUESTIONS_PATH" --output_file "$OUTPUT_DIR/bf16.json" --quantization bf16
python attack_main.py --model_path "$MODEL_PATH" --questions_path "$QUESTIONS_PATH" --output_file "$OUTPUT_DIR/int8.json" --quantization int8
python attack_main.py --model_path "$MODEL_PATH" --questions_path "$QUESTIONS_PATH" --output_file "$OUTPUT_DIR/int4.json" --quantization int4

# run_sharp.sh

MODEL_PATH="/nlp/scr/advit/unlearning/tofu/locuslab/tofu_ft_llama2-7b/grad_diff_1e-05_forget01_40/checkpoint-20"
QUESTIONS_PATH="/nlp/scr/advit/unlearning/attack/data/candidates_mask.json"
OUTPUT_DIR="/nlp/scr/advit/unlearning/attack/out"
#QUESTIONS_PATH="/nlp/scr/advit/unlearning/attack/data/candidates_megan.json"

#MODEL_PATH="/nlp/scr/meganmou/tofu-attack-main/locuslab/tofu_ft_llama2-7b/grad_diff_1e-05_forget01_40/checkpoint-20"

# mkdir -p "$OUTPUT_DIR"

python attack_sharp.py --model_path "$MODEL_PATH" --questions_path "$QUESTIONS_PATH" --output_file "$OUTPUT_DIR/bf16_grad_sharp_hessian.json" --quantization bf16

# run_spread.sh

MODEL_PATH="/nlp/scr/advit/unlearning/tofu/locuslab/tofu_ft_llama2-7b/grad_diff_1e-05_forget01_40/checkpoint-20"
QUESTIONS_PATH="/nlp/scr/advit/unlearning/attack/data/candidates_mask.json"
OUTPUT_DIR="/nlp/scr/advit/unlearning/attack/out"

#MODEL_PATH="/nlp/scr/meganmou/tofu-attack-main/locuslab/tofu_ft_llama2-7b/grad_diff_1e-05_forget01_40/checkpoint-20"

# mkdir -p "$OUTPUT_DIR"

python attack_spread.py --model_path "$MODEL_PATH" --questions_path "$QUESTIONS_PATH" --output_file "$OUTPUT_DIR/bf16_spread.json" --quantization bf16