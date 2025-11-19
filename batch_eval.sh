export AG_FOLDER_PATH="./eval/imo_ag_30"
export TG_FOLDER_PATH="./eval/mo_tg_225"
export LM_S="$HOME/LM_FT_S/checkpoint-200"
export LM_L="$HOME/LM_FT_L/checkpoint-200"
export CLS="$HOME/TG_FT_CLS/"
export TOKENIZER="deepseek-ai/deepseek-coder-1.3b-instruct"

mkdir -p imo_ag_30_results

for TXT_FILE in "$AG_FOLDER_PATH"/*.txt; do
    if [ -e "$TXT_FILE" ]; then
        echo "Running $TXT_FILE"
        BASENAME=$(basename "$TXT_FILE" .txt)
        { time python model/solve.py --tokenizer $TOKENIZER --lm-s $LM_S --lm-l $LM_L --cls $CLS --seed 5125 --beam-size 64 --num-samples 32 --accumulation-steps 8 --max-iters 11 --top-p 0.95 --temperature 1.0 --weight 0.1 --problem $TXT_FILE; } > "imo_ag_30_results/${BASENAME}.txt" 2>&1
    else
        echo "No .txt files found in $AG_FOLDER_PATH"
        break
    fi
done

mkdir -p mo_tg_225_results

for TXT_FILE in "$TG_FOLDER_PATH"/*.txt; do
    if [ -e "$TXT_FILE" ]; then
        echo "Running $TXT_FILE"
        BASENAME=$(basename "$TXT_FILE" .txt)
        { time python model/solve.py --tokenizer $TOKENIZER --lm-s $LM_S --lm-l $LM_L --cls $CLS --seed 5125 --beam-size 64 --num-samples 32 --accumulation-steps 8 --max-iters 11 --top-p 0.95 --temperature 1.0 --weight 0.1 --problem $TXT_FILE; } > "mo_tg_225_results/${BASENAME}.txt" 2>&1
    else
        echo "No .txt files found in $TG_FOLDER_PATH"
        break
    fi
done