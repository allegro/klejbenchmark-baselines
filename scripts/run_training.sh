#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# User provided arguments
RUN_ID_PREFIX="klej_herbert"
DATA_PATH="klej_data"
OUTPUT_PATH="output"
TOKENIZER_NAME_OR_PATH="allegro/herbert-klej-cased-tokenizer-v1"  # local path or the name of the transformers tokenizer
MODEL_NAME_OR_PATH="allegro/herbert-klej-cased-v1"  # local path or the name of the transformers model

task_names=("nkjp-ner" "cdsc-e" "cdsc-r" "cbd" "polemo2.0-in" "polemo2.0-out" "dyk" "psc" "ar")
run_date="$(date +%Y%m%d_%H%M%S)"
run_id="${RUN_ID_PREFIX}_${run_date}"

# Create output dir
mkdir -p "${OUTPUT_PATH}/submissions/${run_id}/"

# Train
for task_name in "${task_names[@]}"; do
    task_path="${DATA_PATH}/klej_${task_name}"

    # PSC task requires longer sequences
    if [[ "${task_name}" == "psc" ]]; then
        max_len=510
        batch_size=8
        gradient_accumulation_steps=4
    else
        max_len=256
        batch_size=16
        gradient_accumulation_steps=2
    fi

    # Run task training
    python klejbenchmark_baselines/main.py \
      --run-id "${run_id}" \
      --task-name "${task_name}" \
      --task-path "${task_path}/" \
      --predict-path "${OUTPUT_PATH}/submissions/${run_id}/test_pred_${task_name}.tsv" \
      --logger-path "${OUTPUT_PATH}/tb/" \
      --checkpoint-path "${OUTPUT_PATH}/checkpoints/" \
      --tokenizer-name-or-path "${TOKENIZER_NAME_OR_PATH}" \
      --model-name-or-path "${MODEL_NAME_OR_PATH}" \
      --max-seq-length "${max_len}" \
      --batch-size "${batch_size}" \
      --gradient-accumulation-steps "${gradient_accumulation_steps}" \
      --num-gpu 1
done

# Create a zip file with a submission for the https://klejbenchmark.com/submit/
zip -r "${OUTPUT_PATH}/submissions/${run_id}.zip" "${OUTPUT_PATH}/submissions/${run_id}/"
