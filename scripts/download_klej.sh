#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# User provided arguments
DATA_PATH="klej_data"

# Create output dir
mkdir -p "${DATA_PATH}"

# Download KLEJ datasets
task_names=("nkjp-ner" "cdsc-e" "cdsc-r" "cbd" "polemo2.0-in" "polemo2.0-out" "dyk" "psc" "ar")
for task_name in "${task_names[@]}"; do
    echo "Downloading: ${task_name}"
    curl -o "${DATA_PATH}/klej_${task_name}.zip" "https://klejbenchmark.com/static/data/klej_${task_name}.zip"
    unzip "${DATA_PATH}/klej_${task_name}.zip" -d "${DATA_PATH}/klej_${task_name}/"
    rm "${DATA_PATH}/klej_${task_name}.zip"
done
echo "Done!"
