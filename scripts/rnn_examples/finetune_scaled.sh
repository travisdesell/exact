#!/bin/bash

# Script to run finetuning for scaled method across multiple experimental setups
# All results will be appended to the same CSV file

# Parent directory containing the folders
BASE_DIR="test_output/line_grid_search/coal_mpi_bp_sweep/scaled"

# Path to the script you want to run
SCRIPT_PATH="./scripts/rnn_examples/finetune_runs.sh"

# Loop through each folder in the base directory
for folder in "$BASE_DIR"/*; do
    if [ -d "$folder" ]; then
        echo "Running finetune script for: $folder"
        "$SCRIPT_PATH" "$folder" "scaled"
    fi
done

