#!/bin/bash

# Script to automate finetuning of RNN genomes across multiple runs
# Usage: ./finetune_runs.sh [base_directory] [output_base_directory] [finetune_iterations]

# Default parameters
BASE_DIR=${1:-"test_output/sure_outputs/coal_mpi_bp_sweep/const/bp_iter_0"}
OUTPUT_BASE_DIR=${2:-"${BASE_DIR}/finetuned"}
FINETUNE_ITERATIONS=${3:-100}

# Training and testing data
TRAINING_FILES="datasets/2018_coal/burner_[0-9].csv"
VALIDATION_FILES="datasets/2018_coal/burner_1[0-1].csv"

# Finetuning parameters
USE_STOCHASTIC=true
TIME_OFFSET=1

# Weight update parameters (matching training configuration)
WEIGHT_UPDATE="adagrad"
EPS=0.000001
BETA1=0.99

# Executable path (relative to script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXECUTABLE="${PROJECT_ROOT}/build/rnn_examples/finetune_new"

# Logging
STD_MESSAGE_LEVEL="INFO"
FILE_MESSAGE_LEVEL="INFO"

echo "=========================================="
echo "Finetuning RNN Genomes Across Multiple Runs"
echo "=========================================="
echo "Base directory: ${BASE_DIR}"
echo "Output base directory: ${OUTPUT_BASE_DIR}"
echo "Finetune iterations: ${FINETUNE_ITERATIONS}"
echo "Stochastic: ${USE_STOCHASTIC}"
echo "=========================================="
echo ""

# Check if executable exists
if [ ! -f "${EXECUTABLE}" ]; then
    echo "Error: Executable not found at ${EXECUTABLE}"
    echo "Please compile the project first: cd build && make finetune_new"
    exit 1
fi

# Check if base directory exists
if [ ! -d "${BASE_DIR}" ]; then
    echo "Error: Base directory not found: ${BASE_DIR}"
    exit 1
fi

# Change to project root for relative paths
cd "${PROJECT_ROOT}"

# Process each run directory
for RUN_NUM in {1..10}; do
    RUN_DIR="${BASE_DIR}/run_${RUN_NUM}"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/run_${RUN_NUM}"
    
    echo ""
    echo "=========================================="
    echo "Processing run_${RUN_NUM}"
    echo "=========================================="
    echo "Genome directory: ${RUN_DIR}"
    echo "Output directory: ${OUTPUT_DIR}"
    echo ""
    
    # Check if run directory exists
    if [ ! -d "${RUN_DIR}" ]; then
        echo "Warning: Run directory not found: ${RUN_DIR}"
        echo "Skipping run_${RUN_NUM}..."
        continue
    fi
    
    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    
    # Run finetuning
    echo "Starting finetuning for run_${RUN_NUM}..."
    echo ""
    
    "${EXECUTABLE}" \
        --genome_directory "${RUN_DIR}" \
        --training_filenames ${TRAINING_FILES} \
        --testing_filenames ${VALIDATION_FILES} \
        --validation_filenames ${VALIDATION_FILES} \
        --output_directory "${OUTPUT_DIR}" \
        --time_offset ${TIME_OFFSET} \
        --finetune_iterations ${FINETUNE_ITERATIONS} \
        --weight_update ${WEIGHT_UPDATE} \
        --eps ${EPS} \
        --beta1 ${BETA1} \
        --std_message_level ${STD_MESSAGE_LEVEL} \
        --file_message_level ${FILE_MESSAGE_LEVEL}
    
    EXIT_CODE=$?
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Finetuning completed successfully for run_${RUN_NUM}!"
        echo "Output saved to: ${OUTPUT_DIR}"
        echo "=========================================="
    else
        echo ""
        echo "=========================================="
        echo "Finetuning failed for run_${RUN_NUM} with exit code: ${EXIT_CODE}"
        echo "=========================================="
        # Continue with next run instead of exiting
    fi
done

echo ""
echo "=========================================="
echo "All runs processed!"
echo "=========================================="

