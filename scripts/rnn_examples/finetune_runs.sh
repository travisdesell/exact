#!/bin/bash

# Script to automate finetuning of RNN genomes across multiple runs
# Usage: ./finetune_runs.sh [base_directory] [method_name] [output_base_directory] [finetune_iterations] [csv_output_file]

# Default parameters
BASE_DIR=${1:-"test_output/sure_outputs/coal_mpi_bp_sweep/const/bp_iter_0"}
METHOD_NAME=${2:-""}
OUTPUT_BASE_DIR=${3:-"${BASE_DIR}/finetuned"}
FINETUNE_ITERATIONS=${4:-100}
CSV_OUTPUT_FILE=${5:-"finetune_results.csv"}

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

# Extract experimental setup name (last folder name from base_directory)
EXPERIMENTAL_SETUP=$(basename "${BASE_DIR}")

# Initialize CSV file with header if it doesn't exist
CSV_PATH="${PROJECT_ROOT}/${CSV_OUTPUT_FILE}"
if [ ! -f "${CSV_PATH}" ]; then
    echo "experimental_setup,run_number,method_name,initial_mse,initial_mae,final_mse,final_mae,mse_difference,mae_difference" > "${CSV_PATH}"
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
    
    # Run finetuning and capture output
    echo "Starting finetuning for run_${RUN_NUM}..."
    echo ""
    
    # Capture both stdout and stderr to parse metrics
    OUTPUT_LOG=$(mktemp)
    "${EXECUTABLE}" \
        --genome_directory "${RUN_DIR}" \
        --training_filenames ${TRAINING_FILES} \
        --validation_filenames ${VALIDATION_FILES} \
        --output_directory "${OUTPUT_DIR}" \
        --time_offset ${TIME_OFFSET} \
        --finetune_iterations ${FINETUNE_ITERATIONS} \
        --weight_update ${WEIGHT_UPDATE} \
        --eps ${EPS} \
        --beta1 ${BETA1} \
        --std_message_level ${STD_MESSAGE_LEVEL} \
        --file_message_level ${FILE_MESSAGE_LEVEL} 2>&1 | tee "${OUTPUT_LOG}"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Finetuning completed successfully for run_${RUN_NUM}!"
        echo "Output saved to: ${OUTPUT_DIR}"
        echo "=========================================="
        
        # Parse metrics from output (get the last occurrence of each metric)
        INITIAL_MSE=$(grep "Initial Testing MSE:" "${OUTPUT_LOG}" | tail -1 | sed 's/.*Initial Testing MSE: *//' | sed 's/[[:space:]]*$//')
        INITIAL_MAE=$(grep "Initial Testing MAE:" "${OUTPUT_LOG}" | tail -1 | sed 's/.*Initial Testing MAE: *//' | sed 's/[[:space:]]*$//')
        FINAL_MSE=$(grep "Final Testing MSE:" "${OUTPUT_LOG}" | tail -1 | sed 's/.*Final Testing MSE: *//' | sed 's/[[:space:]]*$//')
        FINAL_MAE=$(grep "Final Testing MAE:" "${OUTPUT_LOG}" | tail -1 | sed 's/.*Final Testing MAE: *//' | sed 's/[[:space:]]*$//')
        
        # Calculate differences using awk (more portable than bc)
        if [ -n "${INITIAL_MSE}" ] && [ -n "${FINAL_MSE}" ] && [ "${INITIAL_MSE}" != "" ] && [ "${FINAL_MSE}" != "" ]; then
            MSE_DIFF=$(awk "BEGIN {printf \"%.6f\", ${INITIAL_MSE} - ${FINAL_MSE}}")
        else
            MSE_DIFF=""
        fi
        
        if [ -n "${INITIAL_MAE}" ] && [ -n "${FINAL_MAE}" ] && [ "${INITIAL_MAE}" != "" ] && [ "${FINAL_MAE}" != "" ]; then
            MAE_DIFF=$(awk "BEGIN {printf \"%.6f\", ${INITIAL_MAE} - ${FINAL_MAE}}")
        else
            MAE_DIFF=""
        fi
        
        # Write to CSV
        echo "${EXPERIMENTAL_SETUP},${RUN_NUM},${METHOD_NAME},${INITIAL_MSE},${INITIAL_MAE},${FINAL_MSE},${FINAL_MAE},${MSE_DIFF},${MAE_DIFF}" >> "${CSV_PATH}"
        echo "Metrics saved to CSV: ${CSV_PATH}"
    else
        echo ""
        echo "=========================================="
        echo "Finetuning failed for run_${RUN_NUM} with exit code: ${EXIT_CODE}"
        echo "=========================================="
        # Continue with next run instead of exiting
    fi
    
    # Clean up temp file
    rm -f "${OUTPUT_LOG}"
done

echo ""
echo "=========================================="
echo "All runs processed!"
echo "=========================================="

