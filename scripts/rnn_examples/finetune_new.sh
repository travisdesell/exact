#!/bin/bash

# Script to automate finetuning of RNN genomes
# Usage: ./finetune_new.sh [genome_directory] [output_directory] [finetune_iterations]

# Default parameters
GENOME_DIR=${1:-"test_output/sure_outputs/coal_mpi_bp_sweep/const/bp_iter_0/run_1"}
OUTPUT_DIR=${2:-"${GENOME_DIR}/finetuned"}
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
echo "Finetuning RNN Genome"
echo "=========================================="
echo "Genome directory: ${GENOME_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
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

# Check if genome directory exists
if [ ! -d "${GENOME_DIR}" ]; then
    echo "Error: Genome directory not found: ${GENOME_DIR}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Change to project root for relative paths
cd "${PROJECT_ROOT}"

# Run finetuning
echo "Starting finetuning..."
echo ""

"${EXECUTABLE}" \
    --genome_directory "${GENOME_DIR}" \
    --training_filenames ${TRAINING_FILES} \
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
    echo "Finetuning completed successfully!"
    echo "Output saved to: ${OUTPUT_DIR}"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Finetuning failed with exit code: ${EXIT_CODE}"
    echo "=========================================="
    exit ${EXIT_CODE}
fi

