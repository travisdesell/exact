#!/bin/bash

# Script to run finetuning for rand method across multiple experimental setups
# All results will be appended to the same CSV file

./scripts/rnn_examples/finetune_runs.sh test_output/line_grid_search/coal_mpi_bp_sweep/rand/bpmin_0_bpmax_16 "rand"
./scripts/rnn_examples/finetune_runs.sh test_output/line_grid_search/coal_mpi_bp_sweep/rand/bpmin_4_bpmax_12 "rand"
./scripts/rnn_examples/finetune_runs.sh test_output/line_grid_search/coal_mpi_bp_sweep/rand/bpmin_8_bpmax_24 "rand"
./scripts/rnn_examples/finetune_runs.sh test_output/line_grid_search/coal_mpi_bp_sweep/rand/bpmin_12_bpmax_20 "rand"
./scripts/rnn_examples/finetune_runs.sh test_output/line_grid_search/coal_mpi_bp_sweep/rand/bpmin_24_bpmax_40 "rand"
./scripts/rnn_examples/finetune_runs.sh test_output/line_grid_search/coal_mpi_bp_sweep/rand/bpmin_28_bpmax_36 "rand"
