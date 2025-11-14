#!/bin/bash

# Script to run finetuning for const method across multiple experimental setups
# All results will be appended to the same CSV file

./scripts/rnn_examples/finetune_runs.sh test_output/line_grid_search/coal_mpi_bp_sweep/const/bp_iter_0 "const"
./scripts/rnn_examples/finetune_runs.sh test_output/line_grid_search/coal_mpi_bp_sweep/const/bp_iter_1 "const"
./scripts/rnn_examples/finetune_runs.sh test_output/line_grid_search/coal_mpi_bp_sweep/const/bp_iter_2 "const"
./scripts/rnn_examples/finetune_runs.sh test_output/line_grid_search/coal_mpi_bp_sweep/const/bp_iter_4 "const"
./scripts/rnn_examples/finetune_runs.sh test_output/line_grid_search/coal_mpi_bp_sweep/const/bp_iter_8 "const"
./scripts/rnn_examples/finetune_runs.sh test_output/line_grid_search/coal_mpi_bp_sweep/const/bp_iter_16 "const"
./scripts/rnn_examples/finetune_runs.sh test_output/line_grid_search/coal_mpi_bp_sweep/const/bp_iter_32 "const"
