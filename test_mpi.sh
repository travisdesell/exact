#!/bin/bash
#SBATCH  --nodes=N_NODES
#SBATCH  --overcommit
#SBATCH  --cpus-per-task=36
#SBATCH  --exclude theocho
#SBATCH  --time=12:00:00
#SBATCH  -A examm
#SBATCH  --partition=debug
#SBATCH  -J examm_arch_N_NODES_nodes_bp_BPI_control
#SBATCH  -o /home/jak5763/exact/results/slurm_out/%x.%j.out
#SBATCH  -e /home/jak5763/exact/results/slurm_out/%x.%j.err
#SBATCH  --mem=0
MAX_GENOMES=40000

root=/home/josh/development/exact_old/
build=Release
results_dir=$root/results/N_NODESnBPIb_control/

let n_nodes=1
let node_size=36

let np=$n_nodes*$node_size

for i in $(seq 0 0); do
  mkdir -p $results_dir/$i

  export OMPI_MCA_mpi_yield_when_idle=1 && mpirun --use-hwthread-cpus -np $np $root/$1/mpi/examm_mpi \
    --training_filenames $root/datasets/2018_coal/burner_[0].csv --test_filenames \
    $root/datasets/2018_coal/burner_1[0].csv \
    --time_offset 1 \
    --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int \
    --output_parameter_names Main_Flm_Int \
    --number_islands 8 \
    --population_size 16 \
    --min_mutations 1 \
    --max_mutations 1 \
    --min_intra_crossover_parents 2 \
    --max_intra_crossover_parents 4 \
    --min_inter_crossover_parents 2 \
    --max_inter_crossover_parents 4 \
    --max_genomes $MAX_GENOMES \
    --bp_iterations 0 \
    --output_directory $results_dir/$i \
    --possible_node_types simple UGRNN MGU GRU delta LSTM \
    --std_message_level INFO \
    --file_message_level INFO 
done
