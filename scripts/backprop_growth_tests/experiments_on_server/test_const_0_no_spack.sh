#!/bin/bash -l
#SBATCH --account=neuroevolution
#SBATCH --mail-user=slack:@dv6943
#SBATCH --mail-type=ALL
#SBATCH -t 0-01:30:00
#SBATCH --output=%n_%x_%j.out
#SBATCH --error=%x_%a_%j.err
#SBATCH --partition=tier3
#SBATCH -n 18
#SBATCH --mem-per-cpu=10g

BP_MIN=0
BP_MAX=50
INPUT_PARAMETERS="Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int"
OUTPUT_PARAMETERS="Main_Flm_Int"
bpiter=0
i=1
exp_name="/home/dv6943/exact/test_output/line_grid_search/coal_mpi_bp_sweep/const/bp_iter_${bpiter}/run_${i}"
cd /home/dv6943/exact/build
export ASAN_OPTIONS=abort_on_error=1:disable_coredump=0
export LSAN_OPTIONS=verbosity=1:log_threads=1
ulimit -c unlimited
srun mpi/examm_mpi \
    --training_filenames /home/dv6943/exact/datasets/2018_coal/burner_[0-9].csv --validation_filenames /home/dv6943/exact/datasets/2018_coal/burner_1[0-1].csv \
    --time_offset 1 \
    --input_parameter_names $INPUT_PARAMETERS \
    --output_parameter_names $OUTPUT_PARAMETERS \
    --number_islands 10 \
    --island_size 10 \
    --max_wallclock_seconds 3600 \
    --bp_min $BP_MIN \
    --bp_max $BP_MAX \
    --backprop_iterations_type "const" \
    --bp_iterations $bpiter \
    --output_directory "$exp_name" \
    --num_mutations 2 \
    --weight_update adagrad \
    --eps 0.000001 \
    --beta1 0.99 \
    --sequence_length 50 \
    --possible_node_types simple UGRNN MGU GRU delta LSTM \
    --genome_size_log 0 \
    --save_genome_option the_best \
    --std_message_level INFO \
    --file_message_level NONE
