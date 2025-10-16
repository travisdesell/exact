#!/bin/bash -l
#SBATCH --account=neuroevolution
#SBATCH --mail-user=slack:@dv6943
#SBATCH --mail-type=ALL
#SBATCH -t 0-01:30:00
#SBATCH --output=%n_%x_%j.out
#SBATCH --error=%x_%a_%j.err
#SBATCH --partition=tier3
#SBATCH -n 18
#SBATCH --mem-per-cpu=100g
#module load openmpi-1.10-x86_64
# spack load gcc/lhqcen5
# spack load cmake/pbddesj
# spack load libtiff/gnxev37
# spack load openmpi/xcunp5q

cd /home/dv6943/exact/build

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
    --backprop_iterations_type "scaled" \
    --bp_slope $a \
    --bp_exponent $b \
    --output_directory "$exp_name" \
    --num_mutations 2 \
    --weight_update adagrad \
    --eps 0.000001 \
    --beta1 0.99 \
    --sequence_length 50 \
    --possible_node_types simple UGRNN MGU GRU delta LSTM \
    --save_genome_option the_best \
    --std_message_level INFO \
    --file_message_level NONE
