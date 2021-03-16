#!/bin/bash -l
# NOTE the -l flag!
#

# This is an example job file for a Serial Multi-Process job.
# Note that all of the following statements below that begin
# with #SBATCH are actually commands to the SLURM scheduler.
# Please copy this file to your home directory and modify it
# to suit your needs.
# 
# If you need any help, please email rc-help@rit.edu
#

# Name of the job - You'll probably want to customize this.
#SBATCH -J bottle_x

#SBATCH -A examm

# Standard out and Standard Error output files
#SBATCH -o bottle_%x_%j.output
#SBATCH -e bottle_%x_%j.error

#To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user jak5763@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

# Request 5 hours run time MAX, anything over will be KILLED
#SBATCH -t 5-00:0:0

# Put the job in the "work" partition and request FOUR cores for one task
# "work" is the default partition so it can be omitted without issue.

## Please not that each node on the cluster is 36 cores
#SBATCH -p tier3
#SBATCH --nodes x


# Job memory requirements in MB
#SBATCH --mem-per-cpu=5000


#module load module_future
#module load openmpi-1.10-x86_64
module load openmpi
module load gcc
EXAMM="/home/jak5763/exact"
DATA_DIR="/home/jak5763/exact/datasets/2020_wind_turbine"

# EXAMM="/home/zl7069/git/weight_initialize/exact"
# DATA_DIR="/home/zl7069/datasets/2020_wind_turbine"

MAX_GENOME=20000

for fold in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20;
do
    # REPOPULATION_METHOD = "bestGenome"
    exp_name=${EXAMM}/results/bottle/$n_nodes/$fold
    mkdir -p $exp_name
    echo "\tIteration: "$exp_name
    echo "\t###-------------------###"

    INPUT_PARAMETERS="Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int"
    OUTPUT_PARAMETERS="Main_Flm_Int"
    
    time srun $EXAMM/build/mpi/examm_mpi_stress_test \
        --training_filenames ../datasets/2018_coal/burner_[0-12].csv \
        --test_filenames ../datasets/2018_coal/burner_1[0-1].csv 
        --time_offset 1 \
        --input_parameter_names $INPUT_PARAMETERS \
        --output_parameter_names $OUTPUT_PARAMETERS \
        --number_islands 10 \
        --population_size 50 \
        --max_genomes $MAX_GENOME \
        --speciation_method "island" \
        --weight_initialize $WEIGET_INITIALIZE \
        --weight_inheritance $WEIGHT_INHERITANCE \
        --new_component_weight $NEW_COMPONENT_WEIGHT  \
        --normalize min_max \
        --extinction_event_generation_number 0 \
        --islands_to_exterminate 0 \
        --bp_iterations $bp_epoch \
        --output_directory $exp_name \
        --possible_node_types simple UGRNN MGU GRU delta LSTM \
        --std_message_level WARNING --file_message_level WARNING
done

