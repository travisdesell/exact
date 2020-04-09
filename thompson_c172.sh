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
#SBATCH -J examm

#SBATCH -A examm

# Standard out and Standard Error output files
#SBATCH -o examm_test_%A.output
#SBATCH -e examm_test_%A.error

#To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user zl7069@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

# Request 5 hours run time MAX, anything over will be KILLED
#SBATCH -t 1-5:0:0

# Put the job in the "work" partition and request FOUR cores for one task
# "work" is the default partition so it can be omitted without issue.

## Please not that each node on the cluster is 36 cores
#SBATCH -p tier3 -n 32


# Job memory requirements in MB
#SBATCH --mem-per-cpu=5000


#module load module_future
#module load openmpi-1.10-x86_64
module load openmpi
module load gcc
module load libtiff

EXAMM="/home/zl7069/git/thompson/exact"


for folder in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
    flight="c172"
    
    if [ $flight = "c172" ]; then
        echo "\tUsing c172 dateset"
        INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
    elif [ $flight = "pa28" ]; then
        echo "\tUsing pa28 dataset"
        INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
    elif [ $flight="pa44" ]; then
        echo "\tUsing pa44 dataset"
        INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_MAP E1_OilP E1_OilT E1_RPM E2_CHT1 E2_EGT1 E2_EGT2 E2_EGT3 E2_EGT4 E2_FFlow E2_OilP E2_OilT E2_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
    else 
        echo "\tERROR: wrong flight type!"
        exit 1
    fi 

    # OUTPUT_PARAMETERS="E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4"
    OUTPUT_PARAMETERS="Pitch Roll AltMSL IAS LatAc NormAc "
    exp_name="$EXAMM/thompson_results/$flight"
    mkdir -p $exp_name
    echo "\tIteration: "$exp_name
    echo "\t###-------------------###"
    srun --use-hwthread-cpus $EXAMM/build/mpi/examm_mpi \
        --training_filenames $EXAMM/datasets/2019_ngafid_transfer/${flight}_file_[1-10].csv \
        --test_filenames $EXAMM/datasets/2019_ngafid_transfer/${flight}_file_[11-12].csv \
        --time_offset 1 \
        --input_parameter_names $INPUT_PARAMETERS \
        --output_parameter_names $OUTPUT_PARAMETERS \
        --normalize min_max \
        --number_islands 10 \
        --population_size 5 \
        --max_genomes 10000 \
        --speciation_method "island" \
        --extinction_event_generation_number 400 \
        --island_ranking_method "EraseWorst" \
        --repopulation_method "bestGenome" \
        --islands_to_exterminate 0 \
        --repopulation_mutations 2 \
        --bp_iterations 2 \
        --output_directory $exp_name \
        --possible_node_types simple UGRNN MGU GRU delta LSTM \
        --std_message_level ERROR \
        --file_message_level ERROR \
        --use_thompson_sampling 1
done
