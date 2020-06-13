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
#SBATCH -J examm_eras

#SBATCH -A examm    #just examm examm_thompson is not a valid account

# Standard out and Standard Error output files
#SBATCH -o examm_test_%A.output
#SBATCH -e examm_test_%A.error

#To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user jtm5356@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

# Request 5 hours run time MAX, anything over will be KILLED
#SBATCH -t 4-0     # if you want to run for 4 days 1-72:0:0

# Put the job in the "work" partition and request FOUR cores for one task
# "work" is the default partition so it can be omitted without issue.

## Please note that each node on the cluster is 36 cores
#SBATCH -p tier3 -n 36


# Job memory requirements in MB
#SBATCH --mem-per-cpu=5G  # I like to mem with a suffix [K|M|G|T] 5000

#module load module_future
#module load openmpi-1.10-x86_64
module load openmpi
module load gcc
module load libtiff

EXAMM="/home/jtm5356/exact/"
          
exp_name="$EXAMM/thompson_results/$flight/decayrate_${decayrate}/$folder"
mkdir -p $exp_name
echo "\tIteration: "$exp_name
echo "\t###-------------------###"
srun $EXAMM/build/mpi/examm_mpi \
--training_filenames $EXAMM/datasets/2019_ngafid_transfer/${flight}_file_[1-10].csv \
--test_filenames $EXAMM/datasets/2019_ngafid_transfer/${flight}_file_[11-12].csv \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--normalize min_max \
--number_islands 10 \
--population_size 5 \
--max_genomes 16000 \
--speciation_method "island" \
--extinction_event_generation_number 400 \
--island_ranking_method "EraseWorst" \
--repopulation_method "bestGenome" \
--islands_to_exterminate 0 \
--repopulation_mutations 2 \
--bp_iterations 2 \
--output_directory $exp_name \
--std_message_level ERROR \
--file_message_level ERROR \
--use_number_mutations_thompson_sampling $use_thompson \
--number_mutations_sampling_decay_rate $decayrate \
--max_number_mutations 10

