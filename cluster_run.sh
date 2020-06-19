#!/bin/bash -l
# NOTE the -l flag!
#

#SBATCH -J examm_thompson

#SBATCH -A examm    #just examm examm_thompson is not a valid account

#SBATCH -o cluster_logs/examm_test_%A.output
#SBATCH -e cluster_logs/examm_test_%A.error

#SBATCH --mail-user jak5763@rit.edu

#SBATCH --mail-type=ALL

#SBATCH -t 5-0     # if you want to run for 4 days 1-72:0:0

# Put the job in the "work" partition and request FOUR cores for one task
# "work" is the default partition so it can be omitted without issue.

## Please not that each node on the cluster is 36 cores
#SBATCH -p tier3 -n 288


# Job memory requirements in MB
#SBATCH --mem-per-cpu=4G  # I like to mem with a suffix [K|M|G|T] 5000

#module load module_future
#module load openmpi-1.10-x86_64
#module load openmpi
#module load gcc
#module load libtiff

EXAMM="/Users/josh/Programming/exact_thompson"

for decayrate in "0.999" "0.99" "0.98" "0.97"; do
    for aircraft in "c172"; do
        for use_thompson in 1; do
           for folder in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
                flight=$aircraft
                
                if [ $flight = "c172" ]; then
                    echo "\tUsing c172 dateset"
                    INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
                    OUTPUT_PARAMETERS="E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT"
                elif [ $flight = "pa28" ]; then
                    echo "\tUsing pa28 dataset"
                    INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
                    OUTPUT_PARAMETERS="E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM"
                elif [ $flight="pa44" ]; then
                    echo "\tUsing pa44 dataset"
                    INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_MAP E1_OilP E1_OilT E1_RPM E2_CHT1 E2_EGT1 E2_EGT2 E2_EGT3 E2_EGT4 E2_FFlow E2_OilP E2_OilT E2_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
                    OUTPUT_PARAMETERS="E1_CHT1 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_MAP E1_OilP E1_OilT E1_RPM E2_CHT1 E2_EGT1 E2_EGT2 E2_EGT3 E2_EGT4 E2_FFlow E2_OilP E2_OilT E2_RPM"
                else 
                    echo "\tERROR: wrong flight type!"
                    exit 1
                fi 
            
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
                    --max_genomes 16200 \
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
<<<<<<< HEAD
                    --number_mutations_sampling_decay_rate $decayrate \
=======
    		        --number_mutations_sampling_decay_rate $decayrate \
>>>>>>> e481eeda40e7b5e7590c4bc464f01680fd89877d
                    --max_number_mutations 5
            done
        done
    done
done
