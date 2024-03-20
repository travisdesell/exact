#!/usr/bin/zsh
# This is an example of running EXAMM MPI version on c172 dataset
#
# The c172 dataset is not normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization

INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
OUTPUT_PARAMETERS="Pitch"

for i in 0 1 2 3 4 5 6 7 8 9; do
  exp_name="ground_truth_experiments/results/source_genomes/$i"
  mkdir -p $exp_name
  echo $exp_name
  mpirun -np 5 Release/mpi/examm_mpi \
    --training_filenames datasets/2019_ngafid_transfer/c172_file_[1-9].csv \
    --test_filenames datasets/2019_ngafid_transfer/c172_file_1[0-2].csv \
    --time_offset 1 \
    --input_parameter_names ${=INPUT_PARAMETERS} \
    --output_parameter_names ${=OUTPUT_PARAMETERS} \
    --number_islands 8 \
    --island_size 8 \
    --max_genomes 10000 \
    --bp_iterations 5 \
    --num_mutations 2 \
    --normalize min_max \
    --output_directory $exp_name \
    --possible_node_types simple UGRNN MGU GRU delta LSTM \
    --std_message_level ERROR \
    --file_message_level INFO &
done
wait
