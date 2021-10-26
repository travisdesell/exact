#!/bin/sh
# This is an example of running ONENET MPI version on coal dataset
#
# The coal dataset is normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization

INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
OUTPUT_PARAMETERS="Pitch"

cd build
for i in 0
# for i in 0 
do
    NUM_GEN=1200
    ELITESIZE=20
    GENERATESIZE=50
    BP=10
    CHUNK=50

    exp_name="../test_output/onenet_mpi/chunk_$CHUNK/gen_$NUM_GEN/bp_$BP/generated_$GENERATESIZE/elite_$ELITESIZE/$i"
    mkdir -p $exp_name
    echo "Running base EXAMM code with coal dataset, results will be saved to: "$exp_name
    echo "###-------------------###"

    mpirun -np 3 ./mpi/onenet_mpi \
    --training_filenames ../datasets/2019_ngafid_transfer/c172_file_[1-9].csv ../datasets/2019_ngafid_transfer/c172_file_1[0-2].csv \
    --time_offset 1 \
    --speciation_method "onenet" \
    --input_parameter_names $INPUT_PARAMETERS \
    --output_parameter_names $OUTPUT_PARAMETERS \
    --number_islands 2 \
    --elite_population_size $ELITESIZE \
    --num_generations $NUM_GEN \
    --generation_genomes $GENERATESIZE \
    --time_series_length $CHUNK \
    --bp_iterations $BP \
    --normalize min_max \
    --output_directory $exp_name \
    --possible_node_types simple UGRNN MGU GRU delta LSTM \
    --std_message_level ERROR \
    --file_message_level ERROR

    # NUM_GEN=1200
    # ELITESIZE=20
    # GENERATESIZE=50
    # BP=1
    # CHUNK=50

    # exp_name="../test_output/onenet_mpi/chunk_$CHUNK/gen_$NUM_GEN/bp_$BP/generated_$GENERATESIZE/elite_$ELITESIZE/$i"
    # mkdir -p $exp_name
    # echo "Running base EXAMM code with coal dataset, results will be saved to: "$exp_name
    # echo "###-------------------###"

    # mpirun -np 16 ./mpi/onenet_mpi \
    # --training_filenames ../datasets/2019_ngafid_transfer/c172_file_[1-9].csv ../datasets/2019_ngafid_transfer/c172_file_1[0-2].csv \
    # --time_offset 1 \
    # --speciation_method "onenet" \
    # --input_parameter_names $INPUT_PARAMETERS \
    # --output_parameter_names $OUTPUT_PARAMETERS \
    # --number_islands 2 \
    # --elite_population_size $ELITESIZE \
    # --num_generations $NUM_GEN \
    # --generation_genomes $GENERATESIZE \
    # --time_series_length $CHUNK \
    # --bp_iterations $BP \
    # --normalize min_max \
    # --output_directory $exp_name \
    # --possible_node_types simple UGRNN MGU GRU delta LSTM \
    # --std_message_level ERROR \
    # --file_message_level ERROR

    # NUM_GEN=1200
    # ELITESIZE=10
    # GENERATESIZE=20
    # BP=10
    # CHUNK=50

    # exp_name="../test_output/onenet_mpi/chunk_$CHUNK/gen_$NUM_GEN/bp_$BP/generated_$GENERATESIZE/elite_$ELITESIZE/$i"
    # mkdir -p $exp_name
    # echo "Running base EXAMM code with coal dataset, results will be saved to: "$exp_name
    # echo "###-------------------###"

    # mpirun -np 16 ./mpi/onenet_mpi \
    # --training_filenames ../datasets/2019_ngafid_transfer/c172_file_[1-9].csv ../datasets/2019_ngafid_transfer/c172_file_1[0-2].csv \
    # --time_offset 1 \
    # --speciation_method "onenet" \
    # --input_parameter_names $INPUT_PARAMETERS \
    # --output_parameter_names $OUTPUT_PARAMETERS \
    # --number_islands 2 \
    # --elite_population_size $ELITESIZE \
    # --num_generations $NUM_GEN \
    # --generation_genomes $GENERATESIZE \
    # --time_series_length $CHUNK \
    # --bp_iterations $BP \
    # --normalize min_max \
    # --output_directory $exp_name \
    # --possible_node_types simple UGRNN MGU GRU delta LSTM \
    # --std_message_level ERROR \
    # --file_message_level ERROR
done