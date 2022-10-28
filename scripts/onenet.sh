#!/bin/sh
# This is an example of running ONENET MPI version on coal dataset
#
# The coal dataset is normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization

#!/bin/sh
# This is an example of running ONENET MPI version on coal dataset
#
# The coal dataset is normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization
    # --exponential_smooth_alpha 0.5 \
INPUT_PARAMETERS="Ba_avg Rt_avg DCs_avg Cm_avg P_avg S_avg Cosphi_avg Db1t_avg Db2t_avg Dst_avg Gb1t_avg Gb2t_avg Git_avg Gost_avg Ya_avg Yt_avg Ws_avg Wa_avg Ot_avg Nf_avg Nu_avg Rbt_avg"
OUTPUT_PARAMETERS="P_avg"
DATA_DIR="/home/jefhai/Dropbox/D2S2Lab/Datasets/2020_wind_turbine"
    # --data_smooth_method "ma" \
    # --moving_average_smooth_window 3 \
cd build
for i in 0 
do
    NUM_GEN=1000
    ELITESIZE=5
    GENERATESIZE=10
    BP=10
    NUM_ISLANDS=10
    CHUNK=25
    VALIDATION_SIZE=50
    NUM_TRAINING_SETS=300
    EXTINCT=200

    # exp_name="/home/jefhai/Dropbox/Research/experiment_results/onenet/onenet_mpi/wind_repopulation/chunk_$CHUNK/gen_$NUM_GEN/bp_$BP/generated_$GENERATESIZE/elite_$ELITESIZE/$i"
    exp_name="../results/wind_debug/"

    mkdir -p $exp_name
    echo "Running base EXAMM code with wind dataset, results will be saved to: "$exp_name
    echo "###-------------------###"

    mpirun -np 4 ./mpi/examm_mpi \
    --training_filenames $DATA_DIR/turbine_R80711_2017-2020_[1-9].csv $DATA_DIR/turbine_R80711_2017-2020_1[0-9].csv $DATA_DIR/turbine_R80711_2017-2020_2[0-9].csv \
    --test_filenames $DATA_DIR/turbine_R80711_2017-2020_3[0-1].csv \
    --time_offset 1 \
    --speciation_method "island" \
    --genome_bin "/home/jefhai/Documents/git/cluster_results/onenet_mpi/wind_pre/chunk_25/max_genome_20000/bp_10/0/rnn_genome_18462.bin" \
    --transfer_learning_version "v1" \
    --epigenetic_weights \
    --input_parameter_names $INPUT_PARAMETERS \
    --output_parameter_names $OUTPUT_PARAMETERS \
    --number_islands $NUM_ISLANDS \
    --population_size 10 \
    --max_genomes 20000 \
    --elite_population_size $ELITESIZE \
    --num_generations $NUM_GEN \
    --generation_genomes $GENERATESIZE \
    --time_series_length $CHUNK \
    --validation_size $VALIDATION_SIZE \
    --num_training_sets $NUM_TRAINING_SETS \
    --extinction_event_generation_number $EXTINCT \
    --island_ranking_method "EraseWorst" \
    --repopulation_method "bestGenome" \
    --islands_to_exterminate 1 \
    --repopulation_mutations 2 \
    --bp_iterations $BP \
    --normalize min_max \
    --output_directory $exp_name \
    --possible_node_types simple UGRNN MGU GRU delta LSTM \
    --std_message_level INFO \
    --file_message_level INFO

done