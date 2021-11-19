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
DATA_DIR="/Users/zimenglyu/Dropbox/D2S2Lab/Datasets/2020_wind_turbine"

cd build
for i in 0 1 2 3 4
do
    NUM_GEN=1000
    ELITESIZE=20
    GENERATESIZE=100
    BP=10
    CHUNK=50
    VALIDATION_SIZE=50
    NUM_TRAINING_SETS=300

    exp_name="/Users/zimenglyu/Dropbox/Research/experiment_results/onenet/onenet_mpi/wind_debug/chunk_$CHUNK/gen_$NUM_GEN/bp_$BP/generated_$GENERATESIZE/elite_$ELITESIZE/$i"
    mkdir -p $exp_name
    echo "Running base EXAMM code with wind dataset, results will be saved to: "$exp_name
    echo "###-------------------###"

    mpirun -np 16 ./mpi/onenet_mpi \
    --training_filenames $DATA_DIR/turbine_R80711_2017-2020_[1-9].csv $DATA_DIR/turbine_R80711_2017-2020_1[0-9].csv $DATA_DIR/turbine_R80711_2017-2020_2[0-9].csv $DATA_DIR/turbine_R80711_2017-2020_3[0-1].csv \
    --time_offset 1 \
    --data_smooth_method "ma" \
    --moving_average_smooth_window 3 \
    --speciation_method "onenet" \
    --input_parameter_names $INPUT_PARAMETERS \
    --output_parameter_names $OUTPUT_PARAMETERS \
    --number_islands 2 \
    --elite_population_size $ELITESIZE \
    --num_generations $NUM_GEN \
    --generation_genomes $GENERATESIZE \
    --time_series_length $CHUNK \
    --validation_size $VALIDATION_SIZE \
    --num_training_sets $NUM_TRAINING_SETS \
    --bp_iterations $BP \
    --normalize min_max \
    --output_directory $exp_name \
    --possible_node_types simple UGRNN MGU GRU delta LSTM \
    --std_message_level ERROR \
    --file_message_level ERROR

done