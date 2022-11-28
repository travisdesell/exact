#!/bin/sh
# This is an example of running EXAMM MPI version on wind dataset
#
# The wind dataset is not normalized
# To run datasets that's not normalized, make sure to add arguments:
#    --normalize min_max for Min Max normalization, or
#    --normalize avg_std_dev for Z-score normalization

cd build

INPUT_PARAMETERS="Ba_avg Rt_avg DCs_avg Cm_avg P_avg S_avg Cosphi_avg Db1t_avg Db2t_avg Dst_avg Gb1t_avg Gb2t_avg Git_avg Gost_avg Ya_avg Yt_avg Ws_avg Wa_avg Ot_avg Nf_avg Nu_avg Rbt_avg"
OUTPUT_PARAMETERS="P_avg"

exp_name="../test_output/wind_mpi"
mkdir -p $exp_name
echo "Running base EXAMM code with wind dataset, results will be saved to: "$exp_name
echo "###-------------------###"

mpirun -np 4 ./mpi/examm_mpi \
--training_filenames ../datasets/2020_wind_engine/turbine_R80711_2017-2020_[1-9].csv ../datasets/2020_wind_engine/turbine_R80711_2017-2020_1[0-9].csv ../datasets/2020_wind_engine/turbine_R80711_2017-2020_2[0-4].csv \
--test_filenames ../datasets/2020_wind_engine/turbine_R80711_2017-2020_2[5-9].csv ../datasets/2020_wind_engine/turbine_R80711_2017-2020_3[0-1].csv \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--number_islands 10 \
--population_size 10 \
--max_genomes 2000 \
--bp_iterations 5 \
--num_mutations 2 \
--normalize min_max \
--output_directory $exp_name \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--std_message_level INFO \
--file_message_level INFO
