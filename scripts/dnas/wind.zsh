#!/bin/zsh

INPUT_PARAMETERS="Ba_avg Rt_avg DCs_avg Cm_avg P_avg S_avg Cosphi_avg Db1t_avg Db2t_avg Dst_avg Gb1t_avg Gb2t_avg Git_avg Gost_avg Ya_avg Yt_avg Ws_avg Wa_avg Ot_avg Nf_avg Nu_avg Rbt_avg"


offset=1

run_examm() {
  output_dir=results/v0/$bp_epoch/$fold
  mkdir -p $output_dir
  mpirun -np 32 Release/mpi/examm_mpi \
      --training_filenames datasets/2020_wind_engine/turbine_R80711_2017-2020_[1-9].csv datasets/2020_wind_engine/turbine_R80711_2017-2020_1[0-9].csv datasets/2020_wind_engine/turbine_R80711_2017-2020_2[0-4].csv \
      --test_filenames datasets/2020_wind_engine/turbine_R80711_2017-2020_2[5-9].csv datasets/2020_wind_engine/turbine_R80711_2017-2020_3[0-1].csv \
      --time_offset $offset \
      --input_parameter_names ${=INPUT_PARAMETERS} \
      --output_parameter_names ${=output_params} \
      --bp_iterations $bp_epoch \
      --normalize min_max \
      --max_recurrent_depth 1 \
      --output_directory $output_dir \
      --log_filename fitness.csv \
      --learning_rate 0.01 \
      --std_message_level INFO \
      --file_message_level INFO \
      --max_genomes 10000 \
      --island_size 32 \
      --number_islands 4

  touch $output_dir/completed
}


for output_params in "Cm_avg" "P_avg"; do
  for bp_epoch in 2 4 8 16 32; do
    for fold in 0 1 2 3 4 5 6 7 8 9; do
      run_examm
    done
  done
done
