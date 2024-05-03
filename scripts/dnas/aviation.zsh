#!/bin/zsh

INPUT_PARAMETERS='AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd'

offset=1

run_examm() {
  output_dir=results/v0/$bp_epoch/$fold
  mkdir -p $output_dir
  mpirun -np 32 Release/mpi/examm_mpi \
      --training_filenames datasets/2019_ngafid_transfer/c172_file_[1-9].csv \
      --test_filenames datasets/2019_ngafid_transfer/c172_file_1[0-2].csv \
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

for output_params in "E1_CHT1" "Pitch"; do
  for bp_epoch in 2 4 8 16 32; do
    for fold in 0 1 2 3 4 5 6 7 8 9; do
      run_examm
    done
  done
done
