#!/bin/zsh

INPUT_PARAMETERS='AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd'
OUTPUT_PARAMETERS='E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4'

offset=1

run_examm() {
  output_dir=initial_integration_experiments/results/debug/$crystalize_iters/$bp_epoch/$k/$fold
  mkdir -p $output_dir
  mpirun -np 63 --use-hwthread-cpus Release/mpi/examm_mpi \
      --training_filenames datasets/2019_ngafid_transfer/c172_file_[1-9].csv \
      --test_filenames datasets/2019_ngafid_transfer/c172_file_1[0-2].csv \
      --time_offset $offset \
      --possible_node_types dnas \
      --stochastic 1 \
      --input_parameter_names ${=INPUT_PARAMETERS} \
      --output_parameter_names ${=OUTPUT_PARAMETERS} \
      --bp_iterations $bp_epoch \
      --normalize min_max \
      --num_hidden_layers $SIZE \
      --hidden_layer_size $SIZE \
      --validation_sequence_length 100 \
      --max_recurrent_depth 1 \
      --output_directory $output_dir \
      --log_filename fitness.csv \
      --learning_rate 0.01 \
      --std_message_level INFO \
      --file_message_level WARNING \
      --crystalize_iters $crystalize_iters \
      --max_genomes 8192 \
      --island_size 32 \
      --number_islands 4 \
      --stochastic \
      --dnas_k $k

  # best_genome_file=( $output_dir/rnn_genome_*.bin([-1]) )
  # BP_ITERS=$crystalize_iters CRYSTALIZE_ITERS=$crystalize_iters GENOME=$best_genome_file OUTPUT_DIRECTORY=$output_dir k=$k ./initial_integration_experiments/post_training_dnas.zsh
}

CELL_TYPE='dnas'
for crystalize_iters in 128; do
  for bp_epoch in 8; do
    for k in 1; do
      for fold in 0; do
        run_examm
      done
 #      wait
 #      for fold in 4 5 6 7; do
 #        run_examm &
 #      done
 #      wait
    done
  done
done
