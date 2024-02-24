#!/bin/zsh

INPUT_PARAMETERS='AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd'
OUTPUT_PARAMETERS='E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4'

offset=1

run_examm() {
  output_dir=results/control_v8/$bp_epoch/$fold
  mkdir -p $output_dir
  mpirun -np 14 build/mpi/examm_mpi \
      --training_filenames datasets/2019_ngafid_transfer/c172_file_[1-9].csv \
      --test_filenames datasets/2019_ngafid_transfer/c172_file_1[0-2].csv \
      --time_offset $offset \
      --possible_node_types lstm mgu gru ugrnn delta simple \
      --stochastic 0 \
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
      --max_genomes $max_genomes \
      --island_size 32 \
      --number_islands 4 \
      --synchronous

  # best_genome_file=( $output_dir/rnn_genome_*.bin([-1]) )
  # BP_ITERS=$crystalize_iters CRYSTALIZE_ITERS=$crystalize_iters GENOME=$best_genome_file OUTPUT_DIRECTORY=$output_dir k=$k initial_integration_experiments/post_training_dnas.zsh
}

# bp_ge=(8 8192 16 4096 32 2048)
bp_ge=(8 8192)

for bp_epoch max_genomes in "${(@kv)bp_ge}"; do
  for fold in $(seq 0 1); do
     run_examm
   done
done
