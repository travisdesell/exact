#!/bin/zsh

#SBATCH  --nodes=1
#SBATCH  --ntasks-per-node=36
#SBATCH  --exclude theocho
#SBATCH  --time=23:00:00
#SBATCH  -A examm
#SBATCH  --partition=tier3
#SBATCH  -J examm_dnas_experimental
#SBATCH  -o /home/jak5763/exact/results/slurm_out/%x.%j.out
#SBATCH  -e /home/jak5763/exact/results/slurm_out/%x.%j.err
#SBATCH  --mem=0

INPUT_PARAMETERS='AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd'
OUTPUT_PARAMETERS='E1_CHT1'

offset=1

run_examm() {
  output_dir=results/synchronous/$max_genomes/$fold
  mkdir -p $output_dir
  srun -n 36 Release/mpi/examm_mpi \
      --training_filenames datasets/2019_ngafid_transfer/c172_file_[1-9].csv \
      --test_filenames datasets/2019_ngafid_transfer/c172_file_1[0-2].csv \
      --time_offset $offset \
      --possible_node_types simple ugrnn gru mgu lstm delta \
      --input_parameter_names ${=INPUT_PARAMETERS} \
      --output_parameter_names ${=OUTPUT_PARAMETERS} \
      --bp_iterations $bp_epoch \
      --normalize min_max \
      --num_hidden_layers $SIZE \
      --hidden_layer_size $SIZE \
      --max_recurrent_depth 10 \
      --output_directory $output_dir \
      --log_filename fitness.csv \
      --learning_rate 0.01 \
      --std_message_level WARNING \
      --file_message_level WARNING \
      --crystalize_iters $crystalize_iters \
      --max_genomes $max_genomes \
      --island_size 10 \
      --number_islands 10 \
      --num_mutations 1 \
      --synchronous
}

run_group() {
  for fold in $(seq 0 9); do
    run_examm
  done
}

let base_genomes=100000
let max_genomes=$base_genomes/$bp_epoch
run_group
