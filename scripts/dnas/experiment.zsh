#!/bin/zsh
#SBATCH -n 1
#SBATCH -A examm
#SBATCH --partition=tier3
#SBATCH -o /home/jak5763/exact/aistats/slurm_out/%x.%j.out
#SBATCH -e /home/jak5763/exact/aistats/slurm_out/%x.%j.err
#SBATCH --mem=10G

spack load gcc
spack load openmpi
spack load /5aoa7oi
spack load /dd7nzzh

for i in $(seq 0 19); do
  export i=$i
  export output_dir=/home/jak5763/exact/aistats/$control/maxt$maxt/crystal$crystal/bp$bp/$i

  if [ "$control" = "control" ]; then
      node_types="simple UGRNN MGU GRU delta LSTM"
  else
      node_types="DNAS"
  fi

  echo $node_types $control

  export node_types=$node_types

  # ./run_examm.zsh

  best_genome_file=( $output_dir/rnn_genome_*.bin([-1]) )
  export BP_ITERS=1
  export GENOME=$best_genome_file
  ./post_training.zsh
done
