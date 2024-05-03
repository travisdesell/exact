#!/usr/bin/zsh
#
for crystalize_iters in 64 128 256 512; do
  for bp_epoch in 8 16 32 64 128; do
    for k in 1; do
      for fold in 0 1 2 3 4 5 6 7; do
        output_dir=initial_integration_experiments/results/v2/$crystalize_iters/$bp_epoch/$k/$fold
        tail -1 $output_dir/fitness_log.csv
      done
    done
  done
done
