bp=(1 2 3 4 5 10 15 20 30 40 50 100 150 200)
for bp_epoch in $bp; do
  for synchronous in "async" "synchronous"; do
    for scramble_weights in "epigenetic_weights" "no_epigenetic_weights"; do
      bp_epoch=$bp_epoch synchronous="$synchronous" scramble_weights="$scramble_weights" sbatch examm_bias_exp.zsh
    done
  done
done
