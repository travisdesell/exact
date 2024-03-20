bp=(1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 100)
for bp_epoch in $bp; do
  for synchronous in "async" "synchronous"; do
    for scramble_weights in "epigenetic_weights" "no_epigenetic_weights"; do
      bp_epoch=$bp_epoch synchronous="$synchronous" scramble_weights="$scramble_weights" sbatch examm_bias_exp.zsh
    done
  done
done
