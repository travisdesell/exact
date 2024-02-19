bp_ge=(8 8192 16 4096 32 2048 64 1024)
for bp_epoch max_genomes in "${(@kv)bp_ge}"; do
  bp_epoch=$bp_epoch max_genomes=$max_genomes sbatch dnas_cluster.zsh
  bp_epoch=$bp_epoch max_genomes=$max_genomes sbatch dnas_r2_cluster.zsh
  bp_epoch=$bp_epoch max_genomes=$max_genomes sbatch dnas_control.zsh
done
