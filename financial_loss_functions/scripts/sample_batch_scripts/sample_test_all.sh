#!/bin/bash -l

#SBATCH --job-name=<JOB NAME>
#SBATCH --comment="<DESCRIPTION>"

#SBATCH --account=<ACC. NAME>
#SBATCH --partition=<PARTITION NAME>		# Partition to run your job on

#SBATCH --output=<OUTPUT FILE PATH>		    # Output file
#SBATCH --error=<ERROR FILE PATH>			# Error file

#SBATCH --time=0-28:00:00		# Time limit - d:hh:mm:ss
#SBATCH --nodes=1			    # How many nodes to run on
#SBATCH --ntasks-per-node=8     # Ranks per node
#SBATCH --cpus-per-task=2		# Number of CPUs per task
#SBATCH --mem-per-cpu=4g		# Memory per CPU
#SBATCH --gres=gpu:2            # GPUs per node


# Running Code
source /home/<USER_NAME>/miniconda3/etc/profile.d/conda.sh
conda activate <ENV_NAME>

srun python -m scripts.run_evaluation -pm one_model \
-m <MODEL_NAME1>-<LOSS_NAME1> <MODEL_NAME2>-<LOSS_NAME2> -mpi