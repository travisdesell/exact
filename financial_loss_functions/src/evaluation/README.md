# Evaluation on Test Data
[<- Back to Main README](/financial_loss_functions/README.md)

## Evaluate Selected Candidates on the Test Data

### Run Evaluation on Test Data
```bash
python -m scripts.run_evaluation [--prev_mode PREVIOUS MODE] [OPTIONS]
```

#### Arguments Reference
| Flag | Long Flag | Choices / Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `-pm` | `--prev_mode` | `one_model`, `one` | `one_model` | The grid mode used in the previous tuning stage. |
| `-m` | `--model_losses` | `[<MODEL_NAME1>-<LOSS_NAME1>]` | None | List of model-loss combinations to run the test evaluation |
| `-mpi` | `--mpi` | *None* | None | Flag to use mpi for distributed evaluation if grid mode is "one_model" |

#### Examples:
1. Run a selected collection of model-loss combinations on a HPC cluster.
```bash
srun python -m scripts.run_evaluation --model_losses <MODEL_NAME1>-<LOSS_NAME1> <MODEL_NAME2>-<LOSS_NAME2> --mpi
```

To run this sequentially (not recommended):
```bash
python -m scripts.run_evaluation --model_losses <MODEL_NAME1>-<LOSS_NAME1> <MODEL_NAME2>-<LOSS_NAME2>
```

2. Run specific model-loss combination (no mpi, flag will be ignored):
```bash
python -m scripts.run_evaluation --model_losses <MODEL_NAME1>-<LOSS_NAME1>
```

3. Run specific model-loss combination when the previously used mode was 'one':
```bash
python -m scripts.run_evaluation --prev_mode one --model_losses <MODEL_NAME1>-<LOSS_NAME1>
``

4. To get help
```bash
python -m scripts.run_evaluation --help
```

#### Example SBATCH Script
```bash
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
```
All SBATCH examples can be found at `./scripts/sample_batch_scripts/`. The example above can be found at [sample_batch_script](../../scripts/sample_batch_scripts/sample_test_all.sh).