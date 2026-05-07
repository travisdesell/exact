# Hyperparameter Optimization and Loss Functions
[<- Back to Main README](/financial_loss_functions/README.md)

## Tuning and Evaluation on Validation Data
The hyperparameter tuning and validation pipeline can be run in two modes. When --grid_mode is `one_model`, the model name provided will be matched with all available custom loss functions. These model-loss combinations can be run in parallel using the `-mpi` flag. The `-t` --tune flag must be present to tune the model hyperparameters on the validation data, and if no --tune flag is present the default hyperparameters will be used for evaluation without tuning. 

Each model-loss combination is tuned for optimal hyperparameters and then evaluated on the validation data. In hyperparameter tuning and evaluation, the model-loss combinations follow an expanding window walk-forward protocol over the validation period with non-overlapping windows. At each walk step, it also accounts for Bid-Ask Spread costs on the first day of the holding period (at the time of rebalancing). The tuning objective and all portfolio performance metrics are calculated net-of-transaction costs.

The `one_model` grid mode is the default way to run this pipeline and the `one` mode can be used when compute resources are low or for debugging. When --grid_mode is `one`, the model name and loss name must be provided. This single model-loss combination will be tuned if `-t` flag is present. This single model-loss combination is run sequentially and should not be run using mpi.

### Run Hyperparameter Optimization
Run the tuning pipeline using the following command structure:
```bash
python -m scripts.run_training_grid [--grid_mode MODE] [--model MODEL] [OPTIONS]
```

#### Arguments Reference
| Flag | Long Flag | Choices / Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `-gm` | `--grid_mode` | `one_model`, `one` | `one_model` | The scope of the model-loss search. |
| `-lm` | `--loss_mode` | `all`, `custom` | `custom` |  Use all only custom losses or all objectives and custom losses |
| `-m`| `--model` | *string* | None | Name of model.|
| `-l` | `--loss` | *string* | None | **Required** if grid mode `one`. Name of loss function.|
| `-t` | `--tune` | *None* | None | Flag to tune all models specified by the grid mode. |
| `-mpi` | `--mpi` | *None* | None | Flag to use mpi for distributed tuning if grid mode is "one_model" |

#### Examples:
1. Run all model-loss combinations for a neural network architecture with tuning on a HPC cluster.
```bash
srun python -m scripts.run_training --model '<Model Name>' --tune --mpi
```

To run this sequentially (not recommended):
```bash
python -m scripts.run_training --model '<Model Name>' --tune
```

2. Run specific model-loss combination with tuning (no mpi, flag will be ignored):
```bash
python -m scripts.run_training --grid_mode one --model '<Model Name>' --loss_name '<Loss Name>' --tune
```

3. To get help
```bash
python -m scripts.run_training --help
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

#SBATCH --time=0-24:00:00		# Time limit - d:hh:mm:ss
#SBATCH --nodes=1			    # How many nodes to run on
#SBATCH --ntasks-per-node=4     # Ranks per node
#SBATCH --cpus-per-task=2		# Number of CPUs per task
#SBATCH --mem-per-cpu=4g		# Memory per CPU
#SBATCH --gres=gpu:1            # GPUs per node


# Running Code
source /home/<USER_NAME>/miniconda3/etc/profile.d/conda.sh
conda activate <ENV_NAME>

srun python -m scripts.run_training -gm one_model -m <MODEL_NAME> -t -mpi
```
All SBATCH examples can be found at `./scripts/sample_batch_scripts/`. The example above can be found at [sample_batch_script](../../scripts/sample_batch_scripts/sample_one_model_tune.sh).

## Available Models and Loss Functions
### Neural Network Models
- **LSTM**
    - BaseLSTM
    - AttentionLSTM
    - InvertedAttentionLSTM
    - VSN-LSTM
- **Transformer**
    - TemporalTransformer
    - PatchTST
    - DeformTime

### Loss Functions
- **Objectives**
    - raw_sharpe_objective
    - differentiable_sharpe_loss
    - rms_sharpe_objective
    - smooth_neglog_sharpe_loss
    - raw_sortino_loss
    - rms_sortino_loss
    - smooth_neglog_sortino_objective
    - raw_omega_ratio
    - smooth_omega_objective
    - raw_calmar_objective
    - smooth_calmar_objective

- **Regularizers**
    - smooth_mdd_regularizer
    - cvar_topk_regularizer
    - smooth_cvar_regularizer
    - risk_parity_regularizer
    - hhi_regularizer
    - hhi_signed_regularizer
    - entropy_conc_regularizer

- **Custom Loss Functions**
    - custom_loss_1 to custom_loss_16
    - By default only custom_loss_10 (**Custom Loss A**) and custom_loss_11 (**Custom Loss B**) are used, because all other custom losses were a part of an iterative process of designing these candidate loss functions.

This library structure for Neural Network Models and Loss Functions was designed to facilitate a modular "plug-and-play" style of experimentation. New models, loss terms or loss functions can be easily added using our decorator system. See [registry.py](../models/registry.py) at "src/models/registry.py" and the hyperparameters ranges must be added in [hparams.json](../../config/hparams.json) at "config/hparams.json".   