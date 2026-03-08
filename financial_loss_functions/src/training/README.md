# Training and Loss Functions
[<- Back to Main README](/financial_loss_functions/README.md)

## Training
The training pipeline uses a grid of candidate model architectures to train and evaulate with different loss functions. The grid can be run in three mode `all`, `one_model` or `one_loss`. The `all` mode runs all models with specified losses, the `one_model` mode runs one model with speicified losses, and the `one_loss` mode runs all models with one specified loss function. The loss mode consists of two modes, `all` or `custom`. The `all` modes uses all available losses, including objectives as loss functions, and the `custom` mode uses only custom combination of loss functions (objectives and regularizers). 

### Run Candidates Training Grid
Run the training pipeline using the following command structure:
```bash
python -m scripts.run_training_grid [--grid_mode MODE] [OPTIONS]
```

#### Arguments Reference
| Flag | Long Flag | Choices / Type | Default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `-gm` | `--grid_mode` | `all`, `one_model`, `one_loss`, `one` | `all` | The scope of the grid search. |
| `-lm` | `--loss_mode` | `all`, `custom` | `custom` |  Use all available loss functions or only custom combinations |
| `-m`| `--model` | *string* | None | **Required** if grid mode is `one_model` or `one`. Name of model.|
| `-l` | `--loss` | *string* | None | **Required** if grid mode is `one_loss` or `one`. Name of loss function.|

#### Examples:
1. Run everything (Uses defaults: grid_mode='all', loss_mode='all')
```bash
python -m scripts.run_training_grid
```

2. Run specific model with all available loss functions
```bash
python -m scripts.run_training_grid --grid_mode one_model --model '<Model Name>' --loss_mode all
```

3. Run specific loss function with all available models
```bash
python -m scripts.run_training_grid --grid_mode one_loss --loss '<Loss Name>'
```

4. Run one model with one loss function
```bash
python -m scripts.run_training_grid --grid_mode one --model '<Model Name>' --loss '<Loss Name>'
```

5. To get help
```bash
python -m scripts.run_training_grid --help
```


## Available Models and Loss Functions
### Models
- lstm
    - BaseLSTM: Baseline LSTM architecture 
    - AttentionLSTM: LSTM with with attention heads
- transformer
    - TemporalTransformer: LSTM + Transformer encoder
    - DeformTime
    - TFT

### Loss Functions
#### Objectives:
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

#### Regularizers:
- smooth_mdd_regularizer
- cvar_topk_regularizer
- smooth_cvar_regularizer
- risk_parity_regularizer
- hhi_regularizer
- hhi_signed_regularizer
- entropy_conc_regularizer

#### Custom Combinations:
- custom_loss_6
- custom_loss_7
- custom_loss_8
- cutoms_loss_9
- custom_loss_10
- custom_loss_11