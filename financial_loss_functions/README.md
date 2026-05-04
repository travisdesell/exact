# Financially Guided Deep Portfolio Optimization

## Overview
Portfolio optimization in real‑world financial markets is notoriously difficult due to non‑stationarity, noisy data and regime changes. Standard predict‑then‑optimize methods first forecast returns and then solve for weights, compounding prediction errors and often failing under regime shifts. We introduce "Financially Guided Deep Portfolio Optimization": An end-to-end framework for robust portfolio optimization that embeds financial objectives and portfolio construction objectives in hyperparameter tuning as well as in training, allowing a neural network to learn asset allocation weights by backpropagation. The neural network models are trained to directly optimize risk-adjusted metrics while controlling tail-risk and diversification. <!-- [ADD QUICK RESULTS] -->
   
## Key Features
- **End‑to‑End Learning**: Neural network outputs portfolio weights for the out-of-sample holding period directly. No intermediate forecasting.
- **Custom Financially Guided Losses**: Combines differentiable surrogates of risk-adjusted metrics and portfolio construction objectives.
- **Expanding Window Walk-Forward**: The strategy used in the research is: long-only, quarterly rebalancing. Models are retrained from scratch each quarter using all past data to predict portfolio weights for the next quarter, using the past 3 quarters as inference input. 
- **Hyperparameter Optimization**: Uses Optuna for hyperparameter tuning and maximizes the 95% lower confidence bound of the mean Information Ratio across all walk steps (Maximin Optimization).
- **Multiple Nueral Architectures**: Compares 7 models: BaseLSTM, AttentionLSTM, InvertedAttentionLSTM, TemporalTransformer, PatchTST, DeformTime, and VSN‑LSTM.
- **Benchmark Comparisons**: Includes tradional benchmarks like Nested Clustered Optimization, Hierarchial Risk Parity, Mean Variance Portfolio Optimization and Minimum Variance Portfolio Optimization, along with the S&P 500 and Equal Weight Portfolio.
- **Financial & Statistical Robustness**: Includes model performance analysis, pareto efficient frontiers, and statistical signficance test.

## Data
This research uses daily stock data from the Center for Research in Security Prices (CRSP) for 50 constituent stocks of the S&P 500 index, spanning the period from 7 December 2007 to 29 December 2023 (only trading days). The feature set comprises five features for each stock: daily returns, daily change in volatility, bid‑ask spread (BA spread), and turnover. Also, there is an additional column providing the daily return of the S&P 500 index itself. The full timeline is split chronologically into three non‑overlapping sets:
- Training set: 7 December 2007 - 6 February 2020
- Validation set: 7 February 2020 - 31 December 2021
- Test set: 3 January 2022 - 29 December 2023

Since the CRSP dataset cannot be distributed, we provide a synthetic sample data set stored in "data/raw/sample" for tests.


## Installation
### Prerequisites
- Python 3.13.5
- Miniconda (recommended)

### Clone Git Repository
```bash
git clone -b loss-functions https://github.com/travisdesell/exact.git
cd exact/financial_loss_functions
```

### Install Dependencies
1. Create and activate a virtual environment. You can use either venv or conda, but Miniconda is recommended.

venv:
```bash
python -m venv <env_name>
source <env_name>/bin/activate  # On Windows, use: <env_name>\Scripts\activate
```
OR
conda:
```bash
conda create -n <env_name> python=3.13.5
conda activate <env_name>
```

2. Install python dependecies
```bash
# Install the required packages
pip install -r requirements.txt
```

3. If using HPC cluster:
```bash
pip uninstall mpi4py
conda install -c conda-forge mpi4py
```
Note: This was done to avoid any conflicts with OpenMPI versions. This project does not require high speed communication as it is an "embarrassingly parallel" distributed computing framework.

### Setting Up Environment Variables
1. Create your local environment file:
```bash
cp .env.example .env
```

2. If you are using a CRSP or CRSP equivalent dataset update the .env file in root directory with raw data directory name.

```bash
CRSP_DIR = "<equivalent-data-directory>" # change to specific name if using equivalent dataset
```

## Quick Start
### 1. Run data processing
```bash
python -m scripts.run_processing
```

### 2. Run hyperparameter tuning and evaluation on validation data
```bash
python -m scripts.run_training -gm one_model -m AttentionLSTM -t
```

### 3. Run evaulation on test data
```bash
python -m scripts.run_evaluation -pm one_model -m AttentionLSTM
```

### 4. Run unit and integration tests
```bash
pytest tests
```

## For more information
- [Hyperparameter Optimization Guide](/financial_loss_functions/src/training/README.md)
- [Evaluation Guide](/financial_loss_functions/src/evaluation/README.md)

## Hardware Recommendations
- Minimum: 
    - CPU: 4 Cores
    - RAM: 16GB
    - GPU: CUDA NVIDIA GPU (12GB VRAM) or MPS Macbook Pro GPU
- Ideal (High Performance Computing Cluster):
    - CPU: 8 Cores
    - RAM: 32 GB
    - GPU: NVIDIA H100 (40GB VRAM)

## Using CRSP Equivalent Datasets
Currently, a sythetic CRSP-like dataset is stored in `data/raw/sample`. If any other CRSP equivalent dataset is being used, place the directory in `data/raw/` and update the CRSP_DIR environment variable in the .env file with the name of the equivalent data directory.

.env
```bash
CRSP_DIR = "<equivalent-data-directory>" # Add directory name here if using CRSP like dataset
```

Since we use pre-split data (train, val, test), the new files must follow the structure shown below and the config/paths.json must be updated to reflect the correct file names. 

- data/
    - raw/
        - `<equivalent-data-directory>/`
            - `<train_name>.csv`
            - `<validation_name>.csv`
            - `<test_name>.csv`

<!-- ## Methodology

## Results

## Repository Structure -->

## Acknowledgments
- We gratefully acknowledge the use of the RIT Research Computing's HPC cluster at Rochester Institute of Technology, which provided essential computational resources for this research.
- This research uses data from the Center for Research in Security Prices (CRSP), accessed through Rochester Institute of Technology.
- This work would not have been possible without the contributions of the open‑source software community.

## Authors
**Rahul Keneth Fernandes**  
Email: rahulkfernandes@gmail.com  
Github: [@rahulkfernandes](https://github.com/rahulkfernandes)   

**Dr. Travis Desell**  
Email: tjdvse@rit.edu  
Github: [@travisdesell](https://github.com/travisdesell)

**Institution:** Rochester Institute of Technology, Rochester, New York 