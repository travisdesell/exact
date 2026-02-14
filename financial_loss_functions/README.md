# Financially Guided Neural Networks for Robust Portfolio Optimization

## Description
This repository contains the implementation for the capstone project "Financially Guided Neural Networks for Robust Portfolio Optimization," developed by Rahul Kenneth Fernandes and Atharva Atul Vaidya under the guidance of Dr. Travis Desell at Rochester Institute of Technology. The project introduces an end-to-end deep learning framework that embeds financial principles of portfolio construction theory directly into neural network training objectives to create stable, diversified portfolios. Our models output normalized allocation weights and optimize differentiable surrogates for key metrics like Sharpe/Sortino ratios, Conditional Value-at-Risk (CVaR), Maximum Drawdown (MDD), risk parity, and concentration penalties—enabling gradient-based optimization without intermediate forecasting. The goal is to develop a custom loss function to train neural networks to generate robust, diversified portfolios that generalize across market regimes.

**Authors:** Rahul Kenneth fernandes, Atharva Atul Vaidya  
**Advisors:** Dr. Travis Desell   
**Institution:** Rochester Institute of Technology    

## Prerequisites
- Python 3.13.5
- Free [Fred API Key](https://fred.stlouisfed.org)

## Hardware Recommendations
- RAM: 16GB
- GPU: CUDA compatible NVIDIA GPU or MPS MacOS GPU

## Installation

### Clone Git Repository
```bash
git clone -b loss-functions https://github.com/travisdesell/exact.git
cd exact/financial_loss_functions
```

### Install Dependencies
1. Create and activate a virtual environment (optional but recommended) You can use either venv or conda.

venv:
```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv <env_name>
source <env_name>/bin/activate  # On Windows, use: <env_name>\Scripts\activate
```
OR

conda
```bash
conda create -n <env_name> python=3.13.5
conda activate <env_name>
```

2. Install python dependecies
```bash
# Install the required packages
pip install -r requirements.txt
```

### Setting Up Environment Variables
1. Create your local environment file:
```bash
cp .env.example .env
```
2. Update the .env file in root directory with your Fred API key


## Project Pipeline
1. Data Collection (`scripts.run_data_collection`): Fetches raw macro-economic data from FRED API
2. Data Processing (`scripts.run_processing`): Preprocesses raw data and places the files in data/processed/
3. Training (`scripts.run_training`): Trains a grid of models along with different loss functions on the processed data, runs classicial approaches of portfolio optimization, then compares and evaluates all methods.

## Quick Start
### 1. Run macro-economic data collection
```bash
python -m scripts.run_macro_collection
```

### 2. Run data processing
```bash
python -m scripts.run_processing
```

### 3. Run candidates training grid
Run everything (Uses defaults: grid_mode='all', loss_mode='all')
```bash
python -m scripts.run_training
```

### Run tests
```bash
pytest tests
```

## For Detailed Usage Instructions
- [Data Collection and Ingestion Guide](/financial_loss_functions/src/data_collection/README.md)
- [Training Guide](/financial_loss_functions/src/training/README.md)

## Directory Structure
```text
financial_loss_functions                    # Root directory for this project
├── config
│   └── paths.json
├── data
│   ├── processed
│   └── raw
│       ├── macro                           # gitignored, since data can be acquired
│       │   ├── Consumption_Orders_Inventories.csv
│       │   ├── Housing.csv
│       │   ├── Labor_Market.csv
│       │   ├── Money_Credit.csv
│       │   ├── Output_Income.csv
│       │   ├── Prices.csv
│       │   ├── Rates_FX.csv
│       │   └── Stock_Market.csv
│       └── sample                          # Contains synthetic CRSP-like sample data
│           ├── combined_predictors_test.csv
│           ├── combined_predictors_train.csv
│           └── combined_predictors_validation.csv
├── exploration
│   ├── crsp_exp.ipynb                      # Exploration of the CRSP dataset
│   └── fred_series_analysis.ipynb          # Exploration of the macro-economic data
├── pytest.ini
├── README.md                               # This file
├── requirements.txt                        # Python dependecies
├── scripts
│   ├── run_macro_collection.py             # Data collection
│   ├── run_processing.py                   # Data cleaning and processing
│   ├── run_training.py                     # All model training
│   └── utils.py
├── src
│   ├── __init__.py
│   ├── data_collection
│   │   ├── const.py                        # Contains fixed series IDs for FRED API
│   │   └── macro_api.py                    # Collects data from FRED API
│   ├── data_processing
│   │   ├── loading.py
│   │   ├── pipeline.py                     # Runs processing pipeline
│   │   └── preprocess.py
│   ├── models
│   │   ├── cov_models.py                   # Covariance-based classicial models
│   │   ├── examm.py                        # Python wrapper to run EXAMM model
│   │   └── pipeline.py                     # Runs training pipeline for all models
│   └── utils.py
├── tests
│   ├── cov_models_tests.py
│   ├── integration
│   │   └── test_macro_data_coll.py
│   └── unit
│       ├── test_cov_models.py
│       ├── test_loading.py
│       ├── test_macro_api.py
│       ├── test_preprocess.py
│       └── test_utils.py
├── .env                                   # Environment variables
└── .env.example                           # Template for storing environment variables
```

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

## Contact

**Rahul Keneth Fernandes**  
Email: rf4074@rit.edu  
Github: [@rahulkfernandes](https://github.com/rahulkfernandes)  

**Atharva Atul Vaidya**  
Email: aav6986@rit.edu  
Github: [@v-atharva](https://github.com/v-atharva)  