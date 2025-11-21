# Financially Guided Neural Networks for Robust Portfolio Optimization

**Authors:** Rahul Kenneth fernandes, Atharva Atul Vaidya
**Advisors:** Dr. Travis Desell 
**Institution:** Rochester Institute of Technology  

## Prerequisites
- Python 3.13.5
- Free [Fred API Key](https://fred.stlouisfed.org) 

## Installation

### Clone Git Repository
```bash
git clone -b loss-functions https://github.com/travisdesell/exact.git
cd exact/financial_loss_functions
```

### Install Dependencies
```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

### Setting Up Environment Variables
1. Create your local environment file:
```bash
cp .env.example .env
```
2. Update the .env file in root directory with your Fred API key and absolute location of data directory

## Usage

### 1. Run macro-economic data collection
```bash
python -m scripts.run_macro_collection
```

### 2. Run data processing
```bash
python -m scripts.run_processing
```

### Run tests
```bash
pytest tests
```

## Directory Structure
- `exact/`
    - `loss_functions/`: Root directory for this project
        - `data/`
            - `processed/`:
            - `raw/`:
                - `2023_sp_500_select_50/`: Contains CRSP dataset for 50 selected companies from S&P 500
                - `macro/`: Contains CSV files of macro-economic economic data
                - `sample/`: Contains sample data
        - `exploration/`
            - `crsp_exp.ipynb`: Exploration of the CRSP dataset
            - `fred_series_analysis.ipynb`: Exploration of the macro-economic data
        - `scripts/`
            - `run_macro_collection.py`: Data collection
            - `run_processing.py`: Data cleaning and processing
        - `src/`
            - `data_collection/`
                - `const.py`: Contains fixed series IDs for FRED API
                - `macro_api.py`: Collects data from FRED API
            - `data_processing/`
                - `pipeline.py`: Runs processing pipeline
                - `preprocess.py`: Cleaning and preprocessing
            - `models/`
                - `cov_models.py`: Covariance-based classicial models
                - `examm.py`: Python wrapper to run EXAMM model
            - `__init__.py`
            - `main.py`
            - `utils.py`: Utilities functions
        - `tests/`
            - `integration/`
                - `test_macro_data_coll.py`
            - `unit/`
                - `test_cov_models.py`
                - `test_macro_api.py`
                - `test_preprocess.py`
                - `test_utils.py`
        - `.env`: Environment variables
        - `.env.example`: Template  for storing environment variables
        - `pytest.ini`
        - `README.md`: This file
        - `requirements.txt`: Python dependecies
    - `.gitignore`


## Contact
**Rahul Keneth Fernandes**
Email: rf4074@rit.edu
Github: [@rahulkfernandes](https://github.com/rahulkfernandes)

**Atharva Atul Vaidya**
Email: aav6986@rit.edu
Github: [@v-atharva](https://github.com/v-atharva)