# DEPRECATED: Data Collection & CRSP Data Set
[<- Back to Main README](/financial_loss_functions/README.md)

This project utilizes daily stock data of 50 stocks from the S&P500 along with macro-economic data from Federal Reserve pf St. Louis API (FRED). As the CRSP data cannot be redistributed, a synthetic sample dataset is provided. Instruction to collected macro-economic data and placement of a CRSP or CRSP-like dataset if given below.

### Macro-Economic Data Collection
#### 1. Get free API key from FRED API
- Free [Fred API Key](https://fred.stlouisfed.org)

#### 2. Create your local environment file and update .env with the FRED API key
```bash
cp .env.example .env
```
.env
```bash
FRED_KEY = "<API-KEY>" # API key from Fred API
```

#### Run macro-economic data collection
```bash
python -m scripts.run_macro_collection
```

### Injesting CRSP or CRSP Equivalent Dataset
Currently, a sythetic CRSP-like dataset is stored in `data/raw/sample`. If using a CRSP or CRSP equivalent dataset, place the directory in `data/raw/` and update the CRSP_DIR environment variable in the .env file with the name of the equivalent data directory.

#### 1. Update .env
.env
```bash
CRSP_DIR = "<equivalent-data-directory>" # Add directory name here if using CRSP like dataset
```

#### 2. Data directory placement and data split
Since we use pre-split data (train, val, test), the new files must follow the structure shown below and the config/paths.json must be updated to reflect the correct file names. 

- data/
    - raw/
        - `<equivalent-data-directory>/`
            - `<train_name>.csv`
            - `<validation_name>.csv`
            - `<test_name>.csv`

#### 3. Update config.paths.json
Update config/paths.json with the names of the files in the `<equivalent-data-directory>/`.

```
"raw_files": {
        "train": "<train_name>.csv",
        "val": "<validation_name>.csv",
        "test": "<test_name>.csv"
    },
```