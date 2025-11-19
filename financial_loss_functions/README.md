# Financially Guided Neural Networks for Robust Portfolio Optimization

## Prerequisites
- Python 3.13.5

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
cp .env.example ../.env
```
2. Update the .env file with your Fred API key and absolute location of data directory

## Usage

### To run macro-economic data collection
```bash
python -m data_collectors.macro_api
```
