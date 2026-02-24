from pathlib import Path
from src.utils.io import reset_data_stage, save_to_csv
from src.data_processing.loading import load_raw_crsp_datasets, load_macro_data
from src.data_processing.preprocess_crsp import (
    clean_inplace,
    get_only_returns,
    Preprocessor
)

def run_processing_pipeline(paths_config: dict, features_config: dict):
    """
    Data Processing Pipeline entry point

    @param paths_config Dict Dictionary containing paths
    @param features_config Dict Dictionary containing features information
    """
    print('=' * 20, ' Data Processing Pipeline ', '=' * 20)
    
    # Reset directory
    reset_data_stage(Path(paths_config['data']['processed_dir']))
    
    # -------------------- Data Loading -------------------- #
    # Extracting raw data paths for crsp
    crsp_data_path = Path(paths_config['data']['crsp_dir'])
    train_path = crsp_data_path / paths_config['raw_files']['train']
    val_path = crsp_data_path / paths_config['raw_files']['val']
    test_path = crsp_data_path / paths_config['raw_files']['test']
    
    # Loading CRSP Dataset
    train_data, val_data, test_data = load_raw_crsp_datasets(
        train_path,
        val_path,
        test_path
    )

    # Loading raw macro-economic data
    # macro_dir_path = Path(paths_config['data']['raw_macro_dir']) 
    # raw_macro = load_macro_data(macro_dir_path) ### LOAD MACRO ###
    
    # -------------------- CRSP Cleaning -------------------- #
    train_data, val_data, test_data = clean_inplace(train_data, val_data, test_data)

    # -------------------- CRSP Split Returns and S&P 500 Returns -------------------- #
    # Common processing (realized returns)
    ret_train, ret_val, ret_test = get_only_returns(train_data, val_data, test_data)
    save_to_csv(
        ret_train,
        Path(paths_config['processed_paths']['returns_train'])
    )
    save_to_csv(
        ret_val,
        Path(paths_config['processed_paths']['returns_val'])
    )
    save_to_csv(
        ret_test,
        Path(paths_config['processed_paths']['returns_test'])
    )
    
    print('Realized returns extracted and saved.')

    # Extract and Save S&P 500 returns for benchmarks
    sp500_col_name = features_config['sp500_returns']
    
    save_to_csv(
        train_data[sp500_col_name],
        Path(paths_config['processed_paths']['benchmark_train'])
    )

    save_to_csv(
        val_data[sp500_col_name],
        Path(paths_config['processed_paths']['benchmark_val'])
    )

    save_to_csv(
        test_data[sp500_col_name],
        Path(paths_config['processed_paths']['benchmark_test'])
    )

    # -------------------- Macro Cleaning -------------------- #
    # TODO: For Atharva
    # 1. Combine all macro csv files (Already done using  MacroCombiner.combine_macro_data())
    # 2. Make Macro daily and match CRSP (Already done) [ONLY Forward fill, DO NOT backfill, its data leak!]
    # 3. Run feature selection for macro data using crsp train (Use features_config['macro_per_stock'] for number)
    # 4. Provide all three splits of macro data to the Preprocessor.process_*() methods (same as before)
    # 5. Dump a json of sorted common features list in data/processed/updated_common_features.json at the end
    #       Use Preprocessor.get_common_features(), use path in paths_config['data']['processed_paths'][updated_common_features']
    # Notes:
    #   Put all macro code in src/data_processing/preprocess_macro.py, just to put everything in one place
    #   !All file io operations must happen in this file (src/data_processing/pipeline.py)!
    #   Its important for now, for the columns to look like <ticker1>_<feature1>,..., <macro1>, <macro2>,...<macroN>
    #   Also see Preprocessor._update_common_features()

    #### Old Macro Code ####
    # # Process macro-economic data and align with CRSP dates
    # macro_preprocessor = MacroPreprocessor()
    # combined_macro = macro_preprocessor.combine_macro_data(raw_macro)
    # daily_macro = macro_preprocessor.to_daily(combined_macro)
    # macro_train, macro_val, macro_test = macro_preprocessor.split_by_crsp_dates(
    #     daily_macro,
    #     train_data.index,
    #     val_data.index,
    #     test_data.index
    # )
    # print('Macro data processed and aligned with CRSP splits.')

    # -------------------- Macro Feature Selection -------------------- #

    # -------------------- Preporcessing -------------------- #

    # Preprocessing for NN
    nn_preprocessor = Preprocessor(
        common_features = features_config['common_features']
    )
    processed_train = nn_preprocessor.process_train_data(train_data)
    processed_val = nn_preprocessor.process_split_data(val_data)
    processed_test = nn_preprocessor.process_split_data(test_data)

    save_to_csv(
        processed_train,
        Path(paths_config['processed_paths']['processed_train'])
    )
    save_to_csv(
        processed_val,
        Path(paths_config['processed_paths']['processed_val'])
    )
    save_to_csv(
        processed_test,
        Path(paths_config['processed_paths']['processed_test'])
    )

    print('Preprocessing for Neural Networks completed.')

    print('=' * 20, ' All Data Processing Completed! ', '=' * 20)
