from typing import Dict
from pathlib import Path
from src.utils import reset_data_stage, save_to_csv
from src.data_processing.loading import load_raw_crsp_datasets, load_macro_data
from src.data_processing.preprocess import (
    MacroCombiner,
    clean_inplace,
    get_only_returns,
    Preprocessor
)


def run_processing_pipeline(paths_config: Dict, features_config: Dict):
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

    # Extracting macro-economic directory path
    # macro_dir_path = Path(paths_config['data']['raw_macro_dir']) 
    
    #### Uncomment All these lines to add macro-ecoomic data ####

    # # Loading raw macro-economic data
    # raw_macro = load_macro_data(macro_dir_path). #### MACRO REMOVED FOR TESTING
    
    # -------------------- Cleaning -------------------- #
    train_data, val_data, test_data = clean_inplace(train_data, val_data, test_data)

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

    # -------------------- Preporcessing -------------------- #
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
