import json
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
from src.data_processing.preprocess_macro import (
    select_macro_for_pipeline,
    prepare_macro_splits,
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
    crsp_data_path = Path(paths_config['data']['crsp_dir'])
    train_path = crsp_data_path / paths_config['raw_files']['train']
    val_path = crsp_data_path / paths_config['raw_files']['val']
    test_path = crsp_data_path / paths_config['raw_files']['test']
    
    train_data, val_data, test_data = load_raw_crsp_datasets(
        train_path,
        val_path,
        test_path
    )

    # -------------------- CRSP Cleaning -------------------- #
    train_data, val_data, test_data = clean_inplace(train_data, val_data, test_data)

    # -------------------- CRSP Returns -------------------- #
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

    # -------------------- Macro Feature Selection & Alignment -------------------- #
    macro_dir_path = Path(paths_config['data']['raw_macro_dir'])
    macro_train, macro_val, macro_test = None, None, None

    if macro_dir_path.is_dir() and any(macro_dir_path.glob('*.csv')):
        raw_macro = load_macro_data(macro_dir_path)

        fs_config = features_config.get('feature_selection', {})
        lags = fs_config.get('lags', [10, 30, 50, 60])
        low_corr = fs_config.get('low_corr_threshold', 0.1)
        top_k = features_config.get('macro_per_stock', 2)

        # Build returns DataFrame with _RET columns for feature selection
        ret_cols_train = ret_train.copy()
        ret_cols_train.columns = [f'{c}_RET' for c in ret_cols_train.columns]

        filtered_macro, selected_cols = select_macro_for_pipeline(
            raw_macro=raw_macro,
            returns_train=ret_cols_train,
            lags=lags,
            top_k=top_k,
            low_corr_threshold=low_corr,
        )
        print(f'Feature selection complete: {len(selected_cols)} macro features selected.')

        macro_train, macro_val, macro_test = prepare_macro_splits(
            raw_macro=raw_macro,
            train_index=train_data.index,
            val_index=val_data.index,
            test_index=test_data.index,
            selected_cols=selected_cols,
        )
        print(f'Macro data aligned: train={macro_train.shape}, val={macro_val.shape}, test={macro_test.shape}')
    else:
        print('No macro data found, skipping macro feature selection.')

    # -------------------- Preprocessing -------------------- #
    nn_preprocessor = Preprocessor(
        common_features=features_config['common_features']
    )
    processed_train = nn_preprocessor.process_train_data(train_data, macro_data=macro_train)
    processed_val = nn_preprocessor.process_split_data(val_data, macro_data=macro_val)
    processed_test = nn_preprocessor.process_split_data(test_data, macro_data=macro_test)

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

    # Dump updated common features for downstream pipeline stages
    updated_features_path = paths_config['processed_paths'].get('updated_common_features')
    if updated_features_path:
        updated = sorted(nn_preprocessor.common_features or [])
        with open(updated_features_path, 'w') as f:
            json.dump(updated, f, indent=2)
        print(f'Updated common features ({len(updated)}) saved to {updated_features_path}')

    print('=' * 20, ' All Data Processing Completed! ', '=' * 20)
