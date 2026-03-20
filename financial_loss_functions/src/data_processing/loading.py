import pandas as pd
from pathlib import Path
from src.utils.io import check_if_files_exist

def load_raw_crsp_datasets(
        train_path: str, val_path: str, test_path: str
    )-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all CRSP datasets files from a directory which are split into train,
    validation and test.

    @param train_path str Path to raw train data file
    @param val_path str Path to raw validation data file
    @param test_path str Path to raw test data file
    
    @return Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] Raw train, val and test data
    """ 
    # Load split datasets
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    
    return train_data, val_data, test_data

def load_csv_files(paths_dict: dict[str, str], index_dt: bool = False) -> dict[str, pd.DataFrame]:
    """
    Loads csv data files. Provide dictionary of 
    name key and path strings value to be loaded.

    @param paths_dict dict[str, str] dictionary of name key and path strings value to be loaded
    
    @return dict[str, pd.DataFrame] dictionary of name key and loaded dataframe as value
    """
        
    loaded_dfs = {}
    for name, f_path in paths_dict.items():
        temp_df = pd.read_csv(f_path, index_col=0) # Can use parse_dates=True here,but
        if index_dt:
            temp_df.index = pd.to_datetime(temp_df.index) #.but pd.to_datetime for control.
        loaded_dfs[name] = temp_df

    return loaded_dfs

def load_macro_data(macro_dir_path: str) -> dict[str, pd.DataFrame]:
    """
    Loads macro-economic data csv files from given directory path.

    @param macro_dir_path str 
        Path to directory where macro-ecnomic data is store as separate csv files

    @return dict[str, pd.DataFrame] Contains category name as key and dataframe as value
    """
    
    file_paths = list(macro_dir_path.glob('*.csv')) # since data is collected as csv files

    if len(file_paths) == 0:
        raise FileNotFoundError(f'No CSVs not found in directory: {macro_dir_path}')

    macro_files = {}
    for f_path in file_paths:
        macro_files[f_path.stem] = f_path
    
    macro_data_dict = load_csv_files(macro_files)
    return macro_data_dict

def find_artifact_files(
        prefix: str, suffixes: list[str], dir_path: str | Path, ext: str
    ) -> dict[str, str]:
    paths_temp = []
    for suff in suffixes:
        paths_temp.append(
            (suff, dir_path / f'{prefix}_{suff}{ext}')
        )
    
    avg_perf_paths = {}
    existence = check_if_files_exist([tup[1] for tup in paths_temp])
    for path, status in existence.items():
        for tup in paths_temp:
            if path == tup[1]:
                if status:
                    avg_perf_paths.update(
                        {tup[0]: path}
                    )
                else:
                    print(f'{prefix.upper()} file for {tup[0]} not found at {tup[0]}. Skipping!')

    return avg_perf_paths