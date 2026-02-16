import os
import json
import torch
import shutil
import pandas as pd
from pathlib import Path

def create_directory(path: str) -> None:
    """
    Create a directory if it doesn't exist

    @param path str Path to directory
    """
    if not os.path.exists(path): 
        os.makedirs(path)
        print(f'{path} Directory Created!')

def delete_file(file_path: str) -> None:
    """
    Delete file at given path

    @param file_path str Path to file to be deleted
    """
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        raise FileNotFoundError(f'File does not exist at {file_path}')

def delete_directory(dir_path: str) -> None:
    """
    Delete folder at given path

    @param dir_path str Path to directory to be deleted
    """
    try:
        shutil.rmtree(dir_path)  # Delete the folder and all its contents
        print(f"Folder '{dir_path}' and its contents have been deleted.")
    except FileNotFoundError:
        print(f"Folder '{dir_path}' does not exist.")
    except Exception as e:
        print(f'An error occurred: {e}')

def check_if_files_exist(paths_list: list[str]) -> None:
    """
    Check if all files exist
    
    @param paths_list List[str] List of file path strings to be checked for existance
    """
    for path in paths_list:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Required file not found: {path}'
            )
        
def data_dir_check(path: str) -> bool:
    run_permission = False
    if os.path.exists(path):
        print(path, ', Directory Exists!!!!')
        choice = input('Are you sure you want to overwrite it? (Y/N): ').strip()
        if choice == 'Y':
            delete_directory(path)
            create_directory(path)
            run_permission = True
        else:
            print('Aborted. Directory not modified.')
            run_permission = False
    else:
        create_directory(path)
        run_permission = True
    
    return run_permission

def save_to_csv(data: pd.DataFrame, output_path:str) -> None:
    data.to_csv(output_path, sep=',')

def reset_data_stage(dir_path: str) -> None:
    """
    Docstring for reset_data_stage
    
    @param dir_path str Directory path string which contains data for the particular stage
    """
    if os.path.exists(dir_path):
        print(dir_path, ', Directory exists. Overwriting.')
        delete_directory(dir_path)
        create_directory(dir_path)

    else:
        create_directory(dir_path)
        print(dir_path, ', Directory created.')

def load_path_config(path: str, crsp_data_dir: str | None = None) -> dict:
    """
    Loads config.json and adds name of the CRSP data directory if needed.

    Parameters
    ----------
    path: str
        Path to config file
    crsp_data_dir: str
        Name of the directory where the CRSP data is stored

    Returns
    -------
    config: Dict
        Config dictionary containg paths to files and directories
    """
    with open(path, 'r') as f:
        config = json.load(f)

    config_path = Path(path).resolve()
    repo_root = config_path.parent.parent

    # Resolve base directories
    raw_dir = config['data']['raw_dir']
    raw_root = Path(raw_dir)
    if not raw_root.is_absolute():
        raw_root = (repo_root / raw_root).resolve()

    processed_dir = Path(config['data']['processed_dir'])
    if not processed_dir.is_absolute():
        processed_dir = (repo_root / processed_dir).resolve()
    config['data']['processed_dir'] = str(processed_dir)

    raw_macro_dir = Path(config['data']['raw_macro_dir'])
    if not raw_macro_dir.is_absolute():
        raw_macro_dir = (repo_root / raw_macro_dir).resolve()
    config['data']['raw_macro_dir'] = str(raw_macro_dir)

    # Resolve CRSP directory
    if crsp_data_dir:
        if os.path.isabs(crsp_data_dir):
            crsp_dir = Path(crsp_data_dir).resolve()
        else:
            crsp_dir = (raw_root / crsp_data_dir).resolve()
    else:
        default_dir = (raw_root / '2023_sp_500_select_50').resolve()
        if not default_dir.is_dir():
            raise FileNotFoundError(
                f'CRSP directory not provided and default path missing: {default_dir}'
            )
        crsp_dir = default_dir
    config['data']['crsp_dir'] = str(crsp_dir)

    # Make processed file paths absolute
    processed_paths = {}
    for key, rel_path in config.get('processed_paths', {}).items():
        p = Path(rel_path)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        processed_paths[key] = str(p)
    config['processed_paths'] = processed_paths

    return config

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        config = json.load(f)
    return config


def get_best_device() -> torch.device:
    if torch.backends.mps.is_available():
        print('Using mps for GPU acceleration.')
        return torch.device('mps')
    
    elif torch.cuda.is_available():
        print('Using cuda for GPU acceleration.')
        return torch.device('cuda')
    
    else:
        print('No GPU acceleration. Using CPU.')
        return torch.device('cpu')

def extract_req_cols(columns_list: list, suffix: str) -> list:
    """
    Extract required columns based on the suffix in the column names. e.g., NSDN_RETURN

    @param columns_list list List of all column names.
    @param suffix str 
        Suffix str to extract its respective columns. e.g., VOL_CHANGE, RETURN
    
    @return required_cols List of required column names for the given suffix
    """
    required_cols = [col for col in columns_list if suffix in col]
    return required_cols