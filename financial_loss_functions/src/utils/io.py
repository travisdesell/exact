import os
import json
import shutil
import pickle
import pandas as pd
from pathlib import Path

def create_directory(path: str | Path) -> None:
    """
    Create a directory if it doesn't exist

    Args:
        path (str | Path) Path to directory
    """
    if not os.path.exists(path): 
        os.makedirs(path, exist_ok=True)
        print(f'{path} Directory Created!')

def delete_file(file_path: str | Path) -> None:
    """
    Delete file at given path

    Args:
        file_path (str | Path): Path to file to be deleted
    """
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print(f'WARNING: File at {file_path} not found.')

def delete_directory(dir_path: str | Path) -> None:
    """
    Delete folder at given path.

    Args:
        dir_path (str | Path): Path to directory to be deleted
    """
    try:
        shutil.rmtree(dir_path)  # Delete the folder and all its contents
        print(f"Folder '{dir_path}' and its contents have been deleted.")
    except FileNotFoundError:
        print(f"Folder '{dir_path}' does not exist.")
    except Exception as e:
        print(f'An error occurred: {e}')

def check_if_files_exist(
        paths_list: list[str | Path]
    ) -> dict[str | Path, bool]:
    """
    Check if all files exist.
    
    Args:
        paths_list (List[str | Path]): List of file path strings to be checked for existence.
    """
    existence = {}
    for path in paths_list:
        if os.path.exists(path):
            existence[path] = True
        else:
            existence[path] = False
    
    return existence

def raise_file_not_found(path: str | Path):
    if not os.path.exists(path):
        raise FileNotFoundError(
                f'Required file not found: {path}'
            )

def data_dir_check(path: str) -> bool:
    """
    Function to check if data directory exists and to overwrite depending on user input.

    Args:
        path (str): Path to data directory.
    
    Returns:
        run_permission (bool): Bool value to run or stop executing code if user input is N (no).
    """
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

def save_to_csv(data: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save dataframe to csv file.

    Args:
        data (pd.DataFrame): DataFrame to be saved as csv.
        output_path (str | Path): Output path where the file should be saved.
    """
    data.to_csv(output_path, sep=',')

def save_to_json(data: dict, output_file: str) -> None:
    """
    Saves dictionary to a json file.

    Args:
        data_dict (Dict): Dictionary containing data to be saved
        output_file (str | Path): Path to output file
    """
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        json_file.close()

def save_pickle_temp(data: dict, output_file: str | Path):
    """
    Saves a dictionary to a pickle file for temporary usage.
    
    Args:
        data_dict (Dict): Dictionary containing data to be saved
        output_file (str | Path): Path to output file
    """
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

def load_pickle_temp(file_path: str | Path) -> dict:
    """
    Loads saved pickle temp file

    Args:
        file_path (str | Path): Pickle file to be loaded
    
    Returns:
        pkl_data (dict): Loaded data from the pickle file
    """
    with open(file_path, 'rb') as f:
        pkl_data = pickle.load(f)
    
    return pkl_data

def reset_data_stage(dir_path: str | Path) -> None:
    """
    Docstring for reset_data_stage
    
    Args:
        dir_path (str): Directory path string which contains data for the particular stage
    """
    if os.path.exists(dir_path):
        print(dir_path, ', Directory exists. Overwriting.')
        delete_directory(dir_path)
        create_directory(dir_path)

    else:
        create_directory(dir_path)
        print(dir_path, ', Directory created.')

def load_json(path: str | Path) -> dict:
    """
    Load a single json file.

    Args:
        path (str | Path): Path to json file that should be loaded as a dict.
    
    Returns:
        config (dict): Loaded json file as dict.
    """
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def load_path_config(path: str, crsp_data_dir: str | None = None) -> dict:
    """
    Loads config.json and adds name of the CRSP data directory if needed.

    Args:
        path (str): Path to config file
        crsp_data_dir (str): Name of the directory where the CRSP data is stored

    Returns:
        config (dict): Config dictionary containg paths to files and directories.
    """
    config = load_json(path)

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

def artifact_paths_setup(paths_config: dict[str, dict]) -> dict[str, Path]:
    """
    Create artifact directories based on a configuration dictionary.

    Args:
        paths_config: Dictionary containing an 'artifacts' key with sub-directory 
        names and paths.

    Returns:
        Dictionary mapping artifact names to Path objects.
    """
    artifacts_paths = {}
    for name, path in paths_config['artifacts'].items():
        dir_path = Path(path)
        create_directory(dir_path)
        artifacts_paths[name] = dir_path
    
    return artifacts_paths