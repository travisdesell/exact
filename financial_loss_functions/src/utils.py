import os
import shutil
import pandas as pd
from typing import List

def create_directory(path: str) -> None:
    """
    Create a directory if it doesn't exist

    Parameters:
        path (str): Path to directory
    """
    if not os.path.exists(path): 
        os.makedirs(path)
        print(f'{path} Directory Created!')

def delete_file(file_path: str):
    """
    Delete file at given path
    """
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        raise FileNotFoundError(f'File does not exist at {file_path}')

def delete_directory(dir_path: str) -> None:
    """
    Delete folder at given path

    Parameters:
        dir_path (str): Path to directory
    """
    try:
        shutil.rmtree(dir_path)  # Delete the folder and all its contents
        print(f"Folder '{dir_path}' and its contents have been deleted.")
    except FileNotFoundError:
        print(f"Folder '{dir_path}' does not exist.")
    except Exception as e:
        print(f'An error occurred: {e}')

def check_if_files_exist(paths_list: List[str]):
    # Check if all files exist
    for path in paths_list:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Required file not found: {path}'
            )
        
def data_dir_check(path: str):
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

def save_to_csv(data: pd.DataFrame, output_path:str):
    data.to_csv(output_path, sep=',')

def reset_data_stage(dir_path: str):
    if os.path.exists(dir_path):
        print(dir_path, ', Directory exists. Overwriting.')
        delete_directory(dir_path)
        create_directory(dir_path)

    else:
        create_directory(dir_path)
        print(dir_path, ', Directory created.')
    