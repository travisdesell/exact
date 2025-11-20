import os
import shutil

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

def data_dir_check(macro_path: str):
    run_permission = False
    if os.path.exists(macro_path):
        print(macro_path, ', Directory Exists!!!!')
        choice = input('Are you sure you want to overwrite it? (Y/N): ').strip()
        if choice == 'Y':
            delete_directory(macro_path)
            create_directory(macro_path)
            run_permission = True
        else:
            print('Aborted. Directory not modified.')
            run_permission = False
    else:
        create_directory(macro_path)
        run_permission = True
    
    return run_permission