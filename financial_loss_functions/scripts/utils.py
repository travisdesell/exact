import json
import os
from pathlib import Path
from typing import Dict

def load_path_config(path: str, crsp_data_dir: str | None = None) -> Dict:
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

def load_config(path: str) -> Dict:
    with open(path, 'r') as f:
        config = json.load(f)
    return config
