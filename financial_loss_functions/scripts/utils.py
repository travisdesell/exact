import json
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
    
    if crsp_data_dir:
        config['data']['crsp_dir'] = config['data']['raw_dir'] + crsp_data_dir
    
    return config