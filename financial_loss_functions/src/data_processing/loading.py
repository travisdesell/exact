import pandas as pd
from pathlib import Path
from src.utils.io import check_if_files_exist, load_json

def load_raw_crsp_datasets(
        train_path: str, val_path: str, test_path: str
    )-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all CRSP datasets files from a directory which are split into train,
    validation and test.

    Args:
        train_path (str): Path to raw train data file.
        val_path (str): Path to raw validation data file.
        test_path (str): Path to raw test data file.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Raw train, val and test data.
    """ 
    # Load split datasets
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    
    return train_data, val_data, test_data

def load_single_csv(file_path: str | Path, index_dt: bool = True) -> pd.DataFrame:
    """
    Load a single CSV file.

    Args:
        file_path (str | Path): Path to csv file to be loaded.
        index_dt (bool): Convert index to Pandas DateTime if index is a date. Default = True.
    
    Returns:
        df (pd.DataFrame): Loaded csv file as a pandas dataframe.
    """
    df = pd.read_csv(file_path, index_col=0)
    if index_dt:
        df.index = pd.to_datetime(df.index)
    
    return df

def load_csv_files(
        paths_dict: dict[str, str | Path], index_dt: bool = True
    ) -> dict[str, pd.DataFrame]:
    """
    Loads a collection of csv data files. Provide dictionary of 
    name key and path strings value to be loaded. 
    This is done to maintain strict ordering of input and output of files.

    Args:
        paths_dict (dict[str, str | Path]): Dictionary of name key and path strings value to be loaded.
        index_dt (bool): Convert index of each file to Pandas DateTime if index is a date. Default = True.

    Returns:
        loaded_dfs (dict[str, pd.DataFrame]): Dictionary of name key and loaded dataframe as value.
    """
        
    loaded_dfs = {}
    for name, f_path in paths_dict.items():
        temp_df = load_single_csv(f_path, index_dt)
        loaded_dfs[name] = temp_df

    return loaded_dfs

def load_macro_data(macro_dir_path: str | Path) -> dict[str, pd.DataFrame]:
    """
    Loads macro-economic data csv files from given directory path.

    Args:
        macro_dir_path (str | Path): Path to directory where macro-economic data 
        is stored as separate csv files.

    Returns:
        macro_data_dict (dict[str, pd.DataFrame]): Contains category name as key and dataframe as value.
    """
    
    file_paths = list(macro_dir_path.glob('*.csv')) # since data is collected as csv files

    if len(file_paths) == 0:
        raise FileNotFoundError(f'No CSVs not found in directory: {macro_dir_path}')

    macro_files = {}
    for f_path in file_paths:
        macro_files[f_path.stem] = f_path
    
    macro_data_dict = load_csv_files(macro_files)
    return macro_data_dict


class ArtifactDataExtractor:
    """
    Class to extract artifacts data from the artifacts directory. Can extract average performances,
    optimized hyperparameters and daily return performances.
    """
    def __init__(
            self,
            prev_grid_mode: str,
            artifacts_paths: dict[str, Path | str]
        ):
        """
        Initialize ArtifactsExtractor instatnce to extract files like daily returns, 
        average portfolio performances and optimized hyperparameter from artifacts directory. 

        Args:
            prev_grid_mode (str): Grid mode used in the previous pipeline. Either 'one_model' or 'one'.
            artifacts_paths (dict[str, Path | str]): Dictionary containing paths of artifact files.
        
        Raises:
            ValueError: If Incorrect mode is provided. Must be 'one_model' or 'one'.
        """
        self.prev_grid_mode = prev_grid_mode
        self.artifacts_paths = artifacts_paths

        if self.prev_grid_mode not in ['one_model', 'one']:
            raise ValueError('Incorrect mode arguments while running at entry point.')

    def find_artifact_files(
        self, prefix: str, suffixes: list[str], dir_path: str | Path, ext: str
    ) -> dict[str, str]:
        """
        Find the artifact files in the directory and return a dict with files and thei existence status.

        Args:
            prefix (str): String prefix given to files while saving them.
            suffixes (str): List of model name suffixes given while saving files.
            dir_path (str | Path): Path to directory being searched for the files.
            ext (str): File extentions like '.csv' or '.json'.
        
        Returns:
            arti_paths (dict[str, str]): Dictonary containing existence status and file paths. 
        """
        paths_temp = []
        for suff in suffixes:
            paths_temp.append(
                (suff, dir_path / f'{prefix}_{suff}{ext}')
            )
        
        arti_paths = {}
        existence = check_if_files_exist([tup[1] for tup in paths_temp])
        for path, status in existence.items():
            for tup in paths_temp:
                if path == tup[1]:
                    if status:
                        arti_paths.update(
                            {tup[0]: path}
                        )
                    else:
                        print(f'{prefix.upper()} file for {tup[0]} not found at {tup[0]}. Skipping!')

        return arti_paths
    
    def agg_avg_perf(
            self, avg_perf_prefix: str, model_names: list[str]
        ) -> pd.DataFrame:
        """
        Load and aggregate average model portfolio performance files using the given prefix 
        and list of model names (suffixes).

        Args:
            avg_perf_prefix (str): Prefix for the average performance files.
            model_names (list[str]): List of model names which are the suffix to their files.
        
        Returns:
            all_avg_perf (pd.DataFrame): Dataframe containing aggregated average model portfolio performances.
        
        Raises:
            RuntimeError: If no average performance files are found.
            ValueError: This method does not work when previoud grid mode is 'one'.
        """
        if self.prev_grid_mode == 'one_model':
            # Build paths
            avg_perf_paths = self.find_artifact_files(
                avg_perf_prefix,
                model_names,
                self.artifacts_paths['avg_perf_dir'],
                '.csv' # Average Performance files at csv
            )
            if len(avg_perf_paths) == 0:
                raise RuntimeError(
                    'No average Performance files found. Run training and tuning first.'
                )
            
            # Load CSV files
            avg_perf_dfs = load_csv_files(avg_perf_paths, index_dt=False)

            # Combine all files into one dataframe
            all_avg_perf = pd.concat(avg_perf_dfs.values(), axis=0)
            all_avg_perf = all_avg_perf[~all_avg_perf.index.duplicated(keep='first')]

        elif self.prev_grid_mode == 'one':
            raise ValueError('`aggregate_avg_perf()`, does not work for `one` mode.')

        return all_avg_perf
    
    def agg_opti_hparams(
            self, opti_hparams_prefix: str, model_names: list[str]
        ) -> dict:
        """
        Load and aggregate optimized hyperparemeter json files using the given prefix 
        and list of model names (suffixes).

        Args:
            opti_hparams_prefix (str): Prefix for the optimized hyperparameter files.
            model_names (list[str]): List of model names which are the suffix to their files.
        
        Returns:
            optimized_hparams (dict): Dictionary containing the optimized hyperparemeters 
                for each model loss combination.
        
        Raises:
            ValueError: If more than one model+loss combination is present in the model_names list, 
                when previous grid mode is 'one'.
        """
        if self.prev_grid_mode == 'one' and len(model_names) != 1:
            raise ValueError(
                'Provided grid mode is `one`, but more than one model-loss provided.'
            )
        
        # Build paths
        opti_paths = self.find_artifact_files(
            opti_hparams_prefix,
            model_names,
            self.artifacts_paths['hparams_dir'],
            '.json' # Optimized Hyperparameter files are json
        )

        # load files if paths exist
        if len(opti_paths) != 0:
            optimized_hparams = {}
            for path in opti_paths.values():
                optimized_hparams.update(load_json(path))
        else:
            print(
                'WARNING: Models not tuned! No optimized hyperparameters found.',
                'Tune models using `python -m scripts.run_training <options> -t`'
            )
            optimized_hparams = None

        return optimized_hparams
    
    def agg_daily_rets(self, rets_prefix: str, model_names: list[str]) -> dict:
        """
        Load and aggregate the daily returns files using the given prefix 
        and list of model names (suffixes).

        Args:
            rets_prefix (str): Prefix for the daily returns files.
            model_names (list[str]): List of model names which are the suffix to their files.

        Returns:
            daily_rets (dict): Dictionary containing the daily returns for each model+loss 
                combination for each evaluation window.
        """
        # Build paths
        rets_paths = self.find_artifact_files(
            rets_prefix,
            model_names,
            self.artifacts_paths['wfv_rets_dir'],
            '.json'
        )

        # load returns json files if the paths exist
        if len(rets_paths) != 0:
            daily_rets = {}
            for path in rets_paths.values():
                daily_rets.update(load_json(path))
        else:
            print(
                'WARNING: No daily returns found.'
            )
            daily_rets = None
        
        return daily_rets