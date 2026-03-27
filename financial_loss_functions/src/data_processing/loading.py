import pandas as pd
from pathlib import Path
from src.utils.io import check_if_files_exist, load_json

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


class ArtifactDataExtractor:
    def __init__(
            self,
            prev_grid_mode: str,
            artifacts_paths: dict[str, Path|str],
            model_names: list[str]|None = None,

        ):
        self.prev_grid_mode = prev_grid_mode
        self.artifacts_paths = artifacts_paths
        self.model_names = model_names # Model names are ignored for 'all' mode

        if prev_grid_mode == 'one_model' and not model_names:
            raise ValueError(
                'List of model names must be provided as suffixes for `one_model` prev_grid_mode.'
            )

    def find_artifact_files(
        self, prefix: str, suffixes: list[str], dir_path: str | Path, ext: str
    ) -> dict[str, str]:
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
    
    def _build_avg_perf_paths(
            self, prefix: str, suffixes: list[str]
        ) -> dict[str, Path|str]:
        # Average Performance files
        avg_perf_paths = self.find_artifact_files(
            prefix,
            suffixes,
            self.artifacts_paths['avg_perf_dir'],
            '.csv' # Average Performance files at csv
        )
        if len(avg_perf_paths) == 0:
            raise RuntimeError(
                'No average Performance files found. Run training and tuning first.'
            )
        
        return avg_perf_paths
    
    def _build_opti_hparams_paths(
            self, prefix: str, suffixes: list[str]
        ) -> dict[str, Path|str] | None:
        # Optimized Hyperparameter files
        opti_paths = self.find_artifact_files(
            prefix,
            suffixes,
            self.artifacts_paths['hparams_dir'],
            '.json' # Optimized Hyperparameter files are json
        )

        if len(opti_paths) == 0:
            print(
                'WARNING: Models not tuned! Using default hyperparameters.',
                'Tune models using `python -m scripts.run_training`'
            )
            opti_paths = None
        
        return opti_paths
    
    def aggregate_avg_perf(self, avg_perf_prefix: str):
        if self.prev_grid_mode == 'all':
            avg_perf_paths = self._build_avg_perf_paths(
                avg_perf_prefix, ['all']
            )
            if len(avg_perf_paths) > 1:
                raise RuntimeError('More than 1 file found for `all` mode.')
            
            avg_perf_dfs = load_csv_files(avg_perf_paths)

            all_avg_perf = avg_perf_dfs.values()[0] # There should be only 1 file
        
        elif self.prev_grid_mode == 'one_model':

            # Build paths and load files
            avg_perf_paths = self._build_avg_perf_paths(
                avg_perf_prefix, self.model_names
            )
            avg_perf_dfs = load_csv_files(avg_perf_paths)

            # Combine all files into one dataframe
            all_avg_perf = pd.concat(avg_perf_dfs.values(), axis=0)
            all_avg_perf = all_avg_perf[~all_avg_perf.index.duplicated(keep='first')]

        elif self.prev_grid_mode == 'one':
            print('!Mode not implemented yet!')
            exit()
        else:
            raise RuntimeError('Incorrect mode arguments while running at entry point.')
        
        return all_avg_perf
    
    def aggregate_optimized_hparams(self, opti_hparams_prefix: str):
        if self.prev_grid_mode == 'all':
            opti_paths = self._build_opti_hparams_paths(
                opti_hparams_prefix, ['all']
            )
            
            if opti_paths:
                optimized_hparams = {}
                for path in opti_paths.values():
                    optimized_hparams.update(load_json(path))
            else:
                optimized_hparams = None
        
        elif self.prev_grid_mode == 'one_model':
            
            # Build paths and load files
            opti_paths = self._build_opti_hparams_paths(
                opti_hparams_prefix, self.model_names
            )
            
            if opti_paths:
                optimized_hparams = {}
                for path in opti_paths.values():
                    optimized_hparams.update(load_json(path))
            else:
                optimized_hparams = None


        elif self.prev_grid_mode == 'one':
            print('!Mode not implemented yet!')
            exit()
        else:
            raise RuntimeError('Incorrect mode arguments while running at entry point.')
        
        return optimized_hparams