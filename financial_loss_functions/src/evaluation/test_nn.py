import sys
import torch
import traceback
import numpy as np
import pandas as pd
from typing import Callable, Type, Any
from src.utils.constants import MODEL_LOSS_SEP
from src.utils.formatting import reformat_hparams
from src.training.train_nn import WalkerGridUtilities, Walker
from src.utils.io import save_pickle_temp, load_pickle_temp, delete_file

class WFTester(WalkerGridUtilities):
    """
    Class to evaluate selected neural network model + loss function combinations on 
    the unseen test data.

    Attributes:
        temp_wts_prefix (str): This is the prefix to be used while saving temporary portfolio 
            allocation weight files used during distributed evaluation using mpi.
        temp_losses_prefix (str): This is the prefix to be used while saving temporary model 
            train-eval loss curve files used during distributed evaluation using mpi.
        seed_list_key (str): This is the string key in the hparams config file that contains the 
            list of seeds.
    """
    temp_wts_prefix = 'test_temp_alloc_wts'
    temp_losses_prefix = 'test_temp_losses'
    seed_list_key = 'seed_list'
    
    def __init__(
            self,
            model_lib: dict[str, dict[str, Type]],
            loss_lib: dict[str, dict[str, dict[str, Callable]]],
            hparams_config: dict[str, Any],
            num_steps: int,
            common_features: list[str],
            torch_device: torch.device | str,
            mpi: bool = False,
            temp_dir: str | None = None,
            optimized_hparams: dict | None = None,
            enable_diagnostics: bool = False
        ):
        """
        Initialize WFTester object to evaluate the selected model-loss combinations 
        on the test data. This object can run sequentially as well as parallel using
        mpi for distributed evaluation.

        train_eval_* methods can be run only once with each instance.

        Args:
            model_lib (dict[str, dict[str, Type]]): Neural network model architecture 
                library as a dict.
            loss_lib (dict[str, dict[str, dict[str, Callable]]]): Loss functions library 
                as a dict.
            hparams_config (dict[str, Any]): Dictionary containing default hyperparameters 
                and tuning ranges.
            num_steps: (int): Number of walk forward steps to be taken.
            common_features (list[str]): List of common features in the dataset, eg., S&P500 returns.
                This is used for reshaping for different types of broadcasting + reshaping.
            torch_device (torch.device | str): Device to run the PyTorch models.
            mpi (bool): Toggle the use of mpi for distributed evaluation of model-loss combinations.
            temp_dir: (str | None): Directory to save temporary files after a rank has completed its
                work.
            optimized_hparams (dict | None): Dictionary containing optimized hyperparameters for each 
                model-loss combination. Default = False.
            enable_diagnostics (bool): Toggle to print statements about memory usuage during 
                train_eval_* methods. Default = False.
        """
        super().__init__(
            model_lib, loss_lib, hparams_config, num_steps, 
            common_features, torch_device, mpi, temp_dir
        )

        self.optimized_hparams = optimized_hparams
        self.enable_diagnostics = enable_diagnostics

        self.seed_list = self.hparams_config.get(self.seed_list_key)
        if self.seed_list:
            self.multi_seed = True
        else:
            self.multi_seed = False
        
        self.train_eval_losses = {}
        self.all_alloc_weights = {}
    
    def _merge_all_results(
            self,
            size: int,
            temp_wts_prefix: str,
            temp_losses_prefix: str
        ) -> tuple[dict, dict]:
        """
        Merge all temporary allocation weights and train-eval losses into two combined dicts, 
        if rank is 0, i.e., main process. After the merging is done, it deletes the 
        temporary pkl files.

        Args:
            size (int): Size of the mpi comm world.
            temp_wts_prefix (str): The prefix used while saving temporary portfolio 
                allocation weight files.
            temp_losses_prefix (str): The prefix used while saving temporary model 
                train-eval loss curve files.
        
        Returns:
            tuple[dict, dict]: Dictionary [0] is the combined portfolio allocation weights
                and dictionary [1] is the combined train-eval loss curve values.
        """
        all_alloc_weights = {}
        train_eval_losses = {} #### Must be same as in constructor ####
        
        for r in range(size):
            # Load all temp alloc wt files
            rank_alloc_weights = load_pickle_temp(
                self.temp_dir / f'{temp_wts_prefix}_{r}.pkl'
            )
            # Merge into self.all_alloc_weights
            for model_loss, models_dict in rank_alloc_weights.items():
                all_alloc_weights[model_loss] = models_dict
            
            rank_losses = load_pickle_temp(
                self.temp_dir / f'{temp_losses_prefix}_{r}.pkl'
            )
            # Merge into self.train_val_losses
            for model_loss, losses_dict in rank_losses.items():
                train_eval_losses[model_loss] = losses_dict

        # Delete all temp files
        for r in range(size):
            delete_file(self.temp_dir / f'{temp_wts_prefix}_{r}.pkl')
            delete_file(self.temp_dir / f'{temp_losses_prefix}_{r}.pkl')

        print('All temp files merged and then deleted.')

        return all_alloc_weights, train_eval_losses

    def _build_combos(
            self, selected_combos: list[tuple[str, str]]
        ) -> list[tuple[str, Type, str, Callable]]:
        """
        Collect classes and functions of the selected model-loss combinations from the 
        NNModelLibrary and the LossFunctionLibrary. Build a list of tuples with all the names, 
        classes, and functions.

        Args:
            selected_combos (list[tuple[str, str]]): The string names of models and loss functions 
                in the format [(<model_name>, <loss_name>),...]
        
        Returns:
            all_combos (list[tuple[str, Type, str, Callable]]): List of tuples containing the loss name, 
                loss function, model name and model class.
        """
        # Collect models and losses from libraries
        collected_from_libraries = {}

        models = set()
        losses = set()
        
        for modl_loss in selected_combos:
            models.add(modl_loss[0])
            losses.add(modl_loss[1])

        for modl in models:
            temp_model_cls = self._search_model(modl)
            
            if not temp_model_cls: # model not found
                raise RuntimeError(f'{modl} MODEL NOT FOUND IN LIBRARY!')
            else:
                collected_from_libraries[modl] = temp_model_cls
        
        for loss in losses:
            temp_loss_func = self._search_loss_func(loss)

            if not temp_loss_func:
                raise RuntimeError(f'{loss} LOSS FUNCTION NOT FOUND IN LIBRARY!')
            else:
                collected_from_libraries[loss] = temp_loss_func
        
        # Build list of combos and their names
        all_combos = []
        for modl_loss in selected_combos:
            # (loss_name, loss_func, model_name, model_class)
            all_combos.append((
                modl_loss[0], collected_from_libraries[modl_loss[0]],
                modl_loss[1], collected_from_libraries[modl_loss[1]]
            ))
        
        return all_combos
        
    def _walker_helper(
        self,
        model_name: str,
        model_class: Type,
        loss_name: str,
        loss_func: Callable,
        train_data: np.ndarray,
        rets_train: np.ndarray, 
        test_data: np.ndarray,
        rets_test: np.ndarray
    ) -> tuple[
        np.ndarray | dict[str, np.ndarray], 
        dict[str, list] | dict[str, dict[str, list]]
        ]:
        """
        Walk the given model loss combination over the test data set with retraining at each step.

        Args:
            model_name (str): Name of the neural network model architecture.
            model_class (Type): Class of the neural network model architecture.
            loss_name (str): Name of the loss function to be used for the training.
            loss_func (Callable): Loss function to be used for the training.
            train_data (np.ndarray): Train data split that contains all the features.
            rets_train (np.ndarray): Train data split that contains only returns data for all stocks.
            test_data (np.ndarray): Test data split that contains all the features.
            rets_test (np.ndarray): Test data split that contains only the returns data for all stocks.
        
        Returns:
            tuple[np.ndarray | dict[str, np.ndarray], dict[str, list] | dict[str, dict[str, list]]]:
                Tuple containing the alloc_weights ([0]), train_val_losses ([1]).
        """
        model_loss_name = f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
        
        if self.enable_diagnostics:
            print(f'\n[Before training {model_name} with {loss_name}]')
            self._memory_diagnostics()

        # Gather best hyperparameters or use defaults
        if self.optimized_hparams and model_loss_name in self.optimized_hparams:
            combo_hparams = self.optimized_hparams[model_loss_name]
            
        else:
            print('!No optimized hyperparameters provided, using defaults!')
            model_cfg = self.hparams_config[self.mdls_hparams_name][model_name]
            loss_cfg = self.hparams_config[self.ls_hparams_name].get(loss_name, {})

            combo_hparams = reformat_hparams(model_cfg, loss_cfg)

        #### FOR TESTING ####
        # combo_hparams['train']['epochs'] = 1
        ####################
        
        if self.multi_seed:
            seed_ls_len = len(self.seed_list)
            alloc_weights = {}
            train_val_losses = {}
            for idx, seed in enumerate(self.seed_list, 1):
                print(
                    '\n','-'*20,
                    f'Running WF for {model_name}-{loss_name}, seed {seed}, {idx}/{seed_ls_len}',
                    '-'*20
                )
                walker = Walker(
                    self.num_steps,
                    model_name,
                    model_class,
                    loss_name,
                    loss_func,
                    combo_hparams,
                    self.torch_device,
                    self.reshaper,
                    seed
                )
                seed_alloc_weights = walker.walk_1_model(
                    train_data,
                    rets_train, 
                    test_data,
                    rets_test
                )

                seed_train_val_losses = walker.get_train_eval_losses()

                alloc_weights[seed] = seed_alloc_weights
                train_val_losses[seed] = seed_train_val_losses
        else:
            print(f'No seed list provided. Running {model_name} only once.')
            walker = Walker(
                self.num_steps,
                model_name,
                model_class,
                loss_name,
                loss_func,
                combo_hparams,
                self.torch_device,
                self.reshaper
            )

            alloc_weights = walker.walk_1_model(
                train_data,
                rets_train, 
                test_data,
                rets_test
            )

            train_val_losses = walker.get_train_eval_losses()
        
        if self.enable_diagnostics:
            print(f'\n[After training {model_name} with {loss_name}]')
            self._memory_diagnostics()

        return alloc_weights, train_val_losses
    
    def test_all(
            self,
            selected_combos: list[tuple[str, str]],
            train_data: pd.DataFrame,
            rets_train: pd.DataFrame, 
            test_data: pd.DataFrame,
            rets_test: pd.DataFrame,
            comm = None,
            global_rank = None,
            size = None
        ) -> dict[str, dict[str, np.ndarray]] | None:
        """
        Run evaluation of all the selected model-loss combinations on the test data. 
        This is an entry point method for this process. It can run sequentially as well as
        in parallel using mpi to distribute the model-loss combinations across nodes and gpus.
        
        Args:
            selected_combos (list[tuple[str, str]]): Selected model-loss combinations to be evaluated 
                on the test data.
            train_data (pd.DataFrame): Train data split that contains all the features. 
                Feature columns must be in the format <ticker>_<feature> and <comon_feature>.
            rets_train (pd.DataFrame): Train data split that contains only returns data for all stocks.
                Returns columns must be in the format <ticker>.
            test_data (pd.DataFrame): Test data split that contains all the features.
                Feature columns must be in the format <ticker>_<feature> and <comon_feature>.
            rets_test (pd.DataFrame): Test data split that contains only the returns data for all stocks.
                Returns columns must be in the format <ticker>.
            comm: MPI Communication object. Default = None.
            global_rank: Global rank of the current rank this is being executed on. Default = None.
            size: Size of the mpi communication world, i.e., number of ranks. Default = None.

        Returns:
            all_alloc_weights (dict[str, dict[str, np.ndarray]] | None): Portfolio allocation weights 
                for all the portfolio optimizer models and for all output windows. 
                Return is None only for non-zero ranks.
        """

        self._data_check(train_data, rets_test)
        self._trained_check()

        # Extract feature data
        self.reshaper.extract_features(train_data.columns)
        
        # Convert all dataframes to arrays
        train_data, rets_train, test_data, rets_test = self._convert_datasets_to_np(
            train_data, rets_train, test_data, rets_test
        )

        # Pre-search models and loss functions from library (to avoid redundant searches)
        all_combos = self._build_combos(selected_combos)
        total_train_count = len(all_combos)
        
        if self.mpi == False:
            for idx, (model_name, model_class, loss_name, loss_func) in enumerate(all_combos, 1):
                print(
                    '\n', '-'*10,
                    f' Testing {model_name}-{loss_name}, {idx}/{total_train_count}',
                    '-'*10
                )
                try:        
                    alloc_weights, train_eval_losses = self._walker_helper(
                        model_name,
                        model_class, 
                        loss_name,
                        loss_func,
                        train_data,
                        rets_train, 
                        test_data,
                        rets_test
                    )
                    self.all_alloc_weights[
                        f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
                    ] = alloc_weights
                    self.train_eval_losses[
                        f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
                    ] = train_eval_losses

                except Exception as error:
                    print(
                        f'DEBUG: Error while testing {model_name} with {loss_name}. Skipping.',
                        error
                    )
                    traceback.print_exc()
                    continue

            return self.all_alloc_weights
        else:
            self._mpi_setup_check([comm, global_rank, size])

            this_ranks_combos = self._select_ranks_combos(all_combos, global_rank, size)

            # Print summary on each rank
            print(f'Rank {global_rank}: Testing {len(this_ranks_combos)} combos.')
            sys.stdout.flush()

            # Local results dictionary
            local_alloc_weights = {}
            local_train_eval_losses = {}
            for idx, (model_name, model_class, loss_name, loss_func) in enumerate(this_ranks_combos, 1):
                print(f'\nRank {global_rank}: {idx}/{len(this_ranks_combos)} - {model_name} - {loss_name}')
                try:        
                    alloc_weights, train_eval_losses = self._walker_helper(
                        model_name,
                        model_class, 
                        loss_name,
                        loss_func,
                        train_data,
                        rets_train, 
                        test_data,
                        rets_test
                    )
                    local_alloc_weights[
                        f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
                    ] = alloc_weights
                    local_train_eval_losses[
                        f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
                    ] = train_eval_losses

                except Exception as error:
                    print(
                        f'DEBUG: Error while testing {model_name} with {loss_name}. Skipping.',
                        error
                    )
                    traceback.print_exc()
                    continue
            
            # Save local results to a rank‑specific file
            save_pickle_temp(
                local_alloc_weights,
                self.temp_dir / f'{self.temp_wts_prefix}_{global_rank}.pkl'
            )
            save_pickle_temp(
                local_train_eval_losses,
                self.temp_dir / f'{self.temp_losses_prefix}_{global_rank}.pkl'
            )

            # Wait for all ranks to finish
            comm.Barrier()

            # Rank 0 collects and merges all files
            if global_rank == 0:
                self.all_alloc_weights, self.train_eval_losses = self._merge_all_results(
                    size,
                    self.temp_wts_prefix,
                    self.temp_losses_prefix
                )
                
                return self.all_alloc_weights
            else:
                return None
    
    def test_one(
            self,
            model_name: str,
            loss_name: str,
            train_data: pd.DataFrame,
            rets_train: pd.DataFrame, 
            test_data: pd.DataFrame,
            rets_test: pd.DataFrame,
        ) -> dict[str, dict[str, np.ndarray]]:
            """
            Run evaluation of one model-loss combination on the test data. 
            This is an entry point method for this process. It can run only sequentially.
            
            Args:
                model_name (str): Name of the neural network model architecture.
                loss_name (str): Name of the loss function to be used with the neural network model.
                train_data (pd.DataFrame): Train data split that contains all the features. 
                    Feature columns must be in the format <ticker>_<feature> and <comon_feature>.
                rets_train (pd.DataFrame): Train data split that contains only returns data for all stocks.
                    Returns columns must be in the format <ticker>.
                test_data (pd.DataFrame): Test data split that contains all the features.
                    Feature columns must be in the format <ticker>_<feature> and <comon_feature>.
                rets_test (pd.DataFrame): Test data split that contains only the returns data for all stocks.
                    Returns columns must be in the format <ticker>.

            Returns:
                all_alloc_weights (dict[str, dict[str, np.ndarray]]): Portfolio allocation weights for one 
                    model-loss portfolio optimizer for all output windows.
            """

            self._data_check(train_data, rets_test)
            self._trained_check()

            # Extract feature data
            self.reshaper.extract_features(train_data.columns)
            
            # Convert all dataframes to arrays
            train_data, rets_train, test_data, rets_test = self._convert_datasets_to_np(
                train_data, rets_train, test_data, rets_test
            )

            # Search libraries
            model_class = self._search_model(model_name)
            if model_class is None:
                raise KeyError(f'Model {model_name} not found.')
            
            loss_func = self._search_loss_func(loss_name)
            if loss_func is None:
                raise KeyError(f'Loss Function {loss_name} not found.')
            
            print('\n', '-'*10,f' Testing {model_name}-{loss_name}', '-'*10)
            try:        
                alloc_weights, train_eval_losses = self._walker_helper(
                    model_name,
                    model_class, 
                    loss_name,
                    loss_func,
                    train_data,
                    rets_train, 
                    test_data,
                    rets_test
                )
                self.all_alloc_weights[
                    f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
                ] = alloc_weights
                self.train_eval_losses[
                    f'{model_name}{MODEL_LOSS_SEP}{loss_name}'
                ] = train_eval_losses

            except Exception as error:
                print(
                    f'DEBUG: Error while testing {model_name} with {loss_name}. Skipping.',
                    error
                )
                traceback.print_exc()

            return self.all_alloc_weights
    
    def get_train_val_losses(self) -> dict[str, dict[str, list[float]]]:
        """
        Get the train-eval loss curves from training at each walk step. 
        This method reformats the internal dictionary.

        Returns:
            reformatted_dict (dict[str, dict[str, list[float]]]): Dictionary containing the
                train-eval losses at each walk step.
        
        Raises:
            RunTimeError: If models are not yet training and evaluated on the test data
        """
        if self.train_eval_losses:
            reformatted_dict = {}
            
            if self.multi_seed:
                for model_loss, seeds_dict in self.train_eval_losses.items():
                    reformatted_dict.setdefault(model_loss, {})
                    for seed, step_losses in seeds_dict.items():
                        train_losses = []
                        eval_losses = []
                        for step in step_losses:
                            train_losses.append(step['train'])
                            eval_losses.append(step['eval'][0]) # 0 since all evaulation is done on single windows
                        
                        reformatted_dict[model_loss][seed] = {
                            'train': train_losses,
                            'eval': eval_losses
                        }
            else:
                for model_loss, step_losses in self.train_eval_losses.items():
                    train_losses = []
                    eval_losses = []
                    for step in step_losses:
                        train_losses.append(step['train'])
                        eval_losses.append(step['eval'][0])
                    
                    reformatted_dict[model_loss] = {
                        'train': train_losses,
                        'eval': eval_losses
                    }

        else:
            raise RuntimeError('Models not trained yet. Run training first.')
        
        return reformatted_dict