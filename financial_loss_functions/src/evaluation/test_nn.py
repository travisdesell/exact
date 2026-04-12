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
    def __init__(
            self,
            model_lib: dict[str, dict[str, Type]],
            loss_lib: dict[str, dict[str, dict[str, Callable]]],
            hparams_config: dict[str, dict[str, Any]],
            num_steps: int,
            common_features: list[str],
            torch_device: torch.device | str,
            mpi: bool = False,
            temp_dir: str | None = None,
            optimized_hparams: dict = None,
            enable_diagnostics: bool = False
        ):

        super().__init__(
            model_lib, loss_lib, hparams_config, num_steps, 
            common_features, torch_device, mpi, temp_dir
        )

        self.optimized_hparams = optimized_hparams
        self.enable_diagnostics = enable_diagnostics
    
    def _merge_all_results(
            self,
            size,
            temp_wts_prefix,
            temp_losses_prefix
        ):
        """Merge all results into one dict if rank is 0, i.e., main process."""
        self.all_alloc_weights = {}
        self.train_infer_losses = {}
                
        for r in range(size):
            # Load all temp alloc wt files
            rank_alloc_weights = load_pickle_temp(
                self.temp_dir / f'{temp_wts_prefix}_{r}.pkl'
            )
            # Merge into self.all_alloc_weights
            for model_loss, models_dict in rank_alloc_weights.items():
                self.all_alloc_weights[model_loss] = models_dict
            
            rank_losses = load_pickle_temp(
                self.temp_dir / f'{temp_losses_prefix}_{r}.pkl'
            )
            # Merge into self.train_val_losses
            for model_loss, losses_dict in rank_losses.items():
                self.train_infer_losses[model_loss] = losses_dict

        # Delete all temp files
        for r in range(size):
            delete_file(self.temp_dir / f'{temp_wts_prefix}_{r}.pkl')
            delete_file(self.temp_dir / f'{temp_losses_prefix}_{r}.pkl')

        print('All temp files merged and then deleted.')

    def _build_combos(
            self, selected_combos: list[tuple[str, str]]
        ) -> list[tuple[str, Type, str, Callable]]:
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
    ) -> tuple[np.ndarray, dict[str, list], dict | None]:
        
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
        # combo_hparams['train']['epochs'] = 5
        ####################
        final_walker = Walker(
            self.num_steps,
            model_name,
            model_class,
            loss_name,
            loss_func,
            combo_hparams,
            self.torch_device,
            self.reshaper
        )

        alloc_weights = final_walker.walk_1_model(
            train_data,
            rets_train, 
            test_data,
            rets_test
        )

        train_val_losses = final_walker.get_train_eval_losses()
        
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
        ) -> dict[str, dict[str, np.ndarray]]:

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
            raise RuntimeError('MPI VERSION NOT IMPLEMENTED YET! EXITING...')
    
    def test_one(
            self,
            model_name: str,
            loss_name: str,
            train_data: pd.DataFrame,
            rets_train: pd.DataFrame, 
            test_data: pd.DataFrame,
            rets_test: pd.DataFrame,
        ) -> dict[str, dict[str, np.ndarray]]:

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