import os
import inspect
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Type, Any
from src.data_processing.preprocess_crsp import preprocessor2
from concurrent.futures import ProcessPoolExecutor, as_completed

class TradModelsTrainer:
    models_hparams = 'trad_models'
    
    def __init__(
            self,
            model_lib: dict[str, Type],
            hparams_config: dict[str, dict[str, Any]],
            max_workers: int
        ):
        self.model_lib = model_lib
        self.hparams_config = hparams_config
        self.max_workers = max_workers if max_workers > 0 else max(1, os.cpu_count() - 1)
        
        self.all_alloc_weights: dict[str, list[pd.Series | np.ndarray]] = {}

    def _train_one_model(
            self, model_name, model_class: Type, filtered_kwargs: dict
        ) -> pd.Series:

        # Get hyperparameters of the current model
        current_hparams = self.hparams_config[self.models_hparams].get(model_name) or {}
        # print('Model hyperparameters:\n', current_hparams)
        
        model_obj = model_class(**current_hparams)
        alloc_weights = model_obj.calculate_weights(**filtered_kwargs)
        return alloc_weights
    
    def _process_train_1_ds(self, returns_is: pd.DataFrame):
        """
        Preprocess one dataset slice, train all models on it and collect all allocation weights
        """
        returns_is_cov, returns_is_corr = preprocessor2(returns_is)
        payload = {
            'cov': returns_is_cov,
            'corr': returns_is_corr,
            'returns': returns_is
        }

        # Loop over every model
        slice_results = {}
        for model_name, model_class in self.model_lib.items():
            
            # print('\n', '-'*10, f' Training {model_name} ', '-'*10)

            # To inspect args of the calculate_weights method and provide it with the relevant args
            sig = inspect.signature(model_class.calculate_weights)

            filtered_kwargs = {
                k: v for k, v in payload.items() 
                if k in sig.parameters
            }

            if len(filtered_kwargs) == 0:
                raise ValueError(f'Required parameters for {model_name} do not exist in payload.')
            try:
                alloc_weights = self._train_one_model(model_name, model_class, filtered_kwargs)

                if isinstance(alloc_weights, pd.Series):
                    slice_results[model_name] = alloc_weights.to_numpy()
                else:
                    slice_results[model_name] = alloc_weights

            except Exception as error:
                print(
                    f'DEBUG: Error while training {model_name}. Skipping.',
                    error
                )
                # slice_results[model_name] = None
                continue
        
        return slice_results
    
    def _stack_weights(self):
        self.all_alloc_weights = {
            name: np.vstack(weights) 
            for name, weights in self.all_alloc_weights.items()
        }
    
    @staticmethod
    def _build_dataset(
        in_sample_idx: tuple[int, int], # (Start, End)
        out_sample_idx: tuple[int, int],
        returns_train: pd.DataFrame, 
        returns_val: pd.DataFrame,
        returns_test: pd.DataFrame | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Dataset builder function for covariance based models (tradional).
        Combines and slices to create in-sample and out-of-sample datasets.
        """
        
        #### If train data must be sliced or shifted, it must be implmented here 
        # after grabbing index from dataset.py
        if returns_test is None:
            returns_is = pd.concat(
                [returns_train, returns_val.iloc[in_sample_idx[0]: in_sample_idx[1]]]
            )
            
            # iloc[200:250] gives rows 200-249 (Exactly 50 rows)
            returns_oos = returns_val.iloc[out_sample_idx[0]: out_sample_idx[1]]
        
        elif returns_test is not None and isinstance(returns_test, pd.DataFrame):
            returns_is = pd.concat(
                [returns_train, returns_val, returns_test.iloc[in_sample_idx[0]: in_sample_idx[1]]]
            )
            returns_oos = returns_test.iloc[out_sample_idx[0]: out_sample_idx[1]]
        else:
            raise ValueError('Incorrect type for test returns.')

        # Sorting in alphabetical order
        return returns_is.sort_index(axis=1), returns_oos.sort_index(axis=1)

    def train_all(
            self,
            in_sample_indexes: list[tuple],
            out_sample_indexes: list[tuple],
            returns_train: pd.DataFrame,
            returns_val: pd.DataFrame,
            returns_test: pd.DataFrame | None = None
        ) -> dict[str, np.ndarray]:
        
        num_slices = len(in_sample_indexes)
        
        # 1. Slice-First
        # Prepare small data packets in the main thread to minimize IPC overhead
        prepared_slices = []
        for i in range(num_slices):
            returns_is, _ = self._build_dataset(
                in_sample_indexes[i],
                out_sample_indexes[i],
                returns_train,
                returns_val,
                returns_test
            )
            prepared_slices.append((i, returns_is))

        # 2. Parallel Execution
        # Pre-allocate to guarantee chronological order
        ordered_results = [None] * num_slices
        

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # We pass the bound method. Python pickles 'self' automatically.
            futures = {
                executor.submit(self._process_train_1_ds, data): idx 
                for idx, data in prepared_slices
            }

            for future in tqdm(
                as_completed(futures),
                total=num_slices,
                desc=f'Training tradional models on {num_slices} slices', unit='slice'
            ):
                idx = futures[future]
                try:
                    # Place result in the correct chronological slot
                    ordered_results[idx] = future.result()
                except Exception as e:
                    print(f"Slice {idx} failed with error: {e}")

        # 3. Synchronous State Update
        # Update self.all_alloc_weights in order
        for slice_dict in ordered_results:
            if slice_dict is None: continue
            for model_name, weights in slice_dict.items():
                self.all_alloc_weights.setdefault(model_name, []).append(weights)

        self._stack_weights()
        return self.all_alloc_weights