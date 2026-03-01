import gc
import os
import time
import copy
import torch
import psutil
import inspect
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils.device import set_seed
from torch.utils.data import DataLoader
from src.data_processing.dataset import build_dataset
from src.data_processing.preprocess_crsp import preprocessor2
from typing import Callable, Type, Any, TYPE_CHECKING, Optional

# For type hints. To avoid circular dependencies
if TYPE_CHECKING:
    from src.data_processing.dataset import WindowDataset

class Trainer:
    """
    Class to train provided models with provided hyperparameters.
    """
    def __init__(
        self, 
        model,  # Model class, not instance
        optimizer: torch.optim.Optimizer,
        loss: Callable,
        model_hparams: dict[str, Any],      # Specific to model architecture
        optimizer_hparams: dict[str, Any],  # Specific to optimizer
        train_hparams: dict[str, Any],      # Generic training params (epochs, batch_size, etc.)
        in_size: int,
        num_stocks: int,
        device: torch.device | str,
        scheduler_hparams: dict[str, Any] | None = None,
        loss_hparams: dict[str, Any] | None = None
    ):
        """
        Initialize Trainer instance to train given model.

        @param model torch.nn.Module
            Pytorch neural network class to be trained and evaluated
        @param optimizer torch.optim
            Pytorch optimization class to be used to loss optimization
        @param loss Callable
            Custom loss function
        @param model_hparams dict
            Dictionary containing hyperparameters required for model initialization
        @param optimizer_hparams dict
            Dictionary containing hyperparameters required for optimizer initialization
        @param train_hparams Dict
            Dictionary containing hyperparameters required for training
        @param loss_hparams dict | None
            Dictionary containing hyperparameters for loss functions. Default = None
        @param in_size int
            Size of input window
        @param num_stocks int
            Number of stocks, i.e, number of output nodes 
        """

        if isinstance(device, torch.device) :
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            raise ValueError(
                'Incorrect type provided for torch device, must be `str` or `torch.device`.'
            )
        
        print('Model hyperparameters:\n', model_hparams)
        print('Optimizer hyperparameters:\n', optimizer_hparams)
        print('Training hyperparameters:\n', train_hparams)
        print('Scheduler hyperparameters:\n', scheduler_hparams)
        print('Loss Function hyperparameters:\n', loss_hparams)
        
        # Initialize model with its specific hyperparameters
        self.model = model(
            input_size=in_size,
            num_stocks=num_stocks,
            **model_hparams  # Unpack all model-specific hyperparams
        ).to(self.device)
        
        # Initialize optimizer with its specific hyperparameters
        self.optimizer = optimizer(
            self.model.parameters(),
            **optimizer_hparams
        )

        if scheduler_hparams:
            # 2. Initialize Scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min',       # We want to minimize loss
                **scheduler_hparams
            )

            self.lr_schedule = True
        else:
            self.lr_schedule = False

        self.loss = loss

        self.train_hparams = train_hparams
        self.loss_hparams = loss_hparams or {}
        
        self.train_losses = [] # Stores average losses, for plotting
        self.val_losses = [] # Stores average losses, for plotting
        
        self.eval_losses = [] # For out of sample eval. Stores batch losses, not average
        
        self.avg_train_loss = None
        self.avg_eval_loss = None

        self.eval_alloc_weights = []

        # For Early Stopping
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
    
    def train(
            self, train_ds: 'WindowDataset', val_ds: Optional['WindowDataset'] = None
        ):
        """
        Train inistalized model using a train data split.

        @param train_ds WindowDataset
            Training data split converted to windowed dataset tensors
        """
        start_time = time.time()

        # Pull hyperparameters with sensible defaults
        patience = self.train_hparams.get('early_stop_patience', 20)
        min_delta = self.train_hparams.get('early_stop_min_delta', 1e-3)
        early_stopping = self.train_hparams.get('early_stopping', True)
        
        train_loader = DataLoader(
            train_ds,
            batch_size=self.train_hparams['train_batch_size'],
            shuffle=True
        )

        for epoch in range(self.train_hparams['epochs']):
            epoch_start = time.time()
            self.model.train()
            total_loss_sum = 0.0
            total_samples = 0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                weights = self.model(xb)  # (B, N)
                loss = self.loss(weights, yb, **self.loss_hparams)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.train_hparams.get('clip_grad_norm', 0.5)
                )
                self.optimizer.step()

                batch_size = xb.size(0)

                total_loss_sum += loss.item() * batch_size
                total_samples += batch_size

            epoch_avg_loss = total_loss_sum / total_samples
            self.train_losses.append(epoch_avg_loss)
            status_msg = f'Epoch {epoch} | Train Loss: {epoch_avg_loss:.4f}'
            self.avg_train_loss = epoch_avg_loss

            # --- Validation & Early Stopping Logic ---
            if val_ds is not None:
                avg_val_loss = self.validate(val_ds)
                self.val_losses.append(avg_val_loss)

                # --- STEP THE SCHEDULER HERE ---
                # It takes the current validation loss to decide if it should drop the LR
                if self.lr_schedule:
                    self.scheduler.step(avg_val_loss)

                if early_stopping:
                    # Check for improvement
                    if avg_val_loss < (self.best_val_loss - min_delta):
                        self.best_val_loss = avg_val_loss
                        self.patience_counter = 0
                        # Deep copy the weights so we can return to this point later
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                    else:
                        self.patience_counter += 1
                    
                    status_msg = status_msg + f' | Val Loss: {avg_val_loss:.4f}'

                    if self.patience_counter >= patience:
                        print(f'\n--- Early Stopping Triggered at Epoch {epoch} ---')
                        # Load the "Best" weights back into the model
                        self.model.load_state_dict(self.best_model_state)
                        break
                        
                else:
                    status_msg = status_msg + f' | Val Loss: {avg_val_loss:.4f}'
            
            print(status_msg + f' | Time: {round(time.time() - epoch_start, 3)}s')
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        print(f'Training Complete. Best Val Loss: {self.best_val_loss:.4f}')
            
        end_time = time.time()
        time_taken = round(end_time - start_time, 3)
        print(f'Average Train Loss: {self.avg_train_loss:.4f}, Time Taken: {time_taken}s')

    def validate(self, val_ds: 'WindowDataset'):
        """
        Validation method to run on each training epoch
        """
        val_loader = DataLoader(
            val_ds,
            batch_size=self.train_hparams['val_batch_size'],
            shuffle=False
        )

        self.model.eval()
        with torch.no_grad():
            total_loss, total_samples = 0.0, 0

            for xb, yb in val_loader:
                b = xb.size(0)
                xb, yb = xb.to(self.device), yb.to(self.device)
                weights = self.model(xb)
                loss = self.loss(weights, yb, **self.loss_hparams)

                # --- store per-batch loss ---
                # self.eval_losses.append(loss.item())

                # --- accumulate weighted sum for overall avg ---
                total_loss += loss.item() * b
                total_samples += b

            # --- weighted average over all samples ---
            avg_val_loss = total_loss / total_samples
        
        return avg_val_loss


    def evaluate(self, split_ds: 'WindowDataset'):
        """
        Evaluate the trained model using a validation data split.
        
        @param val_ds WindowDataset
            Validation data split converted to windowed dataset tensors
        """
        start_time = time.time()
        eval_loader = DataLoader(
            split_ds,
            # batch_size=self.train_hparams['val_batch_size'],
            batch_size=1, # To iterate over each evaluation sample
            shuffle=False
        )

        # --- evaluation ---
        self.model.eval()
        with torch.no_grad():
            self.eval_losses = []
            total_loss, total_samples = 0.0, 0

            for xb, yb in eval_loader:
                b = xb.size(0)
                xb, yb = xb.to(self.device), yb.to(self.device)
                weights = self.model(xb)
                loss = self.loss(weights, yb, **self.loss_hparams)

                # detach & move to CPU BEFORE appending
                self.eval_alloc_weights.append(weights.detach().cpu()) 

                # --- store per-batch loss ---
                batch_loss = loss.item()
                self.eval_losses.append(batch_loss)

                # --- accumulate weighted sum for overall avg ---
                total_loss += batch_loss * b
                total_samples += b

            # --- weighted average over all samples ---
            self.avg_eval_loss = total_loss / total_samples
        
        end_time = time.time()
        time_taken = round(end_time-start_time, 3)
        print(f'Average Eval Loss: {self.avg_eval_loss:.4f}, Time Taken: {time_taken}')

    def get_eval_alloc_weights(self) -> np.ndarray:
        """
        Getter for allocation weights as numpy array
        
        @return np.ndarray Portfolio allocation weights for each validation window
        """
        if self.eval_alloc_weights:
            wt_array = []
            for w in self.eval_alloc_weights:
                wt_array.append(w.numpy())
            return np.vstack(wt_array)
        else:
            print('Model must be trained and validated.')
            return None
    
    # def device_cleanup(self):
    #     if self.device_name == 'mps':
    #         try:
    #             # Empty MPS cache
    #             torch.mps.empty_cache()
            
    #         except Exception as e:
    #             print(f'MPS cleanup not available. Error: {e}')
            
    #     elif self.device_name == 'cuda':
    #         torch.cuda.empty_cache()
    #         torch.cuda.ipc_collect()
        

class CandidatesGrid:
    models_hparams = 'nn_models'
    losses_hparams = 'losses'
    
    def __init__(
            self,
            model_lib: dict[str, dict[str, Type]],
            loss_lib: dict[str, dict[str, dict[str, Callable]]],
            hparams_config: dict[str, dict[str, Any]],
            torch_device: torch.device | str,
            loss_mode: str = 'all',
            enable_diagnostics: bool = False
        ):
        """
        This runs either all models and loss functions or 
        one model with all loss functions or all models with on loss function.
        It should not run one model for one loss function as this class is for a grid.

        train_eval_* methods can be run only once with each instance
        """
        self.model_lib = model_lib
        self.loss_lib = loss_lib
        self.hparams_config = hparams_config

        if loss_mode not in ['all', 'custom']:
            raise ValueError('Incorrect Loss Mode. Mode must be `all` or `custom`')
        else:
            self.loss_mode = loss_mode
        self.enable_diagnostics = enable_diagnostics

        self.torch_device = torch_device
        
        self.all_alloc_weights: dict[str, dict[str, np.ndarray]] = {}
        self.train_val_losses: dict[str, dict[str, list[float]]] = {}
    
    def required_reshapes(self, train_data, returns_train, val_data, returns_val):
        # Implement different reshaping for different models if needed.
        # Move Reshaper instance from pipeline.py to here
        pass

    def _train_eval_helper(
            self,
            model_name: str,
            model_class: Type,
            loss_name: str,
            loss_func: Callable,
            train_ds: 'WindowDataset',
            val_ds: 'WindowDataset',
            X_train_shape: torch.Size,
            y_train_shape: torch.Size
        ) -> np.ndarray:
        #### Hyperparamater searching can be done here ####

        set_seed(50) # Per model seed for fair comparison
        if self.enable_diagnostics:
            print(f'\n[Before training {model_name} with {loss_name}]')
            self._memory_diagnostics()

        trainer = Trainer(
            model=model_class,
            optimizer=torch.optim.AdamW,
            loss=loss_func,
            model_hparams=self.hparams_config[
                self.models_hparams
            ][model_name]['model'],
            optimizer_hparams=self.hparams_config[
                self.models_hparams
            ][model_name]['optimizer'],
            train_hparams=self.hparams_config[
                self.models_hparams
            ][model_name]['train'],
            in_size=X_train_shape[2],
            num_stocks=y_train_shape[2],
            scheduler_hparams=self.hparams_config[
                self.models_hparams
            ][model_name]['scheduler'],
            loss_hparams=self.hparams_config[self.losses_hparams].get(loss_name),
            device=self.torch_device
        )
        trainer.train(train_ds, val_ds)
        trainer.evaluate(val_ds)

        # To send all loss curves back to pipeline
        self.train_val_losses[f'{model_name}-{loss_name}'] = {
            'train': trainer.train_losses,
            'val': trainer.val_losses,
            'eval':trainer.eval_losses
        }

        alloc_weights = trainer.get_eval_alloc_weights()
        # trainer.device_cleanup()
        del trainer

        if self.enable_diagnostics:
            print(f'\n[After training {model_name} with {loss_name}]')
            self._memory_diagnostics()
        
        return alloc_weights
    
    def _memory_diagnostics(self):
        """Print memory usage diagnostics"""
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / 1024 ** 3
        
        print(f"  Process memory: {mem_gb:.2f} GB")
        
        # if DEVICE == 'cuda':
        #     print(f"  GPU allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        #     print(f"  GPU cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Count Trainer instances
        trainer_count = 0
        for obj in gc.get_objects():
            if type(obj).__name__ == 'Trainer':
                trainer_count += 1
        
        if trainer_count > 0:
            print(f"  WARNING: {trainer_count} Trainer instances still in memory!")

    def _trained_check(self):
        """
        Check if training has been run before. 
        New instance of the class must be created for every training grid.
        """
        if len(self.all_alloc_weights) != 0:
            raise RuntimeError(
                'Allocation weights already predicted. Create new instance of this class.'
            )

    def _calc_total_models(self, grid_mode: str) -> int:
        """
        Calculate total number of models to be trained based on the grid and loss modes
        """
        if grid_mode == 'all':
            len_custom_losses = len(
                self.loss_lib['custom']['__default__']
            ) if 'custom' in self.loss_lib else 0
            
            if self.loss_mode == 'custom':
                len_losses = len_custom_losses
            else:
                len_losses = len(self.loss_lib['objectives']['__default__']) + \
                    len_custom_losses
            
            total_train_count = (
                    len_losses
                ) * sum(len(models_dict) for models_dict in self.model_lib.values())
        
        elif grid_mode == 'one_model':
            len_custom_losses = len(
                self.loss_lib['custom']['__default__']
            ) if 'custom' in self.loss_lib else 0
            
            if self.loss_mode == 'custom':
                total_train_count = len_custom_losses
            else:
                total_train_count = (
                    len(self.loss_lib['objectives']['__default__']) +  len_custom_losses
                )
        elif grid_mode == 'one_loss':
            total_train_count = sum(
                len(models_dict) for models_dict in self.model_lib.values()
            )

        else:
            print('Incorrect usage of `calc_total_models` method!')
            total_train_count = 0
        
        return total_train_count

    def train_eval_grid(
            self, train_ds: 'WindowDataset', val_ds: 'WindowDataset'
        ) -> dict[str, dict[str, np.ndarray]]:
        """Loops over Loss functions first with a nested loop for models"""
        self._trained_check()

        X_train_shape, y_train_shape = train_ds.get_X_y_shapes()
        
        total_train_count = self._calc_total_models('all')
        print(f'\nTraining {total_train_count} models.')
        progress_count = 1
        
        # Grid with custom loss functions
        if 'custom' in self.loss_lib:            
            print('Training all models with all custom loss functions...')

            custom_combos = self.loss_lib['custom']['__default__'] # Custom combos have no category

            # Loop over loss functions
            for loss_name, loss_func in custom_combos.items():
                self.all_alloc_weights.setdefault(loss_name, {})

                for category, models_dict in self.model_lib.items():
                    # Loop over models
                    for model_name, model_class in models_dict.items():
                        print(
                            '\n', '-'*10,
                            f' Training {model_name} - {loss_name}, {progress_count}/{total_train_count}',
                            '-'*10
                        )
                        try: 
                            
                            alloc_weights = self._train_eval_helper(
                                model_name,
                                model_class, 
                                loss_name,
                                loss_func,
                                train_ds,
                                val_ds,
                                X_train_shape,
                                y_train_shape
                            )
                            self.all_alloc_weights[loss_name][model_name] = alloc_weights

                        except Exception as error:
                            print(
                                f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.', error
                            )
                            continue
                        finally:
                            progress_count += 1
        
        else:
            print('\nNo custom loss functions provided. Moving to objectives.')
        
        if self.loss_mode == 'custom':
            return self.all_alloc_weights

        else: # mode == 'all'
            # Grid with only objectives
            print('\nTraining all models with all objectives (only) as loss functions...')
            objectives = self.loss_lib['objectives']['__default__'] # objectives have no category
            
            # Loop over loss functions
            for loss_name, loss_func in objectives.items():
                self.all_alloc_weights.setdefault(loss_name, {})
                
                for category, models_dict in self.model_lib.items():
                    # Loop over models
                    for model_name, model_class in models_dict.items():
                        print(
                            '\n', '-'*10,
                              f' Training {model_name} - {loss_name}, {progress_count}/{total_train_count}',
                              '-'*10
                        )
                        
                        try: 
                            
                            alloc_weights = self._train_eval_helper(
                                model_name,
                                model_class, 
                                loss_name,
                                loss_func,
                                train_ds,
                                val_ds,
                                X_train_shape,
                                y_train_shape
                            )
                            self.all_alloc_weights[loss_name][model_name] = alloc_weights

                        except Exception as error:
                            print(
                                f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.',
                                error
                            )
                            continue
                        finally:
                            progress_count += 1
                
            return self.all_alloc_weights
    
    def _search_model(self, model_name: str) -> Type | None:
        """Search for required model"""
        for _, models_dict in self.model_lib.items():
            if model_name in models_dict:
                return models_dict[model_name]
        return None

    def train_eval_one_model(
            self, model_name: str, train_ds: 'WindowDataset', val_ds: 'WindowDataset'
        ) -> dict[str, dict[str, np.ndarray]]:

        self._trained_check()
        
        # Search for model
        model_class = self._search_model(model_name)
        if not model_class: # model not found
            raise RuntimeError(f'{model_name} MODEL NOT FOUND IN LIBRARY!')
        
        X_train_shape, y_train_shape = train_ds.get_X_y_shapes()

        total_train_count = self._calc_total_models('one_model')
        print(f'\nTraining {total_train_count} models.')
        progress_count = 1
        
        # Grid with custom loss functions
        if 'custom' in self.loss_lib:
            print('\nTraining all models with all custom loss functions...')
            custom_combos = self.loss_lib['custom']['__default__'] # Custom combos have no category

            # Loop over loss functions
            for loss_name, loss_func in custom_combos.items():
                self.all_alloc_weights.setdefault(loss_name, {})

                print(
                    '\n', '-'*10,
                    f' Training {model_name} - {loss_name}, {progress_count}/{total_train_count}',
                    '-'*10
                )
                try:        
                    alloc_weights = self._train_eval_helper(
                        model_name,
                        model_class, 
                        loss_name,
                        loss_func,
                        train_ds,
                        val_ds,
                        X_train_shape,
                        y_train_shape
                    )
                    self.all_alloc_weights[loss_name][model_name] = alloc_weights

                except Exception as error:
                    print(
                        f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.',
                        error
                    )
                    continue
                finally:
                    progress_count += 1

        else:
            print('\nNo custom loss functions provided. Moving to objectives.')
        
        if self.loss_mode == 'custom':
            return self.all_alloc_weights

        else: # mode == 'all'
            # Grid with only objectives
            print('\nTraining all models with all objectives (only) as loss functions...')
            objectives = self.loss_lib['objectives']['__default__'] # objectives have no category
            
            # Loop over loss functions
            for loss_name, loss_func in objectives.items():
                self.all_alloc_weights.setdefault(loss_name, {})
                print(
                    '\n', '-'*10,
                    f' Training {model_name} - {loss_name}, {progress_count}/{total_train_count}',
                    '-'*10
                )
                try:        
                    alloc_weights = self._train_eval_helper(
                        model_name,
                        model_class, 
                        loss_name,
                        loss_func,
                        train_ds,
                        val_ds,
                        X_train_shape,
                        y_train_shape
                    )
                    self.all_alloc_weights[loss_name][model_name] = alloc_weights

                except Exception as error:
                    print(
                        f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.',
                        error
                    )
                    continue
                finally:
                    progress_count += 1
            
            return self.all_alloc_weights

    def _search_loss_func(self, loss_name: str) -> Callable | None:
        """Search for required loss function"""
        for _, cat_dict in self.loss_lib.items():
            for _, sub_cat_dict in cat_dict.items():
                if loss_name in sub_cat_dict:
                    return sub_cat_dict[loss_name]
        
        # objectives = self.loss_lib['objectives']['__default__']
        # if loss_name in objectives:
        #     return objectives[loss_name]
        
        # custom_combos = self.loss_lib['custom']['__default__']
        # if loss_name in custom_combos:
        #     return custom_combos[loss_name]
        
        return None

    def train_eval_one_loss(
            self, loss_name: str, train_ds: 'WindowDataset', val_ds: 'WindowDataset'
        ) -> dict[str, dict[str, np.ndarray]]:
        
        self._trained_check()
    
        loss_func = self._search_loss_func(loss_name)
        
        if not loss_func: # loss function not found
            raise RuntimeError(f'{loss_name} LOSS FUNCTION NOT FOUND IN LIBRARY!')

        self.all_alloc_weights.setdefault(loss_name, {})

        X_train_shape, y_train_shape = train_ds.get_X_y_shapes()

        total_train_count = self._calc_total_models('one_loss')
        print(f'\nTraining {total_train_count} models.')
        progress_count = 1
        
        for category, models_dict in self.model_lib.items():
            # Loop over models
            for model_name, model_class in models_dict.items():
                print(
                    '\n', '-'*10,
                    f' Training {model_name} - {loss_name}, {progress_count}/{total_train_count}',
                    '-'*10
                )
                try: 
                    
                    alloc_weights = self._train_eval_helper(
                        model_name,
                        model_class, 
                        loss_name,
                        loss_func,
                        train_ds,
                        val_ds,
                        X_train_shape,
                        y_train_shape
                    )
                    self.all_alloc_weights[loss_name][model_name] = alloc_weights

                except Exception as error:
                    print(
                        f'DEBUG: Error while training {model_name} with {loss_name}. Skipping.',
                        error
                    )
                    continue
                finally:
                    progress_count += 1

        return self.all_alloc_weights
    
    def get_train_val_losses(self) -> dict[str, dict[str, list[float]]]:
        return self.train_val_losses


class TradModelsTrainer:
    models_hparams = 'trad_models'
    
    def __init__(
            self, model_lib: dict[str, Type], hparams_config: dict[str, dict[str, Any]]
        ):
        self.model_lib = model_lib
        self.hparams_config = hparams_config
        
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
        for model_name, model_class in self.model_lib.items():
            
            # print('\n', '-'*10, f' Training {model_name} ', '-'*10)
            
            self.all_alloc_weights.setdefault(model_name, [])

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
                    self.all_alloc_weights[model_name].append(alloc_weights.to_numpy())
                else:
                    self.all_alloc_weights[model_name].append(alloc_weights)

            except Exception as error:
                print(
                    f'DEBUG: Error while training {model_name}. Skipping.',
                    error
                )
                continue
    
    def _stack_weights(self):
        self.all_alloc_weights = {
            name: np.vstack(weights) 
            for name, weights in self.all_alloc_weights.items()
        }
    
    def train_all(
            self,
            in_sample_indexes: list[tuple],
            out_sample_indexes: list[tuple],
            returns_train: pd.DataFrame,
            returns_val: pd.DataFrame,
            returns_test: pd.DataFrame | None = None
        ) -> dict[str, list[pd.Series | np.ndarray]]:
        
        num_slices = len(in_sample_indexes)
        if returns_test is None: # To use Validation Set (Combines Train + in-sample Val)
            # Calculate indexes for in-sample and out-of-sample to match the neural networks

            # Loop over dataset slices
            for i in tqdm(
                range(num_slices),
                desc=f'Training tradional models on {num_slices} slices',
                unit='slice'
            ): # len(in-sample) = len(out-of-sample)
                returns_is, _ = build_dataset(
                    in_sample_indexes[i],
                    out_sample_indexes[i],
                    returns_train,
                    returns_val
                )
                
                self._process_train_1_ds(returns_is)     
        
        else: # To use Test Set (Combines Train + Val + in-sample Test)

            for i in tqdm(
                range(num_slices),
                desc=f'Training tradional models on {num_slices} slices',
                unit='slice'
            ):
                returns_is, _ = build_dataset(
                    in_sample_indexes[i],
                    out_sample_indexes[i],
                    returns_train,
                    returns_val,
                    returns_test
                )

                self._process_train_1_ds(returns_is)
        self._stack_weights()
        return self.all_alloc_weights